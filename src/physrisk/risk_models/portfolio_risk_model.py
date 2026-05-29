from collections import defaultdict
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import scipy.interpolate

from physrisk.api.v1.common import Asset as APIAsset
from physrisk.api.v1.impact_req_resp import (
    RiskMeasureDefinition,
    ScoreBasedRiskMeasureDefinition,
)
from physrisk.api.v1.scoring_schemes import Category
from physrisk.kernel.assets import Asset
from physrisk.kernel.financial_model import DefaultFinancialModel, FinancialDataProvider
from physrisk.kernel.hazards import Hazard
from physrisk.kernel.impact import AssetImpactResult, ImpactKey
from physrisk.kernel.impact_aggregator import aggregate_impacts
from physrisk.kernel.risk import (
    Measure,
    MeasureKey,
    PortfolioRiskMeasureCalculator,
    Quantity,
    QuantityType,
    RiskQuantityKey,
)


class MissingData(str, Enum):
    NO_MISSING = "no_missing"
    FILL_WITH_MEAN = "fill_with_mean"


class FinancialDataStore(FinancialDataProvider):
    def __init__(
        self,
        assets: Sequence[APIAsset],
        missing_data_strategy: MissingData = MissingData.FILL_WITH_MEAN,
    ):
        self.data = {
            asset.id: asset.financial for asset in assets if asset.financial is not None
        }
        revenue = [
            d.revenue_attributable
            for d in self.data.values()
            if d.revenue_attributable is not None
        ]
        insurable_value = [
            d.total_insurable_value
            for d in self.data.values()
            if d.total_insurable_value is not None
        ]
        if missing_data_strategy == MissingData.NO_MISSING:
            if len(revenue) < len(assets):
                raise ValueError(
                    "Missing revenue data for some assets and missing_data_strategy is set to NO_MISSING."
                )
            if len(insurable_value) < len(assets):
                raise ValueError(
                    "Missing insurable value data for some assets and missing_data_strategy is set to NO_MISSING."
                )
        # require single currency across all assets for simplicity. In case of missing data, fill with mean value across assets.
        currencies = set(d.ccy for d in self.data.values())
        if len(currencies) > 1:
            raise ValueError(
                "Multiple currencies found in financial data; not supported."
            )

        self.mean_revenue = (sum(revenue) / len(revenue)) if revenue else 100.0
        self.mean_insurable_value = (
            (sum(insurable_value) / len(insurable_value)) if insurable_value else 100.0
        )
        self.currency = currencies.pop() if currencies else "EUR"

    def revenue_attributable_to_asset(self, asset: Asset, currency: str) -> float:
        if asset.id is None:
            raise ValueError("Asset id is required to retrieve financial details.")
        financial_details = self.data.get(asset.id)
        if financial_details is None or financial_details.revenue_attributable is None:
            return self.mean_revenue
        if financial_details.ccy != currency:
            raise ValueError(
                f"Currency mismatch for asset with id {asset.id}: expected {currency}, got {financial_details.ccy}"
            )
        return financial_details.revenue_attributable

    def total_insurable_value(self, asset: Asset, currency: str) -> float:
        if asset.id is None:
            raise ValueError("Asset id is required to retrieve financial details.")
        financial_details = self.data.get(asset.id)
        if financial_details is None or financial_details.total_insurable_value is None:
            return self.mean_insurable_value
        if financial_details.ccy != currency:
            raise ValueError(
                f"Currency mismatch for asset with id {asset.id}: expected {currency}, got {financial_details.ccy}"
            )
        return financial_details.total_insurable_value


class CompanyRiskMeasureCalculator(PortfolioRiskMeasureCalculator):
    """Impacts can be:
    1) asset damage
    2) business disruption, comprising
        - impact on revenue and
        - increase in costs.
    Impacts specified in an ImpactDistrib are all relative quantities.
    Asset damage is specified as a fraction of the asset total insurable value.
    Revenue decrease is specified as a fraction of revenue (attributed to the asset in question).
    Cost increases are specified as a fraction of revenue.
    Financial data is applied to the relative quantities and a Monte Carlo-based approach is used
    to aggregate over assets and hazards.
    Finally, scores are assigned based on the aggregate quantities.
    """

    def __init__(self,
                n_events: int = 50000,
                event_batch_sz: int = 1000):
        self._n_events = n_events
        self._event_batch_sz = event_batch_sz
        self._definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[],
            values=[],
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="portfolio_level_score",
                    label="Aggregated damage/business disruption score.",
                    description=(
                        "Portfolio level score inferred from aggregated damage and business disruption."
                    ),
                    units="",
                )
            ],
        )

    def get_definition(self, hazard_type: Optional[type[Hazard]] = None):
        return self._definition

    def calculate_risk_measures(
        self,
        financial_data_provider: FinancialDataProvider,
        asset_level_measures: dict[MeasureKey, Measure] = {},
        impacts: dict[ImpactKey, list[AssetImpactResult]] = {},
    ) -> tuple[dict[MeasureKey, Measure], dict[tuple[str, int | None], dict[RiskQuantityKey, Quantity]]]:
        financial_model = DefaultFinancialModel(
            financial_data_provider, downtime_config=[]
        )
        impacts_by_year_scen: dict[tuple[str, int | None], list[MeasureKey]] = (
            defaultdict(list)
        )
        for mk in asset_level_measures.keys():
            impacts_by_year_scen[(mk.scenario, mk.year)].append(mk)
        measures: dict[MeasureKey, Measure] = {}
        all_portfolio_quantities: dict[tuple[str, int | None], dict[RiskQuantityKey, Quantity]] = {}
        for scenario, year in impacts_by_year_scen.keys():
            portfolio_quantities = aggregate_impacts(
                impacts, financial_model, scenario, year, n_events=self._n_events, event_batch_sz=self._event_batch_sz
            )
            all_portfolio_quantities[(scenario, year)] = portfolio_quantities
            damage, revenue_loss, costs_increase = (
                portfolio_quantities[RiskQuantityKey(quantity=qt)]
                for qt in [
                    QuantityType.DAMAGE,
                    QuantityType.REVENUE_LOSS,
                    QuantityType.COSTS_INCREASE,
                ]
            )
            score = self.calculate_scores(
                damage.values, revenue_loss.values, costs_increase.values
            )
            measures[MeasureKey(None, scenario, year, None, None)] = Measure(
                score=Category(round(score)),
                measure_0=score,
                definition=self._definition,
            )
        return measures, all_portfolio_quantities

    def asset_level_measures_required(self) -> bool:
        return True

    def portfolio_quantities_required(self) -> bool:
        return False

    def calculate_scores(
        self, damage: np.ndarray, revenue_loss: np.ndarray, costs_increase: np.ndarray
    ):
        # very simple model
        # take as a parameter EBITDA / revenue = 0.2
        # EBITA shock: decrease in EBITDA as fraction of EBITDA
        # assume shock is equal to revenue loss as fraction of revenue
        damage_shock = np.array([0, 0.1, 0.2, 0.5, 1.0])
        ebitda_shock = np.array([0, 0.1, 0.2, 0.5, 1.0])
        score_matrix = np.array(
            [
                [0, 0.5, 1, 2, 4],
                [0.5, 1, 1.5, 2, 4],
                [1, 1.5, 1.5, 3, 4],
                [2, 2, 3, 3, 4],
                [4, 4, 4, 4, 4],
            ]
        )
        ebitda = (
            revenue_loss + costs_increase * 5
        )  # estimate for EBITDA as a fraction of revenue
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (damage_shock, ebitda_shock), score_matrix
        )
        scores = interpolator(np.stack([damage, ebitda], axis=1))
        final_score = np.quantile(scores, 0.99)
        return final_score
