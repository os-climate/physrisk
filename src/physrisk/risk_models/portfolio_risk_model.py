from collections import defaultdict
import statistics
from typing import Optional

from physrisk.api.v1.impact_req_resp import (
    RiskMeasureDefinition,
    ScoreBasedRiskMeasureDefinition,
)
from physrisk.api.v1.scoring_schemes import Category
from physrisk.kernel.financial_model import DefaultFinancialModel
from physrisk.kernel.hazards import Hazard
from physrisk.kernel.impact import AssetImpactResult, ImpactKey
from physrisk.kernel.impact_aggregator import aggregate_impacts
from physrisk.kernel.risk import Measure, MeasureKey, PortfolioRiskMeasureCalculator, Quantity, RiskQuantityKey


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

    def __init__(self):
        self._definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[],
            values=[],
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="portfolio_level_score",
                    label="Aggregated damage/business disruption score.",
                    description=("Portfolio level score inferred from aggregated damage and business disruption."),
                    units="",
                )
            ],
        )

    def get_definition(self, hazard_type: Optional[type[Hazard]] = None):
        return self._definition

    def calculate_risk_measures(
        self,
        asset_level_measures: dict[MeasureKey, Measure] = {},
        impacts: dict[ImpactKey, list[AssetImpactResult]] = {},
    ) -> dict[MeasureKey, Measure]:
        financial_data_provider = 
        financial_model = DefaultFinancialModel()
        aggregated_impacts = aggregate_impacts(impacts, )


    def asset_level_measures_required(self) -> bool:
        return True

    def portfolio_quantities_required(self) -> bool:
        return False
