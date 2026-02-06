from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
import numpy.typing as npt

from physrisk.api.v1.impact_req_resp import Category, ScoreBasedRiskMeasureDefinition
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import HazardModel
from physrisk.kernel.hazards import Hazard
from physrisk.kernel.impact import AssetImpactResult, ImpactKey, calculate_impacts
from physrisk.kernel.vulnerability_model import VulnerabilityModels

# from asyncio import ALL_COMPLETED
# import concurrent.futures


Impact = Dict[Tuple[Asset, type], AssetImpactResult]  # the key is (Asset, Hazard type)


class BatchId(NamedTuple):
    scenario: str
    key_year: Optional[int]


class QuantityType(str, Enum):
    DAMAGE = "damage"
    REVENUE_LOSS = "revenue_loss"
    COSTS = "costs"
    TIV = "tiv"
    REVENUE = "revenue"


class RiskQuantityKey(NamedTuple):
    quantity_type: Optional[QuantityType] = None
    agg_id: Optional[str] = None
    hazard_type: Optional[str] = None


class RiskQuantity(NamedTuple):
    quantity_type: Optional[QuantityType] = None
    agg_id: Optional[str] = None
    hazard_type: Optional[str] = None


@dataclass(frozen=True)
class Quantity:
    values: npt.NDArray[np.floating[Any]]
    percentiles: npt.NDArray[np.floating[Any]]
    percentile_values: npt.NDArray[np.floating[Any]]
    mean: float


class MeasureKey(NamedTuple):
    asset: Optional[Asset]
    prosp_scen: str  # prospective scenario
    year: Optional[int]
    hazard_type: Optional[Type[Hazard]]


@dataclass
class Measure:
    score: Category
    measure_0: float
    definition: ScoreBasedRiskMeasureDefinition  # reference to single instance of ScoreBasedRiskMeasureDefinition


class PortfolioRiskMeasureCalculator(Protocol):
    """Class to calculate portfolio-level score-based risk measures, either
    from a set of asset-level score-based risk measures or from portfolio-level
    """

    def get_definition(
        self, hazard_type: Optional[type[Hazard]] = None
    ) -> ScoreBasedRiskMeasureDefinition: ...

    def calculate_risk_measures(
        self,
        asset_level_measures: dict[MeasureKey, Measure] = {},
        portfolio_quantities: dict[RiskQuantityKey, RiskQuantity] = {},
    ) -> dict[MeasureKey, Measure]: ...

    def asset_level_measures_required(self) -> bool: ...

    def portfolio_quantities_required(self) -> bool: ...


class NullAssetBasedPortfolioRiskMeasureCalculator(PortfolioRiskMeasureCalculator):
    """Calculates portfolio score-based risk measures from asset-level score-based risk measures only."""

    def get_definition(self, hazard_type: Optional[Type[Hazard]] = None):
        return ScoreBasedRiskMeasureDefinition(
            hazard_types=[], values=[], underlying_measures=[]
        )

    def calculate_risk_measures(
        self,
        asset_level_measures: dict[MeasureKey, Measure] = {},
        portfolio_quantities: dict[RiskQuantityKey, RiskQuantity] = {},
    ) -> dict[MeasureKey, Measure]:
        return {}

    def asset_level_measures_required(self) -> bool:
        return True

    def portfolio_quantities_required(self) -> bool:
        return False


class RiskModel:
    """Base class for a risk model (i.e. a calculation of risk that makes use of hazard and vulnerability
    models)."""

    def __init__(
        self, hazard_model: HazardModel, vulnerability_models: VulnerabilityModels
    ):
        self._hazard_model = hazard_model
        self._vulnerability_models = vulnerability_models

    def calculate_risk_measures(
        self, assets: Sequence[Asset], prosp_scens: Sequence[str], years: Sequence[int]
    ): ...

    def _calculate_all_impacts(
        self,
        assets: Sequence[Asset],
        prosp_scens: Sequence[str],
        years: Sequence[int],
        include_histo: bool = False,
    ):
        # ensure "historical" is present, e.g. needed for risk measures
        scenarios = list(
            set(["historical"] + list(prosp_scens)) if include_histo else prosp_scens
        )
        impact_results = calculate_impacts(
            assets,
            self._hazard_model,
            self._vulnerability_models,
            scenarios=scenarios,
            years=years,
        )
        return impact_results


class RiskMeasureCalculator(Protocol):
    def calc_measure(
        self,
        hazard_type: Type[Hazard],
        base_impact: Sequence[AssetImpactResult],
        impact: Sequence[AssetImpactResult],
    ) -> Optional[Measure]:
        """Calculate the Measure (score-based risk measure) for the hazard,
        given the base (i.e. historical) and future asset-level impacts. Most often
        there may be a single impact for a given type of hazard, but in general
        there can be multiple corresponding to different vulnerability models.

        Args:
            hazard_type (Type[Hazard]): Hazard type.
            base_impacts (AssetImpactResult): Historical asset-level impacts.
            impacts (AssetImpactResult): Future asset-level impacts.

        Returns:
            Optional[Measure]: Score-based risk measure.
        """
        ...

    def get_definition(
        self, hazard_type: Type[Hazard]
    ) -> ScoreBasedRiskMeasureDefinition: ...

    def supported_hazards(self) -> Set[type]: ...

    def aggregate_risk_measures(
        self,
        measures: Dict[MeasureKey, Measure],
        assets: Sequence[Asset],
        prosp_scens: Sequence[str],
        years: Sequence[int],
    ) -> Dict[MeasureKey, Measure]:
        """The RiskMeasureCalculator can aggregate child hazards into parent hazards
        or proxy one hazard to another. If no aggregation or proxying is needed, the measures
        input is returned unchanged.

        Args:
            measures (Dict[MeasureKey, Measure]): Score-based risk measures.
            prosp_scens (Sequence[str]): Requested prospective scenarios.
            years (Sequence[int]): Requested prospective years.

        Returns:
            Dict[MeasureKey, Measure]: Aggregated or proxied score-based risk measures.
        """
        return measures


class RiskMeasuresFactory(Protocol):
    def asset_calculators(
        self, use_case_id: str
    ) -> Dict[Type[Asset], RiskMeasureCalculator]:
        pass

    def portfolio_calculator(self, use_case_id: str) -> PortfolioRiskMeasureCalculator:
        pass


class AssetLevelRiskModel(RiskModel):
    def __init__(
        self,
        hazard_model: HazardModel,
        vulnerability_models: VulnerabilityModels,
        measure_calculators: Dict[type[Asset], RiskMeasureCalculator],
        portfolio_measure_calculator: PortfolioRiskMeasureCalculator = NullAssetBasedPortfolioRiskMeasureCalculator(),
    ):
        """Risk model that calculates risk measures at the asset level for a sequence
        of assets.

        Args:
            hazard_model (HazardModel): The hazard model.
            vulnerability_models (Dict[type, Sequence[VulnerabilityModelBase]]): Vulnerability models for asset types.
            measure_calculators (Dict[type, RiskMeasureCalculator]): Risk measure calculators for asset types.
        """
        super().__init__(hazard_model, vulnerability_models)
        self.asset_level_measures_required = (
            portfolio_measure_calculator.asset_level_measures_required
        )
        self.portfolio_quantities_required = (
            portfolio_measure_calculator.portfolio_quantities_required
        )
        self._asset_level_measure_calculators = measure_calculators
        self._portfolio_measure_calculator = portfolio_measure_calculator

    def calculate_impacts(
        self, assets: Sequence[Asset], prosp_scens: Sequence[str], years: Sequence[int]
    ):
        impacts = self._calculate_all_impacts(assets, prosp_scens, years)
        return impacts

    def populate_measure_definitions(
        self, assets: Sequence[Asset]
    ) -> Tuple[
        Dict[Type[Hazard], List[str]], Dict[ScoreBasedRiskMeasureDefinition, str]
    ]:
        # the identifiers of the score-based risk measures used for each asset, for each hazard type
        measure_ids_for_hazard: Dict[Type[Hazard], List[str]] = {}
        # one
        calcs_by_asset = [self._calculator_for_asset(asset) for asset in assets]
        # match to specific asset and if no match then use the generic calculator assigned to Asset
        used_calcs = {c for c in calcs_by_asset if c is not None}

        all_supported_hazards = set(
            h for c in used_calcs for h in c.supported_hazards()
        )
        # get all measures
        measure_id_lookup = {
            cal: f"measure_{i}"
            for (i, cal) in enumerate(
                sorted(
                    set(
                        item
                        for item in (
                            cal.get_definition(hazard_type=hazard_type)
                            for hazard_type in all_supported_hazards
                            for cal in used_calcs
                        )
                        if item is not None
                    ),
                    key=lambda c: list(sorted(c.hazard_types))[0],
                )
            )
        }

        if not isinstance(
            self._portfolio_measure_calculator,
            NullAssetBasedPortfolioRiskMeasureCalculator,
        ):
            measure_id_lookup[self._portfolio_measure_calculator.get_definition()] = (
                "portfolio_measure_0"
            )

        def get_measure_id(
            measure_calc: Union[RiskMeasureCalculator, None], hazard_type: type
        ):
            if measure_calc is None:
                return "na"
            measure = measure_calc.get_definition(hazard_type=hazard_type)
            return measure_id_lookup[measure] if measure is not None else "na"

        for hazard_type in all_supported_hazards:
            measure_ids = [get_measure_id(calc, hazard_type) for calc in calcs_by_asset]
            measure_ids_for_hazard[hazard_type] = measure_ids
        return measure_ids_for_hazard, measure_id_lookup

    def calculate_risk_measures(
        self, assets: Sequence[Asset], scenarios: Sequence[str], years: Sequence[int]
    ):
        impacts = self._calculate_all_impacts(
            assets, scenarios, years, include_histo=True
        )
        measures: Dict[MeasureKey, Measure] = {}
        aggregated_measures: Dict[MeasureKey, Measure] = {}
        measure_calc_assets: Dict[RiskMeasureCalculator, List[Asset]] = defaultdict(
            list
        )
        for asset in assets:
            measure_calc = self._calculator_for_asset(asset)
            if measure_calc is not None:
                measure_calc_assets[measure_calc].append(asset)
        for measure_calc, assets_for_calc in measure_calc_assets.items():
            for asset in assets_for_calc:
                for scenario in scenarios:
                    for year in [None] if scenario == "historical" else years:
                        for hazard_type in measure_calc.supported_hazards():
                            base_impacts = impacts.get(
                                ImpactKey(
                                    asset=asset,
                                    hazard_type=hazard_type,
                                    scenario="historical",
                                    key_year=None,
                                )
                            )
                            # the future impact might also be the historical if that is also specified
                            fut_impacts = impacts.get(
                                ImpactKey(
                                    asset=asset,
                                    hazard_type=hazard_type,
                                    scenario=scenario,
                                    key_year=year,
                                )
                            )
                            if base_impacts is None or fut_impacts is None:
                                # should only happen if we are working with limited hazard scope
                                continue
                            if len(base_impacts) == 0 or len(fut_impacts) == 0:
                                continue
                            # if there are multiple impacts (e.g. from multiple vulnerability models), we
                            # pass to the measure calculator. It will aggregate as it sees fit.
                            risk_ind = measure_calc.calc_measure(
                                hazard_type, base_impacts, fut_impacts
                            )
                            if risk_ind is not None:
                                measures[
                                    MeasureKey(asset, scenario, year, hazard_type)
                                ] = risk_ind
            aggregated_measures.update(
                measure_calc.aggregate_risk_measures(measures, assets, scenarios, years)
            )
        portfolio_measures = self._portfolio_measure_calculator.calculate_risk_measures(
            aggregated_measures
        )
        aggregated_measures.update(portfolio_measures)
        return impacts, aggregated_measures

    def _calculator_for_asset(self, asset: Asset) -> Optional[RiskMeasureCalculator]:
        return self._asset_level_measure_calculators.get(
            type(asset), self._asset_level_measure_calculators.get(Asset, None)
        )
