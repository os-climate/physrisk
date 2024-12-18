import concurrent.futures
from dataclasses import dataclass
from typing import (
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

from physrisk.api.v1.impact_req_resp import Category, ScoreBasedRiskMeasureDefinition
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import HazardModel
from physrisk.kernel.hazards import Hazard, all_hazards
from physrisk.kernel.impact import AssetImpactResult, ImpactKey, calculate_impacts
from physrisk.kernel.vulnerability_model import VulnerabilityModels

# from asyncio import ALL_COMPLETED
# import concurrent.futures


Impact = Dict[Tuple[Asset, type], AssetImpactResult]  # the key is (Asset, Hazard type)


class BatchId(NamedTuple):
    scenario: str
    key_year: Optional[int]


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
        scenarios = (
            set(["historical"] + list(prosp_scens)) if include_histo else prosp_scens
        )
        impact_results: Dict[ImpactKey, List[AssetImpactResult]] = {}

        # in case of multiple calculation, run on separate threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            tagged_futures = {
                executor.submit(
                    self._calculate_single_impact, assets, scenario, year
                ): BatchId(scenario, None if scenario == "historical" else year)
                for scenario in scenarios
                for year in years
            }
            for future in concurrent.futures.as_completed(tagged_futures):
                tag = tagged_futures[future]
                try:
                    res = future.result()
                    # flatten to use single key
                    for temp_key, value in res.items():
                        key = ImpactKey(
                            asset=temp_key.asset,
                            hazard_type=temp_key.hazard_type,
                            scenario=tag.scenario,
                            key_year=tag.key_year,
                        )
                        impact_results[key] = value

                except Exception as exc:
                    print("%r generated an exception: %s" % (tag, exc))
        return impact_results

    def _calculate_single_impact(
        self, assets: Sequence[Asset], scenario: str, year: int
    ):
        """Calculate impacts for a single scenario and year."""
        return calculate_impacts(
            assets,
            self._hazard_model,
            self._vulnerability_models,
            scenario=scenario,
            year=year,
        )


class MeasureKey(NamedTuple):
    asset: Asset
    prosp_scen: str  # prospective scenario
    year: Optional[int]
    hazard_type: type


@dataclass
class Measure:
    score: Category
    measure_0: float
    definition: ScoreBasedRiskMeasureDefinition  # reference to single instance of ScoreBasedRiskMeasureDefinition


class RiskMeasureCalculator(Protocol):
    def calc_measure(
        self,
        hazard_type: Type[Hazard],
        base_impact: AssetImpactResult,
        impact: AssetImpactResult,
    ) -> Optional[Measure]:
        """Calculate the Measure (score-based risk measure) for the hazard,
        given the base (i.e. historical) and future asset-level impact.

        Args:
            hazard_type (Type[Hazard]): Hazard type.
            base_impact (AssetImpactResult): Historical asset-level impact.
            impact (AssetImpactResult): Future asset-level impact.

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
    def calculators(self, use_case_id: str) -> Dict[Type[Asset], RiskMeasureCalculator]:
        pass


class AssetLevelRiskModel(RiskModel):
    def __init__(
        self,
        hazard_model: HazardModel,
        vulnerability_models: VulnerabilityModels,
        measure_calculators: Dict[type, RiskMeasureCalculator],
    ):
        """Risk model that calculates risk measures at the asset level for a sequence
        of assets.

        Args:
            hazard_model (HazardModel): The hazard model.
            vulnerability_models (Dict[type, Sequence[VulnerabilityModelBase]]): Vulnerability models for asset types.
            measure_calculators (Dict[type, RiskMeasureCalculator]): Risk measure calculators for asset types.
        """
        super().__init__(hazard_model, vulnerability_models)
        self._measure_calculators = measure_calculators

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
        hazards = all_hazards()
        # the identifiers of the score-based risk measures used for each asset, for each hazard type
        measure_ids_for_hazard: Dict[Type[Hazard], List[str]] = {}
        # one
        calcs_by_asset = [
            self._measure_calculators.get(
                type(asset), self._measure_calculators.get(Asset, None)
            )
            for asset in assets
        ]
        # match to specific asset and if no match then use the generic calculator assigned to Asset
        used_calcs = {c for c in calcs_by_asset if c is not None}
        # get all measures
        measure_id_lookup = {
            cal: f"measure_{i}"
            for (i, cal) in enumerate(
                set(
                    item
                    for item in (
                        cal.get_definition(hazard_type=hazard_type)
                        for hazard_type in hazards
                        for cal in used_calcs
                    )
                    if item is not None
                )
            )
        }

        def get_measure_id(
            measure_calc: Union[RiskMeasureCalculator, None], hazard_type: type
        ):
            if measure_calc is None:
                return "na"
            measure = measure_calc.get_definition(hazard_type=hazard_type)
            return measure_id_lookup[measure] if measure is not None else "na"

        for hazard_type in hazards:
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
        for asset in assets:
            if type(asset) not in self._measure_calculators:
                continue
            measure_calc = self._measure_calculators[type(asset)]
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
                        risk_inds = [
                            measure_calc.calc_measure(
                                hazard_type, base_impact, prosp_impact
                            )
                            for base_impact, prosp_impact in zip(
                                base_impacts, fut_impacts
                            )
                        ]
                        risk_ind = [
                            risk_ind for risk_ind in risk_inds if risk_ind is not None
                        ]
                        if len(risk_ind) > 0:
                            # TODO: Aggregate  measures instead of picking the first value.
                            measures[MeasureKey(asset, scenario, year, hazard_type)] = (
                                risk_ind[0]
                            )
        aggregated_measures.update(
            measure_calc.aggregate_risk_measures(measures, assets, scenarios, years)
        )
        return impacts, aggregated_measures
