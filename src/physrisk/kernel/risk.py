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
from physrisk.kernel import calculation
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import HazardModel
from physrisk.kernel.hazards import Hazard, all_hazards
from physrisk.kernel.impact import AssetImpactResult, ImpactKey, calculate_impacts
from physrisk.kernel.impact_distrib import EmptyImpactDistrib
from physrisk.kernel.vulnerability_model import (
    DictBasedVulnerabilityModelsFactory,
    VulnerabilityModels,
)
# from asyncio import ALL_COMPLETED
# import concurrent.futures


Impact = Dict[Tuple[Asset, type], AssetImpactResult]  # the key is (Asset, Hazard type)


class BatchId(NamedTuple):
    scenario: str
    key_year: Optional[int]


class RiskModel:
    """Base class for a risk model.

    That is, a calculation of risk that makes use of hazard and vulnerability models).
    """

    def __init__(
        self,
        hazard_model: HazardModel,
        vulnerability_models: Optional[VulnerabilityModels] = None,
        use_case_id: Optional[str] = "DEFAULT",
    ):
        """Initialize a RiskModel instance.

        Parameter:
        ---------
            hazard_model (HazardModel): The hazard model to be used for risk calculations.
            vulnerability_models (Optional[VulnerabilityModels]): Optional vulnerability models; if not provided, will use default.
            use_case_id (Optional[str]): Use case identifier to determine vulnerability models if not provided.

        Raise:
        -----
            ValueError: If neither vulnerability_models nor use_case_id is provided.

        """
        super().__init__()
        if vulnerability_models is None and use_case_id is None:
            raise ValueError(
                "Either vulnerability_models or use_case_id must be provided."
            )

        self._hazard_model = hazard_model

        if use_case_id is None:
            self.use_case_id = "DEFAULT"
        else:
            self.use_case_id = use_case_id

        if vulnerability_models is None:
            factory = DictBasedVulnerabilityModelsFactory(self.use_case_id)
            self._vulnerability_models = factory.vulnerability_models()
        else:
            self._vulnerability_models = vulnerability_models

    def calculate_risk_measures(
        self, assets: Sequence[Asset], prosp_scens: Sequence[str], years: Sequence[int]
    ):
        """Calculate risk measures for a set of assets, scenarios, and years."""
        ...

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
    year: int
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
    """Protocol for selecting risk measure calculators."""

    def calculators(
        self, use_case_id: str = ""
    ) -> Dict[Type[Asset], RiskMeasureCalculator]:
        """Get risk measure calculators for asset types.

        Args:
        ----
            use_case_id (Optional[str]): Optional use case ID to filter calculators.

        """
        pass


class AssetLevelRiskModel(RiskModel):
    """Risk model that calculates risk measures at the asset level for various assets."""

    def __init__(
        self,
        hazard_model: HazardModel,
        vulnerability_models: Optional[VulnerabilityModels] = None,
        measure_calculators: Optional[Dict[type, RiskMeasureCalculator]] = None,
        use_case_id: Optional[str] = None,
    ):
        """Risk model that calculates risk measures at the asset level for a sequence of assets.

        Args:
        ----
            hazard_model (HazardModel): The hazard model.
            vulnerability_models (VulnerabilityModels): Vulnerability models for asset types.
            measure_calculators (Dict[type, RiskMeasureCalculator]): Risk measure calculators for asset types.
            use_case_id (str): 'use case' identifier used to get the measure calculators and/or vulnerability
            models if they are not provided.

        """
        if vulnerability_models is None and use_case_id is None:
            raise ValueError(
                "Either vulnerability_models or use_case_id must be provided."
            )

        if measure_calculators is None and use_case_id is None:
            raise ValueError(
                "Either measure_calculators or use_case_id must be provided."
            )

        if use_case_id is None:
            self.use_case_id = "DEFAULT"
        else:
            self.use_case_id = use_case_id

        if measure_calculators is None:
            measure_factory = calculation.DefaultMeasuresFactory()
            measure_calculators = measure_factory.calculators(self.use_case_id)

        if vulnerability_models is None:
            vulnerability_factory = DictBasedVulnerabilityModelsFactory(
                self.use_case_id
            )
            vulnerability_models = vulnerability_factory.vulnerability_models()

        super().__init__(hazard_model, vulnerability_models, use_case_id)
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
        self, assets: Sequence[Asset], prosp_scens: Sequence[str], years: Sequence[int]
    ):
        """Calculate risk measures for a set of assets, scenarios, and years, according to the selected method calculation.

        For the Default Method:
        The impact of the historical scenario is chosen as the base impact, and risk measures are
        calculated using the calc_measure function defined in the RealEstateToyRiskMeasures class. This method performs
        calculations differently depending on whether the hazard is chronic heat or another type. The difference between
        the two methods is that calc_measure_cooling uses mean impacts for calculations, while calc_measure_acute uses
        exceedance curves. In both cases, a Measure object is returned, which contains a score (REDFLAG, HIGH, MEDIUM, LOW),
        measures_0 (future_loss), and a definition.

        For the stress_test method:
        An StressTestImpact object is chosen as the base impact, and risk measures are calculated using the calc_measure
        defined in the ThermalPowerPlantsRiskMeasures class. In this method  the base impact is first used to obtain the
        percentiles (norisk, p50, p75, p90), which are used to evaluate the impact via its mean_intensity.
        This method also returns a Measure object with a score (HIGH, MEDIUM, LOW, NORISK, NODATA),
        measures_0 (mean_intensity), and a definition.

        Args:
        ----
            assets (Sequence[Asset]): List of assets.
            prosp_scens (Sequence[str]): List of prospective scenarios.
            years (Sequence[int]): List of years for the calculations.

        Return:
        ------
            Tuple[
                Dict[Tuple[Asset, type], AssetImpactResult],
                Dict[MeasureKey, Measure]
            ]: A tuple containing:
                - A dictionary mapping asset and hazard type tuples to impact results.
                - A dictionary mapping MeasureKeys to calculated measures.

        """
        impacts = self._calculate_all_impacts(
            assets, prosp_scens, years, include_histo=True
        )
        measures: Dict[MeasureKey, Measure] = {}
        aggregated_measures: Dict[MeasureKey, Measure] = {}
        for asset in assets:
            if type(asset) not in self._measure_calculators:
                continue
            measure_calc = self._measure_calculators[type(asset)]
            hazards = measure_calc.supported_hazards()
            for prosp_scen in prosp_scens:
                for year in years:
                    for hazard_type in hazards:
                        base_impacts = impacts.get(
                            ImpactKey(
                                asset=asset,
                                hazard_type=hazard_type,
                                scenario="historical",
                                key_year=None,
                            ),
                            [EmptyImpactDistrib()],
                        )
                        prosp_impacts = impacts.get(
                            ImpactKey(
                                asset=asset,
                                hazard_type=hazard_type,
                                scenario=prosp_scen,
                                key_year=year,
                            ),
                            [EmptyImpactDistrib()],
                        )
                        risk_inds = [
                            measure_calc.calc_measure(
                                hazard_type, base_impact, prosp_impact
                            )
                            for base_impact, prosp_impact in zip(
                                base_impacts, prosp_impacts
                            )
                        ]
                        risk_ind = [
                            risk_ind for risk_ind in risk_inds if risk_ind is not None
                        ]
                        if len(risk_ind) > 0:
                            # TODO: Aggregate  measures instead of picking the first value.
                            measures[
                                MeasureKey(asset, prosp_scen, year, hazard_type)
                            ] = risk_ind[0]
            aggregated_measures.update(
                measure_calc.aggregate_risk_measures(
                    measures, assets, prosp_scens, years
                )
            )
        return impacts, aggregated_measures
