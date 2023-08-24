from typing import Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple

from physrisk.api.v1.impact_req_resp import RiskMeasureResult
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import HazardModel
from physrisk.kernel.impact import AssetImpactResult, calculate_impacts
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.vulnerability_model import VulnerabilityModelBase

# from asyncio import ALL_COMPLETED
# import concurrent.futures


Impact = Dict[Tuple[Asset, type], AssetImpactResult]  # the key is (Asset, Hazard type)


class BatchId(NamedTuple):
    scenario: str
    key_year: Optional[int]


class RiskModel:
    """Base class for a risk model (i.e. a calculation of risk that makes use of hazard and vulnerability
    models)."""

    def __init__(self, hazard_model: HazardModel, vulnerability_models: Dict[type, Sequence[VulnerabilityModelBase]]):
        self._hazard_model = hazard_model
        self._vulnerability_models = vulnerability_models

    def calculate_risk_measures(self, assets: Sequence[Asset], prosp_scens: Sequence[str], years: Sequence[int]):
        ...

    def _calculate_all_impacts(self, assets: Sequence[Asset], prosp_scens: Sequence[str], years: Sequence[int]):
        scenarios = set(["historical"] + list(prosp_scens))
        impact_results: Dict[BatchId, Impact] = {}
        items = [(scenario, year) for scenario in scenarios for year in years]
        for scenario, year in items:
            key_year = None if scenario == "historical" else year
            impact_results[BatchId(scenario, key_year)] = self._calculate_single_impact(assets, scenario, year)
        return impact_results
        # consider parallelizing using approach similar to:
        # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        #    future_to_url = {executor.submit(self._calculate_single_impact, assets, scenario, year): \
        #                      (scenario, year) for scenario in scenarios for year in years}
        #    for future in concurrent.futures.as_completed(future_to_url):
        #        tag = future_to_url[future]
        #        try:
        #            data = future.result()
        #        except Exception as exc:
        #            print('%r generated an exception: %s' % (tag, exc))
        #        else:
        #            ...

    def _calculate_single_impact(self, assets: Sequence[Asset], scenario: str, year: int):
        """Calculate impacts for a single scenario and year."""
        return calculate_impacts(assets, self._hazard_model, self._vulnerability_models, scenario=scenario, year=year)


class MeasureKey(NamedTuple):
    asset: Asset
    prosp_scen: str  # prospective scenario
    year: int
    hazard_type: type


class RiskMeasureCalculator(Protocol):
    def calc_measure(self, hazard_type: type, base_impact: ImpactDistrib, impact: ImpactDistrib) -> RiskMeasureResult:
        ...

    def supported_hazards(self) -> List[type]:
        ...


class AssetLevelRiskModel(RiskModel):
    def __init__(
        self,
        hazard_model: HazardModel,
        vulnerability_models: Dict[type, Sequence[VulnerabilityModelBase]],
        measure_calculators: Dict[type, RiskMeasureCalculator],
    ):
        super().__init__(hazard_model, vulnerability_models)
        self._measure_calculators = measure_calculators

    def calculate_impacts(self, assets: Sequence[Asset], prosp_scens: Sequence[str], years: Sequence[int]):
        impacts = self._calculate_all_impacts(assets, prosp_scens, years)
        return impacts

    def calculate_risk_measures(self, assets: Sequence[Asset], prosp_scens: Sequence[str], years: Sequence[int]):
        impacts = self._calculate_all_impacts(assets, prosp_scens, years)
        measures: Dict[MeasureKey, RiskMeasureResult] = {}
        for asset in assets:
            if type(asset) not in self._measure_calculators:
                continue
            measure_calc = self._measure_calculators[type(asset)]
            for prosp_scen in prosp_scens:
                for year in years:
                    scenario_impacts = impacts[(prosp_scen, year)]
                    for hazard_type in measure_calc.supported_hazards():
                        key = (asset, hazard_type)
                        if key in scenario_impacts:
                            base_impact = impacts[("historical", None)][key].impact
                            impact = scenario_impacts[key].impact
                            risk_ind = measure_calc.calc_measure(hazard_type, base_impact, impact)
                            measures[MeasureKey(asset, prosp_scen, year, hazard_type)] = risk_ind
                            # if the fractional loss is material and materially increases
        return impacts, measures
