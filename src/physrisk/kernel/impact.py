import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_event_distrib import HazardEventDistrib
from physrisk.kernel.hazard_model import (
    HazardDataFailedResponse,
    HazardDataRequest,
    HazardDataResponse,
    HazardModel,
)
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.vulnerability_distrib import VulnerabilityDistrib
from physrisk.kernel.vulnerability_model import (
    DataRequester,
    VulnerabilityModelAcuteBase,
    VulnerabilityModelBase,
    VulnerabilityModels,
)
from physrisk.utils.helpers import get_iterable

logger = logging.getLogger(__name__)


class ImpactKey(NamedTuple):
    asset: Asset
    hazard_type: type
    # these additional key items can be set to None, for example
    # if the calculation is for a given scenario and year
    # impact_type: Optional[str] = None # consider adding: whether damage or disruption
    scenario: str
    key_year: Optional[int] = None  # this is None for 'historical' scenario

    def __repr__(self) -> str:
        asset_id = self.asset.id if self.asset.id is not None else "no_id"
        return (
            f"ImpactKey(asset={asset_id},hazard_type={self.hazard_type.__name__},"
            f"scenario={self.scenario},key_year={self.key_year})"
        )


@dataclass
class AssetImpactResult:
    impact: ImpactDistrib
    vulnerability: Optional[VulnerabilityDistrib] = None
    event: Optional[HazardEventDistrib] = None
    hazard_data: Optional[Sequence[HazardDataResponse]] = (
        None  # optional detailed results for drill-down
    )


def calculate_impacts(  # noqa: C901
    assets: Iterable[Asset],
    hazard_model: HazardModel,
    vulnerability_models: VulnerabilityModels,
    *,
    scenarios: Sequence[str],
    years: Sequence[int],
) -> Dict[ImpactKey, List[AssetImpactResult]]:
    """Calculate asset level impacts."""

    model_assets: Dict[DataRequester, List[Asset]] = defaultdict(
        list
    )  # list of assets to be modelled using vulnerability model

    for asset in assets:
        asset_type = type(asset)
        mappings = vulnerability_models.vuln_model_for_asset_of_type(asset_type)
        for mapping in mappings:
            model_assets[mapping].append(asset)
    results: Dict[ImpactKey, List[AssetImpactResult]] = {}

    scen_year_asset_requests, responses = _download_data_consolidated(
        hazard_model, model_assets, scenarios, years
    )

    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     tagged_futures = {
    #         executor.submit(
    #             self._calculate_single_impact, assets, scenario, year
    #         ): BatchId(scenario, None if scenario == "historical" else year)
    #         for scenario in scenarios
    #         for year in years
    #     }
    #     for future in concurrent.futures.as_completed(tagged_futures):
    #         tag = tagged_futures[future]
    #         try:
    #             res = future.result()
    #             # flatten to use single key
    #             for temp_key, value in res.items():
    #                 key = ImpactKey(
    #                     asset=temp_key.asset,
    #                     hazard_type=temp_key.hazard_type,
    #                     scenario=tag.scenario,
    #                     key_year=tag.key_year,
    #                 )
    #                 impact_results[key] = value

    #         except Exception as exc:
    #             print("%r generated an exception: %s" % (tag, exc))

    logging.info("Calculating impacts")
    summary: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
    for model, assets in model_assets.items():
        assert isinstance(model, VulnerabilityModelBase)
        summary[model.hazard_type.__name__].append(
            (type(model).__name__, len(assets), type(assets[0]).__name__)
        )
    logging.info("Applying vulnerability models:")
    for k, vl in summary.items():
        logging.info(f"{k}:")
        for v in vl:
            logging.info(f"{v[1]} {v[2]}{'s' if v[1] > 1 else ''}: {v[0]}")
    for scenario in scenarios:
        logging.info(f"Scenario {scenario}")
        for year in [-1] if scenario == "historical" else years:
            if scenario != "historical":
                logging.info(f"Year {year}")
            asset_requests = scen_year_asset_requests[ScenarioYear(scenario, year)]
            for model, assets in model_assets.items():
                assert isinstance(model, VulnerabilityModelBase)
                for asset in assets:
                    requests = asset_requests[(model, asset)]
                    hazard_data = [responses[req] for req in get_iterable(requests)]

                    results.setdefault(
                        ImpactKey(
                            asset=asset,
                            hazard_type=model.hazard_type,
                            scenario=scenario,
                            key_year=None if year == -1 else year,
                        ),
                        [],
                    )

                    if any(
                        isinstance(hd, HazardDataFailedResponse) for hd in hazard_data
                    ):
                        # the failed responses should have been logged already
                        continue
                    try:
                        if isinstance(model, VulnerabilityModelAcuteBase):
                            impact, vul, event = model.get_impact_details(
                                asset, hazard_data
                            )
                            results[
                                ImpactKey(
                                    asset=asset,
                                    hazard_type=model.hazard_type,
                                    scenario=scenario,
                                    key_year=None if year == -1 else year,
                                )
                            ].append(
                                AssetImpactResult(
                                    impact,
                                    vulnerability=vul,
                                    event=event,
                                    hazard_data=hazard_data,
                                )
                            )
                        elif isinstance(model, VulnerabilityModelBase):
                            impact = model.get_impact(asset, hazard_data)
                            results[
                                ImpactKey(
                                    asset=asset,
                                    hazard_type=model.hazard_type,
                                    scenario=scenario,
                                    key_year=None if year == -1 else year,
                                )
                            ].append(AssetImpactResult(impact, hazard_data=hazard_data))
                    except Exception as e:
                        logger.exception(e)
    return results


class ScenarioYear(NamedTuple):
    scenario: str
    key_year: Optional[int] = None


def _download_data_consolidated(
    hazard_model: HazardModel,
    requester_assets: Dict[DataRequester, List[Asset]],
    scenarios: Sequence[str],
    years: Sequence[int],
):
    """As an important performance optimization, data requests are consolidated for all requesters
    (e.g. vulnerability model) because different requesters may query the same hazard data sets
    note that key for a single request is (requester, asset).
    """
    # the list of requests for each requester and asset
    scen_year_asset_requests: Dict[
        ScenarioYear,
        Dict[
            Tuple[DataRequester, Asset],
            Union[HazardDataRequest, Sequence[HazardDataRequest]],
        ],
    ] = {}
    for scenario in scenarios:
        for year in [-1] if scenario == "historical" else years:
            asset_requests: Dict[
                Tuple[DataRequester, Asset],
                Union[HazardDataRequest, Sequence[HazardDataRequest]],
            ] = {}
            for requester, assets in requester_assets.items():
                for asset in assets:
                    asset_requests[(requester, asset)] = requester.get_data_requests(
                        asset, scenario=scenario, year=year
                    )
            scen_year_asset_requests[ScenarioYear(scenario, year)] = asset_requests

    logging.info("Retrieving hazard data")
    flattened_requests = [
        req
        for asset_requests in scen_year_asset_requests.values()
        for requests in asset_requests.values()
        for req in get_iterable(requests)
    ]
    responses = hazard_model.get_hazard_data(flattened_requests)
    return scen_year_asset_requests, responses
