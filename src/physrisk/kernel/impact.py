import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_event_distrib import HazardEventDistrib
from physrisk.kernel.hazard_model import HazardDataRequest, HazardDataResponse, HazardModel
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.vulnerability_distrib import VulnerabilityDistrib
from physrisk.kernel.vulnerability_model import DataRequester, VulnerabilityModelAcuteBase, VulnerabilityModelBase
from physrisk.utils.helpers import get_iterable


class ImpactKey(NamedTuple):
    asset: Asset
    hazard_type: type
    # consider adding type: whether damage or disruption


@dataclass
class AssetImpactResult:
    impact: ImpactDistrib
    vulnerability: Optional[VulnerabilityDistrib] = None
    event: Optional[HazardEventDistrib] = None
    hazard_data: Optional[Iterable[HazardDataResponse]] = None  # optional detailed results for drill-down


def calculate_impacts(  # noqa: C901
    assets: Iterable[Asset],
    hazard_model: HazardModel,
    vulnerability_models: Dict[type, Sequence[VulnerabilityModelBase]],
    *,
    scenario: str,
    year: int,
) -> Dict[ImpactKey, AssetImpactResult]:
    """Calculate asset level impacts."""

    model_assets: Dict[DataRequester, List[Asset]] = defaultdict(
        list
    )  # list of assets to be modelled using vulnerability model

    for asset in assets:
        asset_type = type(asset)
        mappings = vulnerability_models[asset_type]
        for m in mappings:
            model_assets[m].append(asset)
    results = {}

    asset_requests, responses = _request_consolidated(hazard_model, model_assets, scenario, year)

    logging.info("Calculating impacts")
    for model, assets in model_assets.items():
        logging.info(
            "Applying vulnerability model {0} to {1} assets of type {2}".format(
                type(model).__name__, len(assets), type(assets[0]).__name__
            )
        )
        for asset in assets:
            requests = asset_requests[(model, asset)]
            hazard_data = [responses[req] for req in get_iterable(requests)]
            if isinstance(model, VulnerabilityModelAcuteBase):
                impact, vul, event = model.get_impact_details(asset, hazard_data)
                results[ImpactKey(asset, model.hazard_type)] = AssetImpactResult(
                    impact, vulnerability=vul, event=event, hazard_data=hazard_data
                )
            elif isinstance(model, VulnerabilityModelBase):
                impact = model.get_impact(asset, hazard_data)
                results[ImpactKey(asset, model.hazard_type)] = AssetImpactResult(impact, hazard_data=hazard_data)
    return results


def _request_consolidated(
    hazard_model: HazardModel, requester_assets: Dict[DataRequester, List[Asset]], scenario: str, year: int
):
    """As an important performance optimization, data requests are consolidated for all requesters
    (e.g. vulnerability mode) because different requesters may query the same hazard data sets
    note that key for a single request is (requester, asset).
    """
    # the list of requests for each requester and asset
    asset_requests: Dict[Tuple[DataRequester, Asset], Union[HazardDataRequest, Iterable[HazardDataRequest]]] = {}

    logging.info("Generating hazard data requests for requesters")
    for requester, assets in requester_assets.items():
        for asset in assets:
            asset_requests[(requester, asset)] = requester.get_data_requests(asset, scenario=scenario, year=year)

    logging.info("Retrieving hazard data")
    flattened_requests = [req for requests in asset_requests.values() for req in get_iterable(requests)]
    responses = hazard_model.get_hazard_events(flattened_requests)
    return asset_requests, responses
