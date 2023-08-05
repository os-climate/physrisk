import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from physrisk.hazard_models.embedded import get_default_source_paths
from physrisk.kernel.exposure import Category, DataRequester, ExposureMeasure

from ..data.pregenerated_hazard_model import ZarrHazardModel
from ..utils.helpers import get_iterable
from ..vulnerability_models import power_generating_asset_models as pgam
from ..vulnerability_models.chronic_heat_models import ChronicHeatGZNModel
from ..vulnerability_models.real_estate_models import (
    RealEstateCoastalInundationModel,
    RealEstateRiverineInundationModel,
)
from .assets import Asset, IndustrialActivity, PowerGeneratingAsset, RealEstateAsset, TestAsset
from .hazard_event_distrib import HazardEventDistrib
from .hazard_model import HazardDataRequest, HazardDataResponse, HazardModel
from .impact_distrib import ImpactDistrib
from .vulnerability_distrib import VulnerabilityDistrib
from .vulnerability_model import VulnerabilityModelAcuteBase, VulnerabilityModelBase


@dataclass
class AssetExposureResult:
    hazard_categories: Dict[type, Tuple[Category, float]]


class AssetImpactResult:
    def __init__(
        self,
        impact: ImpactDistrib,
        vulnerability: Optional[VulnerabilityDistrib] = None,
        event: Optional[HazardEventDistrib] = None,
        hazard_data: Optional[Iterable[HazardDataResponse]] = None,
    ):
        self.impact = impact
        # optional detailed results for drill-down
        self.hazard_data = hazard_data
        self.vulnerability = vulnerability
        self.event = event


def get_default_hazard_model():
    # Model that gets hazard event data from Zarr storage
    return ZarrHazardModel(source_paths=get_default_source_paths())


def get_default_vulnerability_models():
    """Get default exposure/vulnerability models for different asset types."""
    return {
        PowerGeneratingAsset: [pgam.InundationModel()],
        RealEstateAsset: [RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
        IndustrialActivity: [ChronicHeatGZNModel()],
        TestAsset: [pgam.TemperatureModel()],
    }


def calculate_exposures(
    assets: List[Asset], hazard_model: HazardModel, exposure_measure: ExposureMeasure, scenario: str, year: int
) -> Dict[Asset, AssetExposureResult]:
    requester_assets: Dict[DataRequester, List[Asset]] = {exposure_measure: assets}
    assetRequests, responses = _request_consolidated(hazard_model, requester_assets, scenario, year)

    logging.info(
        "Applying exposure measure {0} to {1} assets of type {2}".format(
            type(exposure_measure).__name__, len(assets), type(assets[0]).__name__
        )
    )
    result: Dict[Asset, AssetExposureResult] = {}

    for asset in assets:
        requests = assetRequests[(exposure_measure, asset)]  # (ordered) requests for a given asset
        hazard_data = [responses[req] for req in get_iterable(requests)]
        result[asset] = AssetExposureResult(hazard_categories=exposure_measure.get_exposures(asset, hazard_data))

    return result


def calculate_impacts(  # noqa: C901
    assets: Iterable[Asset],
    hazard_model: Optional[HazardModel] = None,
    vulnerability_models: Optional[Any] = None,  #: Optional[Dict[type, Sequence[VulnerabilityModelBase]]] = None,
    *,
    scenario: str,
    year: int,
) -> Dict[Tuple[Asset, type], AssetImpactResult]:
    """Calculate asset level impacts."""

    if hazard_model is None:
        hazard_model = get_default_hazard_model()
    # the models that apply to asset of a particular type
    if vulnerability_models is None:
        vulnerability_models = get_default_vulnerability_models()
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
            "Applying model {0} to {1} assets of type {2}".format(
                type(model).__name__, len(assets), type(assets[0]).__name__
            )
        )
        for asset in assets:
            requests = asset_requests[(model, asset)]
            hazard_data = [responses[req] for req in get_iterable(requests)]
            if isinstance(model, VulnerabilityModelAcuteBase):
                impact, vul, event = model.get_impact_details(asset, hazard_data)
                results[(asset, model.hazard_type)] = AssetImpactResult(
                    impact, vulnerability=vul, event=event, hazard_data=hazard_data
                )
            elif isinstance(model, VulnerabilityModelBase):
                impact = model.get_impact(asset, hazard_data)
                results[(asset, model.hazard_type)] = AssetImpactResult(impact, hazard_data=hazard_data)
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
