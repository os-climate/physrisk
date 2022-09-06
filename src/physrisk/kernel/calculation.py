import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..data.hazard_data_provider import (
    get_source_path_osc_chronic_heat,
    get_source_path_wri_coastal_inundation,
    get_source_path_wri_riverine_inundation,
)
from ..data.pregenerated_hazard_model import ZarrHazardModel
from ..models import power_generating_asset_models as pgam
from ..models.real_estate_models import RealEstateCoastalInundationModel, RealEstateRiverineInundationModel
from ..utils.helpers import get_iterable
from .assets import Asset, PowerGeneratingAsset, RealEstateAsset, TestAsset
from .hazard_event_distrib import HazardEventDistrib
from .hazard_model import HazardDataResponse, HazardModel
from .hazards import ChronicHeat, CoastalInundation, RiverineInundation
from .impact_distrib import ImpactDistrib
from .vulnerability_distrib import VulnerabilityDistrib
from .vulnerability_model import VulnerabilityModelAcuteBase, VulnerabilityModelBase


class AssetImpactResult:
    def __init__(
        self,
        impact: ImpactDistrib,
        vulnerability: VulnerabilityDistrib = None,
        event: HazardEventDistrib = None,
        hazard_data: Iterable[HazardDataResponse] = None,
    ):
        self.impact = impact
        # optional detailed results for drill-dowwn
        self.hazard_data = hazard_data
        self.vulnerability = vulnerability
        self.event = event


def get_default_zarr_source_paths():
    return {
        RiverineInundation: get_source_path_wri_riverine_inundation,
        CoastalInundation: get_source_path_wri_coastal_inundation,
        ChronicHeat: get_source_path_osc_chronic_heat,
    }


def get_default_hazard_model():
    # Model that gets hazard event data from Zarr storage
    return ZarrHazardModel(get_default_zarr_source_paths())


def get_default_vulnerability_models():
    """Get default exposure/vulnerability models for different asset types."""
    return {
        PowerGeneratingAsset: [pgam.InundationModel()],
        RealEstateAsset: [RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
        TestAsset: [pgam.TemperatureModel()],
    }


def calculate_impacts(
    assets: Iterable[Asset],
    hazard_model: Optional[HazardModel] = None,
    vulnerability_models: Optional[Any] = None,  #: Optional[Dict[type, Sequence[VulnerabilityModelBase]]] = None,
    *,
    scenario: str,
    year: int,
) -> Dict[Tuple[Asset, type], AssetImpactResult]:
    """ """
    if hazard_model is None:
        hazard_model = get_default_hazard_model()

    # the models that apply to asset of a particular type
    if vulnerability_models is None:
        vulnerability_models = get_default_vulnerability_models()

    model_assets: Dict[VulnerabilityModelBase, List[Asset]] = defaultdict(
        list
    )  # list of assets to be modelled using vulnerability model
    for asset in assets:
        asset_type = type(asset)
        mappings = vulnerability_models[asset_type]
        for m in mappings:
            model_assets[m].append(asset)

    results = {}
    for model, assets in model_assets.items():
        logging.info(
            "Applying model {0} to {1} assets of type {2}".format(
                type(model).__name__, len(assets), type(assets[0]).__name__
            )
        )

        event_requests_by_asset = [model.get_data_requests(asset, scenario=scenario, year=year) for asset in assets]

        event_requests = [
            req for event_request_by_asset in event_requests_by_asset for req in get_iterable(event_request_by_asset)
        ]

        responses = hazard_model.get_hazard_events(event_requests)

        for asset, requests in zip(assets, event_requests_by_asset):
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
