import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from physrisk.data.inventory import Inventory
from physrisk.kernel import hazards

from ..data.hazard_data_provider import (
    SourcePath,
    get_source_path_generic,
    get_source_path_osc_chronic_heat,
    get_source_path_wri_coastal_inundation,
    get_source_path_wri_riverine_inundation,
)
from ..data.pregenerated_hazard_model import ZarrHazardModel
from ..models import power_generating_asset_models as pgam
from ..models.chronic_heat_models import ChronicHeatGznModel
from ..models.real_estate_models import RealEstateCoastalInundationModel, RealEstateRiverineInundationModel
from ..utils.helpers import get_iterable
from .assets import Asset, IndustrialActivity, PowerGeneratingAsset, RealEstateAsset, TestAsset
from .hazard_event_distrib import HazardEventDistrib
from .hazard_model import HazardDataRequest, HazardDataResponse, HazardModel
from .hazards import ChronicHeat, CoastalInundation, RiverineInundation
from .impact_distrib import ImpactDistrib
from .vulnerability_distrib import VulnerabilityDistrib
from .vulnerability_model import VulnerabilityModelAcuteBase, VulnerabilityModelBase


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


def get_default_zarr_source_paths():
    return {
        RiverineInundation: get_source_path_wri_riverine_inundation,
        CoastalInundation: get_source_path_wri_coastal_inundation,
        ChronicHeat: get_source_path_osc_chronic_heat,
    }


def get_source_paths_from_inventory(inventory: Inventory, embedded: Optional[Dict[type, SourcePath]] = None):
    source_paths: Dict[type, SourcePath] = {}
    for key, resource in inventory.resources.items():
        source_paths[hazards.hazard_class(resource.type)] = get_source_path_generic(inventory, resource.type, embedded)
    return source_paths


def get_default_hazard_model():
    # Model that gets hazard event data from Zarr storage
    return ZarrHazardModel(get_default_zarr_source_paths())


def get_default_vulnerability_models():
    """Get default exposure/vulnerability models for different asset types."""
    return {
        PowerGeneratingAsset: [pgam.InundationModel()],
        RealEstateAsset: [RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
        IndustrialActivity: [ChronicHeatGznModel()],
        TestAsset: [pgam.TemperatureModel()],
    }


def calculate_impacts(  # noqa: C901
    assets: Iterable[Asset],
    hazard_model: Optional[HazardModel] = None,
    vulnerability_models: Optional[Any] = None,  #: Optional[Dict[type, Sequence[VulnerabilityModelBase]]] = None,
    *,
    scenario: str,
    year: int,
) -> Dict[Tuple[Asset, type], AssetImpactResult]:
    """Calculate asset level impacts"""
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

    # as an important performance optimisation, data requests are consolidated for all vulnerability models
    # because different vulnerability models may query the same hazard data sets
    # note that key for request is [vulnerability model, asset]
    assetRequests: Dict[
        Tuple[VulnerabilityModelBase, Asset], Union[HazardDataRequest, Iterable[HazardDataRequest]]
    ] = {}

    logging.info("Generating vulnerability model hazard data requests")
    for model, assets in model_assets.items():
        for asset in assets:
            assetRequests[(model, asset)] = model.get_data_requests(asset, scenario=scenario, year=year)

    logging.info("Retrieving hazard data")
    event_requests = [req for requests in assetRequests.values() for req in get_iterable(requests)]
    responses = hazard_model.get_hazard_events(event_requests)

    logging.info("Calculating impacts")
    for model, assets in model_assets.items():
        logging.info(
            "Applying model {0} to {1} assets of type {2}".format(
                type(model).__name__, len(assets), type(assets[0]).__name__
            )
        )

        for asset in assets:
            requests = assetRequests[(model, asset)]
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
