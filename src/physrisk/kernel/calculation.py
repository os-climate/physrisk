import logging
from collections import defaultdict

import physrisk.models.power_generating_asset_model as pgam
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel

from ..data.event_provider import get_source_path_wri_riverine_inundation
from .assets import PowerGeneratingAsset
from .events import RiverineInundation


def get_default_zarr_source_paths():
    return {RiverineInundation: get_source_path_wri_riverine_inundation}


def get_default_vulnerability_models():
    """Get default exposure/vulnerability models for different asset types."""
    return {PowerGeneratingAsset: [pgam.InundationModel, pgam.TemperatureModel]}


def calculate_impacts(assets, cache_folder=None, model_properties=None):

    # the types of model that apply to asset of a particular type
    model_mapping = get_default_vulnerability_models()

    # Model that gets hazard event data from Zarr storage
    hazard_model = ZarrHazardModel(get_default_zarr_source_paths())

    model_assets = defaultdict(list)  # list of assets to be modelled using vulnerability model
    for asset in assets:
        asset_type = type(asset)
        mappings = model_mapping[asset_type]
        for m in mappings:
            model_assets[m].append(asset)

    detailed_results = {}
    for model_type, assets in model_assets.items():
        logging.info(
            "Applying model {0} to {1} assets of type {2}".format(
                model_type.__name__, len(assets), type(assets[0]).__name__
            )
        )

        model = model_type() if model_properties is None else model_type(**model_properties)

        event_requests_by_asset = [model.get_event_data_requests(asset) for asset in assets]

        event_requests = [req for event_request_by_asset in event_requests_by_asset for req in event_request_by_asset]

        responses = hazard_model.get_hazard_events(event_requests)

        for asset, requests in zip(assets, event_requests_by_asset):
            hazard_data = [responses[req] for req in requests]
            impact, vul, event = model.get_impacts(asset, hazard_data)
            detailed_results[asset] = DetailedResultItem(vul, event, impact, hazard_data)

    return detailed_results


class DetailedResultItem:
    def __init__(self, vulnerability, event, impact, hazard_data):
        self.hazard_data = hazard_data
        self.vulnerability = vulnerability
        self.event = event
        self.impact = impact
        self.mean_impact = impact.mean_impact()
