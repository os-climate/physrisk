import logging
from collections import defaultdict

from ..data import data_requests as dr
from ..data.event_provider import EventProvider, get_source_path_wri_riverine_inundation
from ..models import InundationModel
from .asset_impact import get_impact_distrib
from .assets import PowerGeneratingAsset
from .events import RiverineInundation


def _get_default_hazard_data_sources(cache_folder=None):
    """Get default hazard data sources for each hazard type."""
    data_sources = {RiverineInundation: EventProvider(get_source_path_wri_riverine_inundation).get_intensity_curves}
    return data_sources


def _get_default_models():
    """Get default exposure/vulnerability models for different asset types."""
    return {PowerGeneratingAsset: [InundationModel]}


def calculate_impacts(assets, cache_folder=None, model_properties=None):

    # the types of model that apply to asset of a particular type
    model_mapping = _get_default_models()

    # the different sources of hazard data
    hazard_data_source = _get_default_hazard_data_sources(cache_folder=cache_folder)

    model_assets = defaultdict(list)
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

        responses = dr.process_requests(event_requests, hazard_data_source)

        for asset, requests in zip(assets, event_requests_by_asset):
            hazard_data = [responses[req] for req in requests]
            vul, event = model.get_distributions(asset, hazard_data)
            impact = get_impact_distrib(event, vul)
            detailed_results[asset] = DetailedResultItem(vul, event, impact, hazard_data)

    return detailed_results


class DetailedResultItem:
    def __init__(self, vulnerability, event, impact, hazard_data):
        self.hazard_data = hazard_data
        self.vulnerability = vulnerability
        self.event = event
        self.impact = impact
        self.mean_impact = impact.mean_impact()
