
from collections import defaultdict
from physrisk.kernel import Asset, AssetEventDistrib, ExceedanceCurve, VulnerabilityDistrib
from physrisk.kernel import Drought, Inundation
from physrisk.kernel import get_impact_distrib
import physrisk.data.data_requests as dr
from physrisk.data import ReturnPeriodEvDataResp
from physrisk.models import InundationModel
from physrisk.data.hazard.event_provider_wri import EventProviderWri
from physrisk.kernel.assets import PowerGeneratingAsset
import logging

def __get_default_hazard_data_sources(cache_folder = None):
    """Get default hazard data sources for each hazard type."""
    return { Inundation : EventProviderWri('web', cache_folder = cache_folder).get_inundation_depth }

def __get_default_models():
    """Get default exposure/vulnerability models for different asset types."""
    return { PowerGeneratingAsset : [ InundationModel ] }

def calculate_impacts(assets, cache_folder = None, model_properties = None):
    
    # the types of model that apply to asset of a particular type
    model_mapping = __get_default_models()

    # the different sources of hazard data
    hazard_data_source = __get_default_hazard_data_sources(cache_folder = cache_folder)

    model_assets = defaultdict(list)
    for asset in assets:
        asset_type = type(asset)
        mappings = model_mapping[asset_type]
        for m in mappings:
            model_assets[m].append(asset)

    detailed_results = {}
    for model_type, assets in model_assets.items():
        logging.info("Applying model {0} to {1} assets of type {2}".format(model_type.__name__, len(assets), type(assets[0]).__name__))
        
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
