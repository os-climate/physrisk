from importlib import import_module
import json
from typing import cast
from urllib import response

from .api.v1.impact_req_resp import (
    AssetImpactRequest,
    AssetImpactResponse,
    Assets
)

from .api.v1.hazard_data import (
    HazardEventAvailabilityRequest,
    HazardEventAvailabilityResponse,
    HazardEventDataRequest,
    HazardEventDataResponse,
    HazardEventDataResponseItem,
    IntensityCurve,
)
from .api.v1.impact_req_resp import (
    AssetImpactRequest
)
from .data.inventory import Inventory
from .data.pregenerated_hazard_model import ZarrHazardModel
from .kernel import Asset, Hazard
from .kernel import calculation as calc
from .kernel.hazard_model import HazardDataRequest
from .kernel.hazard_model import HazardEventDataResponse as hmHazardEventDataResponse


def get(*, request_id, request_dict, store=None):

    if request_id == "get_hazard_data":
        request = HazardEventDataRequest(**request_dict)
        return json.dumps(_get_hazard_data(request, store=store).dict())
    elif request_id == "get_hazard_data_availability":
        request = HazardEventAvailabilityRequest(**request_dict)
        return json.dumps(_get_hazard_data_availability(request).dict())
    else:
        raise ValueError(f"request type '{request_id}' not found")


def _get_hazard_data_availability(request: HazardEventAvailabilityRequest):
    inventory = Inventory()
    models = inventory.models
    response = HazardEventAvailabilityResponse(models=models)  # type: ignore
    return response


def _get_hazard_data(request: HazardEventDataRequest, source_paths=None, store=None):
    hazard_model = _create_hazard_model(interpolation=request.interpolation, source_paths=source_paths, store=store)

    # get hazard event types:
    event_types = Hazard.__subclasses__()
    event_dict = dict((et.__name__, et) for et in event_types)
    event_dict.update((est.__name__, est) for et in event_types for est in et.__subclasses__())

    # flatten list to let event processer decide how to group
    item_requests = []
    all_requests = []
    for item in request.items:
        event_type = event_dict[item.event_type]

        data_requests = [
            HazardDataRequest(event_type, lon, lat, model=item.model, scenario=item.scenario, year=item.year)
            for (lon, lat) in zip(item.longitudes, item.latitudes)
        ]

        all_requests.extend(data_requests)
        item_requests.append(data_requests)

    response_dict = hazard_model.get_hazard_events(all_requests)
    # responses comes back as a dictionary because requests may be executed in different order to list
    # to optimise performance.

    response = HazardEventDataResponse(items=[])

    for i, item in enumerate(request.items):
        requests = item_requests[i]
        resps = (cast(hmHazardEventDataResponse, response_dict[req]) for req in requests)
        intensity_curves = [
            IntensityCurve(intensities=list(resp.intensities), return_periods=list(resp.return_periods))
            for resp in resps
        ]
        response.items.append(
            HazardEventDataResponseItem(
                intensity_curve_set=intensity_curves,
                request_item_id=item.request_item_id,
                event_type=item.event_type,
                model=item.model,
                scenario=item.scenario,
                year=item.year,
            )
        )

    return response

def create_assets(assets: Assets):
    """Create list of Asset objects from the Assets API object:"""
    module = import_module("physrisk.kernel.assets")
    asset_objs = []
    for asset in assets.items:     
        asset_obj = cast(Asset, getattr(module, asset.asset_class)(asset.latitude, asset.longitude,
            type=asset.type, 
            location=asset.location))
        asset_objs.append(asset_obj)
    return asset_objs

def _create_hazard_model(interpolation='floor', source_paths=None, store=None):
    if source_paths is None:
        source_paths = calc.get_default_zarr_source_paths()

    hazard_model = ZarrHazardModel(source_paths, store=store, interpolation=interpolation)

    return hazard_model


def _get_asset_impacts(request: AssetImpactRequest, source_paths=None, store=None):
    hazard_model = _create_hazard_model(source_paths=source_paths, store=store)
    vulnerability_models = calc.get_default_vulnerability_models()

    # we keep API definition of asset separate from internal Asset class; convert by reflection
    # based on asset_class:
    assets = create_assets(request.assets)

    results = calc.calculate_impacts(
        assets, hazard_model, vulnerability_models, scenario=request.scenario, year=request.year
    )

    for asset, result in results.items():
        # always an impact distribution...
        impact = result.impact
        # for acute hazards we can provide the hazard event distributions:
        #if result.hazard_data is not None:
    
    response = AssetImpactResponse()
    
    return response

