import json

from .data.inventory import Inventory
from .data.pregenerated_hazard_model import ZarrHazardModel
from .data_objects.hazard_event_requests import (
    HazardEventAvailabilityRequest,
    HazardEventAvailabilityResponse,
    HazardEventDataRequest,
    HazardEventDataResponse,
    HazardEventDataResponseItem,
    IntensityCurve,
)
from .kernel import Event
from .kernel import calculation as calc
from .kernel.hazard_model import EventDataRequest


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

    if source_paths is None:
        source_paths = calc.get_default_zarr_source_paths()

    hazard_model = ZarrHazardModel(source_paths, store=store)
    # get hazard event types:
    event_types = Event.__subclasses__()
    event_dict = dict((et.__name__, et) for et in event_types)
    event_dict.update((est.__name__, est) for et in event_types for est in et.__subclasses__())

    # flatten list to let event processer decide how to group
    item_requests = []
    all_requests = []
    for item in request.items:
        event_type = event_dict[item.event_type]

        data_requests = [
            EventDataRequest(event_type, lon, lat, model=item.model, scenario=item.scenario, year=item.year)
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
        intensity_curves = [
            IntensityCurve(
                intensities=list(response_dict[req].intensities), return_periods=list(response_dict[req].return_periods)
            )
            for req in requests
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
