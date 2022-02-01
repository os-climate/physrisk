import json
from typing import List, Optional
from physrisk.data.event_provider import EventProvider
from pydantic import BaseModel
import physrisk.data.data_requests as dr
import physrisk.kernel.calculation as calc
from physrisk.kernel.events import Event

class BaseRequest(BaseModel):
    request_id : str

class HazardEventDataRequestItem(BaseModel):
    longitudes: List[float]
    latitudes: List[float]
    request_item_id: str
    event_type: str # e.g. RiverineInundation
    model: str
    scenario: str # e.g. rcp8p5
    year: int

class HazardEventDataRequest(BaseRequest):
    items: List[HazardEventDataRequestItem]

class IntensityCurve(BaseModel):
    intensities: List[float]
    return_periods: List[float]

class HazardEventDataResponseItem(BaseModel):
    intensity_curve_set: List[IntensityCurve]
    request_item_id: str
    event_type: str 
    model: str
    scenario: str 
    year: int

class HazardEventDataResponse(BaseRequest):
    items: List[HazardEventDataResponseItem]
    
def get(request_dict):
    request_id = request_dict["request_id"].lower()

    if request_id == "get_hazard_data":
        request = HazardEventDataRequest(**request_dict)
        return json.dumps(_get_hazard_data(request).dict())
    else:
        raise ValueError('request type ' + request_dict["request_id"] + ' not found')

def _get_hazard_data(request : HazardEventDataRequest, data_sources = None):
    if data_sources is None:
        data_sources = calc._get_default_hazard_data_sources()
    
    # get hazard event types:
    event_types = Event.__subclasses__()
    event_dict = dict((et.__name__, et) for et in event_types)
    event_dict.update((est.__name__, est) for et in event_types for est in et.__subclasses__())

    # flatten list to let event processer decide how to group
    item_requests = []
    all_requests = []
    for item in request.items:
        event_type = event_dict[item.event_type]
        
        data_requests = [dr.EventDataRequest(event_type, lon, lat,
                model = item.model, scenario = item.scenario, year = item.year) 
                for (lon, lat) in zip(item.longitudes, item.latitudes)]

        all_requests.extend(data_requests)
        item_requests.append(data_requests) 

    response_dict = dr.process_requests(all_requests, data_sources)
    # responses comes back as a dictionary because requests may be executed in different order to list
    # to optimise performance. 

    response = HazardEventDataResponse(request_id = request.request_id, items = [])

    for i, item in enumerate(request.items):
        requests = item_requests[i]
        intensity_curves = [IntensityCurve(intensities = list(response_dict[req].intensities),
            return_periods = list(response_dict[req].return_periods)) for req in requests]
        response.items.append(HazardEventDataResponseItem(intensity_curve_set = intensity_curves,
            request_item_id = item.request_item_id,
            event_type = item.event_type, model = item.model, scenario = item.scenario, year = item.year))

    return response




