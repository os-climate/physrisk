import json
from typing import List, Optional
from physrisk.data.event_provider import EventProvider
from pydantic import BaseModel
import physrisk.data.data_requests as dr
import physrisk.kernel.calculation as calc
from physrisk.kernel.events import Event

class BaseRequest(BaseModel):
    request_id : str

class HazardEventDataRequest(BaseRequest):
    longitudes: List[float]
    latitudes: List[float]
    event_type: str # e.g. RiverineInundation
    model : str
    scenario : str # e.g. rcp8p5
    year : int
    
def get_request(request_dict):
    request_id = request_dict["request_id"].lower()

    if request_id == "get_hazard_data":
        request = HazardEventDataRequest(**request_dict)
        return get_hazard_data(request)
    else:
        raise ValueError('request type ' + request_dict["request_id"] + ' not found')


def get_hazard_data(request : HazardEventDataRequest, data_sources = None):
    if data_sources is None:
        data_sources = calc._get_default_hazard_data_sources()
    
    event_types = Event.__subclasses__()
    event_dict = dict((et.__name__, et) for et in event_types)
    event_dict.update((est.__name__, est) for et in event_types for est in et.__subclasses__())

    event_type = event_dict[request.event_type]

    data_requests = [dr.EventDataRequest(event_type, lon, lat,
            scenario = request.scenario, year = request.year, model = request.model) 
            for (lon, lat) in zip(request.longitudes, request.latitudes)] 

    res = dr.process_requests(data_requests, data_sources)

