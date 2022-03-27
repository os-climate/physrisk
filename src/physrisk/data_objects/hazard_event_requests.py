from typing import List, Union

from pydantic import BaseModel


class BaseRequest(BaseModel):
    request_item_id: str


class Scenario(BaseModel):
    id: str
    years: List[int]


class Model(BaseModel):
    event_type: str
    id: str
    scenarios: List[Scenario]


# region HazardAvailability


class HazardAvailabilityItem(BaseModel):
    models: List[Model]


class HazardEventAvailabilityRequest(BaseModel):
    event_types: Union[List[str], None]  # e.g. RiverineInundation


class HazardEventAvailabilityResponse(BaseModel):
    models: List[Model]


# endregion

# region HazardEventData


class HazardEventDataRequestItem(BaseModel):
    longitudes: List[float]
    latitudes: List[float]
    request_item_id: str
    event_type: str  # e.g. RiverineInundation
    model: str
    scenario: str  # e.g. rcp8p5
    year: int


class HazardEventDataRequest(BaseModel):
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


class HazardEventDataResponse(BaseModel):
    items: List[HazardEventDataResponseItem]


# endregion
