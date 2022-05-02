from typing import List, Union

from pydantic import BaseModel


class Scenario(BaseModel):
    """Scenario ID and the list of available years for that scenario e.g. RCP8.5 = 'rcp8.5'"""

    id: str
    years: List[int]


class HazardModel(BaseModel):
    """Provides the scenarios associated with a hazard model."""

    event_type: str
    path: str
    id: str
    display_name: str
    description: str
    filename: str
    scenarios: List[Scenario]


# region HazardAvailability


class HazardEventAvailabilityRequest(BaseModel):
    event_types: Union[List[str], None]  # e.g. RiverineInundation


class HazardEventAvailabilityResponse(BaseModel):
    models: List[HazardModel]


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
