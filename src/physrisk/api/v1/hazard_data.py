from typing import List, Optional, Union

from pydantic import BaseModel

from physrisk.api.v1.common import IntensityCurve


class MapInfo(BaseModel):
    """Provides information about map layer"""

    colormap: str
    filename: Optional[str]


class Period(BaseModel):
    """A period belonging to a scenario"""

    year: int
    map_id: str


class Scenario(BaseModel):
    """Scenario ID and the list of available years for that scenario e.g. RCP8.5 = 'rcp8.5'"""

    id: str
    years: List[int]
    periods: Optional[List[Period]]


class HazardModel(BaseModel):
    """Provides the scenarios associated with a hazard model."""

    event_type: str
    path: str
    id: str
    display_name: str
    description: str
    filename: str
    map: Optional[MapInfo]
    scenarios: List[Scenario]
    units: str


# region HazardAvailability


class HazardEventAvailabilityRequest(BaseModel):
    event_types: Union[List[str], None]  # e.g. RiverineInundation


class HazardEventAvailabilityResponse(BaseModel):
    models: List[HazardModel]
    colormaps: dict


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
    interpolation: str = "floor"
    items: List[HazardEventDataRequestItem]


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
