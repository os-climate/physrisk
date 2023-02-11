from enum import Flag, auto
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from physrisk.api.v1.common import IntensityCurve

# region HazardModel


class Colormap(BaseModel):
    """Provides details of colormap."""

    min_index: Optional[int] = Field(
        1, description="Value of colormap minimum. Constant min for a group of maps can facilitate comparison."
    )
    min_value: float = Field(
        description="Value of colormap minimum. Constant min for a group of maps can facilitate comparison."
    )
    max_index: Optional[int] = Field(
        255, description="Value of colormap maximum. Constant max for a group of maps can facilitate comparison."
    )
    max_value: float = Field(
        description="Value of colormap maximum. Constant max for a group of maps can facilitate comparison."
    )
    name: str = Field(description="Name of colormap, e.g. 'flare', 'heating'.")
    nodata_index: Optional[int] = Field(0, description="Index used for no data.")
    units: str = Field(description="Units, e.g. 'degree days', 'metres'.")


class MapInfo(BaseModel):
    """Provides information about map layer."""

    colormap: Optional[Colormap] = Field(description="Details of colormap.")
    array_name: Optional[str] = Field(
        description="Name of array reprojected to Web Mercator for on-the-fly display or to hash to obtain tile ID. If not supplied, convention is to add '_map' to array_name."  # noqa
    )
    bounds: Optional[List[Tuple[float, float]]] = Field(
        [[-180.0, 85.0], [180.0, 85.0], [180.0, -85.0], [-180.0, -85.0]],
        description="Bounds (top/left, top/right, bottom/right, bottom/left) as degrees. Note applied to map reprojected into Web Mercator CRS.",  # noqa
    )
    # note that the bounds should be consistent with the array attributes
    source: Optional[str] = Field(description="Source of map image: 'map_array' or 'tiles'.")


class Period(BaseModel):
    """Provides information about a period, which currently corresponds to a year, belonging to a scenario."""

    year: int
    map_id: str = Field(description="If present, identifier to be used for looking up map tiles from server.")


class Scenario(BaseModel):
    """Scenario ID and the list of available years for that scenario e.g. RCP8.5 = 'rcp8.5'"""

    id: str
    years: List[int]
    periods: Optional[List[Period]]


def expanded(item: str, key: str, param: str):
    return item and item.replace("{" + key + "}", param)


class HazardModel(BaseModel):
    """Provides scenarios associated with a hazard model."""

    type: Optional[str] = Field(description="Type of hazard.")
    path: str
    id: str
    params: Optional[Dict[str, List[str]]]
    display_name: str
    description: str
    array_name: Optional[str]  # alias of filename
    map: Optional[MapInfo]
    scenarios: List[Scenario]
    units: str

    def expand(self):
        # should be only one key
        if not self.params:
            yield self
            return
        key = list(self.params.keys())[0]
        params = self.params[key]
        for param in params:
            yield self.copy(
                deep=True,
                update={
                    "id": expanded(self.id, key, param),
                    "display_name": expanded(self.display_name, key, param),
                    "array_name": expanded(self.array_name, key, param),
                    "map": self.map.copy(deep=True, update={"array_name": expanded(self.map.array_name, key, param)}),
                },
            )

    def key(self):
        return (self.type, self.id)


# endregion

# region HazardAvailability


class InventorySource(Flag):
    """Source of inventory. Where multiple are selected, order is as shown:
    results from HAZARD_TEST override those of HAZARD and EMBEDDED (to facilitate testing).
    """

    EMBEDDED = auto()  # inventory embedded in physrisk
    HAZARD = auto()  # inventory stored in the S3 hazard location
    HAZARD_TEST = auto()  # inventory stored in the S3 hazard_test location


class HazardEventAvailabilityRequest(BaseModel):
    types: Optional[List[str]]  # e.g. ["RiverineInundation"]
    sources: Optional[List[str]] = Field(
        description="Sources of inventory, can be 'embedded', 'hazard' or 'hazard_test'."
    )


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
