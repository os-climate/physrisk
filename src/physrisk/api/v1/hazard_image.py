from typing import Any, List, NamedTuple, Optional

from pydantic import BaseModel, Field

from physrisk.api.v1.common import BaseHazardRequest


class TileNotAvailableError(KeyError):
    """Raised if the requested tile is not available, most commonly because
    the zoom level requested is not present.
    """

    pass


class Tile(NamedTuple):
    x: int
    y: int
    z: int


class HazardImageInfoRequest(BaseHazardRequest):
    resource: str = Field(description="Hazard resource path (unique identifier).")
    scenario_id: str
    year: int


class HazardImageInfoResponse(BaseHazardRequest):
    all_index_values: List[Any] = Field(
        [], description="The coordinate values of the index dimension."
    )
    available_index_values: List[Any] = Field(
        [],
        description="The coordinate values of the index dimension for which maps are available.",
    )
    index_display_name: str = Field(
        "index", description="The name of the index dimension."
    )
    index_units: str = Field("", description="The units of the index dimension.")


class HazardImageRequest(BaseHazardRequest):
    resource: str = Field(description="Hazard resource path (unique identifier).")
    scenario_id: str
    year: int
    colormap: Optional[str] = Field("heating")
    format: Optional[str] = Field("PNG")
    min_value: Optional[float]
    max_value: Optional[float]
    tile: Optional[Tile]
    index_value: Optional[Any] = Field(
        None, description="(Non-spatial) index of the array to view."
    )


class HazardImageResponse(BaseModel):
    image: bytes
