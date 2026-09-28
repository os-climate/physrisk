from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field

from physrisk.api.v1.common import BaseHazardRequest


class TileNotAvailableError(KeyError):
    """Raised if the requested tile is not available, most commonly because
    the zoom level requested is not present.
    """


class Tile(NamedTuple):
    x: int
    y: int
    z: int


class HazardImageInfoRequest(BaseHazardRequest):
    resource: str = Field(description="Hazard resource path (unique identifier).")
    scenario_id: str
    year: int


class HazardImageInfoResponse(BaseHazardRequest):
    all_index_values: list[Any] = Field(
        [], description="The coordinate values of the index dimension."
    )
    available_index_values: list[Any] = Field(
        [],
        description="The coordinate values of the index dimension for which maps are available.",
    )
    index_display_name: str = Field(
        "index", description="The name of the index dimension."
    )
    index_units: str = Field("", description="The units of the index dimension.")
    max_zoom: int | None = Field(
        default=None,
        description="The maximum zoom level for which tiles are available.",
    )


class HazardImageRequest(BaseHazardRequest):
    resource: str = Field(description="Hazard resource path (unique identifier).")
    scenario_id: str
    year: int
    colormap: str | None = Field("heating")
    format: str | None = Field("PNG")
    min_value: float | None
    max_value: float | None
    tile: Tile | None
    index_value: Any | None = Field(
        None, description="(Non-spatial) index of the array to view."
    )
    scaling: Literal["linear", "log"] | None = Field(
        None,
        description="Value-to-colour scaling: 'linear' or 'log'. "
        "'log' requires min_value > 0.",
    )


class HazardImageResponse(BaseModel):
    image: bytes
