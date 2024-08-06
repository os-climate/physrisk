from typing import NamedTuple, Optional

from pydantic import BaseModel, Field

from physrisk.api.v1.common import BaseHazardRequest

# class Tile(BaseHazardRequest):
#     x: int
#     y: int
#     z: int


class Tile(NamedTuple):
    x: int
    y: int
    z: int


class HazardImageRequest(BaseHazardRequest):
    resource: str = Field(
        description="Full path to the array; formed by '{path}/{id}'."
    )
    scenario_id: str
    year: int
    colormap: Optional[str] = Field("heating")
    format: Optional[str] = Field("PNG")
    min_value: Optional[float]
    max_value: Optional[float]
    tile: Optional[Tile]


class HazardImageResponse(BaseModel):
    image: bytes
