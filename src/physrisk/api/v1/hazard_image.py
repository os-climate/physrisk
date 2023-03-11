from typing import Optional

from pydantic import BaseModel, Field

from physrisk.api.v1.common import BaseHazardRequest


class HazardImageRequest(BaseHazardRequest):
    resource: str = Field(description="Full path to the array; formed by '{path}/{id}'.")
    scenarioId: str
    year: int
    colormap: Optional[str] = Field("heating")
    format: Optional[str] = Field("PNG")
    min_value: Optional[float]
    max_value: Optional[float]


class HazardImageResponse(BaseModel):
    image: bytes
