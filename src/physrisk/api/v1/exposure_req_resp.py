from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from physrisk.api.v1.common import Assets
from physrisk.api.v1.impact_req_resp import CalcSettings


class AssetExposureRequest(BaseModel):
    """Impact calculation request."""

    assets: Assets
    calc_settings: CalcSettings = Field(default_factory=CalcSettings, description="Interpolation method.")
    scenario: str = Field("rcp8p5", description="Name of scenario ('rcp8p5')")
    year: int = Field(
        2050,
        description="Projection year (2030, 2050, 2080). Any year before 2030, e.g. 1980, is treated as historical.",
    )


class Exposure(BaseModel):
    category: str
    value: Optional[float]


class AssetExposure(BaseModel):
    """Impact at asset level. Each asset can have impacts for multiple hazard types."""

    asset_id: Optional[str] = Field(
        None,
        description="""Asset identifier; will appear if provided in the request
        otherwise order of assets in response is identical to order of assets in request.""",
    )
    exposures: Dict[str, Exposure] = Field({}, description="Category (value) for each hazard type (key).")


class AssetExposureResponse(BaseModel):
    """Response to impact request."""

    items: List[AssetExposure]
