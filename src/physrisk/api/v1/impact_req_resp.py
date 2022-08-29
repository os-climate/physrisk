from typing import List

from pydantic import BaseModel, Field

from physrisk.api.v1.common import Asset, Assets

# region Request


class CalcSettings(BaseModel):
    hazard_interp: str = Field("floor", description="Method used for interpolation of hazards: 'lloor' or 'bilinear'.")


class AssetImpactRequest(BaseModel):
    """Model for asset impact calculation request."""

    assets: Assets
    calc_settings: CalcSettings = Field(default_factory=CalcSettings, description="Interpolation method.")
    include_asset_level: bool = Field(True, description="If true, include ")
    include_calc_details: bool = Field(True, description="If true, include impact calculation details.")
    scenario: str = Field("rcp8p5", description="Name of scenario ('rcp8p5')")
    year: int = Field(
        2050,
        description="Projection year (2030, 2050, 2080). Any year before 2030, e.g. 1980, is treated as historical.",
    )


# endregion

# region Response

class AssetLevelImpact(BaseModel):
    """Impact at asset level"""
    asset_id: str
    hazard_type: str = Field('', description="Type of the hazard.")
    impact_type: str = Field('damage', description="""'damage' or 'disruption'. Whether the impact is fractional damage to the asset
        or disruption to the annual economic benefit obtained from the asset, expressed as fractional decrease to an equivalent cash amount  
        """)
    bin_edges: List[float] = Field(True, description="Edges of the bins")
    probs: bool = Field(True, description="If true, include impact calculation details.")

class AcuteHazardCalculationDetails(BaseModel):
    """Details of an acute hazard calculation."""
    

class AssetImpactResponse(BaseModel):
    """Perform calculation"""

    assets: List[AssetLevelImpact]


# endregion
