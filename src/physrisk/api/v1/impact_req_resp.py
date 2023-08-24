from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from physrisk.api.v1.common import Assets, Distribution, ExceedanceCurve, VulnerabilityDistrib


class CalcSettings(BaseModel):
    hazard_interp: str = Field("floor", description="Method used for interpolation of hazards: 'floor' or 'bilinear'.")


class AssetImpactRequest(BaseModel):
    """Impact calculation request."""

    assets: Assets
    calc_settings: CalcSettings = Field(default_factory=CalcSettings, description="Interpolation method.")
    include_asset_level: bool = Field(True, description="If true, include asset-level impacts.")
    include_measures: bool = Field(True, description="If true, include measures.")
    include_calc_details: bool = Field(True, description="If true, include impact calculation details.")
    scenario: str = Field("rcp8p5", description="Name of scenario ('rcp8p5')")
    year: int = Field(
        2050,
        description="Projection year (2030, 2050, 2080). Any year before 2030, e.g. 1980, is treated as historical.",
    )


# region Response


class Category(str, Enum):
    NODATA = "NODATA"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    REDFLAG = "REDFLAG"


class Indicator(BaseModel):
    value: float
    label: str


class RiskMeasureResult(BaseModel):
    """Provides a risk category based on one or more risk indicators.
    A risk indicator is a quantity derived from one or more vulnerability models,
    e.g. the change in 1-in-100 year damage or disruption.
    """

    category: Category = Field(description="Result category.")
    cat_defn: str = Field(description="Definition of the category for the particular indicator.")
    indicators: List[Indicator]
    summary: str = Field(description="Summary of the indicator.")


class AcuteHazardCalculationDetails(BaseModel):
    """Details of an acute hazard calculation."""

    hazard_exceedance: ExceedanceCurve
    hazard_distribution: Distribution
    vulnerability_distribution: VulnerabilityDistrib


class AssetSingleHazardImpact(BaseModel):
    """Impact at level of single asset and single type of hazard."""

    hazard_type: str = Field("", description="Type of the hazard.")
    impact_type: str = Field(
        "damage",
        description="""'damage' or 'disruption'. Whether the impact is fractional damage to the asset
        ('damage') or disruption to the annual economic benefit obtained from the asset ('disruption'), expressed as
        fractional decrease to an equivalent cash amount.""",
    )
    risk_measure: Optional[RiskMeasureResult]
    impact_distribution: Distribution
    impact_exceedance: ExceedanceCurve
    impact_mean: float
    impact_std_deviation: float
    calc_details: Optional[AcuteHazardCalculationDetails] = Field(
        None,
        description="""Details of impact calculation for acute hazard calculations.""",
    )

    class Config:
        arbitrary_types_allowed = True


class AssetLevelImpact(BaseModel):
    """Impact at asset level. Each asset can have impacts for multiple hazard types."""

    asset_id: Optional[str] = Field(
        None,
        description="""Asset identifier; will appear if provided in the request
        otherwise order of assets in response is identical to order of assets in request.""",
    )
    impacts: List[AssetSingleHazardImpact] = Field([], description="Impacts for each hazard type.")


class AssetImpactResponse(BaseModel):
    """Response to impact request."""

    asset_impacts: List[AssetLevelImpact]


# endregion
