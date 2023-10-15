from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from physrisk.api.v1.common import Assets, Distribution, ExceedanceCurve, VulnerabilityDistrib


class CalcSettings(BaseModel):
    hazard_interp: str = Field("floor", description="Method used for interpolation of hazards: 'floor' or 'bilinear'.")


class AssetImpactRequest(BaseModel):
    """Impact calculation request."""

    assets: Assets
    calc_settings: CalcSettings = Field(
        default_factory=CalcSettings, description="Interpolation method."  # type:ignore
    )
    include_asset_level: bool = Field(True, description="If true, include asset-level impacts.")
    include_measures: bool = Field(True, description="If true, include measures.")
    include_calc_details: bool = Field(True, description="If true, include impact calculation details.")
    scenario: str = Field("rcp8p5", description="Name of scenario ('rcp8p5')")
    year: int = Field(
        2050,
        description="Projection year (2030, 2050, 2080). Any year before 2030, e.g. 1980, is treated as historical.",
    )


class Category(int, Enum):
    NODATA = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    REDFLAG = 4


class RiskKey(BaseModel):
    scenario_id: str
    year: str


class RiskMeasureKey(RiskKey):
    risk_measure_id: str = Field("", description="Identifier of the risk measure.")


class AssetsRiskScores(BaseModel):
    """Risk scores for a set of assets, with risk measures used to calculate the measures.
    A single score may be derived from multiple risk measures in principle, the measures are identified
    by the ScoreBasedMeasureDefinition corresponding to the asset.
    In principle multiple measures may be used to compute the score, hence 'measures_0', 'measures_1' etc,
    although no example yet.
    """

    key: RiskKey
    scores: List[int] = Field(None, description="Identifier for the risk measure.")
    measures_0: List[float]
    measures_1: Optional[List[float]]


class AssetRiskMeasures(BaseModel):
    """Risk measures for a set of assets."""

    key: RiskMeasureKey
    measures: List[float]


class AssetScoreModel(BaseModel):
    asset_model_id: List[str]


class RiskMeasureDefinition(BaseModel):
    measure_id: str = Field(None, description="Identifier for the risk measure.")
    measure_index: int = Field(None, description="Identifier for the risk measure.")
    label: str = Field(
        "<short description of the measure, e.g. fractional loss for 1-in-100 year event.",
        description="Value of the score.",
    )
    description: str


class RiskScoreValue(BaseModel):
    value: Category = Field("", description="Value of the score: red, amber, green, nodata.")
    label: str = Field(
        "", description="Short description of value, e.g. material increase in loss for 1-in-100 year event."
    )
    description: str = Field(
        "",
        description="Full description of value including criteria, \
        e.g. change in fractional loss from 1-in-100 year event is greater than 10%.",
    )


class ScoreBasedRiskMeasureDefinition(BaseModel, frozen=True):
    hazard_types: List[str] = Field([], description="Defines the hazards that the measure is used for.")
    values: List[RiskScoreValue] = Field([], description="Defines the set of values that the score can take.")
    child_measure_ids: List[str] = Field(
        [], description="The identifiers of the risk measures used to calculate the score."
    )

    # should be sufficient to pass frozen=True, but does not seem to work (pydantic docs says feature in beta)
    def __hash__(self):
        return id(self)


class HazardRiskMeasures(BaseModel):
    """Risk measures for one particular type of hazard"""

    hazard_type: str
    scores_for_assets: Optional[List[AssetsRiskScores]] = Field(
        [], description="Risk scores for the set of assets for different scenarios and years."
    )
    measures_for_assets: Optional[List[AssetRiskMeasures]] = Field(
        [], description="Risk measures for the set of assets for different scenarios and years."
    )
    score_measure_ids_for_assets: Optional[List[str]] = Field(
        None, description="Identifiers of the score-based risk measures used for each asset."
    )


class RiskMeasures(BaseModel):
    """Risk measures"""

    hazard_risk_measures: List[HazardRiskMeasures]
    score_definitions: Optional[Dict[str, ScoreBasedRiskMeasureDefinition]]
    measures_definitions: Optional[List[RiskMeasureDefinition]]


class AcuteHazardCalculationDetails(BaseModel):
    """Details of an acute hazard calculation."""

    hazard_exceedance: ExceedanceCurve
    hazard_distribution: Distribution
    vulnerability_distribution: VulnerabilityDistrib


class AssetSingleImpact(BaseModel):
    """Impact at level of single asset and single type of hazard."""

    hazard_type: Optional[str] = Field("", description="Type of the hazard.")
    impact_type: Optional[str] = Field(
        "damage",
        description="""'damage' or 'disruption'. Whether the impact is fractional damage to the asset
        ('damage') or disruption to an operation, expressed as
        fractional decrease to an equivalent cash amount.""",
    )
    year: Optional[str] = None
    impact_distribution: Optional[Distribution]
    impact_exceedance: Optional[ExceedanceCurve]
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
    impacts: List[AssetSingleImpact] = Field([], description="Impacts for each hazard type.")


class AssetImpactResponse(BaseModel):
    """Response to impact request."""

    asset_impacts: Optional[List[AssetLevelImpact]] = None
    risk_measures: Optional[RiskMeasures] = None
