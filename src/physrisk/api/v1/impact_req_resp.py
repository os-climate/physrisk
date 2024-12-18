from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field

from physrisk.api.v1.common import (
    Assets,
    Distribution,
    ExceedanceCurve,
    VulnerabilityDistrib,
)
from physrisk.api.v1.hazard_data import Scenario


class CalcSettings(BaseModel):
    hazard_interp: str = Field(
        "floor",
        description="Method used for interpolation of hazards: 'floor' or 'bilinear'.",
    )


class AssetImpactRequest(BaseModel):
    """Impact calculation request."""

    assets: Assets
    calc_settings: CalcSettings = Field(
        default_factory=CalcSettings,  # type:ignore
        description="Interpolation method.",
    )
    include_asset_level: bool = Field(
        True, description="If true, include asset-level impacts."
    )
    include_measures: bool = Field(
        False, description="If true, include calculation of risk measures."
    )
    include_calc_details: bool = Field(
        True, description="If true, include impact calculation details."
    )
    use_case_id: str = Field(
        "",
        description="Identifier for 'use case' used in the risk measures calculation.",
    )
    provider_max_requests: Dict[str, int] = Field(
        {},
        description="The maximum permitted number of requests \
        to external providers. This setting is intended in particular for paid-for data. The key is the provider \
        ID and the value is the maximum permitted requests.",
    )
    scenarios: Optional[Sequence[str]] = Field(
        [], description="Name of scenarios ('rcp8p5')"
    )
    years: Optional[Sequence[int]] = Field(
        [],
        description="""Projection year (2030, 2050, 2080). Any year before 2030,
        e.g. 1980, is treated as historical.""",
    )
    # to be deprecated
    scenario: str = Field("rcp8p5", description="Name of scenario ('rcp8p5')")
    year: int = Field(
        2050,
        description="""Projection years (e.g. 2030, 2050, 2080). Any year before 2030,
        e.g. 1980, is treated as historical.""",
    )


class Category(int, Enum):
    NODATA = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    REDFLAG = 4


class RiskMeasureDefinition(BaseModel):
    measure_id: str = Field("", description="Identifier for the risk measure.")
    label: str = Field(
        "<short description of the measure, e.g. fractional loss for 1-in-100 year event>",
        description="Short label for the measure quantity.",
    )
    description: str
    units: str = Field(
        "", description="Units; if no units an empty string is expected."
    )


class RiskScoreValue(BaseModel):
    value: Category = Field(
        Category.NODATA, description="Value of the score: red, amber, green, nodata."
    )
    label: str = Field(
        "",
        description="Short description of value, e.g. material increase in loss for 1-in-100 year event.",
    )
    description: str = Field(
        "",
        description="Full description of value including criteria, \
        e.g. change in fractional loss from 1-in-100 year event is greater than 10%.",
    )


class ScoreBasedRiskMeasureDefinition(BaseModel):
    hazard_types: List[str] = Field(
        [], description="Defines the hazards that the measure is used for."
    )
    values: List[RiskScoreValue] = Field(
        [], description="Defines the set of values that the score can take."
    )
    underlying_measures: List[RiskMeasureDefinition] = Field(
        [],
        description="Defines the underlying risk measures from which the scores are inferred.",
    )
    # for now underlying measures defined directly rather than by referencing an ID via:
    # underlying_measure_ids: List[str] = Field(
    #    [], description="The identifiers of the underlying risk measures from which the scores are inferred."
    # )

    # It is not enough to pass frozen=True, since not all attributes are hashable
    def __hash__(self):
        return id(self)


class RiskMeasureKey(BaseModel):
    hazard_type: str
    scenario_id: str
    year: str
    measure_id: str


class RiskMeasuresForAssets(BaseModel):
    key: RiskMeasureKey
    scores: List[int] = Field([0], description="Identifier for the risk measure.")
    measures_0: List[float]
    measures_1: Optional[List[float]]


class ScoreBasedRiskMeasureSetDefinition(BaseModel):
    measure_set_id: str
    asset_measure_ids_for_hazard: Dict[str, List[str]]
    score_definitions: Dict[str, ScoreBasedRiskMeasureDefinition]


class RiskMeasures(BaseModel):
    """Risk measures"""

    measures_for_assets: List[RiskMeasuresForAssets]
    score_based_measure_set_defn: ScoreBasedRiskMeasureSetDefinition
    measures_definitions: Optional[List[RiskMeasureDefinition]]
    scenarios: List[Scenario]
    asset_ids: List[str]


class AcuteHazardCalculationDetails(BaseModel):
    """Details of an acute hazard calculation."""

    hazard_exceedance: ExceedanceCurve
    hazard_distribution: Distribution
    vulnerability_distribution: VulnerabilityDistrib
    hazard_path: List[str] = Field(
        ["unknown"], description="Path to the hazard indicator data source."
    )


class ImpactKey(BaseModel):
    hazard_type: str = Field("", description="Type of the hazard.")
    scenario_id: str = Field("", description="Identifier of the scenario.")
    year: str = Field("", description="Year of impact.")


class AssetSingleImpact(BaseModel):
    """Impact at level of single asset and single type of hazard."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    key: ImpactKey

    @computed_field  # deprecated: use key instead
    def hazard_type(self) -> str:
        return self.key.hazard_type

    @computed_field  # deprecated: use key instead
    def year(self) -> str:
        return self.key.year

    impact_type: str = Field(
        "damage",
        description="""'damage' or 'disruption'. Whether the impact is fractional damage to the asset
        ('damage') or disruption to an operation, expressed as
        fractional decrease to an equivalent cash amount.""",
    )
    impact_distribution: Optional[Distribution]
    impact_exceedance: Optional[ExceedanceCurve]
    impact_mean: float
    impact_std_deviation: float
    calc_details: Optional[AcuteHazardCalculationDetails] = Field(
        None,
        description="""Details of impact calculation for acute hazard calculations.""",
    )


class AssetLevelImpact(BaseModel):
    """Impact at asset level. Each asset can have impacts for multiple hazard types."""

    asset_id: Optional[str] = Field(
        None,
        description="""Asset identifier; will appear if provided in the request
        otherwise order of assets in response is identical to order of assets in request.""",
    )
    impacts: List[AssetSingleImpact] = Field(
        [], description="Impacts for each hazard type."
    )


class AssetImpactResponse(BaseModel):
    """Response to impact request."""

    asset_impacts: Optional[List[AssetLevelImpact]] = None
    risk_measures: Optional[RiskMeasures] = None


class RiskMeasuresHelper:
    def __init__(self, risk_measures: RiskMeasures):
        """Helper class to assist in extracting results from a RiskMeasures object.

        Args:
            risk_measures (RiskMeasures): RiskMeasures result.
        """
        self.measures = {self._key(m.key): m for m in risk_measures.measures_for_assets}
        self.measure_definition = risk_measures.score_based_measure_set_defn
        self.measure_set_id = self.measure_definition.measure_set_id

    def _key(self, key: RiskMeasureKey):
        return self.Key(
            hazard_type=key.hazard_type,
            scenario_id=key.scenario_id,
            year=key.year,
            measure_id=key.measure_id,
        )

    def get_measure(self, hazard_type: str, scenario: str, year: int):
        measure_key = self.Key(
            hazard_type=hazard_type,
            scenario_id=scenario,
            year=str(year),
            measure_id=self.measure_set_id,
        )
        measure = self.measures[measure_key]
        asset_scores, asset_measures = (
            measure.scores,
            [measure.measures_0, measure.measures_1],
        )  # scores for each asset
        # measure IDs for each asset (for the hazard type in question)
        measure_ids = self.measure_definition.asset_measure_ids_for_hazard[hazard_type]
        # measure definitions for each asset
        measure_definitions = [
            self.measure_definition.score_definitions[mid] if mid != "na" else None
            for mid in measure_ids
        ]
        return asset_scores, asset_measures, measure_definitions

    def get_score_details(
        self, score: int, definition: ScoreBasedRiskMeasureDefinition
    ):
        rs_value = next(v for v in definition.values if v.value == score)
        return rs_value.label, rs_value.description

    class Key(NamedTuple):  # hashable key for looking up measures
        hazard_type: str
        scenario_id: str
        year: str
        measure_id: str
