import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence, Set, Tuple, Type

import numpy as np

from physrisk.api.v1.impact_req_resp import (
    Category,
    RiskMeasureDefinition,
    RiskScoreValue,
    ScoreBasedRiskMeasureDefinition,
)
from physrisk.kernel.hazard_model import (
    HazardEventDataResponse,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    Drought,
    Fire,
    Hazard,
    Landslide,
    RiverineInundation,
    Subsidence,
    WaterRisk,
    Wind,
)
from physrisk.kernel.impact import AssetImpactResult
from physrisk.kernel.impact_distrib import EmptyImpactDistrib, ImpactDistrib
from physrisk.kernel.risk import Measure, RiskMeasureCalculator


class Threshold(int, Enum):
    ABS_HIGH = 0
    ABS_LOW = 1
    CHANGE = 2


class RealEstateToyRiskMeasures(RiskMeasureCalculator):
    """Toy model for calculating risk measures for real estate assets."""

    def __init__(self):
        self.model_summary = {"*Toy* model for real estate risk assessment."}
        self.return_period = (
            100.0  # criteria based on 1 in 100-year flood or cyclone events
        )
        self.measure_thresholds_acute = {
            Threshold.ABS_HIGH: 0.1,  # fraction
            Threshold.ABS_LOW: 0.03,  # fraction
            Threshold.CHANGE: 0.03,  # fraction
        }
        self.measure_thresholds_cooling = {
            Threshold.ABS_HIGH: 500,  # kWh
            Threshold.ABS_LOW: 300,  # kWh
            Threshold.CHANGE: 0.2,  # fraction
        }

        definition_acute = self._definition_acute()
        definition_cooling = self._definition_cooling()
        self._definition_lookup = {
            RiverineInundation: definition_acute,
            CoastalInundation: definition_acute,
            Wind: definition_acute,
            ChronicHeat: definition_cooling,
        }

    def _definition_acute(self):
        definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[
                RiverineInundation.__name__,
                CoastalInundation.__name__,
                Wind.__name__,
            ],
            values=self._definition_values(self._acute_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measures_0",
                    label=f"1-in-{self.return_period:0.0f} year annual loss.",
                    description=f"1-in-{self.return_period:0.0f} year loss as fraction of asset insured value.",
                )
            ],
        )
        return definition

    def _definition_cooling(self):
        definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[ChronicHeat.__name__],
            values=self._definition_values(self._cooling_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measures_1",
                    label="Expected cooling annual energy consumption (kWh).",
                    description="Expected cooling annual energy consumption (kWh).",
                )
            ],
        )
        return definition

    def _definition_values(self, description: Callable[[Category], str]):
        return [
            RiskScoreValue(
                value=Category.REDFLAG,
                label=(
                    "The asset is very significantly impacted and the impact will increase "
                    "as a result of climate change."
                ),
                description=description(Category.REDFLAG),
            ),
            RiskScoreValue(
                value=Category.HIGH,
                label="The asset is materially impacted and the impact will increase as a result of climate change.",
                description=description(Category.HIGH),
            ),
            RiskScoreValue(
                value=Category.MEDIUM,
                label=(
                    "The asset is materially impacted but the impact will not significantly increase "
                    "as a result of climate change."
                ),
                description=description(Category.MEDIUM),
            ),
            RiskScoreValue(
                value=Category.LOW,
                label="No material impact.",
                description=description(Category.LOW),
            ),
            RiskScoreValue(
                value=Category.NODATA, label="No data.", description="No data."
            ),
        ]

    def _acute_description(self, category: Category):
        if category == Category.LOW:
            description = (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss is less than "
                f"{self.measure_thresholds_acute[Threshold.ABS_LOW] * 100:0.0f}% of asset value."
            )
        elif category == Category.MEDIUM:
            description = (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss is more than "
                f"{self.measure_thresholds_acute[Threshold.ABS_LOW] * 100:0.0f}% but increases by less than "
                f"{self.measure_thresholds_acute[Threshold.CHANGE] * 100:0.0f}% of asset value over historical baseline."
            )
        elif category == Category.HIGH:
            description = (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss is more than "
                f"{self.measure_thresholds_acute[Threshold.ABS_LOW] * 100:0.0f}% and increases by more than "
                f"{self.measure_thresholds_acute[Threshold.CHANGE] * 100:0.0f}% of asset value over historical baseline."
            )
        elif category == Category.REDFLAG:
            description = (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss is more than "
                f"{self.measure_thresholds_acute[Threshold.ABS_HIGH] * 100:0.0f}% and increases by more than "
                f"{self.measure_thresholds_acute[Threshold.CHANGE] * 100:0.0f}% of asset value over historical baseline."
            )
        else:
            description = "No Data"
        return description

    def _cooling_description(self, category: Category):
        if category == Category.LOW:
            description = (
                f"Expected cooling annual energy consumption is less than "
                f"{self.measure_thresholds_cooling[Threshold.ABS_LOW]}kWh."
            )
        elif category == Category.MEDIUM:
            description = (
                f"Expected cooling annual energy consumption is more than "
                f"{self.measure_thresholds_cooling[Threshold.ABS_LOW]}kWh but increases by less than "
                f"{self.measure_thresholds_cooling[Threshold.CHANGE] * 100:0.0f}% over historical baseline."
            )
        elif category == Category.HIGH:
            description = (
                f"Expected cooling annual energy consumption is more than "
                f"{self.measure_thresholds_cooling[Threshold.ABS_LOW]}kWh and increases by more than "
                f"{self.measure_thresholds_cooling[Threshold.CHANGE] * 100:0.0f}% over historical baseline."
            )
        elif category == Category.REDFLAG:
            description = (
                f"Expected cooling annual energy consumption is more than "
                f"{self.measure_thresholds_cooling[Threshold.ABS_HIGH]}kWh and increases by more than "
                f"{self.measure_thresholds_cooling[Threshold.CHANGE] * 100:0.0f}% over historical baseline."
            )
        else:
            description = "No Data"
        return description

    def calc_measure(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        base_impact_res: AssetImpactResult,
        impact_res: AssetImpactResult,
    ) -> Optional[Measure]:
        if isinstance(base_impact_res, EmptyImpactDistrib) or isinstance(
            impact_res, EmptyImpactDistrib
        ):
            return None

        assert isinstance(base_impact_res.impact, ImpactDistrib)

        if hazard_type == ChronicHeat:
            return self.calc_measure_cooling(
                hazard_type, base_impact_res.impact, impact_res.impact
            )
        else:
            return self.calc_measure_acute(
                hazard_type, base_impact_res.impact, impact_res.impact
            )

    def calc_measure_acute(
        self, hazard_type: type, base_impact: ImpactDistrib, impact: ImpactDistrib
    ) -> Measure:
        return_period = 100.0  # criterion based on 1 in 100-year flood events
        histo_loss = base_impact.to_exceedance_curve().get_value(1.0 / return_period)
        future_loss = impact.to_exceedance_curve().get_value(1.0 / return_period)
        loss_change = future_loss - histo_loss

        if (
            future_loss > self.measure_thresholds_acute[Threshold.ABS_HIGH]
            and loss_change > self.measure_thresholds_acute[Threshold.CHANGE]
        ):
            score = Category.REDFLAG
        elif (
            future_loss > self.measure_thresholds_acute[Threshold.ABS_LOW]
            and loss_change > self.measure_thresholds_acute[Threshold.CHANGE]
        ):
            score = Category.HIGH
        elif (
            future_loss > self.measure_thresholds_acute[Threshold.ABS_LOW]
            and loss_change <= self.measure_thresholds_acute[Threshold.CHANGE]
        ):
            score = Category.MEDIUM
        else:
            score = Category.LOW
        return Measure(
            score=score,
            measure_0=future_loss,
            definition=self.get_definition(hazard_type),
        )

    def calc_measure_cooling(
        self, hazard_type: type, base_impact: ImpactDistrib, impact: ImpactDistrib
    ) -> Measure:
        histo_cooling = base_impact.mean_impact()
        future_cooling = impact.mean_impact()
        cooling_change = (future_cooling - histo_cooling) / histo_cooling

        if (
            future_cooling > self.measure_thresholds_cooling[Threshold.ABS_HIGH]
            and cooling_change > self.measure_thresholds_cooling[Threshold.CHANGE]
        ):
            score = Category.REDFLAG
        elif (
            future_cooling > self.measure_thresholds_cooling[Threshold.ABS_LOW]
            and cooling_change > self.measure_thresholds_cooling[Threshold.CHANGE]
        ):
            score = Category.HIGH
        elif (
            future_cooling > self.measure_thresholds_cooling[Threshold.ABS_LOW]
            and cooling_change <= self.measure_thresholds_cooling[Threshold.CHANGE]
        ):
            score = Category.MEDIUM
        else:
            score = Category.LOW
        return Measure(
            score=score,
            measure_0=future_cooling,
            definition=self.get_definition(hazard_type),
        )

    def get_definition(self, hazard_type: type, indicator_id: Optional[str] = None):
        return self._definition_lookup.get(hazard_type, None)

    def supported_hazards(self) -> Set[type[Hazard]]:
        return set([RiverineInundation, CoastalInundation, Wind, ChronicHeat])


@dataclass
class ECBBounds:
    categories: Sequence[Category]
    lower: Sequence[float]
    upper: Sequence[float]


@dataclass
class ECBHazardBounds(ECBBounds):
    hazard_type: Type[Hazard]
    indicator_id: str
    units: str
    indicator_return: float


@dataclass
class ECBImpactBounds(ECBBounds):
    units: str
    # ECBImpactBounds hereda de ECBBounds y no necesita atributos adicionales
    pass


class ECBScoreRiskMeasures(RiskMeasureCalculator):
    """Toy model for calculating risk scores based on "Climate change-related statistical indicators" Statistics Paper Series No. 48 ECB 2024
    Link: https://www.ecb.europa.eu/pub/pdf/scpsps/ecb.sps48~e3fd21dd5a.en.pdf"""

    def __init__(self):
        self.model_summary = {"..."}
        # fmt: off
        # Bounds
        self._bounds = {
            (WaterRisk, "water_stress") : ECBHazardBounds(
                categories=[Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
                hazard_type=WaterRisk,
                indicator_id="water_stress",
                indicator_return=1,
                units="index",
                lower=[float("-inf"), 0.1, 0.2, 0.8],
                upper=[0.1,           0.2, 0.8, float("inf")]
                ), # noqa
        }
        self._bounds[Subsidence, "subsidence_susceptability"] = ECBHazardBounds(
            categories=[Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
            hazard_type=Subsidence,
            indicator_id="subsidence_susceptability",
            indicator_return=1,
            units="index",
            lower=[float("-inf"), 1, 2, 3],
            upper=[1,           2, 3, float("inf")]
        )
        self._bounds[Drought, "cdd"] = ECBHazardBounds(
            categories=[Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
            hazard_type=Drought,
            indicator_id="cdd",
            indicator_return=1,
            units="days",
            lower=[float("-inf"), 15, 20, 40],
            upper=[15,           20, 40, float("inf")]
        )
        self._bounds[Drought, "spi6"] = ECBHazardBounds(
            categories=[Category.HIGH, Category.MEDIUM, Category.LOW, Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
            hazard_type=Drought,
            indicator_id="spi6",
            indicator_return=1,
            units="index",
            lower=[float("-inf"), -2, -1.5, -1, 1, 1.5, 2],
            upper=[-2, -1.5, -1, 1, 1.5, 2, float("inf")]
        )
        self._bounds[Fire, "fire_probability"] = ECBHazardBounds(
            categories=[Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
            hazard_type=Fire,
            indicator_id="fire_probability",
            indicator_return=1,
            units="rate",
            lower=[float("-inf"), 0.001, 0.002, 0.010],
            upper=[0.001, 0.002, 0.010, float("inf")]
        )
        impact_bounds = ECBImpactBounds(
            categories=[Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
            units="rate",
            lower=[float("-inf"), 1e-4, 0.02, 0.06],
            upper=[1e-4,             0.02, 0.06, float("inf")]
        ) # noqa
        self._bounds[RiverineInundation, "flood_depth"] = impact_bounds
        self._bounds[CoastalInundation, "flood_depth"] = impact_bounds
        self._bounds[Landslide, "landslide_susceptability"] = impact_bounds
        self._bounds[Wind, "wind_speed/3s"] = impact_bounds
        self._bounds[Wind, "max_speed"] =  impact_bounds

        self._bounds[ChronicHeat, "mean_degree_days/above/32c"] = ECBImpactBounds(
            categories=[Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
            units="degree days",
            lower=[float("-inf"), 0.0029, 0.00711, 0.01197],
            upper=[0.0029, 0.00711, 0.01197, float("inf")]
        )
        self._bounds[ChronicHeat, "mean_work_loss/high"] = ECBImpactBounds(
            categories=[Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
            units="fractional loss",
            lower=[float("-inf"), 0.01118, 0.03250, 0.07794],
            upper=[0.01118, 0.03250, 0.07794, float("inf")]
        )
        self._bounds[ChronicHeat, "mean_degree_days/above/index"] = ECBImpactBounds(
            categories=[Category.NORISK, Category.LOW, Category.MEDIUM, Category.HIGH],
            units="degree days",
            lower=[float("-inf"), 3.4837, 84.1551, 425.9830],
            upper=[3.4837, 84.1551, 425.9830, float("inf")]
        )
        # fmt: on

        # Definitions
        self._definition_lookup = {}
        self._definition_lookup[WaterRisk, "water_stress"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[WaterRisk.__name__],
                values=self._definition_values(
                    self._bounds[WaterRisk, "water_stress"],
                    self.water_stress_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_water_stress",
                        label="Water stress",
                        description="Scores derived from the World Resources Institute (WRI) "
                        "Aqueduct 4.0 based on the ratio between water demand and water supply.",
                        units="",
                    )
                ],
            )
        )
        self._definition_lookup[Subsidence, "subsidence_susceptability"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[Subsidence.__name__],
                values=self._definition_values(
                    self._bounds[Subsidence, "subsidence_susceptability"],
                    self.subsidence_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_susceptability",
                        label="Subsidence susceptibility",
                        description="Score (1-5) from the Joint Research Centre (JRC) DRMKC RDH based on "
                        "soils' clay, silt, sand and water content.",
                        units="index",
                    )
                ],
            )
        )

        self._definition_lookup[Landslide, "landslide_susceptability"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[Landslide.__name__],
                values=self._definition_values(
                    self._bounds[Landslide, "landslide_susceptability"],
                    self.landslide_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_susceptability",
                        label="Landslide susceptibility",
                        description="Score (1-5) from the Joint Research Centre (JRC) DRMKC RDH based on characteristics "
                        "of the terrain combined with daily maximum precipitation (per return period).",
                        units="index",
                    )
                ],
            )
        )

        self._definition_lookup[Drought, "cdd"] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Drought.__name__],
            values=self._definition_values(
                self._bounds[Drought, "cdd"],
                self.cdd_label_description,
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_cdd",
                    label="Consecutive Dry Days (CDD)",
                    description="Maximum number of consecutive dry days (with precipitation < 1mm per day) from CORDEX-CORE.",
                    units="days",
                )
            ],
        )

        self._definition_lookup[Drought, "spi6"] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Drought.__name__],
            values=self._definition_values(
                self._bounds[Drought, "spi6"],
                self.spi6_label_description,
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_spi6",
                    label="Standardized Precipitation Index (SPI-6)",
                    description="Index comparing cumulated precipitation for 6 months with the long-term precipitation "
                    "distribution from CORDEX-CORE.",
                    units="index",
                )
            ],
        )

        self._definition_lookup[Wind, "max_speed"] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Wind.__name__],
            values=self._definition_values(
                self._bounds[Wind, "max_speed"],
                self.wind_maxspeed_label_description,
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_wind",
                    label="Max speed",
                    description="Data set of maximum sustained (10 metres) wind speed for various return periods "
                    "from the RAIN project.",
                    units="m/s",
                )
            ],
        )

        self._definition_lookup[Wind, "wind_speed/3s"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[Wind.__name__],
                values=self._definition_values(
                    self._bounds[Wind, "wind_speed/3s"],
                    self.wind_3s_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_wind_speed/3s",
                        label="Wind speed (3s)",
                        description="Maximum 10 metres 3 second average gust peak wind speed for different return periods, "
                        "inferred from the Copernicus WISC European storm event set.",
                        units="m/s",
                    )
                ],
            )
        )

        self._definition_lookup[Fire, "fire_probability"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[Fire.__name__],
                values=self._definition_values(
                    self._bounds[Fire, "fire_probability"],
                    self.fire_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_fire",
                        label="Fire probability",
                        description="Fire dataset by Alpha-Klima replicating the paper 'Climate Change Risk Indicators "
                        "for Central Banking: Explainable AI in Fire Risk Estimations'.",
                        units="probability",
                    )
                ],
            )
        )

        self._definition_lookup[RiverineInundation, "flood_depth"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[RiverineInundation.__name__, CoastalInundation.__name__],
                values=self._definition_values(
                    self._bounds[RiverineInundation, "flood_depth"],
                    self.inundation_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_flood_depth",
                        label="Flood depth",
                        description="Water level rise (m) based on the extreme events intensities (per return period).",
                        units="metres",
                    )
                ],
            )
        )

        self._definition_lookup[CoastalInundation, "flood_depth"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[RiverineInundation.__name__, CoastalInundation.__name__],
                values=self._definition_values(
                    self._bounds[CoastalInundation, "flood_depth"],
                    self.inundation_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_flood_depth",
                        label="Flood depth",
                        description="Water level rise (m) based on the extreme events intensities (per return period).",
                        units="metres",
                    )
                ],
            )
        )

        self._definition_lookup[ChronicHeat, "mean_degree_days/above/32c"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[ChronicHeat.__name__],
                values=self._definition_values(
                    self._bounds[ChronicHeat, "mean_degree_days/above/32c"],
                    self.chronic_heat_mdd_32c_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_mean_degree_days/above/32c",
                        label="Mean degree days (above 32C)",
                        description="Mean degree days above 32C.",
                        units="degree days",
                    )
                ],
            )
        )

        self._definition_lookup[ChronicHeat, "mean_degree_days/above/index"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[ChronicHeat.__name__],
                values=self._definition_values(
                    self._bounds[ChronicHeat, "mean_degree_days/above/index"],
                    self.chronic_heat_mdd_index_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_mean_degree_days/above/index",
                        label="Mean degree days (above index)",
                        description="Mean degree days above index.",
                        units="degree days",
                    )
                ],
            )
        )

        self._definition_lookup[ChronicHeat, "mean_work_loss/high"] = (
            ScoreBasedRiskMeasureDefinition(
                hazard_types=[ChronicHeat.__name__],
                values=self._definition_values(
                    self._bounds[ChronicHeat, "mean_work_loss/high"],
                    self.chronic_heat_mwl_label_description,
                ),
                underlying_measures=[
                    RiskMeasureDefinition(
                        measure_id="measure_mean_work_loss/high",
                        label="Mean work loss (high intensity)",
                        description="Mean work loss (high intensity).",
                        units="fraction",
                    )
                ],
            )
        )

    def _definition_values(
        self,
        bounds: ECBBounds,
        label_description: Callable[[ECBBounds, str], Tuple[str, str]],
    ):
        risk_score_values = []
        for category, lower, upper in zip(
            bounds.categories, bounds.lower, bounds.upper
        ):
            if lower == float("-inf"):
                bound_text = f"under {upper}"
            elif upper == float("inf"):
                bound_text = f"over {lower}"
            else:
                bound_text = f"between {lower} and {upper}"

            label, description = label_description(bounds, bound_text)
            rsv = RiskScoreValue(
                value=category,
                label=label,
                description=description,
            )
            risk_score_values.append(rsv)
        return risk_score_values

    def subsidence_label_description(self, bounds: ECBHazardBounds, bound_text: str):
        label = f"Scored by index bands: Value {bound_text} {bounds.units}."
        description = (
            "Score (1-5) from the Joint Research Centre (JRC) DRMKC RDH based on "
            "soils' clay, silt, sand and water content."
        )
        return label, description

    def water_stress_label_description(self, bounds: ECBHazardBounds, bound_text: str):
        label = f"Scored by ratio bands: Ratio {bound_text} {bounds.units}."
        description = (
            "Scores derived from the World Resources Institute (WRI) "
            "Aqueduct 4.0 based on the ratio between water demand and water supply."
        )
        return label, description

    def landslide_label_description(self, bounds: ECBImpactBounds, bound_text: str):
        label = f"Scored by impact on asset value: Impact {bound_text} {bounds.units}."
        description = (
            "Score (1-5) from the Joint Research Centre (JRC) DRMKC RDH based on characteristics "
            "of the terrain combined with daily maximum precipitation (per return period)."
        )
        return label, description

    def cdd_label_description(self, bounds: ECBHazardBounds, bound_text: str):
        label = f"Scored by thresholds based on consecutive dry days a year: Number of cdd {bound_text} {bounds.units}."
        description = "Maximum number of consecutive dry days (with precipitation < 1mm per day) from CORDEX-CORE."
        return label, description

    def spi6_label_description(self, bounds: ECBHazardBounds, bound_text: str):
        label = f"Scored by thresholds based on the IPCC Standard Precipitation Index: Value {bound_text} {bounds.units}."
        description = (
            "Index comparing cumulated precipitation for 6 months with the long-term precipitation "
            "distribution from CORDEX-CORE."
        )
        return label, description

    def fire_label_description(self, bounds: ECBHazardBounds, bound_text: str):
        label = f"Scored by probability thresholds: Probability {bound_text} {bounds.units}."
        description = (
            "Fire dataset by Alpha-Klima replicating the paper 'Climate Change Risk Indicators "
            "for Central Banking: Explainable AI in Fire Risk Estimations'."
        )
        return label, description

    def wind_maxspeed_label_description(self, bounds: ECBImpactBounds, bound_text: str):
        label = f"Scored by impact on asset value: Impact {bound_text} {bounds.units}."
        description = (
            "Data set of maximum sustained (10 metres) wind speed for various return periods "
            "from the RAIN project."
        )
        return label, description

    def wind_3s_label_description(self, bounds: ECBImpactBounds, bound_text: str):
        label = f"Scored by impact on asset value: Impact {bound_text} {bounds.units}."
        description = (
            "Maximum 10 metres 3 second average gust peak wind speed for different return periods, "
            "inferred from the Copernicus WISC European storm event set."
        )
        return label, description

    def inundation_label_description(self, bounds: ECBImpactBounds, bound_text: str):
        label = f"Scored by impact on asset value: Impact {bound_text} {bounds.units}."
        description = "Water level rise (m) based on the extreme events intensities (per return period)."
        return label, description

    def chronic_heat_mdd_32c_label_description(
        self, bounds: ECBImpactBounds, bound_text: str
    ):
        label = f"Scored based on worldwide energy consumption: Value {bound_text} {bounds.units}."
        description = "Mean degree days above 32C."
        return label, description

    def chronic_heat_mdd_index_label_description(
        self, bounds: ECBImpactBounds, bound_text: str
    ):
        label = f"Scored based on worldwide energy consumption: Value {bound_text} {bounds.units}."
        description = "Mean degree days above index."
        return label, description

    def chronic_heat_mwl_label_description(
        self, bounds: ECBImpactBounds, bound_text: str
    ):
        label = f"Scored based on mean work loss (high intensity): Mean work loss {bound_text} {bounds.units}."
        description = "Mean work loss (high intensity)."
        return label, description

    def ground_shaking_label_description(
        self, bounds: ECBImpactBounds, bound_text: str
    ):
        # You can tune wording, but structurally this matches the others
        label = f"Scored based on worldwide Peak Ground Acceleration (PGA): PGA {bound_text} {bounds.units}."
        description = (
            "Peak Ground Acceleration (PGA) with 10% probability of exceedance in 50 years "
            "from the GEM Global Seismic Hazard Model v2023.1."
        )
        return label, description

    def calc_measure(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        histo_impact_res: AssetImpactResult,
        future_impact_res: AssetImpactResult,
    ) -> Optional[Measure]:
        if (
            isinstance(histo_impact_res, EmptyImpactDistrib)
            or isinstance(future_impact_res, EmptyImpactDistrib)
            or future_impact_res.hazard_data is None
            or (hazard_type, indicator_id) not in self._bounds.keys()
        ):
            return None

        bounds = self._bounds[hazard_type, indicator_id]

        if isinstance(bounds, ECBHazardBounds):
            hazard_data = list(future_impact_res.hazard_data)
            if len(hazard_data) > 1:
                raise ValueError("ambiguous hazard data response")
            resp = hazard_data[0]
            if hazard_type.__name__ == "Landslide" and isinstance(
                resp, HazardEventDataResponse
            ):
                index = np.where(resp.return_periods == 100)[0][0]
                param = resp.intensities[index]
            elif isinstance(resp, HazardParameterDataResponse):
                param = resp.parameter
            else:
                raise ValueError("Hazard Data Failed Response")
            if math.isnan(param):
                return Measure(
                    score=Category.NODATA,
                    measure_0=float(param),
                    definition=self.get_definition(hazard_type, indicator_id),
                )
            else:
                index = np.searchsorted(bounds.lower, param, side="left") - 1
                return Measure(
                    score=bounds.categories[index],
                    measure_0=float(param),
                    definition=self.get_definition(hazard_type, indicator_id),
                )

        elif isinstance(bounds, ECBImpactBounds):
            if (
                isinstance(future_impact_res, EmptyImpactDistrib)
                or future_impact_res.hazard_data is None
                or isinstance(future_impact_res.impact, EmptyImpactDistrib)
            ):
                return None
            assert future_impact_res.impact is not None
            assert histo_impact_res.impact is not None

            if not isinstance(future_impact_res.impact, ImpactDistrib):
                raise ValueError(
                    "future_impact_res.impact is not ImpactDistrib instance"
                )
            mean_intensity = future_impact_res.impact.mean_impact()

            if math.isnan(mean_intensity):
                return Measure(
                    score=Category.NODATA,
                    measure_0=float(mean_intensity),
                    definition=self.get_definition(hazard_type, indicator_id),
                )

            index = np.searchsorted(bounds.lower, mean_intensity, side="left") - 1

            return Measure(
                score=bounds.categories[index],
                measure_0=mean_intensity,
                definition=self.get_definition(hazard_type, indicator_id),
            )

        else:
            raise NotImplementedError("impact distribution case not implemented yet")

    def get_definition(
        self, hazard_type: Type[Hazard], indicator_id: Optional[str] = None
    ):
        return self._definition_lookup.get((hazard_type, indicator_id), None)

    def supported_hazards(self) -> Set[type]:
        return set(
            [
                CoastalInundation,
                RiverineInundation,
                WaterRisk,
                Subsidence,
                Drought,
                Landslide,
                Wind,
                Fire,
                ChronicHeat,
            ]
        )
