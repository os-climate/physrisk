from enum import Enum
from typing import Callable, Optional, Set, Type

from physrisk.api.v1.impact_req_resp import (
    Category,
    RiskMeasureDefinition,
    RiskScoreValue,
    ScoreBasedRiskMeasureDefinition,
)
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    Hazard,
    RiverineInundation,
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
                f"{self.measure_thresholds_acute[Threshold.ABS_LOW]*100:0.0f}% of asset value."
            )
        elif category == Category.MEDIUM:
            description = (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss is more than "
                f"{self.measure_thresholds_acute[Threshold.ABS_LOW]*100:0.0f}% but increases by less than "
                f"{self.measure_thresholds_acute[Threshold.CHANGE]*100:0.0f}% of asset value over historical baseline."
            )
        elif category == Category.HIGH:
            description = (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss is more than "
                f"{self.measure_thresholds_acute[Threshold.ABS_LOW]*100:0.0f}% and increases by more than "
                f"{self.measure_thresholds_acute[Threshold.CHANGE]*100:0.0f}% of asset value over historical baseline."
            )
        elif category == Category.REDFLAG:
            description = (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss is more than "
                f"{self.measure_thresholds_acute[Threshold.ABS_HIGH]*100:0.0f}% and increases by more than "
                f"{self.measure_thresholds_acute[Threshold.CHANGE]*100:0.0f}% of asset value over historical baseline."
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
                f"{self.measure_thresholds_cooling[Threshold.CHANGE]*100:0.0f}% over historical baseline."
            )
        elif category == Category.HIGH:
            description = (
                f"Expected cooling annual energy consumption is more than "
                f"{self.measure_thresholds_cooling[Threshold.ABS_LOW]}kWh and increases by more than "
                f"{self.measure_thresholds_cooling[Threshold.CHANGE]*100:0.0f}% over historical baseline."
            )
        elif category == Category.REDFLAG:
            description = (
                f"Expected cooling annual energy consumption is more than "
                f"{self.measure_thresholds_cooling[Threshold.ABS_HIGH]}kWh and increases by more than "
                f"{self.measure_thresholds_cooling[Threshold.CHANGE]*100:0.0f}% over historical baseline."
            )
        else:
            description = "No Data"
        return description

    def calc_measure(
        self,
        hazard_type: Type[Hazard],
        base_impact_res: AssetImpactResult,
        impact_res: AssetImpactResult,
    ) -> Optional[Measure]:
        if isinstance(base_impact_res.impact, EmptyImpactDistrib) or isinstance(
            impact_res.impact, EmptyImpactDistrib
        ):
            return None
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

    def get_definition(self, hazard_type: type):
        return self._definition_lookup.get(hazard_type, None)

    def supported_hazards(self) -> Set[type]:
        return set([RiverineInundation, CoastalInundation, Wind, ChronicHeat])
