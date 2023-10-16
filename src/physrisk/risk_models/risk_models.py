from typing import Set

from physrisk.api.v1.impact_req_resp import Category, RiskScoreValue, ScoreBasedRiskMeasureDefinition
from physrisk.kernel.hazards import CoastalInundation, RiverineInundation, Wind
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.risk import Measure, RiskMeasureCalculator


class RealEstateToyRiskMeasures(RiskMeasureCalculator):
    """Toy model for calculating risk measures for real estate assets."""

    def __init__(self):
        self.model_summary = {"*Toy* model for real estate risk assessment."}
        self.return_period = 100.0  # criteria based on 1 in 100-year flood or cyclone events
        self.measure_thresholds = {
            Category.REDFLAG: 0.3,
            Category.HIGH: 0.1,
            Category.MEDIUM: 0.05,
        }
        definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[RiverineInundation.__name__, CoastalInundation.__name__],
            values=[
                RiskScoreValue(
                    value=Category.REDFLAG,
                    label="Material marginal impact on valuation very likely.",
                    description=self._description(Category.REDFLAG),
                ),
                RiskScoreValue(
                    value=Category.HIGH,
                    label="Material marginal impact on valuation likely.",
                    description=self._description(Category.HIGH),
                ),
                RiskScoreValue(
                    value=Category.MEDIUM,
                    label="Marginal impact on valuation possible.",
                    description=self._description(Category.MEDIUM),
                ),
                RiskScoreValue(
                    value=Category.LOW,
                    label="Marginal impact on valuation very unlikely.",
                    description=self._description(Category.LOW),
                ),
                RiskScoreValue(value=Category.NODATA, label="No data.", description="No data."),
            ],
            child_measure_ids=["annual_loss_{return_period:0.0f}year"],
        )
        self.measure_definitions = [definition]
        self._definition_lookup = {RiverineInundation: definition, CoastalInundation: definition}

    def _description(self, category: Category):
        return (
            (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss "
                f"increases by less than {self.measure_thresholds[Category.MEDIUM]*100:0.0f}% of asset value "
                f"over historical baseline."
            )
            if category == Category.LOW
            else (
                f"Projected 1-in-{self.return_period:0.0f} year annual loss "
                f"increases by at least {self.measure_thresholds[category]*100:0.0f}% of asset value over "
                f"historical baseline."
            )
        )

    def calc_measure(self, hazard_type: type, base_impact: ImpactDistrib, impact: ImpactDistrib) -> Measure:
        return_period = 100.0  # criterion based on 1 in 100-year flood events
        histo_loss = base_impact.to_exceedance_curve().get_value(1.0 / return_period)
        future_loss = impact.to_exceedance_curve().get_value(1.0 / return_period)
        loss_increase = future_loss - histo_loss

        if loss_increase > self.measure_thresholds[Category.REDFLAG]:
            score = Category.REDFLAG
        elif loss_increase > self.measure_thresholds[Category.HIGH]:
            score = Category.HIGH
        elif loss_increase > self.measure_thresholds[Category.MEDIUM]:
            score = Category.MEDIUM
        else:
            score = Category.LOW
        return Measure(score=score, measure_0=loss_increase, definition=self.get_definition(hazard_type))

    def get_definition(self, hazard_type: type):
        return self._definition_lookup.get(hazard_type, None)

    def supported_hazards(self) -> Set[type]:
        return set([RiverineInundation, CoastalInundation, Wind])
