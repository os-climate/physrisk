from typing import List

from physrisk.api.v1.impact_req_resp import Category, Indicator, RiskMeasureResult
from physrisk.kernel.hazards import CoastalInundation, Hazard, HazardKind, RiverineInundation
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.risk import RiskMeasureCalculator


class RealEstateToyRiskMeasures(RiskMeasureCalculator):
    """Toy model for calculating risk measures for real estate assets."""

    def __init__(self):
        self.model_summary = {"*Toy* model for real estate risk assessment."}
        self._category_defns = {
            Category.NODATA: "No information.",
            Category.LOW: "Marginal impact on real estate valuation very unlikely under RCP 8.5.",
            Category.MEDIUM: "Material marginal impact on real estate valuation unlikely under RCP 8.5.",
            Category.HIGH: "Material marginal impact on real estate valuation possible under RCP 8.5.",
            Category.REDFLAG: "Material marginal impact on real estate valuation likely under RCP 8.5 "
            "with possible impact on availability of insurance.",
        }

    def calc_measure(self, hazard_type: type, base_impact: ImpactDistrib, impact: ImpactDistrib) -> RiskMeasureResult:
        if Hazard.kind(base_impact.hazard_type) == HazardKind.acute:
            return_period = 100.0  # criterion based on 1 in 100-year flood events
            histo_loss = base_impact.to_exceedance_curve().get_value(1.0 / return_period)
            future_loss = impact.to_exceedance_curve().get_value(1.0 / return_period)
            loss_increase = future_loss - histo_loss

            if loss_increase > 0.3:
                category = Category.REDFLAG
            elif loss_increase > 0.1:
                category = Category.HIGH
            elif loss_increase > 0.05:
                category = Category.MEDIUM
            else:
                category = Category.LOW

            summary = (
                f"Projected 1-in-{return_period:0.0f} year annual loss "
                f"increases by {loss_increase*100:0.0f}% of asset value over historical baseline. "
            )
            cat_defn = self._category_defns[category]
            indicator = Indicator(value=loss_increase, label=f"{loss_increase * 100:0.0f}%")

            return RiskMeasureResult(category=category, cat_defn=cat_defn, indicators=[indicator], summary=summary)
        else:
            raise NotImplementedError()

    def supported_hazards(self) -> List[type]:
        return [RiverineInundation, CoastalInundation]
