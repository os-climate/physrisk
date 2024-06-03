import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence, Set, Tuple, Type

import numpy as np
from pint import UnitRegistry

from physrisk.api.v1.impact_req_resp import (
    Category,
    RiskMeasureDefinition,
    RiskScoreValue,
    ScoreBasedRiskMeasureDefinition,
)
from physrisk.kernel.hazard_model import HazardEventDataResponse, HazardParameterDataResponse
from physrisk.kernel.hazards import Hazard, Wind
from physrisk.kernel.impact import AssetImpactResult
from physrisk.kernel.risk import Measure, RiskMeasureCalculator

ureg = UnitRegistry()


class Threshold(int, Enum):
    ABS_HIGH = 0
    ABS_LOW = 1
    CHANGE = 2


class BoundsType(int, Enum):
    IMPACT = 0  # fractional damage or disruption
    INDICATOR_VALE = 1  #


@dataclass
class ImpactBounds:
    """Category applies if lower <= value < upper"""

    type: BoundsType  # whether an impact (fractional damage or fractional disruption) or a hazard indicator value
    category: Category
    lower: float
    upper: float


@dataclass
class HazardIndicatorBounds:
    category: Category
    hazard_type: Type[Hazard]
    indicator_id: str
    units: str
    lower: float
    upper: float
    indicator_return: float = 100


class GenericScoreBasedRiskMeasures(RiskMeasureCalculator):
    """A generic score based risk measure. 'Generic' indicates that the user of the score is unknown.
    i.e. it is unknown whether the user owns the assets in question, or interested in the assets from
    the point of view of loan origination or project financing.
    """

    def __init__(self):
        self.model_summary = {"Generic score based risk measure."}
        # fmt: off
        self.wind_bounds = [
            HazardIndicatorBounds(category=Category.NODATA, hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=float("-inf"), upper=63), # noqa
            HazardIndicatorBounds(category=Category.LOW,    hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=63, upper=90), # noqa
            HazardIndicatorBounds(category=Category.MEDIUM, hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=90, upper=119), # noqa
            HazardIndicatorBounds(category=Category.HIGH,   hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=119, upper=178), # noqa
            HazardIndicatorBounds(category=Category.REDFLAG,hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=178, upper=float("inf")) # noqa
        ]
        # fmt: on
        self._definition_lookup = {}
        self._bounds_lookup = {Wind: self._bounds_to_lookup(self.wind_bounds)}
        self._definition_lookup[Wind] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Wind.__name__],
            values=self._definition_values(self.wind_bounds, self.wind_label_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_wind",
                    label="1-in-100 year sustained wind speed.",
                    description="1-in-100 year sustained wind speed.",
                )
            ],
        )

    def _bounds_to_lookup(self, bounds: Sequence[HazardIndicatorBounds]):
        lower_bounds = np.array([b.lower for b in bounds])
        categories = np.array([b.category for b in bounds])
        return (lower_bounds, categories, bounds)

    def _definition_values(
        self,
        bounds: Sequence[HazardIndicatorBounds],
        label_description: Callable[[HazardIndicatorBounds], Tuple[str, str]],
    ):
        risk_score_values = []
        for b in bounds:
            label, description = label_description(b)
            rsv = RiskScoreValue(
                value=b.category,
                label=label,
                description=description,
            )
            risk_score_values.append(rsv)
        return risk_score_values

    def wind_label_description(self, bounds: HazardIndicatorBounds):
        label = f"Max wind speed between {bounds.lower} and {bounds.upper} {bounds.units}"
        description = f"Max sustained wind speed between {bounds.lower} and {bounds.upper} {bounds.units}"
        return label, description

    def calc_measure(
        self, hazard_type: Type[Hazard], base_impact_res: AssetImpactResult, impact_res: AssetImpactResult
    ) -> Measure:
        # in general we want to use the impact distribution, but in certain circumstances we can use
        # the underlying hazard data some care is needed given that vulnerability models are interchangeable
        # (what if the vulnerability model used does not make use of the hazard indicator we require?)
        (lower_bounds, categories, bounds) = self._bounds_lookup[hazard_type]
        if isinstance(bounds[0], HazardIndicatorBounds):
            assert impact_res.hazard_data is not None
            hazard_data = list(impact_res.hazard_data)
            if len(hazard_data) > 1:
                # the vulnerability model makes more than one request: ambiguous
                raise ValueError("ambiguous hazard data response")
            resp = hazard_data[0]
            if isinstance(resp, HazardParameterDataResponse):
                param = resp.parameter
            elif isinstance(resp, HazardEventDataResponse):
                return_period = bounds[0].indicator_return
                param = float(np.interp(return_period, resp.return_periods, resp.intensities))
                if resp.units != "default":
                    param = ureg.convert(param, resp.units, bounds[0].units)
            if math.isnan(param):
                return Measure(
                    score=Category.NODATA, measure_0=float(param), definition=self.get_definition(hazard_type)
                )
            else:
                index = np.searchsorted(lower_bounds, param, side="right") - 1
                return Measure(
                    score=categories[index], measure_0=float(param), definition=self.get_definition(hazard_type)
                )
        else:
            raise NotImplementedError("impact distribution case not implemented yet")

    def get_definition(self, hazard_type: Type[Hazard]):
        return self._definition_lookup.get(hazard_type, None)

    def supported_hazards(self) -> Set[type]:
        return set([Wind])  # RiverineInundation, CoastalInundation,
