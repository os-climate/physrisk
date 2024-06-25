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
from physrisk.kernel.hazards import Hazard, Wind, Hail, Drought, Fire, Precipitation, ChronicHeat
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
            HazardIndicatorBounds(category=Category.LOW,    hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=float("-inf"), upper=90), # noqa
            HazardIndicatorBounds(category=Category.MEDIUM, hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=90, upper=119), # noqa
            HazardIndicatorBounds(category=Category.HIGH,   hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=119, upper=178), # noqa
            HazardIndicatorBounds(category=Category.REDFLAG,hazard_type=Wind, indicator_id="max_speed", indicator_return=100, units="km/h", lower=178, upper=float("inf")) # noqa
        ]
        self.hail_bounds = [
            HazardIndicatorBounds(category=Category.LOW,    hazard_type=Hail, indicator_id="days/above/5cm", indicator_return=1, units="days/year", lower=float("-inf"), upper=1), # noqa
            HazardIndicatorBounds(category=Category.MEDIUM, hazard_type=Hail, indicator_id="days/above/5cm", indicator_return=1, units="days/year", lower=1, upper=2), # noqa
            HazardIndicatorBounds(category=Category.HIGH,   hazard_type=Hail, indicator_id="days/above/5cm", indicator_return=1, units="days/year", lower=2, upper=3), # noqa
            HazardIndicatorBounds(category=Category.REDFLAG,hazard_type=Hail, indicator_id="days/above/5cm", indicator_return=1, units="days/year", lower=3, upper=float("inf")) # noqa
        ]
        self.drought_bounds = [
            HazardIndicatorBounds(category=Category.LOW,    hazard_type=Drought, indicator_id="months/spei3m/below/-2", indicator_return=1, units="months/year", lower=float("-inf"), upper=0.25), # noqa
            HazardIndicatorBounds(category=Category.MEDIUM, hazard_type=Drought, indicator_id="months/spei3m/below/-2", indicator_return=1, units="months/year", lower=0.25, upper=0.5), # noqa
            HazardIndicatorBounds(category=Category.HIGH,   hazard_type=Drought, indicator_id="months/spei3m/below/-2", indicator_return=1, units="months/year", lower=0.5, upper=1), # noqa
            HazardIndicatorBounds(category=Category.REDFLAG,hazard_type=Drought, indicator_id="months/spei3m/below/-2", indicator_return=1, units="months/year", lower=1, upper=float("inf")) # noqa
        ]
        self.fire_bounds = [
            HazardIndicatorBounds(category=Category.LOW,    hazard_type=Fire, indicator_id="fire_probability", indicator_return=1, units="% probability/month", lower=float("-inf"), upper=20), # noqa
            HazardIndicatorBounds(category=Category.MEDIUM, hazard_type=Fire, indicator_id="fire_probability", indicator_return=1, units="% probability/month", lower=20, upper=35), # noqa
            HazardIndicatorBounds(category=Category.HIGH,   hazard_type=Fire, indicator_id="fire_probability", indicator_return=1, units="% probability/month", lower=35, upper=50), # noqa
            HazardIndicatorBounds(category=Category.REDFLAG,hazard_type=Fire, indicator_id="fire_probability", indicator_return=1, units="% probability/month", lower=50, upper=float("inf")) # noqa
        ]
        self.precipitation_bounds = [
            HazardIndicatorBounds(category=Category.LOW,    hazard_type=Precipitation, indicator_id="max/daily/water_equivalent", indicator_return=100, units="mm rainfall/day", lower=float("-inf"), upper=100), # noqa
            HazardIndicatorBounds(category=Category.MEDIUM, hazard_type=Precipitation, indicator_id="max/daily/water_equivalent", indicator_return=100, units="mm rainfall/day", lower=100, upper=130), # noqa
            HazardIndicatorBounds(category=Category.HIGH,   hazard_type=Precipitation, indicator_id="max/daily/water_equivalent", indicator_return=100, units="mm rainfall/day", lower=130, upper=160), # noqa
            HazardIndicatorBounds(category=Category.REDFLAG,hazard_type=Precipitation, indicator_id="max/daily/water_equivalent", indicator_return=100, units="mm rainfall/day", lower=160, upper=float("inf")) # noqa
        ]
        self.heat_bounds = [
            HazardIndicatorBounds(category=Category.LOW,    hazard_type=ChronicHeat, indicator_id="days/above/35c", indicator_return=1, units="days/year", lower=float("-inf"), upper=10), # noqa
            HazardIndicatorBounds(category=Category.MEDIUM, hazard_type=ChronicHeat, indicator_id="days/above/35c", indicator_return=1, units="days/year", lower=10, upper=20), # noqa
            HazardIndicatorBounds(category=Category.HIGH,   hazard_type=ChronicHeat, indicator_id="days/above/35c", indicator_return=1, units="days/year", lower=20, upper=30), # noqa
            HazardIndicatorBounds(category=Category.REDFLAG,hazard_type=ChronicHeat, indicator_id="days/above/35c", indicator_return=1, units="days/year", lower=30, upper=float("inf")) # noqa
        ]
        # fmt: on
        self._definition_lookup = {}
        self._bounds_lookup = {
            Wind: self._bounds_to_lookup(self.wind_bounds),
            Hail: self._bounds_to_lookup(self.hail_bounds),
            Drought: self._bounds_to_lookup(self.drought_bounds),
            Fire: self._bounds_to_lookup(self.fire_bounds),
            Precipitation: self._bounds_to_lookup(self.precipitation_bounds),
            ChronicHeat: self._bounds_to_lookup(self.heat_bounds),
        }
        self._definition_lookup[Wind] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Wind.__name__],
            values=self._definition_values(self.wind_bounds, self.wind_label_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_wind",
                    label="1-in-100 year sustained wind speed.",
                    description=(
                      "This wind speed model computes the maximum 1-minute sustained wind speed (in km/hr) experienced over a 100 year return period"
                      "based on mean wind speed distributions from multiple Global Climate Models (GCMs).",
                  )
                )
            ],
        )
        self._definition_lookup[Hail] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Hail.__name__],
            values=self._definition_values(self.hail_bounds, self.hail_label_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_hail",
                    label="Number of days per year where large hail (>2 inches / 5 cm in diameter) is possible.",
                    description=(
                      "This hail model computes the number of days per annum where hail exceeding 5 cm diameter is possible based on the"
                      "mean distribution of several parameters across multiple bias-corrected and downscaled Global Climate Models (GCMs).",
                  )
                )
            ],
        )
        self._definition_lookup[Drought] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Drought.__name__],
            values=self._definition_values(self.drought_bounds, self.drought_label_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_drought",
                    label="Months per year where the rolling 3-month average Standardized Precipitation Evapotranspiration Index is <2.",
                    description=(
                      "This drought model is based on the Standardized Precipitation-Evapotranspiration Index (SPEI). The SPEl is an"
                      "extension of the Standardized Precipitation Index which also considers Potential Evapotranspiration (PET) in determining drought events."
                      "The SPEl is calculated from a log-logistic probability distribution function of climatic water balance (precipitation minus evapotranspiration)"
                      "over a given time scale. The SPEI itself is a standardized variable with a mean value 0 and standard deviation 1."
                      "This drought model computes the number of months per annum where the 3-month rolling average\nof SPEI is below -2 based on the mean values of"
                      "several parameters from bias-corrected and downscaled multiple Global Climate Models (GCMs).",
                  )
                )
            ],
        )
        self._definition_lookup[Fire] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Fire.__name__],
            values=self._definition_values(self.fire_bounds, self.fire_label_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_fire",
                    label="The maximum value, across all months, of the monthly probability of a wildfire within 100km of the location.",
                    description=(
                      "This fire model computes the maximum monthly probability per annum of a wildfire within 100 km of a given location"
                      "based on several parameters from multiple bias corrected and downscaled Global Climate Models (GCMs)."
                      "For example, if the probability of occurrence of a wildfire is 5%% in July, 20%% in August, 10%% in September"
                      "and 0%% for other months, the hazard indicator value is 20%.",
                  )
                )
            ],
        )
        self._definition_lookup[Precipitation] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Precipitation.__name__],
            values=self._definition_values(self.precipitation_bounds, self.precipitation_label_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_precipitation",
                    label="1-in-100 year maximum daily total water equivalent precipitation (in mm).",
                    description=(
                      "This model computes the maximum daily water equivalent precipitation (in mm) measured at the 100 year return period"
                      "based on the mean of the precipitation distribution from multiple bias corrected and downscaled Global Climate Models (GCMs).",
                  )
                )
            ],
        )
        self._definition_lookup[ChronicHeat] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[ChronicHeat.__name__],
            values=self._definition_values(self.heat_bounds, self.heat_label_description),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_chronicHeat",
                    label="Days per year with temperature > 35°C.",
                    description=(
                      "This heat model computes the number of days exceeding 35°C per annum based on the mean of distribution fits"
                      "to the bias-corrected and downscaled high temperature distribution across multiple Global Climate Models (GCMs).",
                  )
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

    def hail_label_description(self, bounds: HazardIndicatorBounds):
        label = f"Max number of days per year between {bounds.lower} and {bounds.upper} {bounds.units}"
        description = f"Max number of days per year between {bounds.lower} and {bounds.upper} {bounds.units}"
        return label, description

    def drought_label_description(self, bounds: HazardIndicatorBounds):
        label = f"Max months per year between {bounds.lower} and {bounds.upper} {bounds.units}"
        description = f"Max months per year between {bounds.lower} and {bounds.upper} {bounds.units}"
        return label, description

    def fire_label_description(self, bounds: HazardIndicatorBounds):
        label = f"Max value, across all months, of the monthly probability of a wildfire between {bounds.lower} and {bounds.upper} {bounds.units}"
        description = f"Max value, across all months, of the monthly probability of a wildfire between {bounds.lower} and {bounds.upper} {bounds.units}"
        return label, description

    def precipitation_label_description(self, bounds: HazardIndicatorBounds):
        label = (
            f"Max daily total water equivalent precipitation between {bounds.lower} and {bounds.upper} {bounds.units}"
        )
        description = (
            f"Max daily total water equivalent precipitation between {bounds.lower} and {bounds.upper} {bounds.units}"
        )
        return label, description

    def heat_label_description(self, bounds: HazardIndicatorBounds):
        label = f"Max days per year between {bounds.lower} and {bounds.upper} {bounds.units}"
        description = f"Max days per year between {bounds.lower} and {bounds.upper} {bounds.units}"
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
        return set([Wind, Fire, Hail, ChronicHeat, Drought, Precipitation])  # RiverineInundation, CoastalInundation,
