import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, Type

import numpy as np
from pint import UnitRegistry

from physrisk.api.v1.impact_req_resp import (
    Category,
    RiskMeasureDefinition,
    RiskScoreValue,
    ScoreBasedRiskMeasureDefinition,
)
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import (
    HazardEventDataResponse,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    Drought,
    Fire,
    Hail,
    Hazard,
    PluvialInundation,
    Precipitation,
    RiverineInundation,
    Wind,
)
from physrisk.kernel.impact import AssetImpactResult
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.risk import Measure, MeasureKey, RiskMeasureCalculator

ureg = UnitRegistry()


@dataclass
class ImpactBoundsJoint:
    """Category applies if lower <= value < upper"""

    measure1: Any
    return1: float
    measure2: Any
    return2: float
    categories: Sequence[Category]
    lower1: Sequence[float]
    upper1: Sequence[float]
    lower2: Sequence[float]
    upper2: Sequence[float]


@dataclass
class HazardIndicatorBounds:
    hazard_type: Type[Hazard]
    indicator_id: str
    units: str
    indicator_return: float
    categories: List[Category]
    lower: List[float]
    upper: List[float]


class GenericScoreBasedRiskMeasures(RiskMeasureCalculator):
    """A generic score based risk measure. 'Generic' indicates that the user of the score is unknown.
    i.e. it is unknown whether the user owns the assets in question, or interested in the assets from
    the point of view of loan origination or project financing.
    """

    def __init__(self):
        self.model_summary = {"Generic score based risk measure."}
        # fmt: off
        self._bounds = {
            Wind: HazardIndicatorBounds(
                categories=[Category.LOW, Category.MEDIUM, Category.HIGH, Category.REDFLAG], 
                hazard_type=Wind,
                indicator_id="max_speed",
                indicator_return=100,
                units="km/h",
                lower=[float("-inf"), 90,  119, 178],
                upper=[90,            119, 178, float("inf")]
                ), # noqa
            Hail: HazardIndicatorBounds(
                categories=[Category.LOW, Category.MEDIUM, Category.HIGH, Category.REDFLAG], 
                hazard_type=Hail,
                indicator_id="days/above/5cm",
                indicator_return=1,
                units="days/year",
                lower=[float("-inf"), 1, 2, 3],
                upper=[1,             2, 3, float("inf")]
                ), # noqa
            Drought: HazardIndicatorBounds(
                categories=[Category.LOW, Category.MEDIUM, Category.HIGH, Category.REDFLAG], 
                hazard_type=Drought,
                indicator_id="months/spei3m/below/-2",
                indicator_return=1,
                units="months/year",
                lower=[float("-inf"), 0.25, 0.5, 1],
                upper=[0.25,          0.5,  1,   float("inf")]
                ), # noqa
            Fire: HazardIndicatorBounds(
                categories=[Category.LOW, Category.MEDIUM, Category.HIGH, Category.REDFLAG], 
                hazard_type=Fire,
                indicator_id="fire_probability",
                indicator_return=1,
                units="%",
                lower=[float("-inf"), 20, 35, 50],
                upper=[20,            35, 50, float("inf")]
                ), # noqa        
            Precipitation: HazardIndicatorBounds(
                categories=[Category.LOW, Category.MEDIUM, Category.HIGH, Category.REDFLAG], 
                hazard_type=Precipitation,
                indicator_id="max/daily/water_equivalent",
                indicator_return=100,
                units="mm/day",
                lower=[float("-inf"), 100, 130, 160],
                upper=[100,           130, 160, float("inf")]
                ), # noqa
            ChronicHeat: HazardIndicatorBounds(
                categories=[Category.LOW, Category.MEDIUM, Category.HIGH, Category.REDFLAG], 
                hazard_type=ChronicHeat,
                indicator_id="days/above/35c",
                indicator_return=1,
                units="days/year",
                lower=[float("-inf"), 10, 20, 30],
                upper=[10,            20, 30, float("inf")]
                ), # noqa
        }
        acute_bounds = ImpactBoundsJoint(
            categories=[Category.LOW, Category.MEDIUM, Category.HIGH, Category.REDFLAG],
            measure1=self._impact, 
            return1=100, # i.e. 1-in-100 year impact
            measure2=self._delta_impact,
            return2=100, # i.e. change in 1-in-100 year impact from historical to future scenario
            lower1=[float("-inf"), 0.03,          0.03,         0.1], # applies to impact
            upper1=[0.03,          0.1,           0.1,          float("inf")],
            lower2=[float("-inf"), float("-inf"), 0.03,         0.03], # applies to change of impact from baseline to future
            upper2=[0.03,          0.03,          float("inf"), float("inf")],
        ) # noqa
        self._bounds[RiverineInundation] = acute_bounds
        self._bounds[CoastalInundation] = acute_bounds
        self._bounds[PluvialInundation] = acute_bounds
        # fmt: on
        self._definition_lookup = {}
        self._definition_lookup[Wind] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Wind.__name__],
            values=self._definition_values(
                self._bounds[Wind], self.wind_label_description
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_wind",
                    label="1-in-100 year 1 minute sustained wind speed score.",
                    description=("1-in-100 year 1 minute sustained wind speed."),
                    units="km/hr",
                )
            ],
        )
        self._definition_lookup[Hail] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Hail.__name__],
            values=self._definition_values(
                self._bounds[Hail], self.hail_label_description
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_hail",
                    label="Number of days per year where large hail (>2 inches / 5 cm in diameter) is possible.",
                    description=(
                        "Number of days per year where large hail (>2 inches / 5 cm in diameter) is possible."
                    ),
                )
            ],
        )
        self._definition_lookup[Drought] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Drought.__name__],
            values=self._definition_values(
                self._bounds[Drought], self.drought_label_description
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_drought",
                    label=(
                        "Months per year where the rolling 3-month average Standardized Precipitation "
                        "Evapotranspiration Index is <2."
                    ),
                    description=(
                        "Months per year where the rolling 3-month average Standardized Precipitation "
                        "Evapotranspiration Index is <2."
                    ),
                )
            ],
        )
        self._definition_lookup[Fire] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Fire.__name__],
            values=self._definition_values(
                self._bounds[Fire], self.fire_label_description
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_fire",
                    label=("Wildfire probability."),
                    description=(
                        "Maximum value, across all months, of the monthly probability of a wildfire within "
                        "100km of the location."
                    ),
                )
            ],
        )
        self._definition_lookup[Precipitation] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Precipitation.__name__],
            values=self._definition_values(
                self._bounds[Precipitation], self.precipitation_label_description
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_precipitation",
                    label="1-in-100 year maximum daily total water equivalent precipitation.",
                    description=(
                        "1-in-100 year maximum daily total water equivalent precipitation."
                    ),
                    units="mm",
                )
            ],
        )
        self._definition_lookup[ChronicHeat] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[ChronicHeat.__name__],
            values=self._definition_values(
                self._bounds[ChronicHeat], self.heat_label_description
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_chronicHeat",
                    label="Days per year with temperature > 35°C.",
                    description=("Days per year with temperature > 35°C."),
                    units="days/year",
                )
            ],
        )
        acute_definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[
                RiverineInundation.__name__,
                CoastalInundation.__name__,
                PluvialInundation.__name__,
            ],  # Wind.__name__],
            values=self._definition_values_impact(
                lambda category: self._acute_description(category, acute_bounds)
            ),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measures_0",
                    label=f"1-in-{acute_bounds.return1:0.0f} year annual loss.",
                    description=f"1-in-{acute_bounds.return1:0.0f} year loss as fraction of asset insured value.",
                )
            ],
        )
        self._definition_lookup[RiverineInundation] = acute_definition

    def _definition_values(
        self,
        bounds: HazardIndicatorBounds,
        label_description: Callable[
            [HazardIndicatorBounds, float, float], Tuple[str, str]
        ],
    ):
        risk_score_values = []
        for category, lower, upper in zip(
            bounds.categories, bounds.lower, bounds.upper
        ):
            label, description = label_description(bounds, lower, upper)
            rsv = RiskScoreValue(
                value=category,
                label=label,
                description=description,
            )
            risk_score_values.append(rsv)
        return risk_score_values

    def wind_label_description(
        self, bounds: HazardIndicatorBounds, lower: float, upper: float
    ):
        label = f"Max wind speed between {lower} and {upper} {bounds.units}"
        description = (
            f"Max sustained wind speed between {lower} and {upper} {bounds.units}"
        )
        return label, description

    def hail_label_description(
        self, bounds: HazardIndicatorBounds, lower: float, upper: float
    ):
        label = (
            f"Max number of days per year between {lower} and {upper} {bounds.units}"
        )
        description = (
            f"Max number of days per year between {lower} and {upper} {bounds.units}"
        )
        return label, description

    def drought_label_description(
        self, bounds: HazardIndicatorBounds, lower: float, upper: float
    ):
        label = f"Max months per year between {lower} and {upper} {bounds.units}"
        description = f"Max months per year between {lower} and {upper} {bounds.units}"
        return label, description

    def fire_label_description(
        self, bounds: HazardIndicatorBounds, lower: float, upper: float
    ):
        label = (
            f"Max value, across all months, of the monthly probability of a wildfire between "
            f"{lower} and {upper} {bounds.units}"
        )
        description = (
            f"Max value, across all months, of the monthly probability of a wildfire between "
            f"{lower} and {upper} {bounds.units}"
        )
        return label, description

    def precipitation_label_description(
        self, bounds: HazardIndicatorBounds, lower: float, upper: float
    ):
        label = (
            f"Max daily total water equivalent precipitation between {lower} and "
            f"{upper} {bounds.units}"
        )
        description = (
            f"Max daily total water equivalent precipitation between {lower} and "
            f"{upper} {bounds.units}"
        )
        return label, description

    def heat_label_description(
        self, bounds: HazardIndicatorBounds, lower: float, upper: float
    ):
        label = f"Max days per year between {lower} and {upper} {bounds.units}"
        description = f"Max days per year between {lower} and {upper} {bounds.units}"
        return label, description

    def _definition_values_impact(self, description: Callable[[Category], str]):
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

    def _acute_description(self, category: Category, bounds: ImpactBoundsJoint):
        index = bounds.categories.index(category)
        if index > 0:
            description = (
                f"Projected 1-in-{bounds.return1} year annual loss is between {bounds.lower1[index]} "
                f"and {bounds.upper1[index]} and change in projected 1-in-{bounds.return2} annual loss "
                f"is between {bounds.lower2[index]} and {bounds.upper2[index]}."
            )
        else:
            description = "No Data"
        return description

    def calc_measure(
        self,
        hazard_type: Type[Hazard],
        histo_impact_res: AssetImpactResult,
        future_impact_res: AssetImpactResult,
    ) -> Measure:
        # in general we want to use the impact distribution, but in certain circumstances we can use
        # the underlying hazard data some care is needed given that vulnerability models are interchangeable
        # (what if the vulnerability model used does not make use of the hazard indicator we require?)

        bounds = self._bounds[hazard_type]
        if isinstance(bounds, HazardIndicatorBounds):
            assert future_impact_res.hazard_data is not None
            hazard_data = list(future_impact_res.hazard_data)
            if len(hazard_data) > 1:
                # the vulnerability model makes more than one request: ambiguous
                raise ValueError("ambiguous hazard data response")
            resp = hazard_data[0]
            if isinstance(resp, HazardParameterDataResponse):
                param = resp.parameter
            elif isinstance(resp, HazardEventDataResponse):
                return_period = bounds.indicator_return
                param = float(
                    np.interp(return_period, resp.return_periods, resp.intensities)
                )
                if resp.units != "default":
                    param = ureg.convert(param, resp.units, bounds.units)
            if math.isnan(param):
                return Measure(
                    score=Category.NODATA,
                    measure_0=float(param),
                    definition=self.get_definition(hazard_type),
                )
            else:
                index = np.searchsorted(bounds.lower, param, side="right") - 1
                return Measure(
                    score=bounds.categories[index],
                    measure_0=float(param),
                    definition=self.get_definition(hazard_type),
                )
        elif isinstance(bounds, ImpactBoundsJoint):
            assert future_impact_res.impact is not None
            assert histo_impact_res.impact is not None
            measure1 = bounds.measure1(
                histo_impact_res.impact, future_impact_res.impact, bounds.return1
            )
            measure2 = bounds.measure2(
                histo_impact_res.impact, future_impact_res.impact, bounds.return2
            )
            score = Category.NODATA
            for category, lower1, upper1, lower2, upper2 in zip(
                bounds.categories,
                bounds.lower1,
                bounds.upper1,
                bounds.lower2,
                bounds.upper2,
            ):
                if (
                    measure1 >= lower1
                    and measure1 < upper1
                    and measure2 >= lower2
                    and measure2 < upper2
                ):
                    score = category
                    break
            return Measure(
                score=score,
                measure_0=float(measure1),
                definition=self.get_definition(hazard_type),
            )
        else:
            raise NotImplementedError("impact distribution case not implemented yet")

    def _impact(
        self,
        histo_impact: ImpactDistrib,
        future_impact: ImpactDistrib,
        return_period: float,
    ):
        return future_impact.to_exceedance_curve().get_value(1.0 / return_period)

    def _delta_impact(
        self,
        histo_impact: ImpactDistrib,
        future_impact: ImpactDistrib,
        return_period: float,
    ):
        histo_loss = histo_impact.to_exceedance_curve().get_value(1.0 / return_period)
        future_loss = future_impact.to_exceedance_curve().get_value(1.0 / return_period)
        return future_loss - histo_loss

    def get_definition(self, hazard_type: Type[Hazard]):
        return self._definition_lookup.get(hazard_type, None)

    def supported_hazards(self) -> Set[type]:
        return set(
            [
                Wind,
                Fire,
                Hail,
                ChronicHeat,
                Drought,
                Precipitation,
                CoastalInundation,
                PluvialInundation,
                RiverineInundation,
            ]
        )

    def aggregate_risk_measures(
        self,
        measures: Dict[MeasureKey, Measure],
        assets: Sequence[Asset],
        prosp_scens: Sequence[str],
        years: Sequence[int],
    ) -> Dict[MeasureKey, Measure]:
        aggregate_measures = {}
        aggregate_measures.update(measures)
        for asset in assets:
            for scenario in prosp_scens:
                for year in years:
                    # if the Precipitation measures exists but the corresponding PluvialInundation
                    # is not present, proxy PluvialInundation to Precipitation
                    from_key = MeasureKey(asset, scenario, year, PluvialInundation)
                    if from_key not in measures:
                        to_key = MeasureKey(asset, scenario, year, Precipitation)
                        if to_key in measures:
                            aggregate_measures[from_key] = measures[to_key]
        return aggregate_measures
