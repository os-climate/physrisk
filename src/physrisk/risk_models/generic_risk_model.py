import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Set, Type, Union

import numpy as np
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
from physrisk.kernel.impact_distrib import EmptyImpactDistrib, ImpactDistrib
from physrisk.kernel.risk import Measure, MeasureKey, RiskMeasureCalculator
from pint import UnitRegistry

ureg = UnitRegistry()


@dataclass
class HazardIndicatorBounds:
    hazard_type: Type[Hazard]
    indicator_id: str
    units: str
    indicator_return: float
    categories: Sequence[Category]
    lower: Sequence[float]
    upper: Sequence[float]


@dataclass
class ImpactBounds:
    """Category applies if lower <= value < upper."""

    measure: Any
    categories: Sequence[Category]
    lower: Sequence[float]
    upper: Sequence[float]


@dataclass
class ImpactBoundsJoint:
    """Category applies if lower1 <= value1 < upper1 and lower2 <= value2 < upper2."""

    measure1: Any
    return1: float
    measure2: Any
    return2: float
    categories: Sequence[Category]
    lower1: Sequence[float]
    upper1: Sequence[float]
    lower2: Sequence[float]
    upper2: Sequence[float]


class GenericScoreBasedRiskMeasures(RiskMeasureCalculator):
    """A generic score based risk measure. 'Generic' indicates that the user of the score is unknown.
    i.e. it is unknown whether the user owns the assets in question, or interested in the assets from
    the point of view of loan origination or project financing.
    """

    _bounds: Dict[
        Type[Hazard], Union[HazardIndicatorBounds, ImpactBounds, ImpactBoundsJoint]
    ]

    def __init__(self):
        self.model_summary = {"Generic score based risk measure."}
        # fmt: off

        acute_bounds = ImpactBounds(
                categories=[Category.VERY_LOW, Category.LOW, Category.MEDIUM, Category.HIGH, Category.VERY_HIGH],
                measure=self._aal_future,
                lower=[float("-inf"),   0.01/100.,  0.2/100.,   1./100.,   5./100.],  # applies to impact
                upper=[0.01/100.,       0.2/100.,   1./100.,    5./100.,   float("inf")],
        ) # noqa
        chronic_bounds = ImpactBounds(
                categories=[Category.VERY_LOW, Category.LOW, Category.MEDIUM, Category.HIGH, Category.VERY_HIGH],
                measure=self._aal_future,
                lower=[float("-inf"),   0.01/100.,  1./100.,   5./100.,    10./100.],  # applies to impact
                upper=[0.01/100,        1./100.,    5./100.,   10./100.,  float("inf")],
        ) # noqa

        self._bounds = {
            Precipitation: HazardIndicatorBounds(
                categories=[Category.LOW, Category.MEDIUM, Category.HIGH, Category.VERY_HIGH],
                hazard_type=Precipitation,
                indicator_id="max/daily/water_equivalent",
                indicator_return=100,
                units="mm/day",
                lower=[float("-inf"), 100, 130, 160],
                upper=[100,           130, 160, float("inf")]
                ), # noqa
            CoastalInundation: acute_bounds,
            PluvialInundation: acute_bounds,
            RiverineInundation: acute_bounds,
            Fire: acute_bounds,
            Hail: acute_bounds,
            Wind: acute_bounds,
            Drought: chronic_bounds,
            ChronicHeat: chronic_bounds,
        }
        # fmt: on
        self._definition_lookup = {}
        self._definition_lookup[Precipitation] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Precipitation.__name__],
            values=self._definition_values(self._bounds[Precipitation]),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_precipitation",
                    label="1-in-100 year precipitation.",
                    description=(
                        "1-in-100 year maximum daily total water equivalent precipitation."
                    ),
                    units="mm",
                )
            ],
        )
        acute_definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[
                CoastalInundation.__name__,
                PluvialInundation.__name__,
                RiverineInundation.__name__,
                Fire.__name__,
                Hail.__name__,
                Wind.__name__,
            ],
            values=self._definition_values_impact(acute_bounds),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_aal",
                    label="Average annual loss (AAL).",
                    description=(
                        "Average annual loss (AAL). Annual damage as a fraction of the total insurable asset value."
                    ),
                    units="",  # percentage expected
                )
            ],
        )
        self._definition_lookup[CoastalInundation] = acute_definition
        self._definition_lookup[PluvialInundation] = acute_definition
        self._definition_lookup[RiverineInundation] = acute_definition
        self._definition_lookup[Wind] = acute_definition

    def _definition_values(self, bounds: HazardIndicatorBounds):
        return [
            RiskScoreValue(
                value=Category.NODATA, label="No data", description="No data."
            ),
            RiskScoreValue(
                value=Category.LOW,
                label="Low exposure",
                description="Low exposure",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.LOW)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.LOW)]
                    )
                ],
            ),
            RiskScoreValue(
                value=Category.MEDIUM,
                label="Medium exposure",
                description="Medium exposure",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.MEDIUM)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.MEDIUM)]
                    )
                ],
            ),
            RiskScoreValue(
                value=Category.HIGH,
                label="High exposure",
                description="High exposure",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.HIGH)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.HIGH)]
                    )
                ],
            ),
            RiskScoreValue(
                value=Category.VERY_HIGH,
                label=("Very high exposure"),
                description="Very high exposure",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.VERY_HIGH)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.VERY_HIGH)]
                    )
                ],
            ),
        ]

    def _definition_values_impact(self, bounds: ImpactBounds):
        return [
            RiskScoreValue(
                value=Category.NODATA, label="No data", description="No data."
            ),
            RiskScoreValue(
                value=Category.LOW,
                label="Low impact",
                description="Low impact",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.LOW)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.LOW)]
                    )
                ],
            ),
            RiskScoreValue(
                value=Category.MEDIUM,
                label="Medium impact",
                description="Medium impact",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.MEDIUM)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.MEDIUM)]
                    )
                ],
            ),
            RiskScoreValue(
                value=Category.HIGH,
                label="High impact",
                description="High impact",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.HIGH)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.HIGH)]
                    )
                ],
            ),
            RiskScoreValue(
                value=Category.VERY_HIGH,
                label=("Very high impact"),
                description="Very high impact",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.VERY_HIGH)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.VERY_HIGH)]
                    )
                ],
            ),
        ]

    def _definition_values_impact_change(self, description: Callable[[Category], str]):
        return [
            RiskScoreValue(
                value=Category.VERY_HIGH,
                label=("The asset is very materially impacted."),
                description=description(Category.VERY_HIGH),
            ),
            RiskScoreValue(
                value=Category.HIGH,
                label="The asset is materially impacted.",
                description=description(Category.HIGH),
            ),
            RiskScoreValue(
                value=Category.MEDIUM,
                label=("The asset is impacted."),
                description=description(Category.MEDIUM),
            ),
            RiskScoreValue(
                value=Category.LOW,
                label="Little impact.",
                description=description(Category.LOW),
            ),
            RiskScoreValue(
                value=Category.NODATA, label="No data.", description="No data."
            ),
        ]

    def _acute_description_aal(self, category: Category, bounds: ImpactBounds):
        index = int(category)
        if index > 0:
            description = (
                f"Risk score of {index} corresponds to a projected Average Annual Loss "
                f"(AAL) between {bounds.lower[index - 1]} "
                f"and {bounds.upper[index - 1]}."
            )
        else:
            description = "No Data"
        return description

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
        histo_impacts: Sequence[AssetImpactResult],
        future_impacts: Sequence[AssetImpactResult],
    ) -> Measure:
        # in general we want to use the impact distribution, but in certain circumstances we can use
        # the underlying hazard data some care is needed given that vulnerability models are interchangeable
        # (what if the vulnerability model used does not make use of the hazard indicator we require?)
        histo_impact_res, future_impact_res = histo_impacts[0], future_impacts[0]
        bounds = self._bounds[hazard_type]
        if isinstance(bounds, HazardIndicatorBounds):
            assert future_impact_res.hazard_data is not None
            hazard_data = list(future_impact_res.hazard_data)
            if len(hazard_data) > 1:
                # the vulnerability model makes more than one request: ambiguous
                raise ValueError("ambiguous hazard data response")
            resp = hazard_data[0]
            if isinstance(resp, HazardParameterDataResponse):
                # if data is not present for parameters, the array will come through as empty
                # (as opposed to NaN or a fixed size array passed with NaNs)
                param = resp.parameter if any(resp.parameters) else float("nan")
                # undesirable but temporary 'special case'
                # 'fire probability' is only well defined if the pixel size is sufficiently small,
                # i.e. smaller than the area of the wildfire outbreak. This adjustment gives reasonable
                # behaviour using the Jupiter 100 km data set. In principle we could define different
                # indicators, but probably over-engineering given move towards high-resolution maps.
                if resp.path.startswith("fire/jupiter/v1/fire_probability_"):
                    param /= 100
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
                    score=list(bounds.categories)[index],
                    measure_0=float(param),
                    definition=self.get_definition(hazard_type),
                )
        elif isinstance(bounds, ImpactBounds):
            assert future_impact_res.impact is not None
            if isinstance(future_impact_res.impact, EmptyImpactDistrib):
                # can occur in the case where a curve cannot be matched
                return Measure(
                    score=Category.NODATA,
                    measure_0=float("nan"),
                    definition=self.get_definition(hazard_type),
                )
            measure = bounds.measure(
                histo_impact_res.impact, future_impact_res.impact, -1
            )
            score = Category.NODATA
            for category, lower, upper in zip(
                bounds.categories, bounds.lower, bounds.upper
            ):
                if measure >= lower and measure < upper:
                    score = category
                    break
            return Measure(
                score=score,
                measure_0=float(measure),
                definition=self.get_definition(hazard_type),
            )
        elif isinstance(bounds, ImpactBoundsJoint):
            assert future_impact_res.impact is not None
            assert histo_impact_res.impact is not None
            if isinstance(histo_impact_res.impact, EmptyImpactDistrib):
                if hazard_type == PluvialInundation:
                    # there is no alternative pluvial inundation model, so allow for this
                    return Measure(
                        score=Category.NODATA,
                        measure_0=float("nan"),
                        definition=self.get_definition(hazard_type),
                    )
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

    def _aal_future(
        self,
        histo_impact: ImpactDistrib,
        future_impact: ImpactDistrib,
        return_period: float,
    ):
        return future_impact.mean_impact()

    def _bounds_format(self, bound: float) -> Any:
        if bound == float("-inf"):
            return "-inf"
        elif bound == float("inf"):
            return "inf"
        return bound

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
        # if PluvialInundation present at all, we do not want the proxy:
        # is confusing.
        if any(k.hazard_type == PluvialInundation for k in measures.keys()):
            return aggregate_measures
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
