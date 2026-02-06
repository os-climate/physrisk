import math
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence, Set, Type, Union

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


class UnderlingMeasure(Protocol):
    """Calculates the measure(s) underlying the score from the one or more
    impacts.
    """

    def __call__(
        self,
        histo_impacts: Sequence[ImpactDistrib],
        future_impact: Sequence[ImpactDistrib],
    ) -> float: ...


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

    measure: UnderlingMeasure
    categories: Sequence[Category]
    lower: Sequence[float]
    upper: Sequence[float]


@dataclass
class ImpactBoundsJoint:
    """Category applies if lower1 <= value1 < upper1 and lower2 <= value2 < upper2."""

    measure1: UnderlingMeasure
    measure2: UnderlingMeasure
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
                measure=self._delta_aal,
                lower=[float("-inf"),   0.01/100.,  1./100.,   5./100.,    10./100.],  # applies to impact
                upper=[0.01/100,        1./100.,    5./100.,   10./100.,  float("inf")],
        ) # noqa
        # fmt: on

        # The one remaining example of an exposure-based measure is Precipitation, an exemplar special case.
        self._bounds = {
            Precipitation: HazardIndicatorBounds(
                categories=[
                    Category.LOW,
                    Category.MEDIUM,
                    Category.HIGH,
                    Category.VERY_HIGH,
                ],
                hazard_type=Precipitation,
                indicator_id="max/daily/water_equivalent",
                indicator_return=100,
                units="mm/day",
                lower=[float("-inf"), 100, 130, 160],
                upper=[100, 130, 160, float("inf")],
            ),  # noqa
            CoastalInundation: acute_bounds,
            PluvialInundation: acute_bounds,
            RiverineInundation: acute_bounds,
            Fire: acute_bounds,
            Hail: acute_bounds,
            Wind: acute_bounds,
            ChronicHeat: chronic_bounds,
            Drought: chronic_bounds,
        }
        self._definition_lookup = {}
        self._definition_lookup[Precipitation] = ScoreBasedRiskMeasureDefinition(
            hazard_types=[Precipitation.__name__],
            values=self._definition_values_exposure(self._bounds[Precipitation]),
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
                    measure_id="measure_damage_aal",
                    label="Average annual loss (AAL).",
                    description=(
                        "Average annual loss (AAL). Annual damage as a fraction of the asset total insurable value (TIV)."
                        "Where applicable, aggregates asset-restoration downtime loss, also as fraction of TIV."
                    ),
                    units="",  # value is fraction; percentage expected when visualising
                )
            ],
        )
        chronic_definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[
                ChronicHeat.__name__,
                Drought.__name__,
            ],
            values=self._definition_values_impact(chronic_bounds),
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="measure_disruption_aal",
                    label="Average annual loss (AAL).",
                    description=(
                        "Average annual loss (AAL). Annual disruption as a fraction of the total revenue attributable to the asset."
                    ),
                    units="",  # value is fraction;percentage expected when visualising
                )
            ],
        )
        for hazard_type in [
            CoastalInundation,
            PluvialInundation,
            RiverineInundation,
            Fire,
            Hail,
            Wind,
        ]:
            self._definition_lookup[hazard_type] = acute_definition
        for hazard_type in [ChronicHeat, Drought]:
            self._definition_lookup[hazard_type] = chronic_definition

    def _definition_values_exposure(self, bounds: HazardIndicatorBounds):
        return [
            RiskScoreValue(
                value=Category.NODATA, label="No data", description="No data."
            ),
            RiskScoreValue(
                value=Category.VERY_LOW,
                label="Very low exposure",
                description="Very low exposure",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.VERY_LOW)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.VERY_LOW)]
                    )
                ],
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
                label="Very low impact",
                description="Very low impact",
                lower_bound=[
                    self._bounds_format(
                        bounds.lower[bounds.categories.index(Category.VERY_LOW)]
                    )
                ],
                upper_bound=[
                    self._bounds_format(
                        bounds.upper[bounds.categories.index(Category.VERY_LOW)]
                    )
                ],
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

    def calc_measure(
        self,
        hazard_type: type[Hazard],
        histo_impacts: Sequence[AssetImpactResult],
        future_impacts: Sequence[AssetImpactResult],
    ) -> Measure:
        # in general ImpactBounds are preferred over HazardIndicatorBounds, because the score thereby
        # takes account of the vulnerability of the asset as well as the hazard intensity.
        bounds = self._bounds[hazard_type]
        if isinstance(bounds, ImpactBounds):
            # just need the impact part, no hazard indicator data.
            h_impacts = [
                h.impact
                for h in histo_impacts
                if not isinstance(h.impact, EmptyImpactDistrib)
            ]
            f_impacts = [
                f.impact
                for h, f in zip(histo_impacts, future_impacts)
                if not isinstance(h.impact, EmptyImpactDistrib)
            ]
            if len(f_impacts) == 0:
                # if there are no impacts with data, we cannot calculate a measure. This
                # can occur in the case where a curve cannot be matched
                return Measure(
                    score=Category.NODATA,
                    measure_0=float("nan"),
                    definition=self.get_definition(hazard_type),
                )
            measure = bounds.measure(h_impacts, f_impacts)
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
        elif isinstance(bounds, HazardIndicatorBounds):
            if len(future_impacts) > 1:
                raise NotImplementedError(
                    "multiple future impacts not supported for hazard indicator based measures"
                )
            assert future_impacts[0].hazard_data is not None
            hazard_data = list(future_impacts[0].hazard_data)
            if len(hazard_data) > 1:
                # the vulnerability model makes more than one request: ambiguous
                raise ValueError("ambiguous hazard data response")
            resp = hazard_data[0]
            if isinstance(resp, HazardParameterDataResponse):
                # if data is not present for parameters, the array will come through as empty
                # (as opposed to NaN or a fixed size array passed with NaNs)
                param = resp.parameter if any(resp.parameters) else float("nan")
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
        elif isinstance(bounds, ImpactBoundsJoint):
            # just need the impact part, no hazard indicator data.
            h_impacts = [
                h.impact
                for h in histo_impacts
                if not isinstance(h.impact, EmptyImpactDistrib)
            ]
            f_impacts = [
                f.impact
                for h, f in zip(histo_impacts, future_impacts)
                if not isinstance(h.impact, EmptyImpactDistrib)
            ]
            if len(f_impacts) == 0:
                if hazard_type == PluvialInundation:
                    # there is no alternative pluvial inundation model, so allow for this
                    return Measure(
                        score=Category.NODATA,
                        measure_0=float("nan"),
                        definition=self.get_definition(hazard_type),
                    )
            measure1 = bounds.measure1(h_impacts, f_impacts)
            measure2 = bounds.measure2(h_impacts, f_impacts)
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
        histo_impact: Sequence[ImpactDistrib],
        future_impact: Sequence[ImpactDistrib],
    ):
        return sum(f.mean_impact() for f in future_impact)

    def _delta_aal(
        self,
        histo_impacts: Sequence[ImpactDistrib],
        future_impacts: Sequence[ImpactDistrib],
    ):
        return sum(
            f.mean_impact() - h.mean_impact()
            for h, f in zip(histo_impacts, future_impacts)
        )

    def _bounds_format(self, bound: float) -> Any:
        if bound == float("-inf"):
            return "-inf"
        elif bound == float("inf"):
            return "inf"
        return bound

    def _impact(
        self,
        histo_impacts: Sequence[ImpactDistrib],
        future_impacts: Sequence[ImpactDistrib],
        return_period: float,
    ):
        if len(future_impacts) > 1:
            raise NotImplementedError("multiple future impacts not supported")
        return future_impacts[0].to_exceedance_curve().get_value(1.0 / return_period)

    def _delta_impact(
        self,
        histo_impacts: Sequence[ImpactDistrib],
        future_impacts: Sequence[ImpactDistrib],
        return_period: float,
    ):
        if len(future_impacts) > 1:
            raise NotImplementedError("multiple future impacts not supported")
        histo_loss = (
            histo_impacts[0].to_exceedance_curve().get_value(1.0 / return_period)
        )
        future_loss = (
            future_impacts[0].to_exceedance_curve().get_value(1.0 / return_period)
        )
        return future_loss - histo_loss

    def get_definition(self, hazard_type: Type[Hazard]):
        return self._definition_lookup.get(hazard_type, None)

    def supported_hazards(self) -> Set[type]:
        return set(
            [
                CoastalInundation,
                PluvialInundation,
                RiverineInundation,
                ChronicHeat,
                Drought,
                Fire,
                Hail,
                Precipitation,
                Wind,
            ]
        )

    def aggregate_risk_measures(
        self,
        measures: Dict[MeasureKey, Measure],
        assets: Sequence[Asset],
        prosp_scens: Sequence[str],
        years: Sequence[int],
    ) -> Dict[MeasureKey, Measure]:
        return measures
