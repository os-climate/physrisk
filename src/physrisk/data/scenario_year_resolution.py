"""Scenario selection and year interpolation for hazard resources."""

import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence

from typing_extensions import Protocol

from physrisk.api.v1.hazard_data import Scenario


@dataclass(frozen=True)
class ScenarioYear:
    """Scenario identifier and year used as a request or result key.

    Attributes:
        scenario: Climate scenario identifier.
        year: Projection year, or ``-1`` for a historical request.
    """

    scenario: str
    year: int


@dataclass
class WeightedSum:
    """Concrete scenario/year pairs and their coefficients."""

    weights: list[tuple[ScenarioYear, float]]


class ScenarioYearResolver(Protocol):
    """Resolve a single request into concrete weighted inputs from one resource."""

    def __call__(
        self,
        available: Sequence[Scenario],
        requested: ScenarioYear,
    ) -> WeightedSum | None:
        """Return available weighted inputs, or None if unsupported.

        Available scenarios follow the resource's declared order. The caller retains
        the requested pair as the output key, including at the historical anchor.
        """
        ...


def _historical_or_proxy(available: Sequence[Scenario]) -> Scenario | None:
    populated = [scenario for scenario in available if scenario.years]
    historical = next(
        (scenario for scenario in populated if scenario.id == "historical"), None
    )
    # Prefer historical data; otherwise use the scenario with the earliest year.
    # min preserves resource order when earliest years are tied.
    return (
        historical
        if historical is not None
        else min(populated, key=lambda scenario: min(scenario.years), default=None)
    )


def _requested_or_proxy(
    requested: str, available: Sequence[Scenario]
) -> Scenario | None:
    if not available:
        return None
    if requested == "historical":
        return _historical_or_proxy(available)
    selected_id = (
        cmip6_scenario_to_rcp(requested)
        if available[0].id.startswith("rcp") or available[-1].id.startswith("rcp")
        else requested
    )
    return next(
        (scenario for scenario in available if scenario.id == selected_id), None
    )


def resolve_exact_year(
    available: Sequence[Scenario],
    requested: ScenarioYear,
) -> WeightedSum | None:
    """Apply scenario proxies and read exact years."""
    selected = _requested_or_proxy(requested.scenario, available)
    if selected is None or not selected.years:
        return None
    year = min(selected.years) if requested.scenario == "historical" else requested.year
    if year not in selected.years:
        return None
    return WeightedSum([(ScenarioYear(selected.id, year), 1.0)])


class InterpolatedYearResolver(ScenarioYearResolver):
    """Apply scenario proxies and interpolate years using a historical anchor.

    Historical data comes from the earliest historical year, or the earliest year
    of any scenario if historical is absent. It is placed at historical_year on
    the interpolation timeline, independently of its concrete stored year.
    """

    def __init__(self, historical_year: int = 2025):
        """Set the year represented by historical data on the interpolation timeline."""
        self._historical_year = historical_year

    def __call__(
        self,
        available: Sequence[Scenario],
        requested: ScenarioYear,
    ) -> WeightedSum | None:
        selected = _requested_or_proxy(requested.scenario, available)
        if selected is None or not selected.years:
            return None
        historical = _historical_or_proxy(available)
        if historical is None:
            return None
        return interpolate_year(
            requested,
            selected,
            ScenarioYear(historical.id, min(historical.years)),
            self._historical_year,
        )


def interpolate_year(
    requested: ScenarioYear,
    available: Scenario,
    historical: ScenarioYear,
    historical_year: int,
) -> WeightedSum:
    """Interpolate one year with an explicit scenario and historical timeline anchor."""
    if requested.scenario == "historical":
        return WeightedSum([(historical, 1.0)])
    timeline = sorted([historical_year] + available.years)

    def source(year: int) -> ScenarioYear:
        return (
            historical if year == historical_year else ScenarioYear(available.id, year)
        )

    index = bisect_left(timeline, requested.year)
    if index == len(timeline):
        # Extend the slope defined by the latest two timeline years.
        slope = (float(requested.year) - float(timeline[-1])) / (
            float(timeline[-1]) - float(timeline[-2])
        )
        weights = [
            (source(timeline[-2]), -slope),
            (source(timeline[-1]), 1.0 + slope),
        ]
    elif timeline[index] == requested.year:
        weights = [(source(timeline[index]), 1.0)]
    else:
        w1 = (float(timeline[index]) - float(requested.year)) / (
            float(timeline[index]) - float(timeline[index - 1])
        )
        weights = [
            (source(timeline[index - 1]), w1),
            (source(timeline[index]), 1.0 - w1),
        ]
    return WeightedSum(weights)


def cmip6_scenario_to_rcp(scenario: str) -> str:
    """Convention is that CMIP6 scenarios are expressed by identifiers:
    SSP1-2.6: 'ssp126'
    SSP2-4.5: 'ssp245'
    SSP5-8.5: 'ssp585' etc.
    Here we translate to form
    RCP-4.5: 'rcp4p5'
    RCP-8.5: 'rcp8p5' etc.
    """
    match = re.fullmatch(r"ssp([1-5])(\d)(\d)", scenario)
    if match:
        _, second, third = match.groups()
        return f"rcp{second}p{third}"
    else:
        # Handle scenarios that do not match the SSP pattern but are valid RCPs or historical
        valid_scenarios = [
            "rcp2p6",
            "rcp4p5",
            "rcp6p0",
            "rcp8p5",
            "historical",
            "rcp26",
            "rcp45",
            "rcp60",
            "rcp7p0",
            "rcp85",
        ]
        if scenario not in valid_scenarios:
            raise ValueError(f"unexpected scenario {scenario}")
        return scenario
