import re
from collections.abc import Sequence

from physrisk.api.v1.hazard_data import HazardResource, Scenario
from physrisk.data.hazard_data_provider import ScenarioYear


class EmptyResourceError(ValueError):
    """Raised when a hazard resource contains no available scenario and year."""


def _forcing(scenario: str) -> float | None:
    """Extract the radiative-forcing value from an SSP or RCP identifier."""
    ssp = re.fullmatch(r"ssp[1-5](\d)(\d)", scenario)
    if ssp is not None:
        return float(f"{ssp.group(1)}.{ssp.group(2)}")
    rcp = re.fullmatch(r"rcp(\d)p(\d)", scenario)
    if rcp is not None:
        return float(f"{rcp.group(1)}.{rcp.group(2)}")
    return None


def resolve_nearest_scenario_year(
    requested: ScenarioYear,
    resource: HazardResource,
) -> ScenarioYear:
    """Select the closest available scenario and year from a resource.

    Scenario identifiers are matched exactly when possible. Otherwise, recognized
    SSP and RCP scenarios are matched by radiative forcing. The selected year
    minimizes absolute distance from the requested year, with the later year
    winning ties.
    """
    scenarios = tuple(s for s in resource.scenarios if len(s.years) > 0)
    if len(scenarios) == 0:
        raise EmptyResourceError(
            f"resource {resource.path!r} has no available scenario/year"
        )

    selected_scenario = _resolve_nearest_scenario(requested.scenario, scenarios)
    year = _resolve_nearest_year(requested, selected_scenario)
    return ScenarioYear(selected_scenario.id, year)


def _resolve_nearest_scenario(
    requested: str, available: Sequence[Scenario]
) -> Scenario:
    exact = next(
        (scenario for scenario in available if scenario.id == requested),
        None,
    )
    if exact is not None:
        return exact

    if requested == "historical":
        with_forcing = [
            (scenario, forcing)
            for scenario in available
            if (forcing := _forcing(scenario.id)) is not None
        ]
        if len(with_forcing) > 0:
            return min(with_forcing, key=lambda item: (item[1], item[0].id))[0]
        return available[0]

    requested_forcing = _forcing(requested)
    if requested_forcing is not None:
        with_forcing = [
            (scenario, forcing)
            for scenario in available
            if (forcing := _forcing(scenario.id)) is not None
        ]
        if len(with_forcing) > 0:
            return min(
                with_forcing,
                key=lambda item: (
                    abs(item[1] - requested_forcing),
                    -item[1],
                    item[0].id,
                ),
            )[0]

        historical = next(
            (scenario for scenario in available if scenario.id == "historical"),
            None,
        )
        if historical is not None:
            return historical

    return available[-1]


def _resolve_nearest_year(requested: ScenarioYear, selected_scenario: Scenario) -> int:
    if len(selected_scenario.years) == 0:
        raise ValueError("cannot resolve a year from an empty sequence")
    if requested.scenario == "historical":
        if selected_scenario.id == "historical":
            return max(selected_scenario.years)
        return min(selected_scenario.years)
    return min(
        selected_scenario.years,
        key=lambda year: (abs(year - requested.year), -year),
    )
