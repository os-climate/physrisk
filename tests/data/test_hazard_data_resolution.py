import pytest

from physrisk.api.v1.hazard_data import HazardResource, Scenario
from physrisk.data.hazard_data_provider import ScenarioYear
from physrisk.data.hazard_data_resolution import (
    EmptyResourceError,
    resolve_nearest_scenario_year,
)


def resource(*scenarios: tuple[str, tuple[int, ...]]) -> HazardResource:
    return HazardResource(
        hazard_type="TestHazard",
        indicator_id="indicator",
        indicator_model_gcm="test",
        path="resource/{id}/{scenario}/{year}",
        display_name="Test resource",
        description="Test resource",
        scenarios=[
            Scenario(id=scenario, years=list(years)) for scenario, years in scenarios
        ],
        units="test",
    )


@pytest.mark.parametrize(
    "requested,available_resource,expected",
    [
        (
            ScenarioYear("ssp245", 2050),
            resource(("ssp245", (2030, 2050))),
            ScenarioYear("ssp245", 2050),
        ),
        (
            ScenarioYear("ssp245", 2050),
            resource(("rcp4p5", (2050,))),
            ScenarioYear("rcp4p5", 2050),
        ),
        (
            ScenarioYear("ssp370", 2050),
            resource(("rcp4p5", (2050,)), ("rcp8p5", (2050,))),
            ScenarioYear("rcp8p5", 2050),
        ),
        (
            ScenarioYear("ssp245", 2055),
            resource(("ssp245", (2050, 2060))),
            ScenarioYear("ssp245", 2060),
        ),
        (
            ScenarioYear("rcp5p5", 2050),
            resource(("rcp4p5", (2050,)), ("rcp6p5", (2050,))),
            ScenarioYear("rcp6p5", 2050),
        ),
        (
            ScenarioYear("ssp370", 2050),
            resource(("ssp245", (2050,)), ("rcp4p5", (2050,))),
            ScenarioYear("rcp4p5", 2050),
        ),
        (
            ScenarioYear("ssp585", 2080),
            resource(("historical", (2005,))),
            ScenarioYear("historical", 2005),
        ),
        (
            ScenarioYear("historical", -1),
            resource(("historical", (1980, 2005))),
            ScenarioYear("historical", 2005),
        ),
        (
            ScenarioYear("historical", 1980),
            resource(("historical", (1980, 2005))),
            ScenarioYear("historical", 2005),
        ),
        (
            ScenarioYear("historical", -1),
            resource(("ssp585", (2030,)), ("ssp126", (2020, 2030))),
            ScenarioYear("ssp126", 2020),
        ),
        (
            ScenarioYear("unrecognised", 2050),
            resource(
                ("custom-optimistic", (2040,)),
                ("custom-pessimistic", (2060,)),
            ),
            ScenarioYear("custom-pessimistic", 2060),
        ),
    ],
)
def test_nearest_scenario_year_resolution(requested, available_resource, expected):
    assert resolve_nearest_scenario_year(requested, available_resource) == expected


def test_empty_availability_is_reported_to_provider():
    with pytest.raises(EmptyResourceError):
        resolve_nearest_scenario_year(ScenarioYear("ssp585", 2050), resource())
