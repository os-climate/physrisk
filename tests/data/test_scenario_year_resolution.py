"""Regression checks for cascading scenario selection and resource reads."""

from unittest.mock import Mock

import numpy as np
import pytest

from physrisk.api.v1.hazard_data import HazardResource, Scenario
from physrisk.data.hazard_data_provider import (
    CascadingHazardDataProvider,
)
from physrisk.data.scenario_year_resolution import (
    ScenarioYear,
    InterpolatedYearResolver,
    resolve_exact_year,
)
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.kernel.hazard_model import HazardDataRequest, HazardEventDataResponse
from physrisk.kernel.hazards import RiverineInundation


def resource(scenarios):
    return HazardResource(
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        path="{id}_{scenario}_{year}",
        scenarios=[Scenario(id=name, years=years) for name, years in scenarios],
        units="m",
        indicator_model_gcm="",
        display_name="",
        description="",
    )


@pytest.mark.parametrize(
    "requested_years,expected_year",
    [
        pytest.param([2050], 2050, id="no-match-uses-next-resource"),
        pytest.param([2030, 2050], 2030, id="partial-match-claims-coverage"),
    ],
)
async def test_cascade_skips_resources_without_matches_but_partial_matches_claim_coverage(
    requested_years, expected_year
):
    resources = Mock()
    resources.get_resources.return_value = [
        resource([("ssp585", [2030])]),
        resource([("ssp585", [2050])]),
    ]
    reader = Mock()
    reader.in_bounds.return_value = np.array([True])
    reader.get_curves.return_value = (
        np.array([[3.0]]),
        np.array([True]),
        np.array([100.0]),
        "m",
    )
    provider = CascadingHazardDataProvider(
        RiverineInundation,
        resources,
        scenario_year_resolver=resolve_exact_year,
        zarr_reader=reader,
    )
    result = await provider.get_data(
        np.array([0.0]),
        np.array([0.0]),
        indicator_id="flood_depth",
        scenarios=["ssp585"],
        years=requested_years,
    )
    # A partial match still claims the shared mask, preventing later resources
    # from supplying the other year. With no match, the next resource is tried.
    assert list(result) == [ScenarioYear("ssp585", expected_year)]
    np.testing.assert_array_equal(
        result[ScenarioYear("ssp585", expected_year)].values, [[3.0]]
    )


def test_get_hazard_data_preserves_future_request_key_at_historical_anchor():
    anchor = 2025
    r = resource([("historical", [1980]), ("ssp585", [2050])])
    resources = Mock()
    resources.get_resources.return_value = [r]
    resources.hazard_indicators.return_value = {RiverineInundation: [r.indicator_id]}
    reader = Mock()
    reader.in_bounds.return_value = np.array([True])
    reader.get_curves.return_value = (
        np.array([[1.5]]),
        np.array([True]),
        np.array([100.0]),
        "m",
    )
    model = ZarrHazardModel(
        resource_selector=resources,
        reader=reader,
        scenario_year_resolver=InterpolatedYearResolver(historical_year=anchor),
    )
    future = HazardDataRequest(
        RiverineInundation,
        longitude=0.0,
        latitude=0.0,
        indicator_id="flood_depth",
        scenario="ssp585",
        year=anchor,
    )
    result = model.get_hazard_data([future])

    assert isinstance(result[future], HazardEventDataResponse)
    np.testing.assert_array_equal(result[future].intensities, [1.5])
