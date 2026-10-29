"""Coverage and merging regressions for the cascading resource provider."""

from unittest.mock import Mock

import numpy as np
import pytest

from physrisk.api.v1.hazard_data import HazardResource, ScenarioYears
from physrisk.data.hazard_data_provider import (
    CascadingHazardDataProvider,
)
from physrisk.data.scenario_year_resolution import (
    InterpolatedYearResolver,
    ScenarioYear,
    resolve_exact_year,
)
from physrisk.kernel.hazards import RiverineInundation


def resource(name):
    return HazardResource(
        path=f"{name}_{{scenario}}_{{year}}",
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        indicator_model_gcm="",
        display_name="",
        description="",
        units="m",
        scenarios=[
            ScenarioYears(id="historical", years=[1980]),
            ScenarioYears(id="ssp585", years=[2050, 2080]),
        ],
    )


def provider(resources, reader, resolver=resolve_exact_year):
    resource_selector = Mock()
    resource_selector.get_resources.return_value = resources
    return CascadingHazardDataProvider(
        RiverineInundation,
        resource_selector,
        scenario_year_resolver=resolver,
        zarr_reader=reader,
    )


async def test_merge_aligns_partial_coverage_and_sizes_each_requested_year():
    first, second = resource("first"), resource("second")
    reader = Mock()
    reader.in_bounds.side_effect = [
        np.array([True, False, False]),
        np.array([True, False]),
    ]

    values_by_path = {
        "first_ssp585_2050": [[1.0]],
        "second_ssp585_2050": [[2.0, 2.0, 2.0]],
        "first_ssp585_2080": [[1.0, 1.0]],
        "second_ssp585_2080": [[2.0, 2.0]],
    }

    def curves(path, *args):
        values = np.array(values_by_path[path])
        return values, np.array([True]), np.arange(values.shape[1]), "m"

    reader.get_curves.side_effect = curves
    model = provider([first, second], reader)
    coords = np.array([0.0, 1.0, 2.0])
    result = await model.get_data(
        coords,
        coords,
        indicator_id="flood_depth",
        scenarios=["ssp585"],
        years=[2050, 2080],
    )

    near = result[ScenarioYear("ssp585", 2050)]
    far = result[ScenarioYear("ssp585", 2080)]
    assert near.values.shape == (3, 3)
    assert far.values.shape == (3, 2)
    np.testing.assert_array_equal(near.coverage_mask, [True, True, False])
    np.testing.assert_array_equal(far.coverage_mask, near.coverage_mask)
    np.testing.assert_array_equal(near.indices_length[near.coverage_mask], [1, 3])
    np.testing.assert_array_equal(far.indices_length[far.coverage_mask], [2, 2])
    np.testing.assert_array_equal(near.values[0, :1], [1.0])
    np.testing.assert_array_equal(near.values[1], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(far.values[:2], [[1.0, 1.0], [2.0, 2.0]])
    assert near.paths[near.coverage_mask].tolist() == [first.path, second.path]


async def test_merge_rejects_resources_with_different_units():
    reader = Mock()
    reader.in_bounds.side_effect = [np.array([True, False]), np.array([True])]
    reader.get_curves.side_effect = lambda path, *args: (
        np.array([[1.0]]),
        np.array([True]),
        np.array([100.0]),
        "m" if path.startswith("first") else "cm",
    )
    model = provider([resource("first"), resource("second")], reader)

    with pytest.raises(ValueError, match="inconsistent units for scenario/year"):
        await model.get_data(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            indicator_id="flood_depth",
            scenarios=["ssp585"],
            years=[2050],
        )


async def test_read_must_cover_every_coordinate_claimed_by_bounds():
    reader = Mock()
    reader.in_bounds.return_value = np.array([True])
    # Bounds claim the point, but the historical array does not cover it.
    reader.get_curves.side_effect = lambda path, *args: (
        np.array([[1.0]]),
        np.array(["historical" not in path]),
        np.array([100.0]),
        "m",
    )
    model = provider([resource("first")], reader, InterpolatedYearResolver())
    with pytest.raises(ValueError, match="inconsistent geographic coverage"):
        await model.get_data(
            np.array([0.0]),
            np.array([0.0]),
            indicator_id="flood_depth",
            scenarios=["ssp585"],
            years=[2040],
        )


async def test_interpolation_rejects_inputs_with_different_units():
    reader = Mock()
    reader.in_bounds.return_value = np.array([True])
    reader.get_curves.side_effect = lambda path, *args: (
        np.ones((1, 1)),
        np.array([True]),
        np.array([100.0]),
        "m" if "historical" in path else "cm",
    )
    model = provider([resource("first")], reader, InterpolatedYearResolver())
    with pytest.raises(
        ValueError, match="inconsistent units across interpolation inputs"
    ):
        await model.get_data(
            np.array([0.0]),
            np.array([0.0]),
            indicator_id="flood_depth",
            scenarios=["ssp585"],
            years=[2040],
        )
