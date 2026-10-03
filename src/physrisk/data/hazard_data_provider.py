import asyncio
import sys
from dataclasses import dataclass
from typing import (
    Dict,
    MutableMapping,
    Optional,
    Sequence,
    Type,
)

import numpy as np
from shapely import Point
from typing_extensions import Protocol

from physrisk.api.v1.hazard_data import HazardResource
from physrisk.kernel.hazards import Hazard

from .scenario_year_resolution import (
    ScenarioYear,
    ScenarioYearResolver,
    WeightedSum,
)
from .zarr_reader import ZarrReader


@dataclass
class HazardDataHint:
    """Requestors of hazard data may provide a hint which may be taken into account by the Hazard Model.
    A hazard resource path can be specified which uniquely defines the hazard resource; otherwise the resource
    is inferred from the indicator_id."""

    path: Optional[str] = None
    # consider adding: indicator_model_gcm: str

    def group_key(self):
        return self.path


class HazardResourceSelector(Protocol):
    """Selects hazard resources and exposes their available hazard indicators."""

    def hazard_indicators(self) -> dict[type[Hazard], list[str]]:
        """Return each available hazard class and its unique indicator identifiers."""
        ...

    def get_resources(
        self,
        hazard_type: type[Hazard],
        indicator_id: str,
        hint: HazardDataHint | None = None,
    ) -> list[HazardResource]:
        """Return matching resources in cascade order.

        Args:
            hazard_type: Hazard class requested by the caller.
            indicator_id: Identifier of the requested hazard indicator.
            hint: Optional resource-selection hint.

        Returns:
            Matching resources ordered from most to least preferred.
        """
        ...


class DataSourcingError(Exception):
    pass


@dataclass
class ScenarioYearResult:
    values: np.ndarray
    indices: np.ndarray
    indices_length: np.ndarray
    coverage_mask: np.ndarray  # boolean mask giving the part of the original set of lats/lons that this applies to
    units: str
    paths: np.ndarray


class HazardDataProvider(Protocol):
    """Provides hazard data for coordinates, scenarios, and years."""

    async def get_data(
        self,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        *,
        indicator_id: str,
        scenarios: Sequence[str],
        years: Sequence[int],
        hint: HazardDataHint | None = None,
        buffer: int | None = None,
    ) -> dict[ScenarioYear, ScenarioYearResult]:
        """Read hazard data for coordinates, scenarios, and years.

        Args:
            longitudes: Longitude of each requested coordinate.
            latitudes: Latitude of each requested coordinate.
            indicator_id: Identifier of the requested hazard indicator.
            scenarios: Requested scenario identifiers.
            years: Requested projection years.
            hint: Optional resource-selection hint.
            buffer: Radius in metres over which to take the maximum value. ``None``
                performs point reads.

        Returns:
            Results keyed by requested scenario and year and aligned with the input
            coordinates.
        """
        ...


class CascadingHazardDataProvider:
    """Reads hazard data by cascading through sorted sources."""

    def __init__(
        self,
        hazard_type: Type[Hazard],
        resource_selector: HazardResourceSelector,
        scenario_year_resolver: ScenarioYearResolver,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
    ):
        """Provides hazard data.

        Args:
            hazard_type (Type[Hazard]): Hazard type.
            resource_selector: Selects hazard resources in cascade order.
            scenario_year_resolver: Joint scenario selection and interpolation policy.
            store (Optional[MutableMapping], optional): Zarr store instance. Defaults to None.
            zarr_reader (Optional[ZarrReader], optional): ZarrReader instance. Defaults to None.
            interpolation (Optional[str], optional): Interpolation type. Defaults to "floor".

        Raises:
            ValueError: If interpolation not in permitted list.
        """
        self.hazard_type = hazard_type
        self._resource_selector = resource_selector
        self._scenario_year_resolver = scenario_year_resolver
        self._reader = (
            zarr_reader if zarr_reader is not None else ZarrReader(store=store)
        )
        if interpolation not in ["floor", "linear", "max", "min"]:
            raise ValueError("interpolation must be 'floor', 'linear', 'max' or 'min'")
        self._interpolation = interpolation

    async def get_data(
        self,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        *,
        indicator_id: str,
        scenarios: Sequence[str],
        years: Sequence[int],
        hint: Optional[HazardDataHint] = None,
        buffer: Optional[int] = None,
    ) -> Dict[ScenarioYear, ScenarioYearResult]:
        """Returns data for set of latitude and longitudes.

        Args:
            longitudes (np.ndarray): Longitudes.
            latitudes (np.ndarray): Latitudes.
            indicator_id (str): Hazard Indicator ID.
            scenarios (Sequence[str]): Identifier of scenario, e.g. ssp585 (SSP 585), rcp8p5 (RCP 8.5).
            years (Sequence[int]): Projection years, e.g. [2050, 2080].
            hint (Optional[HazardDataHint], optional): Hint. Defaults to None.
            buffer (Optional[int], optional): _description_. Buffer around each point.

        Returns:
            Dict[ScenarioYear, ScenarioYearResult]: Results.

        Raises:
            DataSourcingError: A selected array does not exist.
            ValueError: Selected arrays have inconsistent geographic coverage or units.
        """
        requested_scenario_years = [
            ScenarioYear(scenario, year)
            for scenario in scenarios
            for year in ([-1] if scenario == "historical" else years)
        ]
        if not requested_scenario_years:
            return {}
        resources = self._resource_selector.get_resources(
            self.hazard_type, indicator_id=indicator_id, hint=hint
        )
        unprocessed_mask = np.ones(len(longitudes), dtype=bool)
        partial_results_by_request: dict[ScenarioYear, list[ScenarioYearResult]] = {}
        for resource in resources:
            # within a HazardResource the arrays have the same spatial coverage
            # any array can therefore be used for checking bounds
            if not np.any(unprocessed_mask):
                break
            available_scenarios = resource.scenarios
            weighted_sums_by_request = {
                requested_scenario_year: weighted_sum
                for requested_scenario_year in requested_scenario_years
                if (
                    weighted_sum := self._scenario_year_resolver(
                        available_scenarios, requested_scenario_year
                    )
                )
                is not None
            }
            if not weighted_sums_by_request:
                continue
            for (
                requested_scenario_year,
                weighted_sum,
            ) in weighted_sums_by_request.items():
                if len(weighted_sum.weights) not in (1, 2):
                    raise ValueError(
                        f"Expected one or two weighted inputs for {requested_scenario_year}, "
                        f"got {len(weighted_sum.weights)}"
                    )
            # All arrays in the resource share spatial coverage, so any resolved
            # input can represent it for the bounds check.
            coverage_weighted_sum = next(iter(weighted_sums_by_request.values()))
            coverage_source, _ = coverage_weighted_sum.weights[0]
            coverage_path = resource.path_for_scenario_year(
                coverage_source.scenario, coverage_source.year
            )
            try:
                in_bounds_mask = await asyncio.to_thread(
                    self._reader.in_bounds,
                    coverage_path,
                    longitudes[unprocessed_mask],
                    latitudes[unprocessed_mask],
                    self._interpolation,
                )
            except KeyError as error:
                raise DataSourcingError(
                    f"Dataset not found for hazard type {self.hazard_type.__name__} "
                    f"indicator ID {indicator_id}: {error.args[0]}"
                ) from error
            coverage_mask = unprocessed_mask.copy()
            coverage_mask[unprocessed_mask] &= in_bounds_mask
            if not np.any(coverage_mask):
                continue
            # A partial scenario/year match claims these coordinates for the whole
            # batch; later resources cannot fill its missing scenario/year pairs.
            unprocessed_mask[coverage_mask] = False

            resource_results = await self.get_scenarios_and_years(
                coverage_mask,
                longitudes[coverage_mask],
                latitudes[coverage_mask],
                indicator_id,
                resource,
                weighted_sums_by_request,
                buffer,
            )
            for requested_scenario_year, partial_result in resource_results.items():
                partial_results_by_request.setdefault(
                    requested_scenario_year, []
                ).append(partial_result)

        return self._merge_results(partial_results_by_request)

    @staticmethod
    def _merge_results(
        partial_results_by_request: dict[ScenarioYear, list[ScenarioYearResult]],
    ) -> dict[ScenarioYear, ScenarioYearResult]:
        """Merge geographical partial results for each requested scenario/year."""
        merged_results_by_request: dict[ScenarioYear, ScenarioYearResult] = {}
        for (
            requested_scenario_year,
            resource_results,
        ) in partial_results_by_request.items():
            first_result = resource_results[0]
            if any(
                partial_result.units != first_result.units
                for partial_result in resource_results[1:]
            ):
                raise ValueError(
                    f"inconsistent units for scenario/year {requested_scenario_year}"
                )
            coordinate_count = len(first_result.coverage_mask)
            max_index_length = max(
                int(partial_result.indices_length[0])
                for partial_result in resource_results
            )
            merged_result = ScenarioYearResult(
                values=np.empty((coordinate_count, max_index_length)),
                indices=np.empty(
                    (coordinate_count, max_index_length),
                    dtype=first_result.indices.dtype,
                ),
                indices_length=np.empty(
                    coordinate_count, dtype=first_result.indices_length.dtype
                ),
                coverage_mask=np.zeros(coordinate_count, dtype=bool),
                units=first_result.units,
                paths=np.empty(coordinate_count, dtype=np.object_),
            )
            for partial_result in resource_results:
                index_length = int(partial_result.indices_length[0])
                merged_result.values[partial_result.coverage_mask, :index_length] = (
                    partial_result.values
                )
                merged_result.indices[partial_result.coverage_mask, :index_length] = (
                    partial_result.indices
                )
                merged_result.indices_length[partial_result.coverage_mask] = (
                    partial_result.indices_length
                )
                merged_result.coverage_mask[partial_result.coverage_mask] = True
                merged_result.paths[partial_result.coverage_mask] = partial_result.paths
            merged_results_by_request[requested_scenario_year] = merged_result
        return merged_results_by_request

    async def get_scenarios_and_years(
        self,
        coverage_mask: np.ndarray,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        indicator_id: str,
        resource: HazardResource,
        weighted_sums_by_request: Dict[ScenarioYear, WeightedSum],
        buffer: Optional[int],
    ) -> dict[ScenarioYear, ScenarioYearResult]:
        """Get data for all scenarios and years using just a single HazardResource as the source.
        The importance of this is that interpolation of years is assumed to be feasible within the same resource as this
        is a single model (with consistent meaning of the values).
        """
        results_by_request: Dict[ScenarioYear, ScenarioYearResult] = {}
        expected_units = resource.units
        source_scenario_years = {
            source_scenario_year
            for weighted_sum in weighted_sums_by_request.values()
            for source_scenario_year, _ in weighted_sum.weights
        }
        if len(source_scenario_years) == 0:
            return {}
        results_by_source: Dict[ScenarioYear, ScenarioYearResult] = {}
        try:
            # Any errors should propagate up.
            read_results = await asyncio.gather(
                *(
                    self.get_single_item(
                        source_scenario_year,
                        latitudes,
                        longitudes,
                        buffer,
                        resource.path_for_scenario_year(
                            source_scenario_year.scenario, source_scenario_year.year
                        ),
                    )
                    for source_scenario_year in source_scenario_years
                )
            )
            for (
                source_scenario_year,
                values,
                in_bounds_mask,
                indices,
                units,
            ) in read_results:
                if in_bounds_mask is not None and not np.all(in_bounds_mask):
                    raise ValueError(
                        "inconsistent geographic coverage across scenario/year sources"
                    )
                results_by_source[source_scenario_year] = ScenarioYearResult(
                    values=values,
                    indices=indices,
                    indices_length=np.array([len(indices)], dtype=np.int32),
                    coverage_mask=coverage_mask,
                    units=expected_units if units == "default" else units,
                    paths=np.array([sys.intern(resource.path)], dtype=np.object_),
                )
        except KeyError as error:
            raise DataSourcingError(
                f"Dataset not found for hazard type {self.hazard_type.__name__} "
                + f"indicator ID {indicator_id}: {error.args[0]}"
            ) from error
        for requested_scenario_year, weighted_sum in weighted_sums_by_request.items():
            first_source, first_weight = weighted_sum.weights[0]
            first_result = results_by_source[first_source]
            combined_values = first_result.values * first_weight
            if len(weighted_sum.weights) > 1:
                second_source, second_weight = weighted_sum.weights[1]
                second_result = results_by_source[second_source]
                if first_result.units != second_result.units:
                    raise ValueError(
                        f"inconsistent units across interpolation inputs for {requested_scenario_year}"
                    )
                combined_values = combined_values + second_result.values * second_weight
                # important edge-case: if this is an extrapolation then the result may be non-monotonic
                # if the inputs are either non-decreasing or non-increasing we ensure the same is true
                # for the outputs
                if first_result.values.shape[1] > 1 and (
                    (first_weight > 0 and second_weight < 0)
                    or (first_weight < 0 and second_weight > 0)
                ):
                    if np.all(np.diff(first_result.values) >= 0) and np.all(
                        np.diff(second_result.values) >= 0
                    ):
                        if not np.all(np.diff(combined_values) >= 0):
                            combined_values[:, 1:] -= np.minimum(
                                np.diff(combined_values), 0.0
                            )
                    elif np.all(np.diff(first_result.values) <= 0) and np.all(
                        np.diff(second_result.values) <= 0
                    ):
                        if not np.all(np.diff(combined_values) <= 0):
                            combined_values[:, 1:] -= np.maximum(
                                np.diff(combined_values), 0.0
                            )

            results_by_request[requested_scenario_year] = ScenarioYearResult(
                values=combined_values,
                indices=first_result.indices,
                indices_length=first_result.indices_length,
                coverage_mask=first_result.coverage_mask,
                units=first_result.units,
                paths=first_result.paths,
            )
        return results_by_request

    async def get_single_item(
        self,
        source_scenario_year: ScenarioYear,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        buffer: Optional[int],
        concrete_path: str,
    ):
        in_bounds_mask = None
        if buffer is None:
            values, in_bounds_mask, indices, units = await asyncio.to_thread(
                self._reader.get_curves,
                concrete_path,
                longitudes,
                latitudes,
                self._interpolation,
            )
        else:
            if buffer < 0 or 1000 < buffer:
                raise Exception(
                    "The buffer must be an integer between 0 and 1000 metres."
                )
            values, indices, units = await asyncio.to_thread(
                self._reader.get_max_curves,
                concrete_path,
                [
                    (
                        Point(longitude, latitude)
                        if buffer == 0
                        else Point(longitude, latitude).buffer(
                            ZarrReader._get_equivalent_buffer_in_arc_degrees(
                                latitude, buffer
                            )
                        )
                    )
                    for longitude, latitude in zip(longitudes, latitudes)
                ],
                self._interpolation,
            )  # type: ignore
        return (
            source_scenario_year,
            values,
            in_bounds_mask,
            indices,
            units,
        )
