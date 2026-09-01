import asyncio
import sys
from collections.abc import Sequence

import numpy as np

from physrisk.data.hazard_data_provider import (
    DataSourcingError,
    HazardDataHint,
    HazardResourceProvider,
    ScenarioYear,
    ScenarioYearResolver,
    ScenarioYearResult,
    read_single_item,
)
from physrisk.data.hazard_data_resolution import (
    EmptyResourceError,
)
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazard_model import HazardDataSource
from physrisk.kernel.hazards import Hazard


class HierarchicalHazardDataProvider:
    """Retrieves hazard data from an ordered hierarchy of resources.

    Requested scenario/year pairs are resolved against each resource's available
    data. Resources are processed from highest to lowest priority, with each
    subsequent resource supplying coordinates not covered by preceding resources.
    """

    def __init__(
        self,
        hazard_type: type[Hazard],
        resource_provider: HazardResourceProvider,
        resolver: ScenarioYearResolver,
        *,
        store=None,
        zarr_reader: ZarrReader | None = None,
        interpolation: str = "floor",
    ):
        self.hazard_type = hazard_type
        self._reader = (
            zarr_reader if zarr_reader is not None else ZarrReader(store=store)
        )
        if interpolation not in ["floor", "linear", "max", "min"]:
            raise ValueError("interpolation must be 'floor', 'linear', 'max' or 'min'")
        self._interpolation = interpolation
        self._resource_provider = resource_provider
        self._resolver = resolver

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
        """Read data for requested scenario/year pairs and coordinates."""
        requested_items = tuple(
            ScenarioYear(scenario, year)
            for scenario in scenarios
            for year in ([-1] if scenario == "historical" else years)
        )
        if len(requested_items) == 0:
            return {}

        resources = self._resource_provider.get_resources(
            self.hazard_type, indicator_id, hint=hint
        )
        mask_unprocessed = np.ones(len(longitudes), dtype=bool)
        results: dict[ScenarioYear, list[ScenarioYearResult]] = {}

        for resource in resources:
            if not np.any(mask_unprocessed):
                break

            try:
                resolutions = {
                    requested: self._resolver(requested, resource)
                    for requested in requested_items
                }
            except EmptyResourceError:
                continue

            paths_by_resolved_item = {
                resolved: resource.path_for_scenario_year(
                    resolved.scenario, resolved.year
                )
                for resolved in dict.fromkeys(resolutions.values())
            }
            coverage_path = next(iter(paths_by_resolved_item.values()))
            try:
                mask_in_bounds = await asyncio.to_thread(
                    self._reader.in_bounds,
                    coverage_path,
                    longitudes[mask_unprocessed],
                    latitudes[mask_unprocessed],
                    self._interpolation,
                )
                coverage = mask_unprocessed.copy()
                coverage[mask_unprocessed] &= mask_in_bounds
                if not np.any(coverage):
                    continue
                mask_unprocessed[coverage] = False

                read_results = await asyncio.gather(
                    *(
                        read_single_item(
                            self._reader,
                            self._interpolation,
                            item,
                            latitudes[coverage],
                            longitudes[coverage],
                            buffer,
                            path,
                        )
                        for item, path in paths_by_resolved_item.items()
                    )
                )
            except KeyError as error:
                raise DataSourcingError(
                    f"Dataset not found for hazard type {self.hazard_type.__name__} "
                    f"indicator ID {indicator_id}: {error.args[0]}"
                ) from error

            results_by_resolved_item = {}
            for (
                item,
                values,
                in_bounds_mask,
                indices,
                units,
                concrete_path,
            ) in read_results:
                if in_bounds_mask is not None and not np.all(in_bounds_mask):
                    raise ValueError(
                        "inconsistent geographic coverage across scenario/year sources"
                    )
                source = HazardDataSource(
                    resource_id=sys.intern(resource.path),
                    path=sys.intern(concrete_path),
                    scenario=sys.intern(item.scenario),
                    year=item.year,
                )
                results_by_resolved_item[item] = ScenarioYearResult(
                    values=values,
                    indices=indices,
                    indices_length=np.array([len(indices)], dtype=np.int32),
                    coverage_mask=coverage,
                    units=resource.units if units == "default" else units,
                    paths=np.array([sys.intern(resource.path)], dtype=np.object_),
                    sources=np.array([source], dtype=np.object_),
                )

            for requested, resolved_item in resolutions.items():
                results.setdefault(requested, []).append(
                    results_by_resolved_item[resolved_item]
                )

        return self._merge_results(results)

    @staticmethod
    def _merge_results(
        results: dict[ScenarioYear, list[ScenarioYearResult]],
    ) -> dict[ScenarioYear, ScenarioYearResult]:
        """Merge geographical partial results into full-coverage results."""
        final: dict[ScenarioYear, ScenarioYearResult] = {}
        for key, partial_results in results.items():
            first = partial_results[0]
            if any(partial.units != first.units for partial in partial_results[1:]):
                raise ValueError(f"inconsistent units for scenario/year {key}")
            coordinate_count = len(first.coverage_mask)
            max_index_length = max(
                int(partial.indices_length[0]) for partial in partial_results
            )

            merged = ScenarioYearResult(
                values=np.empty((coordinate_count, max_index_length)),
                indices=np.empty(
                    (coordinate_count, max_index_length), dtype=first.indices.dtype
                ),
                indices_length=np.empty(
                    coordinate_count, dtype=first.indices_length.dtype
                ),
                coverage_mask=np.zeros(coordinate_count, dtype=bool),
                units=first.units,
                paths=np.empty(coordinate_count, dtype=np.object_),
                sources=np.empty(coordinate_count, dtype=np.object_),
            )
            final[key] = merged

            for partial in partial_results:
                indices_length = int(partial.indices_length[0])
                merged.values[partial.coverage_mask, :indices_length] = partial.values
                merged.indices[partial.coverage_mask, :indices_length] = partial.indices
                merged.indices_length[partial.coverage_mask] = partial.indices_length
                merged.coverage_mask[partial.coverage_mask] = True
                merged.paths[partial.coverage_mask] = partial.paths
                assert merged.sources is not None and partial.sources is not None
                merged.sources[partial.coverage_mask] = partial.sources

        return final
