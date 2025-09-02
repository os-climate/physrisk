from abc import ABC
import asyncio
from dataclasses import dataclass
import logging
import sys
from typing import (
    Callable,
    Dict,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import numpy as np
from shapely import Point
from typing_extensions import Protocol

from physrisk.kernel.hazards import Hazard

from .zarr_reader import ZarrReader


logger = logging.getLogger(__name__)


@dataclass
class HazardDataHint:
    """Requestors of hazard data may provide a hint which may be taken into account by the Hazard Model.
    A hazard resource path can be specified which uniquely defines the hazard resource; otherwise the resource
    is inferred from the indicator_id."""

    path: Optional[str] = None
    # consider adding: indicator_model_gcm: str

    def group_key(self):
        return self.path


@dataclass
class ScenarioPaths:
    """For a given Hazard Resource and scenario, gives the available years and
    function to generate the path for each year. If no years are available (because the
    scenario cannot be matched), this will be an empty list.
    """

    years: Sequence[int]
    path: Callable[[int], str]


@dataclass
class ResourcePaths:
    """Provides for a single HazardResource the mappings of scenarios to the ScenarioPaths.
    The ScenarioPaths for each scenario will give the available years and a function to get
    the array path for a year.
    """

    # the path (unique identifier) of the HazardResource
    resource_path: str
    # note the key is the *requested* scenario, not the scenario it may be proxied to
    scenarios: Dict[str, ScenarioPaths]


class SourcePaths(Protocol):
    def hazard_types(self) -> List[Type[Hazard]]:
        """Lists the available hazard types.

        Returns:
            List[Type[Hazard]]: Available hazard types.
        """
        ...

    def resource_paths(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        scenarios: Sequence[str],
        hint: Optional[HazardDataHint] = None,
    ) -> List[ResourcePaths]:
        """Provides a list of ResourcePaths. Each item in the list provides the paths for
        a different HazardResource. The ResourcePaths will be tried in order. If a
        latitude/longitude is out of bounds of one HazardResource

        Args:
            hazard_type (Type[Hazard]): Hazard type.
            indicator_id (str): Hazard indicator identifier.
            scenarios (Sequence[str]): Scenario identifiers.
            hint (Optional[HazardDataHint], optional): Hint to be applied to select path. Generally only
            makes sense if there is just a single scenario. Defaults to None.

        Returns:
            List[ResourcePaths]: List of ResourcePaths to be tried in order.
        """
        ...

    def scenario_paths_for_id(
        self,
        resource_id: str,
        scenarios: Sequence[str],
        map: bool = False,
        map_zoom: Optional[int] = None,
    ) -> Dict[str, ScenarioPaths]:
        """Returns the ScenarioPaths when a unique ID is specified."""
        ...


class DataSourcingError(Exception):
    pass


class ScenarioYear(NamedTuple):
    scenario: str
    year: int


class ScenarioYearRes(NamedTuple):
    scenario: str
    year: int
    resource_index: Optional[int]


@dataclass
class ScenarioYearResult:
    values: np.ndarray
    indices: np.ndarray
    indices_length: np.ndarray
    coverage_mask: np.ndarray  # boolean mask giving the part of the original set of lats/lons that this applies to
    units: str
    paths: np.ndarray


@dataclass
class WeightedSum:
    weights: List[Tuple[ScenarioYear, float]]


class HazardDataProvider(ABC):
    def __init__(
        self,
        hazard_type: Type[Hazard],
        source_paths: SourcePaths,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
        historical_year: int = 2025,
    ):
        """Provides hazard data.

        Args:
            hazard_type (Type[Hazard]): Hazard type.
            source_paths (SourcePath): Provides the source path mappings.
            store (Optional[MutableMapping], optional): Zarr store instance. Defaults to None.
            zarr_reader (Optional[ZarrReader], optional): ZarrReader instance. Defaults to None.
            interpolation (Optional[str], optional): Interpolation type. Defaults to "floor".
            historical_year (int): The year to be considered as 'historical' for purposes of interpolation over years.

        Raises:
            ValueError: If interpolation not in permitted list.
        """
        self.hazard_type = hazard_type
        self.historical_year = historical_year
        self._source_paths = source_paths
        self._reader = (
            zarr_reader if zarr_reader is not None else ZarrReader(store=store)
        )
        if interpolation not in ["floor", "linear", "max", "min"]:
            raise ValueError("interpolation must be 'floor', 'linear', 'max' or 'min'")
        self._interpolation = interpolation

    async def get_data_cascading(
        self,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        *,
        indicator_id: str,
        scenarios: Sequence[str],
        years: Sequence[int],
        hint: Optional[HazardDataHint] = None,
        buffer: Optional[int] = None,
        interpolate_years: bool = False,
    ):
        """Returns data for set of latitude and longitudes.

        Args:
            longitudes (np.ndarray): Longitudes.
            latitudes (np.ndarray): Latitudes.
            indicator_id (str): Hazard Indicator ID.
            scenarios (Sequence[str]): Identifier of scenario, e.g. ssp585 (SSP 585), rcp8p5 (RCP 8.5).
            years (Sequence[int]): Projection years, e.g. [2050, 2080].
            hint (Optional[HazardDataHint], optional): Hint. Defaults to None.
            buffer (Optional[int], optional): _description_. Buffer around each point.
            interpolate_years (bool, optional): If True, interpolate between years. Defaults to False.

        Returns:
            Dict[ScenarioYear, ScenarioYearResult]: Results.
        """
        # For each scenario, we find the list of array paths and available years for each path.
        final_result: Dict[ScenarioYear, ScenarioYearResult] = {}
        # mask_unprocessed is the mask of lats and lons that remain unprocessed.
        # This always has the same length and is updated for each path_item.
        # combined data for each year
        mask_unprocessed = np.ones(len(longitudes), dtype=np.bool)
        resource_paths_set: List[ResourcePaths] = self._source_paths.resource_paths(
            self.hazard_type,
            indicator_id=indicator_id,
            scenarios=scenarios,
            hint=hint,
        )
        results: Dict[ScenarioYearRes, ScenarioYearResult] = {}
        max_dim = 0
        for i, resource_paths in enumerate(resource_paths_set):
            # within a HazardResource the arrays have the same spatial coverage
            # any array can therefore be used for checking bounds
            if not np.any(mask_unprocessed):
                break
            p, y = next(
                (
                    (p, y)
                    for p in resource_paths.scenarios.values()
                    for y in p.years
                    if y in years
                ),
                # use a matching year if there is one. This is done just to facilitate unit testing!
                next(
                    (
                        (p, p.years[0])
                        for p in resource_paths.scenarios.values()
                        if len(p.years) > 0
                    ),
                    # otherwise any valid year; if there are none, no results can be returned.
                    (None, None),
                ),
            )
            if p is None or y is None:
                continue
            set_id = p.path(y)
            mask_in_bounds = await asyncio.to_thread(
                self._reader.in_bounds,
                set_id,
                longitudes[mask_unprocessed],
                latitudes[mask_unprocessed],
            )
            coverage = mask_unprocessed.copy()
            coverage[mask_unprocessed] = coverage[mask_unprocessed] & mask_in_bounds
            mask_unprocessed[mask_unprocessed] = (
                mask_unprocessed[mask_unprocessed] & ~mask_in_bounds
            )

            resource_result = await self.get_scenarios_and_years(
                i,
                coverage,
                longitudes[coverage],
                latitudes[coverage],
                indicator_id,
                resource_paths,
                years,
                buffer,
                interpolate_years,
            )
            if len(resource_result) > 0:
                results.update(resource_result)
                max_dim = max(
                    max_dim, next(iter(resource_result.values())).indices_length[0]
                )

        for k, v in results.items():
            key = ScenarioYear(k.scenario, k.year)
            if key not in final_result:
                values = np.empty((len(longitudes), max_dim))
                indices = np.empty((len(longitudes), max_dim), dtype=v.indices.dtype)
                indices_length = np.empty(
                    (len(longitudes)), dtype=v.indices_length.dtype
                )
                paths = np.empty((len(longitudes)), dtype=np.object_)
                final_result[key] = ScenarioYearResult(
                    values=values,
                    indices=indices,
                    indices_length=indices_length,
                    coverage_mask=np.zeros(
                        len(longitudes), dtype=np.bool
                    ),  # ~mask_unprocessed,
                    units=v.units,
                    paths=paths,
                )
            res = final_result[key]
            indices_length = v.indices_length[0]
            res.values[v.coverage_mask, :indices_length] = v.values
            res.indices[v.coverage_mask, :indices_length] = v.indices
            res.indices_length[v.coverage_mask] = v.indices_length
            res.coverage_mask[v.coverage_mask] = True
            res.paths[v.coverage_mask] = v.paths
        return final_result

    async def get_scenarios_and_years(
        self,
        resource_index: int,
        coverage: np.ndarray,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        indicator_id: str,
        resource_paths: ResourcePaths,
        years: Sequence[int],
        buffer: Optional[int],
        interpolate_years: bool,
    ):
        """Get data for all scenarios and years using just a single HazardResource as the source.
        The importance of this is that interpolation of years is assumed to be feasible within the same resource as this
        is a single model (with consistent meaning of the values).
        """
        result: Dict[ScenarioYear, ScenarioYearResult] = {}
        # Retrieve the data for all available years for the path in question.
        weights: Dict[ScenarioYear, WeightedSum] = {}
        for scenario, paths in resource_paths.scenarios.items():
            if len(paths.years) == 0:
                continue
            requested_years = [-1] if scenario == "historical" else years
            if interpolate_years:
                year_weights = HazardDataProvider._weights(
                    scenario, paths.years, requested_years, self.historical_year
                )
            else:
                year_weights = {
                    ScenarioYear(scenario, y): WeightedSum(
                        weights=[(ScenarioYear(scenario, y), 1.0)]
                    )
                    for y in requested_years
                    if y in paths.years
                }
            weights.update(year_weights)

        all_items = set(w[0] for ws in weights.values() for w in ws.weights)
        if len(all_items) == 0:
            empty: Dict[ScenarioYearRes, ScenarioYearResult] = {}
            return empty
        masks_in_bounds = {}
        targets: Dict[ScenarioYear, ScenarioYearResult] = {}
        try:
            # Any errors should propagate up.
            res = await asyncio.gather(
                *(
                    self.get_single_item(
                        item,
                        latitudes,
                        longitudes,
                        buffer,
                        resource_paths.scenarios[item.scenario].path(item.year),
                    )
                    for item in all_items
                )
            )
            for item, values, mask_in_bounds, indices, units, _path in res:
                targets[item] = ScenarioYearResult(
                    values=values,
                    indices=indices,
                    indices_length=np.array(
                        [len(indices)], dtype=np.int32
                    ),  # a numpy scalar
                    coverage_mask=coverage,
                    units=units,
                    paths=np.array(
                        [sys.intern(resource_paths.resource_path)], dtype=np.object_
                    ),
                )
                masks_in_bounds[item] = mask_in_bounds
        except KeyError as ke:
            raise DataSourcingError(
                f"Dataset not found for hazard type {self.hazard_type.__name__} "
                + f"indicator ID {indicator_id}: {ke.args[0]}"
            )
        for item, sum in weights.items():
            r1, w1 = targets[sum.weights[0][0]], sum.weights[0][1]
            values = r1.values * w1
            if len(sum.weights) > 1:
                r2, w2 = targets[sum.weights[1][0]], sum.weights[1][1]
                values = values + r2.values * w2
                # important edge-case: if this is an extrapolation then the result may be non-monotonic
                # if the inputs are either non-decreasing or non-increasing we ensure the same is true
                # for the outputs
                if r1.values.shape[1] > 1 and (
                    (w1 > 0 and w2 < 0) or (w1 < 0 and w2 > 0)
                ):
                    if np.all(np.diff(r1.values) >= 0) and np.all(
                        np.diff(r2.values) >= 0
                    ):
                        if not np.all(np.diff(values) >= 0):
                            values[:, 1:] -= np.minimum(np.diff(values), 0.0)
                    elif np.all(np.diff(r1.values) <= 0) and np.all(
                        np.diff(r1.values) <= 0
                    ):
                        if not np.all(np.diff(values) <= 0):
                            values[:, 1:] -= np.maximum(np.diff(values), 0.0)

            result[item] = ScenarioYearResult(
                values=values,
                indices=r1.indices,
                indices_length=r1.indices_length,
                coverage_mask=r1.coverage_mask,
                units=r1.units,
                paths=r1.paths,
            )
        # For a given data set, the spatial coverage should be identical between years. If not, something is wrong.
        mask_in_bounds = next(iter(masks_in_bounds.values()))
        if mask_in_bounds is not None and any(
            np.any(mask != mask_in_bounds)
            for mask in masks_in_bounds.values()
            if mask is not None
        ):
            raise ValueError("inconsistent coverage across years")
        return {
            ScenarioYearRes(k.scenario, k.year, resource_index): v
            for k, v in result.items()
        }

    async def get_single_item(
        self,
        item: ScenarioYear,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        buffer: Optional[int],
        path: str,
    ):
        indices, units = [], ""
        mask_in_bounds = None
        if buffer is None:
            values, mask_in_bounds, indices, units = await asyncio.to_thread(
                self._reader.get_curves,
                path,
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
                path,
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
        return item, values, mask_in_bounds, indices, units, path

    @staticmethod
    def _weights(
        scenario: str,
        available_years: Sequence[int],
        requested_years: Sequence[int],
        historical_year: int,
    ) -> Dict[ScenarioYear, WeightedSum]:
        available_with_current = sorted(
            np.array([historical_year] + list(available_years))
        )
        # return i such that a[i-1] < v <= a[i]
        # e.g. with available years: 2025, 2040, 2050, 2060
        # 2045 gives index 2, need 1 and 2
        # 2050 gives index 2, need 2
        # 2060 gives index 3, need 1 and 2
        weights: List[Tuple[ScenarioYear, float]] = []
        indices = np.searchsorted(available_with_current, requested_years, side="left")
        result: Dict[ScenarioYear, WeightedSum] = {}

        if scenario == "historical":
            result[ScenarioYear("historical", -1)] = WeightedSum(
                weights=[(ScenarioYear("historical", -1), 1.0)]
            )
            return result

        def scenario_year(scenario: str, year: int):
            return (
                ScenarioYear("historical", -1)
                if year == historical_year
                else ScenarioYear(scenario, int(year))
            )

        for i, index in enumerate(indices):
            if index == len(available_with_current):
                # linear extrapolation
                # v_e = v_2 + (y_e - y_2) * (v_2 - v_1) / (y_2 - y_1)
                slope = (
                    float(requested_years[i]) - float(available_with_current[-1])
                ) / (
                    float(available_with_current[-1])
                    - float(available_with_current[-2])
                )
                weights = [
                    (scenario_year(scenario, available_with_current[-2]), -slope),
                    (scenario_year(scenario, available_with_current[-1]), 1.0 + slope),
                ]
            elif available_with_current[index] == requested_years[i]:
                # exact match
                weights = [
                    (scenario_year(scenario, available_with_current[index]), 1.0)
                ]
            else:
                # linear interpolation
                w1 = (
                    float(available_with_current[index]) - float(requested_years[i])
                ) / (
                    float(available_with_current[index])
                    - float(available_with_current[index - 1])
                )
                weights = [
                    (
                        scenario_year(scenario, available_with_current[index - 1]),
                        w1,
                    ),
                    (scenario_year(scenario, available_with_current[index]), 1.0 - w1),
                ]
            result[scenario_year(scenario, requested_years[i])] = WeightedSum(
                weights=weights
            )
        return result
