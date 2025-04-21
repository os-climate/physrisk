from abc import ABC
import asyncio
from dataclasses import dataclass
import sys
from typing import Callable, Dict, List, MutableMapping, NamedTuple, Optional, Sequence, Tuple, Type, Union, runtime_checkable

import numpy as np
from shapely import Point
from typing_extensions import Protocol

from physrisk.kernel.hazards import Hazard

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


@dataclass
class Paths:
    """Gives the paths for all of the matching years.
    Years are ordered.
    """
    #year_paths: Dict[int, str]
    years: List[int]
    path: Callable[[int], str]


class SourcePath(Protocol):
    def __call__(
        self,
        *,
        indicator_id: str,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ) -> Union[str, List[Paths]]:
        """_summary_

        Args:
            indicator_id (str): Hazard indicator identifier.
            scenario (str): Scenario identifier.
            year (int): Year. Defaults to None.
            hint (Optional[HazardDataHint], optional): Hint. Defaults to None.

        Returns:
            str: If there is just a single path, return it, otherwise return paths for each year.
        """
        ...


class SourcePaths(Protocol):
    def cascading_year_paths(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        scenario: str,
        hint: Optional[HazardDataHint] = None) -> List[Paths]:
        """_summary_

        Args:
            indicator_id (str): Hazard indicator identifier.
            scenario (str): Scenario identifier.
        """
        ...

    # def cascading_year_source_path(
    #     self,    
    #     hazard_type: Type[Hazard],
    #     indicator_id: str,
    #     scenario: str,
    #     hint: Optional[HazardDataHint] = None,
    # ) -> List[Tuple[List[int], Callable[[int], str]]]:
    #     ...

        
class ScenarioYear(NamedTuple):
    scenario: str
    year: Optional[int]


@dataclass
class ScenarioYearResult:
    values: np.ndarray
    indices: np.ndarray
    mask_unprocessed: np.ndarray
    units: str
    paths: np.ndarray


class HazardDataProvider(ABC):
    def __init__(
        self,
        hazard_type: Type[Hazard],
        source_paths: SourcePaths,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
    ):
        """Provides hazard data.

        Args:
            get_source_path (SourcePath): Provides the source path mappings.
            store (Optional[MutableMapping], optional): Zarr store instance. Defaults to None.
            zarr_reader (Optional[ZarrReader], optional): ZarrReader instance. Defaults to None.
            interpolation (Optional[str], optional): Interpolation type. Defaults to "floor".

        Raises:
            ValueError: If interpolation not in permitted list.
        """
        self.hazard_type = hazard_type
        self._source_paths = source_paths
        self._reader = (
            zarr_reader if zarr_reader is not None else ZarrReader(store=store)
        )
        if interpolation not in ["floor", "linear", "max", "min"]:
            raise ValueError("interpolation must be 'floor', 'linear', 'max' or 'min'")
        self._interpolation = interpolation

    def get_data(
        self,
        longitudes: List[float],
        latitudes: List[float],
        *,
        indicator_id: str,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
        buffer: Optional[int] = None,
    ):
        """Returns data for set of latitude and longitudes.

        Args:
            longitudes (List[float]): List of longitudes.
            latitudes (List[float]): List of latitudes.
            indicator_id (str): Hazard Indicator ID.
            scenario (str): Identifier of scenario, e.g. ssp585 (SSP 585), rcp8p5 (RCP 8.5).
            year (int): Projection year, e.g. 2080.
            hint (Optional[HazardDataHint], optional): Hint. Defaults to None.
            buffer (Optional[int], optional): Buffer around each point
            expressed in metres (within [0, 1000]). Defaults to None.

        Raises:
            Exception: _description_

        Returns:
            values (np.ndarray): Hazard indicator values.
            indices (np.ndarray): Index values.
            units (str). Units
            path: Path to the hazard indicator data source.
        """
        path = self._source_paths.cascading_year_paths(self.hazard_type,
            indicator_id=indicator_id, scenario=scenario, hint=hint
        )[-1 if scenario=="historical" else year]
        if buffer is None:
            values, indices, units = self._reader.get_curves(
                path, longitudes, latitudes, self._interpolation
            )  # type: ignore
        else:
            if buffer < 0 or 1000 < buffer:
                raise Exception(
                    "The buffer must be an integer between 0 and 1000 metres."
                )
            values, indices, units = self._reader.get_max_curves(
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
        return values, indices, units, path
    
    async def get_data_cascading(
        self,
        longitudes: Sequence[float],
        latitudes: Sequence[float],
        *,
        indicator_id: str,
        scenarios: Sequence[str],
        years: Sequence[int],
        hint: Optional[HazardDataHint] = None,
        buffer: Optional[int] = None,
        interpolate_years: bool = False
    ):
        if interpolate_years == True:
            raise NotImplementedError("interpolation not yet implemented")
        # For each scenario, we find the list of array paths and available years for each path.
        longitudes = np.array(longitudes)
        latitudes = np.array(latitudes)
        result: Dict[ScenarioYear, ScenarioYearResult] = {}
        for scenario in scenarios:
            path_set = self._source_paths.cascading_year_paths(self.hazard_type,
                                                          indicator_id=indicator_id,
                                                          scenario=scenario,
                                                          hint=hint)
            # TODO, check transforms are the same across years
            # mask is the mask of lats and lons that remain unprocessed, alwatys has the same length 
            #for year in years:
            mask_unprocessed = np.ones(len(longitudes), dtype=np.bool)
            combined_values: Dict[int, np.ndarray] = {}
            paths = np.empty((len(longitudes)), dtype=np.object_)
            for path_index, path_item in enumerate(path_set): 
                # get the data for all available years for the path in question
                available_years = [y for y in ([-1] if scenario == "historical" else years) if y in path_item.years] 
                if len(available_years) == 0:
                    # can happen if not interpolating years: year just not available
                    continue
                data_for_year: Dict[int: np.ndarray] = {}
                mask_for_year: Dict[int: np.ndarray] = {}
                # get data for each year
                for year in available_years:    
                    try:
                        if buffer is None:
                            values, mask_in_bounds, indices, units = await asyncio.to_thread(
                                self._reader.get_curves,
                                path_item.path(year),
                                longitudes[mask_unprocessed],
                                latitudes[mask_unprocessed],
                                self._interpolation)
                        else:
                            if buffer < 0 or 1000 < buffer:
                                raise Exception(
                                    "The buffer must be an integer between 0 and 1000 metres."
                                )
                            values, indices, units = self._reader.get_max_curves(
                                path_item.path,
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
                        data_for_year[year] = values
                        mask_for_year[year] = mask_in_bounds
                    except KeyError as ke:
                        data_for_year[year] = np.empty(0)
                        mask_for_year[year] = np.ones(len(longitudes), dtype=np.bool)
                    except Exception as e:
                        data_for_year[year] = np.empty(0)
                        mask_for_year[year] = np.ones(len(longitudes), dtype=np.bool)
                # we can choose to interpolate data at this point
                # For a given data set, the spatial coverage should be identical between years. If not, something is wrong.
                mask_in_bounds = mask_for_year[available_years[0]] 
                if any(np.any(mask_for_year[y] != mask_in_bounds) for y in available_years[1:]):
                    raise ValueError("inconsistent coverage across years detected")
                for year in available_years:
                    # if interpolating, we would 
                    if path_index == 0:
                        combined_values[year] = data_for_year[year]
                    else:
                        combined_values[year][mask_unprocessed] = data_for_year[year]

                paths[mask_unprocessed] = sys.intern(path_item.path(year))
                mask_unprocessed[mask_unprocessed] = mask_unprocessed[mask_unprocessed] & ~mask_in_bounds
                if not np.any(mask_unprocessed):
                    break
                # we need to first find the specific set_id of the data sets, as this gives the
            for year in available_years:
                if np.any(mask_unprocessed):
                    combined_values[year][mask_unprocessed] = np.nan
                if len(combined_values[year]) > 0:
                    # if there was an exception, length will be zero: no result
                    result[ScenarioYear(scenario, year)] = ScenarioYearResult(combined_values[year], indices, mask_unprocessed, units, paths)
        return result
    

    @staticmethod
    def _bounding_years(requested_years: np.ndarray, available_years: np.ndarray):
        # available years includes current year
        # 2025, 2040, 2050, 2060
        # 2045 gives 2, need 1 and 2
        # 2050 gives 2, need 2
        # 2060 gives 3, need 1 and 2

        # a[i-1] < v <= a[i]
        # indices of available_years
        indices = np.searchsorted(available_years, requested_years)

        extrap_needed = False
        required_indices = []
        for i, index in enumerate(indices):
            if index == len(available_years):
                extrap_needed = True
            elif available_years[index] == requested_years[i]:
                required_indices.append(index)
            else:
                required_indices.extend([index - 1, index])
        if extrap_needed:
            required_indices.extend([len(available_years) - 2, len(available_years) - 1])
        return np.unique(required_indices)

            
