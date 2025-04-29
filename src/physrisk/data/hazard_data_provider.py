from abc import ABC
import asyncio
from dataclasses import dataclass
import logging
import sys
from typing import Callable, Dict, List, MutableMapping, NamedTuple, Optional, Sequence, Tuple, Type, Union, runtime_checkable

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
    def hazard_types(self) -> List[Type[Hazard]]:
        """Lists the available hazard types.

        Returns:
            List[Type[Hazard]]: Available hazard types.
        """
        ...
    
    def paths(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        scenario: str,
        hint: Optional[HazardDataHint] = None) -> List[Paths]:
        """Get the cascading Paths list for the given hazard type, indicator ID and scenario.
        Each Paths item in the list has the available years and a function to obtain the path for each year.
        Each Paths item covers a different area and can be used to match multiple data sets. 
        
        Args:
            hazard_type (Type[Hazard]): Hazard yype.
            indicator_id (str): Hazard indicator identifier.
            scenario (str): Scenario identifier.
            hint (Optional[HazardDataHint], optional): Hint to be applied to select path. Defaults to None.

        Returns:
            List[Paths]: List of Paths objects to be used in order to obtain the data.
        """
        ...

class DataSourcingError(Exception):
    pass

    
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
        """Returns data for set of latitude and longitudes.

        Args:
            longitudes (Sequence[float]): Longitudes.
            latitudes (Sequence[float]): Latitudes.
            indicator_id (str): Hazard Indicator ID.
            scenarios (Sequence[str]): Identifier of scenario, e.g. ssp585 (SSP 585), rcp8p5 (RCP 8.5).
            years (Sequence[int]): Projection years, e.g. [2050, 2080].
            hint (Optional[HazardDataHint], optional): Hint. Defaults to None.
            buffer (Optional[int], optional): _description_. Buffer around each point.
            interpolate_years (bool, optional): If True, interpolate between years. Defaults to False.

        Returns:
            Dict[ScenarioYear, ScenarioYearResult]: Results.
        """
        if interpolate_years == True:
            raise NotImplementedError("interpolation not yet implemented")
        # For each scenario, we find the list of array paths and available years for each path.
        longitudes = np.array(longitudes)
        latitudes = np.array(latitudes)
        result: Dict[ScenarioYear, ScenarioYearResult] = {}
        # any exception should be allowed to propagate
        await asyncio.gather(*(self.get_scenario(longitudes, latitudes, indicator_id, scenario, years,
                            hint, buffer, interpolate_years, result) for scenario in scenarios))
        # if interpolate_years
        return result
    
    async def get_scenario(self,
                        longitudes: Sequence[float],
                        latitudes: Sequence[float],
                        indicator_id: str,
                        scenario: str,
                        years: Sequence[int],
                        hint: Optional[HazardDataHint],
                        buffer: Optional[int],
                        interpolate_years: bool,
                        result: Dict[ScenarioYear, ScenarioYearResult]):
        path_set = self._source_paths.paths(self.hazard_type,
                                                indicator_id=indicator_id,
                                                scenario=scenario,
                                                hint=hint)
        # mask_unprocessed is the mask of lats and lons that remain unprocessed.
        # This always has the same length and is updated for each path_item. 
        mask_unprocessed = np.ones(len(longitudes), dtype=np.bool)
        # combined data for each year
        combined_values: Dict[int, np.ndarray] = {}
        paths = np.empty((len(longitudes)), dtype=np.object_)
        for path_item in path_set: 
            # Retrieve the data for all available years for the path in question.
            if scenario == "historical":
                required_years = [-1]
            elif interpolate_years:
                required_years = self._bounding_years(years, path_item.years)
            else:
                required_years = [y for y in ([-1] if scenario == "historical" else years) if y in path_item.years] 
            if len(required_years) == 0:
                # can happen if not interpolating years: year just not available
                continue
            data_for_year: Dict[int, np.ndarray] = {}
            mask_for_year: Dict[int, np.ndarray] = {} 
            try:
                # Retrieve data for all years. Any errors should propagate up.
                res = await asyncio.gather(*(self.get_year(latitudes, longitudes, buffer, path_item, year, mask_unprocessed)
                                    for year in required_years))
                for year, values_year, mask_in_bounds_year, indices, units in res:
                    data_for_year[year] = values_year
                    mask_for_year[year] = mask_in_bounds_year
            except KeyError as ke:
                raise DataSourcingError(f"Dataset not found for hazard type {self.hazard_type.__name__} " +
                                        f"indicator ID {indicator_id} and scenario {scenario}: {ke.args[0]}")
            # For a given data set, the spatial coverage should be identical between years. If not, something is wrong.
            mask_in_bounds = mask_for_year[required_years[0]] 
            if any(np.any(mask_for_year[y] != mask_in_bounds) for y in required_years[1:]):
                raise ValueError("inconsistent coverage across years")
            # Interpolate here
            for year in required_years:
                if year not in combined_values:
                    combined_values[year] = data_for_year[year]
                else:
                    combined_values[year][mask_unprocessed] = data_for_year[year]

            paths[mask_unprocessed] = sys.intern(path_item.path(year))
            mask_unprocessed[mask_unprocessed] = mask_unprocessed[mask_unprocessed] & ~mask_in_bounds
            if not np.any(mask_unprocessed):
                break
        # We can choose to interpolate data at this point.
        for year in required_years:
            if np.any(mask_unprocessed):
                combined_values[year][mask_unprocessed] = np.nan
            result[ScenarioYear(scenario, year)] = ScenarioYearResult(combined_values[year], indices, mask_unprocessed, units, paths)

    async def get_year(self,
                        latitudes: Sequence[float],
                        longitudes: Sequence[float],
                        buffer: Optional[int],
                        path_item: Paths,
                        year: int,
                        mask_unprocessed: np.ndarray): 
        indices, units = [], ""
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
            values, indices, units = await asyncio.to_thread(
                self._reader.get_max_curves,
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
                    for longitude, latitude in zip(longitudes[mask_unprocessed], latitudes[mask_unprocessed])
                ],
                self._interpolation) # type: ignore
        return year, values, mask_in_bounds, indices, units

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

            
