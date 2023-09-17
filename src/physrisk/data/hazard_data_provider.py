from abc import ABC
from dataclasses import dataclass
from typing import List, MutableMapping, Optional

from typing_extensions import Protocol

from .zarr_reader import ZarrReader


@dataclass
class HazardDataHint:
    """Requestors of hazard data may provide a hint which may be taken into account by the Hazard Model.
    A hazard resource path can be specified which uniquely defines the hazard resource; otherwise the resource
    is inferred from the indicator_id."""

    path: Optional[str]
    # consider adding: indicator_model_gcm: str

    def group_key(self):
        return self.path


class SourcePath(Protocol):
    """Provides path to hazard event data source. Each source should have its own implementation.
    Args:
        model: model identifier.
        scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
        year: projection year, e.g. 2080.
    """

    def __call__(self, *, indicator_id: str, scenario: str, year: int, hint: Optional[HazardDataHint] = None) -> str:
        ...


class HazardDataProvider(ABC):
    def __init__(
        self,
        get_source_path: SourcePath,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
    ):
        """Create an EventProvider.

        Args:
            get_source_path: provides the path to the hazard event data source depending on year/scenario/model.
        """
        self._get_source_path = get_source_path
        self._reader = zarr_reader if zarr_reader is not None else ZarrReader(store=store)
        if interpolation not in ["floor", "linear"]:
            raise ValueError("interpolation must be 'floor' or 'linear'")
        self._interpolation = interpolation


class AcuteHazardDataProvider(HazardDataProvider):
    """Provides hazard event intensities for a single Hazard (type of hazard event)."""

    def __init__(
        self,
        get_source_path: SourcePath,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
    ):
        super().__init__(get_source_path, store=store, zarr_reader=zarr_reader, interpolation=interpolation)

    def get_intensity_curves(
        self,
        longitudes: List[float],
        latitudes: List[float],
        *,
        indicator_id: str,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        """Get intensity curve for each latitude and longitude coordinate pair.

        Args:
            longitudes: list of longitudes.
            latitudes: list of latitudes.
            model: model identifier.
            scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
            year: projection year, e.g. 2080.

        Returns:
            curves: numpy array of intensity (no. coordinate pairs, no. return periods).
            return_periods: return periods in years.
        """

        path = self._get_source_path(indicator_id=indicator_id, scenario=scenario, year=year, hint=hint)
        curves, return_periods = self._reader.get_curves(
            path, longitudes, latitudes, self._interpolation
        )  # type: ignore
        return curves, return_periods


class ChronicHazardDataProvider(HazardDataProvider):
    """Provides hazard parameters for a single type of chronic hazard."""

    def __init__(
        self,
        get_source_path: SourcePath,
        *,
        store: Optional[MutableMapping] = None,
        zarr_reader: Optional[ZarrReader] = None,
        interpolation: Optional[str] = "floor",
    ):
        super().__init__(get_source_path, store=store, zarr_reader=zarr_reader, interpolation=interpolation)

    def get_parameters(
        self,
        longitudes: List[float],
        latitudes: List[float],
        *,
        indicator_id: str,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        """Get hazard parameters for each latitude and longitude coordinate pair.

        Args:
            longitudes: list of longitudes.
            latitudes: list of latitudes.
            model: model identifier.
            scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
            year: projection year, e.g. 2080.

        Returns:
            parameters: numpy array of parameters
        """

        path = self._get_source_path(indicator_id=indicator_id, scenario=scenario, year=year, hint=hint)
        parameters, _ = self._reader.get_curves(path, longitudes, latitudes, self._interpolation)
        return parameters[:, 0]
