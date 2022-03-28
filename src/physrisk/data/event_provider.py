import os
from typing import List, MutableMapping, Optional

from typing_extensions import Protocol

from .zarr_reader import ZarrReader


class SourcePath(Protocol):
    """Provides path to hazard event data source. Each source should have its own implementation.
    Args:
        year: projection year, e.g. 2080.
        scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
        model: model identifier.
    """

    def __call__(self, *, model: str, scenario: str, year: int) -> str:
        ...


class EventProvider:
    """Provides hazard event intensities for a single Event (type of hazard event)."""

    def __init__(
        self,
        get_source_path: SourcePath,
        *,
        store: Optional[MutableMapping] = None,
    ):
        """Create an EventProvider.

        Args:
            get_source_path: provides the path to the hazard event data source depending on year/scenario/model.
        """
        self._get_source_path = get_source_path
        self._reader = ZarrReader(store=store)

    def get_intensity_curves(
        self, longitudes: List[float], latitudes: List[float], *, model: str, scenario: str, year: int
    ):
        """Get intensity curve for each latitude and longitude coordinate pair.

        Args:
            longitudes: list of longitudes.
            latitudes: list of latitudes.
            year: projection year, e.g. 2080.
            scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
            model: model identifier.

        Returns:
            curves: numpy array of intensity (no. coordinate pairs, no. return periods).
            return_periods: return periods in years.
        """

        path = self._get_source_path(model=model, scenario=scenario, year=year)
        curves, return_periods = self._reader.get_curves(path, longitudes, latitudes)
        return curves, return_periods


# region World Resource Aqueduct Model


def _wri_inundation_prefix():
    return "inundation/wri/v2"


def get_source_path_wri_riverine_inundation(*, model: str, scenario: str, year: int):
    type = "river"
    return os.path.join(_wri_inundation_prefix(), f"inun{type}_{scenario}_{model}_{year}")


# endregion
