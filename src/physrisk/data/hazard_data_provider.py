from abc import ABC
from dataclasses import dataclass
from typing import List, MutableMapping, Optional

from shapely import Point
from typing_extensions import Protocol

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


class SourcePath(Protocol):
    """Provides path to hazard event data source. Each source should have its own implementation.

    Args:
        model: model identifier.
        scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
        year: projection year, e.g. 2080.
    """

    def __call__(
        self,
        *,
        indicator_id: str,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ) -> str: ...


class HazardDataProvider(ABC):
    def __init__(
        self,
        get_source_path: SourcePath,
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
        self._get_source_path = get_source_path
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

        path = self._get_source_path(
            indicator_id=indicator_id, scenario=scenario, year=year, hint=hint
        )
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
