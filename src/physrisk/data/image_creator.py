import io
import logging
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image as Image

from physrisk.api.v1.hazard_data import HazardResource
from physrisk.api.v1.hazard_image import TileNotAvailableError
from physrisk.kernel.hazards import HazardKind, hazard_class
from physrisk.data import colormap_provider
from physrisk.data.hazard_data_provider import CascadingHazardDataProvider, SourcePaths
from physrisk.data.inventory import Inventory
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazard_model import HazardImageCreator, Tile

logger = logging.getLogger(__name__)


class ImageCreator(HazardImageCreator):
    """Convert small arrays into images for map display.
    Intended for arrays <~1500x1500 (otherwise, recommended to use Mapbox tiles - or similar).
    """

    def __init__(
        self,
        inventory: Inventory,
        source_paths: SourcePaths,
        reader: ZarrReader,
        historical_year: int = 2025,
    ):
        self.inventory = inventory
        self.source_paths = source_paths
        self.reader = reader
        self.historical_year = historical_year  # might be needed for interpolation

    def create_image(
        self,
        resource_id: str,
        scenario: str,
        year: int,
        format="PNG",
        colormap: str = "heating",
        tile: Optional[Tile] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        index_value: Optional[Union[str, float]] = None,
        scaling: str = "linear",
    ):
        try:
            scenario_paths = self.source_paths.scenario_paths_for_id(
                resource_id,
                ["historical", scenario],
                True,
                map_zoom=tile.z + 1 if tile is not None else None,
            )
            weighted_sum = next(
                iter(
                    CascadingHazardDataProvider._weights(
                        scenario,
                        scenario_paths[scenario].years,
                        [year],
                        self.historical_year,
                    ).values()
                )
            )
            image = self._to_image(
                {
                    scenario_paths[sy.scenario].path(sy.year): w
                    for sy, w in weighted_sum.weights
                },
                colormap,
                tile=tile,
                index_value=index_value,
                min_value=min_value,
                max_value=max_value,
                scaling=scaling,
            )
        except Exception as e:
            # if we are creating a whole image that does not exist, we log the error
            # and return a empty image; but if creating a tile we let the error propagate
            # because many map controls expect an HTTPException in such cases.
            if tile is None:
                logger.exception(e)
                image = Image.fromarray(np.array([[0]]), mode="RGBA")
            else:
                if isinstance(e, KeyError):
                    raise TileNotAvailableError(e.args[0]) from e
                else:
                    raise
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=format)
        return image_bytes.getvalue()

    def get_info(
        self, resource_id: str, scenario: str, year: int
    ) -> Tuple[Sequence[Any], Sequence[Any], str, str, Optional[int]]:
        resource = self.inventory.resources[resource_id]
        # in principle, depends on the scenario and year, although we assume here that
        # all years have the same index values available.
        scenario_paths = self.source_paths.scenario_paths_for_id(
            resource_id, [scenario], True, map_zoom=1
        )[scenario]
        path = scenario_paths.path(scenario_paths.years[0])
        return _get_image_info(self.reader, resource, path, resource_id)

    def to_file(
        self,
        filename: str,
        path: str,
        format="PNG",
        colormap: str = "heating",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """Create image for path specified and save as file.

        Args:
            filename (str): Filename.
            path (str): Path to array.
            format (str, optional): Image format. Defaults to "PNG".
            colormap (str, optional): Colormap name. Defaults to "heating".
            min_value (Optional[float], optional): Min value. Defaults to None.
            max_value (Optional[float], optional): Max value. Defaults to None.
        """
        image = self._to_image(
            {path: 1.0}, colormap, min_value=min_value, max_value=max_value
        )
        image.save(filename, format=format)

    def _to_image(
        self,
        path_weights: Dict[str, float],
        colormap: str = "heating",
        tile: Optional[Tile] = None,
        index_value: Optional[Union[str, float, int]] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        scaling: str = "linear",
    ) -> Image.Image:
        """Get image for path specified as array of bytes."""

        weighted_arrays = (
            weight * _read_map_array(self.reader, path, tile, index_value)
            for path, weight in path_weights.items()
        )
        data = next(weighted_arrays)
        for weighted_array in weighted_arrays:
            data += weighted_array
        return _render_map_array(
            data,
            colormap,
            min_value=min_value,
            max_value=max_value,
            scaling=scaling,
        )


def to_rgba(  # noqa: C901
    data: np.ndarray,
    get_colors: Callable[[int], List[int]],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    nodata_lower: Optional[float] = None,
    nodata_upper: Optional[float] = None,
    nodata_bin_transparent: bool = False,
    min_bin_transparent: bool = False,
    scaling: str = "linear",
) -> np.ndarray:
    """Convert a two-dimensional data array to an RGBA image."""
    if scaling not in ["linear", "log"]:
        raise ValueError(f"unsupported scaling: {scaling}")

    red = np.zeros(256, dtype=np.uint32)
    green = np.zeros(256, dtype=np.uint32)
    blue = np.zeros(256, dtype=np.uint32)
    alpha = np.zeros(256, dtype=np.uint32)
    for index in range(256):
        red[index], green[index], blue[index], alpha[index] = get_colors(index)
    if nodata_bin_transparent:
        alpha[0] = 0
    if min_bin_transparent:
        alpha[1] = 0

    def apply_palette(indices: np.ndarray) -> np.ndarray:
        return (
            red[indices]
            + (green[indices] << 8)
            + (blue[indices] << 16)
            + (alpha[indices] << 24)
        )

    mask_nodata = np.isnan(data)
    if nodata_lower is not None:
        mask_nodata |= data <= nodata_lower
    if nodata_upper is not None:
        mask_nodata |= data >= nodata_upper

    valid_data = data[~mask_nodata]
    if len(valid_data) == 0:
        return apply_palette(np.zeros(data.shape, dtype=np.uint8))

    if min_value is None:
        min_value = np.min(valid_data)
    if max_value is None:
        max_value = np.max(valid_data)

    if scaling == "log" and min_value <= 0.0:
        raise ValueError("scaling='log' requires a min_value greater than 0.")
    if max_value < min_value:
        raise ValueError("max_value must be greater than or equal to min_value.")

    mask_ge_max = data >= max_value
    mask_le_min = data <= min_value

    if max_value == min_value:
        result = np.where(mask_le_min, 1, 255).astype(np.uint8)
        result[mask_nodata] = 0
        return apply_palette(result)

    if scaling == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log(data, out=data)
        np.add(data, -np.log(min_value), out=data)
        np.multiply(
            data,
            253.0 / (np.log(max_value) - np.log(min_value)),
            out=data,
        )
        np.add(data, 2.0, out=data)
    else:
        np.add(data, -min_value, out=data)
        np.multiply(data, 253.0 / (max_value - min_value), out=data)
        np.add(data, 2.0, out=data)

    result = data.astype(np.uint8, casting="unsafe", copy=False)
    result[mask_ge_max] = 255
    result[mask_le_min] = 1
    result[mask_nodata] = 0
    return apply_palette(result)


def _read_map_array(
    reader: ZarrReader,
    path: str,
    tile: Tile | None,
    index_value: str | float | int | None,
) -> np.ndarray:
    """Read the requested index and tile from one concrete map path."""
    data = get_data(reader, path)
    if len(data.shape) != 3:
        raise ValueError(
            f"map array at {path!r} must have dimensions (index, y, x); "
            f"got shape {data.shape}"
        )

    index_values, _ = reader.get_index_values(data)
    if index_value is not None:
        selected_index_value: str | float | int
        if isinstance(index_values[0], float):
            selected_index_value = float(index_value)
        elif isinstance(index_values[0], int):
            selected_index_value = int(index_value)
        else:
            selected_index_value = str(index_value)
    index = (
        len(index_values) - 1
        if index_value is None
        else index_values.index(selected_index_value)
    )
    if tile is None:
        return data[index, :, :]

    tile_size = 512
    return data[
        index,
        tile_size * tile.y : tile_size * (tile.y + 1),
        tile_size * tile.x : tile_size * (tile.x + 1),
    ]


def _render_map_array(
    data: np.ndarray,
    colormap: str,
    *,
    min_value: float | None,
    max_value: float | None,
    scaling: str,
) -> Image.Image:
    """Render one prepared two-dimensional array as an image."""
    if any(dimension > 4000 for dimension in data.shape):
        raise ValueError("dimension too large (over 4000).")
    map_definition = colormap_provider.colormap(colormap)

    def get_colors(index: int):
        return map_definition[str(index)]

    rgba = to_rgba(
        data,
        get_colors,
        min_value=min_value,
        max_value=max_value,
        scaling=scaling,
    )
    return Image.fromarray(rgba, mode="RGBA")


def _get_image_info(
    reader: ZarrReader,
    resource: HazardResource,
    path: str,
    resource_id: str,
) -> tuple[Sequence[Any], Sequence[Any], str, str, int | None]:
    """Read image metadata from one concrete map path."""
    z = reader.all_data(path)
    all_index_values, index_units = reader.get_index_values(z)
    available_index_values = (
        resource.map.index_values
        if resource.map is not None and resource.map.index_values
        else all_index_values
    )
    hazard_type = hazard_class(resource.hazard_type)
    index_display_name = (
        "return period" if hazard_type.kind == HazardKind.ACUTE else "threshold"
    )
    if index_units == "default":
        if hazard_type.kind == HazardKind.ACUTE:
            index_units = "years"
        elif resource.indicator_id in [
            "days_wbgt_above",
            "mean_degree_days/above/index",
            "weeks_water_temp_above",
        ]:
            index_units = "°C"
        else:
            index_units = ""

    max_zoom = None
    is_pyramid = resource.map and resource.map.source != "map_array"
    if is_pyramid:
        try:
            path_stripped = str(PurePosixPath(path).parent) + "/"
            zoom_levels = [
                int(PurePosixPath(item).name)
                for item in reader.ls(path_stripped)
                if PurePosixPath(item).name.isnumeric()
            ]
            max_zoom = max(zoom_levels)
        except Exception as error:
            logger.warning(
                "Could not obtain max zoom for resource %s: %s", resource_id, error
            )

    return (
        all_index_values,
        available_index_values,
        index_display_name,
        index_units,
        max_zoom,
    )


@lru_cache(maxsize=32)
def get_data(reader: ZarrReader, path: str):
    return reader.all_data(path)
