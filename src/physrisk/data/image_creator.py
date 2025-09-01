from importlib import import_module
import io
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import PIL.Image as Image
import zarr.storage

from physrisk.api.v1.hazard_image import TileNotAvailableError
from physrisk.kernel.hazards import Hazard, HazardKind
from physrisk.data import colormap_provider
from physrisk.data.hazard_data_provider import HazardDataProvider, SourcePaths
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
                    HazardDataProvider._weights(
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
    ) -> Tuple[Sequence[Any], Sequence[Any], str, str]:
        resource = self.inventory.resources[resource_id]
        # in principle, depends on the scenario and year, although we assume here that
        # all years have the same index values available.
        scenario_paths = self.source_paths.scenario_paths_for_id(
            resource_id, [scenario], True, map_zoom=1
        )[scenario]
        path = scenario_paths.path(scenario_paths.years[0])
        z = self.reader.all_data(path)
        all_index_values, index_units = self.reader.get_index_values(z)
        index_dim_name = z.attrs.get("dimensions", ["index"])[0]
        assert isinstance(index_dim_name, str)
        if resource.map and resource.map.index_values:
            available_index_values = resource.map.index_values
        else:
            available_index_values = all_index_values
        physrisk_hazards = import_module("physrisk.kernel.hazards")
        hazard_class = getattr(physrisk_hazards, resource.hazard_type)

        # the attribute requires cleaning before use: do not use for now
        # index_display_name = z.attrs.get(index_dim_name + "_name", index_dim_name.replace("_", " "))
        index_display_name = self._default_index_display_name(
            hazard_class, resource.indicator_id
        )

        if index_units == "default":
            index_units = self._default_index_units(hazard_class, resource.indicator_id)
        return all_index_values, available_index_values, index_display_name, index_units

    def _default_index_display_name(
        self, hazard_class: Type[Hazard], indicator_id: str
    ):
        if hazard_class.kind == HazardKind.ACUTE:
            return "return period"
        else:
            return "threshold"

    def _default_index_units(self, hazard_class: Type[Hazard], indicator_id: str):
        if hazard_class.kind == HazardKind.ACUTE:
            return "years"
        if indicator_id in [
            "days_wbgt_above",
            "mean_degree_days/above/index",
            "weeks_water_temp_above",
        ]:
            return "°C"
        else:
            return ""

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
    ) -> Image.Image:
        """Get image for path specified as array of bytes."""

        tile_size = 512
        index = None

        def get_array(data: zarr.Array, index: Optional[int]):
            if len(data.shape) == 3:
                index_values, _ = self.reader.get_index_values(data)
                if index_value is not None:
                    if isinstance(index_values[0], float):
                        _index_value = float(index_value)
                    elif isinstance(index_values[0], int):
                        _index_value = int(index_value)
                    elif isinstance(index_values[0], str):
                        _index_value = str(index_value)  # type:ignore
                index = (
                    len(index_values) - 1
                    if index_value is None
                    else index_values.index(_index_value)
                )
                if tile is None:
                    # return whole array
                    return data[index, :, :]  # .squeeze(axis=0)
                else:
                    # (from zarr 2.16.0 we can also use block indexing)
                    return data[
                        index,
                        tile_size * tile.y : tile_size * (tile.y + 1),
                        tile_size * tile.x : tile_size * (tile.x + 1),
                    ]

        data = sum(
            weight
            * get_array(
                get_data(
                    self.reader,
                    path,
                ),
                index,
            )
            for path, weight in path_weights.items()
        )

        if any(dim > 4000 for dim in data.shape):
            raise Exception("dimension too large (over 1500).")
        map_defn = colormap_provider.colormap(colormap)

        def get_colors(index: int):
            return map_defn[str(index)]

        rgba = self._to_rgba(data, get_colors, min_value=min_value, max_value=max_value)
        image = Image.fromarray(rgba, mode="RGBA")
        return image

    def _to_rgba(  # noqa: C901
        self,
        data: np.ndarray,
        get_colors: Callable[[int], List[int]],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        nodata_lower: Optional[float] = None,
        nodata_upper: Optional[float] = None,
        nodata_bin_transparent: bool = False,
        min_bin_transparent: bool = False,
    ) -> np.ndarray:
        """Convert the data to an RGBA image using values provided by get_colors.
        We are particular about min and max values, ensuring that these get their own indices
        from the colormap. Thee rules are:
        0: value is nodata
        1: value <= min_value
        2: min_value < value < (max_value - min_value) / 253
        254: (max_value - min_value) / 253 <= value < max_value
        255 is >= max_value

        Args:
            data (np.ndarray): Two dimensional array.
            get_colors (Callable[[int], Tuple[int, int, int]]): When passed an integer index in range 0:256, returns RGB components as integers in range 0:256.
            min_value (Optional[float]): Minimum value. Defaults to None.
            max_value (Optional[float]): Maximum value. Defaults to None.
            nodata_lower (Optional[float], optional): If supplied, values smaller than or equal to nodata_lower threshold are considered nodata. Defaults to None.
            nodata_upper (Optional[float], optional): If supplied, values larger than or equal to nodata_upper threshold are considered nodata. Defaults to None.
            nodata_bin_transparent (bool, optional): If True make no_data bin transparent. Defaults to False.
            min_bin_transparent (bool, optional): If True make min_bin transparent. Defaults to False.

        Returns:
            np.ndarray: RGBA array.
        """  # noqa

        red = np.zeros(256, dtype=np.uint32)
        green = np.zeros(256, dtype=np.uint32)
        blue = np.zeros(256, dtype=np.uint32)
        a = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            (red[i], green[i], blue[i], a[i]) = get_colors(i)
        if nodata_bin_transparent:
            a[0] = 0
        if min_bin_transparent:
            a[1] = 0
        mask_nodata = np.isnan(data)
        if nodata_lower:
            mask_nodata = data <= nodata_lower
        if nodata_upper:
            mask_nodata = (
                (mask_nodata | (data >= nodata_upper))
                if mask_nodata is not None
                else (data >= nodata_upper)
            )

        if min_value is None:
            min_value = np.nanmin(data)
        if max_value is None:
            max_value = np.nanmax(data)

        mask_ge_max = data >= max_value
        mask_le_min = data <= min_value

        np.add(data, -min_value, out=data)
        np.multiply(data, 253.0 / (max_value - min_value), out=data)
        np.add(data, 2.0, out=data)  # np.clip seems a bit slow so we do not use

        result = data.astype(np.uint8, casting="unsafe", copy=False)
        del data

        if mask_nodata is not None:
            result[mask_nodata] = 0
            del mask_nodata

        result[mask_ge_max] = 255
        result[mask_le_min] = 1
        del mask_ge_max, mask_le_min

        final = (
            red[result]
            + (green[result] << 8)
            + (blue[result] << 16)
            + (a[result] << 24)
        )
        return final

    @staticmethod
    def test_store(path: str):
        store = zarr.storage.MemoryStore(root="hazard.zarr")
        root = zarr.open(store=store, mode="w")
        x, y = np.meshgrid(
            (np.arange(1000) - 500.0) / 500.0, (np.arange(1000) - 500.0) / 500.0
        )
        im = np.exp(-(x**2 + y**2))
        z = root.create_dataset(  # type: ignore
            path,
            shape=(1, im.shape[0], im.shape[1]),
            chunks=(1, im.shape[0], im.shape[1]),
            dtype="f4",
        )
        z[0, :, :] = im
        return store


@lru_cache(maxsize=32)
def get_data(reader: ZarrReader, path: str):
    return reader.all_data(path)
