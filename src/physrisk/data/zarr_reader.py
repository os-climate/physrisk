import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import (
    Any,
    Callable,
    Coroutine,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import s3fs
import shapely.ops
import zarr
import zarr.api.asynchronous
from affine import Affine
from pyproj import Transformer
from shapely import MultiPoint, Point, affinity, Polygon
from zarr import AsyncArray, AsyncGroup
from zarr.storage import FsspecStore, StoreLike

from physrisk.utils.concurrency import available_cpu_count
from physrisk.utils.event_loop import BackgroundEventLoop

logger = logging.getLogger(__name__)

T = TypeVar("T")

# All zarr reads in the process run on this loop. zarr decompresses chunks with
# asyncio.to_thread, so the loop's default executor is the decode pool, which also
# runs the CPU-bound parts of reads. A store bound to a loop (e.g. async S3) is
# only ever used from here, and the decode pool is shared by all callers.
_zarr_loop = BackgroundEventLoop("physrisk-zarr")


def configure_zarr_concurrency(
    decode_threads: Optional[int] = None,
    async_concurrency: Optional[int] = None,
) -> None:
    """Set the concurrency of hazard data reads.

    zarr fetches the chunks of each array read concurrently, up to
    ``async_concurrency``, and decompresses them on a pool of ``decode_threads``
    shared by all reads. Decoded chunks are large (e.g. 36 MB for a (9, 1000, 1000)
    float32 chunk), so the decode pool bounds the memory of decoded chunks. Chunks
    downloaded but waiting for a decode thread are limited only per read, by
    ``async_concurrency``, so they grow with the number of arrays read at once when
    downloads outpace decoding (fast network, few CPUs). Both therefore default to
    sizes based on the CPUs available to the process; ``os.cpu_count()``, which zarr
    would otherwise use, oversubscribes a CPU-limited container.

    Args:
        decode_threads: Threads decompressing chunks and running the CPU-bound parts
            of reads. Defaults to the available CPUs.
        async_concurrency: Maximum chunks in flight per array read; zarr's global
            async.concurrency setting. Defaults to twice the available CPUs, at most 8.
            Reduce it if downloaded chunks accumulate waiting for decode.
    """
    if async_concurrency is None:
        async_concurrency = min(8, 2 * available_cpu_count())
    zarr.config.set({"async.concurrency": async_concurrency})
    _zarr_loop.set_executor_workers(decode_threads)


@dataclass(frozen=True)
class ArrayInfo:
    """Shape and attributes of a hazard array, without its data."""

    shape: Tuple[int, ...]
    attrs: Mapping[str, Any]


def get_env(key: str, default: Optional[str] = None) -> str:
    value = os.environ.get(key)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"environment variable {key} not present")
    else:
        return value


class ZarrReader:
    """Reads hazard event data from Zarr files, including OSC-format-specific attributes."""

    # environment variable names:
    __access_key = "OSC_S3_ACCESS_KEY"
    __secret_key = "OSC_S3_SECRET_KEY"
    __S3_bucket = "OSC_S3_BUCKET"  # e.g. os-climate-physical-risk
    __zarr_path = "OSC_S3_HAZARD_PATH"  # hazard-indicators/hazard.zarr")

    def __init__(
        self,
        store: Optional[StoreLike] = None,
        path_provider: Optional[Callable[..., str]] = None,
        get_env: Callable[[str, Optional[str]], str] = get_env,
        cache_arrays: bool = True,
    ):
        """Create a ZarrReader.

        Each read is available as a method (e.g. get_curves) and an asynchronous
        method (e.g. get_curves_async); either way it runs on a loop shared by all
        readers, with concurrency set by configure_zarr_concurrency.

        Args:
            store: if not supplied, create S3 store.
            path_provider: function that provides path to the data set based on a set ID.
            get_env: allows override obtaining of environment variables.
            cache_arrays: keep each opened array, so that its metadata is read from
                the store once rather than on every read. The arrays hold metadata,
                not data. Changes to an array's metadata in the store are then not
                seen by this reader.
        """
        if store is None:
            # if no store is provided, attempt to connect to an S3 bucket
            if get_env is None:
                raise TypeError(
                    "if no store specified, get_env is required to provide credentials"
                )

            store = ZarrReader.create_s3_zarr_store(get_env)

        self._store = store
        self._path_provider = path_provider
        self._root: Optional[AsyncGroup] = None
        # only accessed on the zarr loop, so needs no lock
        self._arrays: Optional[dict[str, AsyncArray]] = {} if cache_arrays else None

    @staticmethod
    def _sync(coro: Coroutine[Any, Any, T]) -> T:
        return _zarr_loop.run_sync(coro)

    @staticmethod
    async def _async(coro: Coroutine[Any, Any, T]) -> T:
        return await _zarr_loop.run(coro)

    async def _group(self) -> AsyncGroup:
        if self._root is None:
            # hazard data is zarr v2 without consolidated metadata
            self._root = await zarr.api.asynchronous.open_group(
                store=self._store, mode="r", use_consolidated=False
            )
        return self._root

    async def _get_node(self, path: str) -> Union[AsyncArray, AsyncGroup]:
        group = await self._group()
        return await group.getitem(path) if path else group

    async def _get_array(self, set_id: str) -> AsyncArray:
        """Return the array identified by a set ID."""
        path = (
            self._path_provider(set_id) if self._path_provider is not None else set_id
        )
        if self._arrays is not None and path in self._arrays:
            return self._arrays[path]
        node = await self._get_node(path)
        if not isinstance(node, AsyncArray):
            raise KeyError(path)
        if self._arrays is not None:
            self._arrays[path] = node
        return node

    def array_info(self, set_id: str) -> ArrayInfo:
        """Return the shape and attributes of an array.

        Args:
            set_id: Path of the array, or set ID converted to one by path_provider.

        Returns:
            Shape and attributes.
        """
        return self._sync(self._array_info(set_id))

    async def _array_info(self, set_id: str) -> ArrayInfo:
        z = await self._get_array(set_id)
        return ArrayInfo(shape=z.shape, attrs=dict(z.attrs))

    def read(self, set_id: str, selection: Any) -> np.ndarray:
        """Read a basic selection, e.g. a tuple of integers and slices, from an array.

        Args:
            set_id: Path of the array, or set ID converted to one by path_provider.
            selection: Selection, as for numpy indexing.

        Returns:
            Selected data.
        """
        return self._sync(self._read(set_id, selection))

    async def read_async(self, set_id: str, selection: Any) -> np.ndarray:
        """Asynchronous version of read."""
        return await self._async(self._read(set_id, selection))

    async def _read(self, set_id: str, selection: Any) -> np.ndarray:
        z = await self._get_array(set_id)
        return np.asarray(await z.getitem(selection))

    @staticmethod
    def _get_spatial_metadata(z: Union[AsyncArray, ArrayInfo]) -> tuple[Affine, str]:
        """Return the affine transform and CRS of an array."""
        t = z.attrs["transform_mat3x3"]
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        crs = z.attrs.get("crs", "epsg:4326")
        return transform, crs

    def ls(self, path: str) -> List[str]:
        """List the arrays and groups in the group at path.

        Args:
            path: Path of a group, relative to the root.

        Returns:
            Paths of the group's members, relative to the root.
        """
        return self._sync(self._ls(path))

    async def _ls(self, path: str) -> List[str]:
        path = path.strip("/")
        group = await self._get_node(path)
        if not isinstance(group, AsyncGroup):
            raise ValueError(f"{path} is not a group")
        return [str(PurePosixPath(path, name)) async for name in group.keys()]

    @classmethod
    def create_s3_zarr_store(
        cls,
        get_env: Callable[[str, Optional[str]], str] = get_env,
        max_pool_connections: int = 64,
    ) -> FsspecStore:
        """Create a read-only store for the hazard data in S3.

        Args:
            get_env: Provides the environment variables naming bucket, path and credentials.
            max_pool_connections: S3 connection pool size. botocore's default of 10
                limits chunk downloads well below what zarr can fetch concurrently.

        Returns:
            The store.
        """
        access_key = get_env(cls.__access_key, "")
        secret_key = get_env(cls.__secret_key, "")
        s3_bucket = get_env(cls.__S3_bucket, "os-climate-physical-risk")
        zarr_path = get_env(cls.__zarr_path, "hazard-indicators/hazard.zarr")

        config_kwargs = {"max_pool_connections": max_pool_connections}
        s3 = (
            s3fs.S3FileSystem(anon=True, asynchronous=True, config_kwargs=config_kwargs)
            if access_key == ""
            else s3fs.S3FileSystem(
                anon=False,
                key=access_key,
                secret=secret_key,
                asynchronous=True,
                config_kwargs=config_kwargs,
            )
        )
        return FsspecStore(
            fs=s3, path=str(PurePosixPath(s3_bucket, zarr_path)), read_only=True
        )

    def get_curves(
        self,
        set_id: str,
        longitudes: Union[np.ndarray, Sequence[float]],
        latitudes: Union[np.ndarray, Sequence[float]],
        interpolation="floor",
    ):
        """Get intensity curve for each latitude and longitude coordinate pair.

        Args:
            set_id: string or tuple representing data set, converted into path by path_provider.
            longitudes: list of longitudes.
            latitudes: list of latitudes.
            interpolation: interpolation method, "floor", "linear", "max" or "min".

        Returns:
            curves: numpy array of intensity (no. coordinate pairs, no. return periods).
            return_periods: return periods in years.
        """
        return self._sync(
            self._get_curves(set_id, longitudes, latitudes, interpolation)
        )

    async def get_curves_async(
        self,
        set_id: str,
        longitudes: Union[np.ndarray, Sequence[float]],
        latitudes: Union[np.ndarray, Sequence[float]],
        interpolation="floor",
    ):
        """Asynchronous version of get_curves."""
        return await self._async(
            self._get_curves(set_id, longitudes, latitudes, interpolation)
        )

    async def _get_curves(self, set_id, longitudes, latitudes, interpolation):
        # assume that calls to this are large, not chatty
        if len(longitudes) != len(latitudes):
            raise ValueError("length of longitudes and latitudes not equal")
        if interpolation not in ["floor", "linear", "max", "min"]:
            raise ValueError(
                "interpolation must have value 'floor', 'linear', 'max' or 'min"
            )
        z = await self._get_array(set_id)
        # in the case of acute risks, index_values will contain the return periods
        index_values, _ = self.get_index_values(z)
        image_coords, in_bounds, selection = await asyncio.to_thread(
            self._curves_selection,
            z,
            len(index_values),
            longitudes,
            latitudes,
            interpolation,
        )
        data = await z.get_coordinate_selection(selection)
        res = await asyncio.to_thread(
            self._curves_from_data,
            data,
            image_coords,
            in_bounds,
            len(index_values),
            interpolation,
        )
        # OSC-specific attributes contain units
        units: str = z.attrs.get("units", "default")
        return res, in_bounds, np.array(index_values), units

    @staticmethod
    def _curves_selection(
        z: Union[AsyncArray, ArrayInfo],
        n_index: int,
        longitudes,
        latitudes,
        interpolation: str,
    ):
        """Image coordinates of the in-bounds points, which points are in bounds, and
        the coordinate selection to read for them."""
        transform, crs = ZarrReader._get_spatial_metadata(z)
        image_coords = ZarrReader._get_coordinates(
            longitudes,
            latitudes,
            crs,
            transform,
            apply_pixel_center_offset=interpolation != "floor",
            shape=z.shape,
        )
        in_bounds = ZarrReader._in_bounds(image_coords, z.shape, interpolation)
        image_coords = image_coords[:, in_bounds]
        if interpolation == "floor":
            image_coords = np.floor(image_coords).astype(int)
            selection = ZarrReader._floor_selection(z.shape, image_coords, n_index)
        else:
            selection = ZarrReader._frac_coordinates_selection(
                z.shape, image_coords, n_index
            )
        return image_coords, in_bounds, selection

    @staticmethod
    def _floor_selection(shape, image_coords: np.ndarray, n_index: int):
        iz = np.tile(np.arange(shape[0]), image_coords.shape[1])
        iy = np.repeat(image_coords[1, :], n_index)
        ix = np.repeat(image_coords[0, :], n_index)
        return iz, iy, ix

    @staticmethod
    def _curves_from_data(
        data: np.ndarray,
        image_coords: np.ndarray,
        in_bounds: np.ndarray,
        n_index: int,
        interpolation: str,
    ) -> np.ndarray:
        res = np.zeros((len(in_bounds), n_index))
        res[~in_bounds] = np.nan
        if interpolation == "floor":
            data = ZarrReader._handle_legacy_nans(data)
            res[in_bounds] = data.reshape([np.count_nonzero(in_bounds), n_index])
        else:
            res[in_bounds, :] = ZarrReader._interpolate_frac_coordinates(
                data, image_coords, interpolation
            )
        return res

    def in_bounds(
        self,
        set_id: str,
        longitudes: Union[np.ndarray, Sequence[float]],
        latitudes: Union[np.ndarray, Sequence[float]],
        interpolation: str = "floor",
    ):
        """Return whether coordinates lie within a hazard array.

        Args:
            set_id: Path of the hazard array.
            longitudes: Longitude of each coordinate.
            latitudes: Latitude of each coordinate.
            interpolation: Sampling mode used for the subsequent read: "floor",
                "linear", "max" or "min". Defaults to "floor", matching get_curves.

        Returns:
            A boolean array aligned with the input coordinates.
        """
        return self._sync(
            self._in_bounds_for(set_id, longitudes, latitudes, interpolation)
        )

    async def in_bounds_async(
        self,
        set_id: str,
        longitudes: Union[np.ndarray, Sequence[float]],
        latitudes: Union[np.ndarray, Sequence[float]],
        interpolation: str = "floor",
    ):
        """Asynchronous version of in_bounds."""
        return await self._async(
            self._in_bounds_for(set_id, longitudes, latitudes, interpolation)
        )

    async def _in_bounds_for(self, set_id, longitudes, latitudes, interpolation):
        if len(longitudes) != len(latitudes):
            raise ValueError("length of longitudes and latitudes not equal")
        z = await self._get_array(set_id)
        return await asyncio.to_thread(
            self._coordinates_in_bounds, z, longitudes, latitudes, interpolation
        )

    @staticmethod
    def _coordinates_in_bounds(z, longitudes, latitudes, interpolation):
        transform, crs = ZarrReader._get_spatial_metadata(z)
        image_coords = ZarrReader._get_coordinates(
            longitudes,
            latitudes,
            crs,
            transform,
            apply_pixel_center_offset=interpolation != "floor",
            shape=z.shape,
        )
        return ZarrReader._in_bounds(image_coords, z.shape, interpolation)

    @staticmethod
    def _in_bounds(
        image_coords: np.ndarray,
        shape: tuple[int, ...],
        interpolation: str,
    ) -> np.ndarray:
        """Return whether raster coordinates lie within an array shape."""
        lower_bound = 0.0 if interpolation == "floor" else -0.5
        return (
            (image_coords[0, :] >= lower_bound)
            & (image_coords[0, :] < shape[2])
            & (image_coords[1, :] >= lower_bound)
            & (image_coords[1, :] < shape[1])
        )

    @staticmethod
    def get_index_dimension_name(z: Union[ArrayInfo, AsyncArray, zarr.Array]) -> str:
        # if dimensions attribute is present, assume that the first index
        # is the non-spatial one.
        return z.attrs.get("dimensions", ["index"])[0]

    def get_index_values(
        self, z: Union[ArrayInfo, AsyncArray, zarr.Array]
    ) -> Tuple[List[Any], str]:
        index_dim_name = self.get_index_dimension_name(z)
        index_values: List[Any] = z.attrs.get(index_dim_name + "_values", [0])
        index_units = z.attrs.get(index_dim_name + "_units", "default")
        if index_values is None:
            index_values = [0]  # type: ignore
        return index_values, index_units

    def get_max_curves(
        self, set_id: str, shapes: Sequence[Polygon], interpolation: str = "floor"
    ):
        """Get maximal intensity curve for a given geometry.

        Args:
            set_id: string or tuple representing data set, converted into path by path_provider.
            shapes: list of shapely.Polygon.
            interpolation: interpolation method, "floor", "linear", "max" or "min".

        Returns:
            curves_max: numpy array of maximum intensity on the grid for a given geometry
            (no. coordinate pairs, no. return periods).
            return_periods: return periods in years.
            units: units.
        """
        return self._sync(self._get_max_curves(set_id, shapes, interpolation))

    async def get_max_curves_async(
        self, set_id: str, shapes: Sequence[Polygon], interpolation: str = "floor"
    ):
        """Asynchronous version of get_max_curves."""
        return await self._async(self._get_max_curves(set_id, shapes, interpolation))

    async def _get_max_curves(self, set_id, shapes, interpolation):
        if interpolation not in ["floor", "linear", "max", "min"]:
            raise ValueError(
                "interpolation must have value 'floor', 'linear', 'max' or 'min"
            )
        z = await self._get_array(set_id)
        # in the case of acute risks, index_values will contain the return periods
        index_values, _ = self.get_index_values(z)
        image_coords, points_per_shape, selection = await asyncio.to_thread(
            self._max_curves_selection, z, len(index_values), shapes, interpolation
        )
        data = await z.get_coordinate_selection(selection)
        curves_max = await asyncio.to_thread(
            self._max_curves_from_data,
            data,
            image_coords,
            points_per_shape,
            len(index_values),
            interpolation,
        )
        units: str = z.attrs.get("units", "default")
        return curves_max, np.array(index_values), units

    @staticmethod
    def _max_curves_selection(
        z: Union[AsyncArray, ArrayInfo],
        n_index: int,
        shapes: Sequence[Polygon],
        interpolation: str,
    ):
        """Image coordinates of the points sampled in each shape, the number of points
        per shape, and the coordinate selection to read for them."""
        transform, crs = ZarrReader._get_spatial_metadata(z)

        if crs.lower() != "epsg:4326":
            transproj = Transformer.from_crs("epsg:4326", crs, always_xy=True).transform
            shapes = [shapely.ops.transform(transproj, shape) for shape in shapes]

        matrix = np.array(~transform).reshape(3, 3).transpose()[:, :-1].reshape(6)

        transformed_shapes = [
            affinity.affine_transform(shape, matrix) for shape in shapes
        ]

        pixel_offset = 0.5 if interpolation != "floor" else 0.0
        multipoints = [
            MultiPoint(
                [
                    (x - pixel_offset, y - pixel_offset)
                    for x in range(
                        int(np.floor(shape.bounds[0])),
                        int(np.ceil(shape.bounds[2])) + 1,
                    )
                    for y in range(
                        int(np.floor(shape.bounds[1])),
                        int(np.ceil(shape.bounds[3])) + 1,
                    )
                ]
            ).intersection(shape)
            for shape in transformed_shapes
        ]
        multipoints = [
            (
                Point(
                    0.5 * (shape.bounds[0] + shape.bounds[2]),
                    0.5 * (shape.bounds[1] + shape.bounds[3]),
                )
                if multipoint.is_empty
                else multipoint
            )
            for shape, multipoint in zip(transformed_shapes, multipoints)
        ]
        multipoints = [
            MultiPoint([(point.x, point.y)]) if isinstance(point, Point) else point
            for point in multipoints
        ]

        if interpolation == "floor":
            image_coords = np.floor(
                np.array(
                    [
                        [point.x, point.y]
                        for multipoint in multipoints
                        for point in multipoint.geoms
                    ]
                ).transpose()
            ).astype(int)
            image_coords[0, :] %= z.shape[2]
            selection = ZarrReader._floor_selection(z.shape, image_coords, n_index)
        else:
            multipoints = [
                multipoint.union(
                    transformed_shape
                    if isinstance(transformed_shape, Point)
                    else MultiPoint(transformed_shape.exterior.coords)
                )
                for transformed_shape, multipoint in zip(
                    transformed_shapes, multipoints
                )
            ]
            image_coords = np.array(
                [
                    [point.x, point.y]
                    for multipoint in multipoints
                    for point in multipoint.geoms
                ]
            ).transpose()
            selection = ZarrReader._frac_coordinates_selection(
                z.shape, image_coords, n_index
            )

        points_per_shape = [len(multipoint.geoms) for multipoint in multipoints]
        return image_coords, points_per_shape, selection

    @staticmethod
    def _max_curves_from_data(
        data: np.ndarray,
        image_coords: np.ndarray,
        points_per_shape: List[int],
        n_index: int,
        interpolation: str,
    ) -> np.ndarray:
        if interpolation == "floor":
            curves = data.reshape(image_coords.shape[1], n_index)
        else:
            curves = ZarrReader._interpolate_frac_coordinates(
                data, image_coords, interpolation
            )
        points_per_shape_cumulated = np.cumsum(points_per_shape)
        return np.array(
            [
                np.nanmax(curves[index - number_of_points : index, :], axis=0)
                for number_of_points, index in zip(
                    points_per_shape, points_per_shape_cumulated
                )
            ]
        )

    def get_max_curves_on_grid(
        self,
        set_id,
        longitudes,
        latitudes,
        interpolation="floor",
        delta_km=1.0,
        n_grid=5,
    ):
        """Get maximal intensity curve for a grid around a given latitude and longitude coordinate pair.
        It is almost equivalent to::

            self.get_max_curves
            (
                set_id,
                [
                    Polygon(
                        (
                            (x - 0.5 * delta_deg, y - 0.5 * delta_deg),
                            (x - 0.5 * delta_deg, y + 0.5 * delta_deg),
                            (x + 0.5 * delta_deg, y + 0.5 * delta_deg),
                            (x + 0.5 * delta_deg, y - 0.5 * delta_deg)
                        )
                    )
                    for x, y in zip(longitudes, latitudes)
                ]
                interpolation
            )


        Args:
            set_id: string or tuple representing data set, converted into path by path_provider.
            longitudes: list of longitudes.
            latitudes: list of latitudes.
            interpolation: interpolation method, "floor", "linear", "max" or "min".
            delta_km: linear distance in kilometres of the side of the square grid surrounding a given position.
            n_grid: number of grid points along the latitude and longitude dimensions used for
                calculating the maximal value.

        Returns:
            curves_max: numpy array of maximum intensity on the grid for a given coordinate pair
            (no. coordinate pairs, no. return periods).
            return_periods: return periods in years.
        """

        kilometres_per_degree = 110.574
        delta_deg = delta_km / kilometres_per_degree

        n_data = len(latitudes)

        grid = np.linspace(-0.5, 0.5, n_grid)
        lats_grid_baseline = np.broadcast_to(
            np.array(latitudes).reshape(n_data, 1, 1), (len(latitudes), n_grid, n_grid)
        )
        lons_grid_baseline = np.broadcast_to(
            np.array(longitudes).reshape(n_data, 1, 1),
            (len(longitudes), n_grid, n_grid),
        )
        lats_grid_offsets = delta_deg * grid.reshape((1, n_grid, 1))
        lons_grid_offsets = (
            delta_deg
            * grid.reshape((1, 1, n_grid))
            / (np.cos((np.pi / 180) * np.array(latitudes)).reshape(n_data, 1, 1))
        )
        lats_grid = lats_grid_baseline + lats_grid_offsets
        lons_grid = lons_grid_baseline + lons_grid_offsets
        curves, _, return_periods, _ = self.get_curves(
            set_id,
            lons_grid.reshape(-1),
            lats_grid.reshape(-1),
            interpolation=interpolation,
        )
        curves_max = np.nanmax(
            curves.reshape((n_data, n_grid * n_grid, len(return_periods))), axis=1
        )
        return curves_max, return_periods

    @staticmethod
    def _linear_interp_frac_coordinates(
        z: zarr.Array, image_coords, return_periods, interpolation="linear"
    ):
        """Return linear interpolated data from fractional row and column coordinates."""
        selection = ZarrReader._frac_coordinates_selection(
            z.shape, image_coords, len(return_periods)
        )
        data = z.get_coordinate_selection(selection)
        return ZarrReader._interpolate_frac_coordinates(
            data, image_coords, interpolation
        )

    @staticmethod
    def _frac_coordinates_selection(shape, image_coords: np.ndarray, n_index: int):
        """Coordinate selection of the four pixels surrounding each fractional
        coordinate, for each index: arrays of shape (points, 4, index)."""
        icx = np.floor(image_coords[0, :]).astype(int)[..., None]
        # note periodic boundary condition
        ix = np.concatenate(
            [
                icx % shape[2],
                icx % shape[2],
                (icx + 1) % shape[2],
                (icx + 1) % shape[2],
            ],
            axis=1,
        )[..., None].repeat(n_index, axis=2)  # points, 4, return_periods

        icy = np.floor(image_coords[1, :]).astype(int)[..., None]
        iy = np.concatenate([icy, icy + 1, icy, icy + 1], axis=1)[..., None].repeat(
            n_index, axis=2
        )

        iz = (
            np.arange(n_index, dtype=int)[None, ...]
            .repeat(4, axis=0)[None, ...]
            .repeat(image_coords.shape[1], axis=0)
        )
        return iz, iy, ix  # index, row, column

    @staticmethod
    def _interpolate_frac_coordinates(
        data: np.ndarray, image_coords: np.ndarray, interpolation: str
    ) -> np.ndarray:
        """Interpolate the data read with _frac_coordinates_selection."""
        icx = np.floor(image_coords[0, :]).astype(int)[..., None]
        icy = np.floor(image_coords[1, :]).astype(int)[..., None]

        data = ZarrReader._handle_legacy_nans(data)

        # retain ability to output arbitrary NaN value, although might be no longer needed as
        # physrisk deals separately, e.g. with removing NaNs before passing back via JSON
        nan_output_value = float("nan")

        if interpolation == "linear":
            xf = image_coords[0, :][..., None] - icx  # type: ignore
            yf = image_coords[1, :][..., None] - icy  # type: ignore
            w0 = (1 - yf) * (1 - xf)
            w1 = yf * (1 - xf)
            w2 = (1 - yf) * xf
            w3 = yf * xf
            w = np.transpose(np.array([w0, w1, w2, w3]), (1, 0, 2))
            mask = 1 - np.isnan(data)
            data = np.nan_to_num(data, 0)
            w_good = w * mask
            w_good_sum = np.transpose(
                np.sum(w_good, axis=1).reshape(
                    tuple([1]) + np.sum(w_good, axis=1).shape
                ),
                axes=(1, 0, 2),
            )
            w_used = np.divide(w_good, np.where(w_good_sum == 0.0, np.nan, w_good_sum))
            return np.nan_to_num(np.sum(w_used * data, axis=1), nan=nan_output_value)

        elif interpolation == "max":
            data = np.where(np.isnan(data), -np.inf, data)
            return np.nan_to_num(
                np.maximum.reduce(
                    [data[:, 0, :], data[:, 1, :], data[:, 2, :], data[:, 3, :]]
                ),
                nan=nan_output_value,
                neginf=nan_output_value,
            )

        elif interpolation == "min":
            data = np.where(np.isnan(data), np.inf, data)
            return np.nan_to_num(
                np.minimum.reduce(
                    [data[:, 0, :], data[:, 1, :], data[:, 2, :], data[:, 3, :]]
                ),
                nan=nan_output_value,
                posinf=nan_output_value,
            )

        else:
            raise ValueError("interpolation must have value 'linear', 'max' or 'min")

    @staticmethod
    def _get_coordinates(
        longitudes,
        latitudes,
        crs: str,
        transform: Affine,
        apply_pixel_center_offset: bool,
        shape: Optional[Tuple[int, int, int]] = None,
    ):
        """Transform geographic coordinates into fractional raster coordinates.

        Args:
            longitudes: Longitude of each coordinate.
            latitudes: Latitude of each coordinate.
            crs: Coordinate reference system of the raster.
            transform: Affine transform from raster to projected coordinates.
            apply_pixel_center_offset: Whether to align raster coordinates with
                pixel centres by subtracting half a pixel.
            shape: Raster shape used to detect the legacy 0–360 longitude layout.

        Returns:
            Fractional raster coordinates with one column per input coordinate.
        """
        if crs.lower() != "epsg:4326":
            transproj = Transformer.from_crs("epsg:4326", crs, always_xy=True)
            x, y = transproj.transform(longitudes, latitudes)
        else:
            x, y = longitudes, latitudes
        coords = np.vstack((x, y, np.ones(len(longitudes))))  # type: ignore
        # check a special legacy case:
        if shape is not None and crs.lower() == "epsg:4326":
            if ((transform * (shape[2], shape[1]))[0] > 180) and (
                (transform * (0, 0))[0] >= 0.0
            ):
                # detect if transform convention expects longitude in range [0, 360]
                coords = np.vstack((x % 360.0, y, np.ones(len(longitudes))))
                logger.warning(
                    "Detected transform convention with longitude in range [0, 360]. Adjusting input longitudes accordingly."
                )
        inv_trans = ~transform
        mat = np.array(inv_trans).reshape(3, 3)
        image_coords = mat @ coords
        if apply_pixel_center_offset:
            image_coords[:2, :] -= 0.5
        return image_coords

    @staticmethod
    def _get_equivalent_buffer_in_arc_degrees(latitude, buffer_in_metres):
        """
        area = radius * radius * cos(p) * dp * dq = buffer_in_metres * buffer_in_metres
        """
        semi_major_axis = 6378137
        semi_minor_axis = 6356752.314245
        degrees_to_radians = np.pi / 180.0
        latitude_in_radians = latitude * degrees_to_radians
        cosinus = np.abs(np.cos(latitude_in_radians))
        sinus = np.abs(np.sin(latitude_in_radians))
        buffer_in_arc_degrees = (
            buffer_in_metres
            * np.sqrt((cosinus / semi_major_axis) ** 2 + (sinus / semi_minor_axis) ** 2)
            / degrees_to_radians
        )
        if 0.0 < cosinus:
            buffer_in_arc_degrees /= np.sqrt(cosinus)
        return buffer_in_arc_degrees

    @staticmethod
    def _handle_legacy_nans(data: np.ndarray):
        # nodata in the zarr files are considered to be
        # 1) float("nan") (which Zarr supports) or 2) nan_value of -9999.0
        # 2 is legacy behaviour: for Zarr better to use float("nan")
        nan_input_value_legacy = -9999.0
        data = np.where(data == nan_input_value_legacy, np.nan, data)
        return data
