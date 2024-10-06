import os
from pathlib import PurePosixPath
from typing import Callable, MutableMapping, Optional, Sequence, Union

import numpy as np
import s3fs
import shapely.ops
import zarr
from affine import Affine
from pyproj import Transformer
from shapely import MultiPoint, Point, affinity, Polygon


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
    __S3_bucket = "OSC_S3_BUCKET"  # e.g. physrisk-hazard-indicators
    __zarr_path = "OSC_S3_HAZARD_PATH"  # hazard/hazard.zarr

    def __init__(
        self,
        store: Optional[MutableMapping] = None,
        path_provider: Optional[Callable[..., str]] = None,
        get_env: Callable[[str, Optional[str]], str] = get_env,
    ):
        """Create a ZarrReader.

        Args:
            store: if not supplied, create S3Map store.
            path_provider: function that provides path to the data set based on a set ID.
            get_env: allows override obtaining of environment variables.
        """
        if store is None:
            # if no store is provided, attempt to connect to an S3 bucket
            if get_env is None:
                raise TypeError(
                    "if no store specified, get_env is required to provide credentials"
                )

            store = ZarrReader.create_s3_zarr_store(get_env)

        self._root = zarr.open(store, mode="r")
        self._path_provider = path_provider
        pass

    def all_data(self, set_id: str):
        path = (
            self._path_provider(set_id) if self._path_provider is not None else set_id
        )
        z = self._root[path]  # e.g. inundation/wri/v2/<filename>
        return z

    @classmethod
    def create_s3_zarr_store(
        cls, get_env: Callable[[str, Optional[str]], str] = get_env
    ):
        access_key = get_env(cls.__access_key, "")
        secret_key = get_env(cls.__secret_key, "")
        s3_bucket = get_env(cls.__S3_bucket, "physrisk-hazard-indicators")
        zarr_path = get_env(cls.__zarr_path, "hazard/hazard.zarr")

        s3 = s3fs.S3FileSystem(anon=False, key=access_key, secret=secret_key)

        store = s3fs.S3Map(
            root=str(PurePosixPath(s3_bucket, zarr_path)),
            s3=s3,
            check=False,
        )
        return store

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
        # assume that calls to this are large, not chatty
        if len(longitudes) != len(latitudes):
            raise ValueError("length of longitudes and latitudes not equal")
        path = (
            self._path_provider(set_id) if self._path_provider is not None else set_id
        )
        z = self._root[path]  # e.g. inundation/wri/v2/<filename>

        # OSC-specific attributes contain transform and return periods
        t = z.attrs["transform_mat3x3"]  # type: ignore
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        crs = z.attrs.get("crs", "epsg:4326")
        units: str = z.attrs.get("units", "default")

        # in the case of acute risks, index_values will contain the return periods
        index_values = self.get_index_values(z)
        image_coords = self._get_coordinates(
            longitudes,
            latitudes,
            crs,
            transform,
            pixel_is_area=interpolation != "floor",
        )

        if interpolation == "floor":
            image_coords = np.floor(image_coords).astype(int)
            image_coords[0, :] %= z.shape[2]
            iz = np.tile(np.arange(z.shape[0]), image_coords.shape[1])  # type: ignore
            iy = np.repeat(image_coords[1, :], len(index_values))
            ix = np.repeat(image_coords[0, :], len(index_values))

            data = z.get_coordinate_selection((iz, iy, ix))  # type: ignore
            return (
                data.reshape([len(longitudes), len(index_values)]),
                np.array(index_values),
                units,
            )

        elif interpolation in ["linear", "max", "min"]:
            res = ZarrReader._linear_interp_frac_coordinates(
                z, image_coords, index_values, interpolation=interpolation
            )
            return res, np.array(index_values), units

        else:
            raise ValueError(
                "interpolation must have value 'floor', 'linear', 'max' or 'min"
            )

    def get_index_values(self, z: zarr.Array):
        index_values = z.attrs.get("index_values", [0])
        if index_values is None:
            index_values = [0]
        return index_values

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
        path = (
            self._path_provider(set_id) if self._path_provider is not None else set_id
        )
        z = self._root[path]  # e.g. inundation/wri/v2/<filename>

        # in the case of acute risks, index_values will contain the return periods
        index_values = self.get_index_values(z)

        t = z.attrs["transform_mat3x3"]  # type: ignore
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        crs = z.attrs.get("crs", "epsg:4326")
        units: str = z.attrs.get("units", "default")

        if crs.lower() != "epsg:4236":
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

            iz = np.tile(np.arange(z.shape[0]), image_coords.shape[1])  # type: ignore
            iy = np.repeat(image_coords[1, :], len(index_values))
            ix = np.repeat(image_coords[0, :], len(index_values))

            curves = z.get_coordinate_selection((iz, iy, ix))
            curves = curves.reshape(image_coords.shape[1], len(index_values))

        elif interpolation in ["linear", "max", "min"]:
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

            curves = ZarrReader._linear_interp_frac_coordinates(
                z, image_coords, index_values, interpolation=interpolation
            )

        else:
            raise ValueError(
                "interpolation must have value 'floor', 'linear', 'max' or 'min"
            )

        numbers_of_points_per_shape = [
            len(multipoint.geoms) for multipoint in multipoints
        ]
        numbers_of_points_per_shape_cumulated = np.cumsum(numbers_of_points_per_shape)
        curves_max = np.array(
            [
                np.nanmax(curves[index - number_of_points_per_shape : index, :], axis=0)
                for number_of_points_per_shape, index in zip(
                    numbers_of_points_per_shape, numbers_of_points_per_shape_cumulated
                )
            ]
        )

        return curves_max, np.array(index_values), units

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
        It is almost equivalent to:
        ```
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
        ```

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
        curves, return_periods, _ = self.get_curves(
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
        z, image_coords, return_periods, interpolation="linear"
    ):
        """Return linear interpolated data from fractional row and column coordinates."""
        icx = np.floor(image_coords[0, :]).astype(int)[..., None]
        # note periodic boundary condition
        ix = np.concatenate(
            [
                icx % z.shape[2],
                icx % z.shape[2],
                (icx + 1) % z.shape[2],
                (icx + 1) % z.shape[2],
            ],
            axis=1,
        )[..., None].repeat(len(return_periods), axis=2)  # points, 4, return_periods

        icy = np.floor(image_coords[1, :]).astype(int)[..., None]
        iy = np.concatenate([icy, icy + 1, icy, icy + 1], axis=1)[..., None].repeat(
            len(return_periods), axis=2
        )

        iz = (
            np.arange(len(return_periods), dtype=int)[None, ...]
            .repeat(4, axis=0)[None, ...]
            .repeat(image_coords.shape[1], axis=0)
        )

        data = z.get_coordinate_selection((iz, iy, ix))  # type: ignore # index, row, column

        # nodata in the zarr files are considered to be
        # 1) float("nan") (which Zarr supports) or 2) nan_value of -9999.0
        # 2 is legacy behaviour: for Zarr better to use float("nan")
        nan_input_value_legacy = -9999.0
        data = np.where(data == nan_input_value_legacy, np.nan, data)
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
        longitudes, latitudes, crs: str, transform: Affine, pixel_is_area: bool
    ):
        if crs.lower() != "epsg:4236":
            transproj = Transformer.from_crs("epsg:4326", crs, always_xy=True)
            x, y = transproj.transform(longitudes, latitudes)
        else:
            x, y = longitudes, latitudes
        coords = np.vstack((x, y, np.ones(len(longitudes))))  # type: ignore
        inv_trans = ~transform
        mat = np.array(inv_trans).reshape(3, 3)
        frac_image_coords = mat @ coords
        if pixel_is_area:
            frac_image_coords[:2, :] -= 0.5
        return frac_image_coords

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
