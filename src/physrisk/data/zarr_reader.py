import os
from pathlib import PurePosixPath
from typing import Callable, MutableMapping, Optional

import numpy as np
import s3fs
import zarr
from affine import Affine


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
    __S3_bucket = "OSC_S3_BUCKET"  # e.g. redhat-osc-physical-landing-647521352890 on staging
    __zarr_path = "OSC_S3_HAZARD_PATH"  # e.g. hazard/hazard.zarr on staging

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
                raise TypeError("if no store specified, get_env is required to provide credentials")

            access_key = get_env(self.__access_key, None)
            secret_key = get_env(self.__secret_key, None)
            s3_bucket = get_env(self.__S3_bucket, "redhat-osc-physical-landing-647521352890")
            zarr_path = get_env(self.__zarr_path, "hazard/hazard.zarr")

            s3 = s3fs.S3FileSystem(anon=False, key=access_key, secret=secret_key)

            store = s3fs.S3Map(
                root=str(PurePosixPath(s3_bucket, zarr_path)),
                s3=s3,
                check=False,
            )

        self._root = zarr.open(store, mode="r")
        self._path_provider = path_provider
        pass

    def get_curves(self, set_id, longitudes, latitudes, interpolation="floor"):
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
        path = self._path_provider(set_id) if self._path_provider is not None else set_id
        z = self._root[path]  # e.g. inundation/wri/v2/<filename>

        # OSC-specific attributes contain tranform and return periods
        t = z.attrs["transform_mat3x3"]  # type: ignore
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

        # in the case of acute risks, index_values will contain the return periods
        index_values = z.attrs.get("index_values", [0])
        if index_values is None:
            index_values = [0]

        image_coords = self._get_coordinates(longitudes, latitudes, transform)

        if interpolation == "floor":
            image_coords = np.floor(image_coords).astype(int)
            iz = np.tile(np.arange(z.shape[0]), image_coords.shape[1])  # type: ignore
            iy = np.repeat(image_coords[1, :], len(index_values))
            ix = np.repeat(image_coords[0, :], len(index_values))

            data = z.get_coordinate_selection((iz, iy, ix))  # type: ignore
            return data.reshape([len(longitudes), len(index_values)]), np.array(index_values)

        elif interpolation in ["linear", "max", "min"]:
            res = ZarrReader._linear_interp_frac_coordinates(z, image_coords, index_values, interpolation=interpolation)
            return res, np.array(index_values)

        else:
            raise ValueError("interpolation must have value 'floor', 'linear', 'max' or 'min")

    def get_max_curves(self, set_id, longitudes, latitudes, interpolation="floor", delta_km=1.0, n_grid=5):
        """Get maximal intensity curve for a grid around a given latitude and longitude coordinate pair.

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
        KILOMETRES_PER_DEGREE = 110.574
        n_data = len(latitudes)
        delta_deg = delta_km / KILOMETRES_PER_DEGREE
        grid = np.linspace(-0.5, 0.5, n_grid)
        lats_grid_baseline = np.broadcast_to(latitudes.reshape((n_data, 1, 1)), (len(latitudes), n_grid, n_grid))
        lons_grid_baseline = np.broadcast_to(longitudes.reshape((n_data, 1, 1)), (len(longitudes), n_grid, n_grid))
        lats_grid_offsets = delta_deg * grid.reshape((1, n_grid, 1))
        lons_grid_offsets = (
            delta_deg * grid.reshape((1, 1, n_grid)) / (np.cos((np.pi / 180) * latitudes).reshape(n_data, 1, 1))
        )
        lats_grid = lats_grid_baseline + lats_grid_offsets
        lons_grid = lons_grid_baseline + lons_grid_offsets
        curves_, return_periods = self.get_curves(
            set_id, lons_grid.reshape(-1), lats_grid.reshape(-1), interpolation=interpolation
        )
        curves_max = np.max(curves_.reshape((n_data, n_grid * n_grid, len(return_periods))), axis=1)
        return curves_max, return_periods

    @staticmethod
    def _linear_interp_frac_coordinates(z, image_coords, return_periods, interpolation="linear"):
        """Return linear interpolated data from fractional row and column coordinates."""
        icx = np.floor(image_coords[0, :]).astype(int)[..., None]
        # note periodic boundary condition
        ix = np.concatenate([icx, icx, (icx + 1) % z.shape[2], (icx + 1) % z.shape[2]], axis=1)[..., None].repeat(
            len(return_periods), axis=2
        )  # points, 4, return_periods

        icy = np.floor(image_coords[1, :]).astype(int)[..., None]
        iy = np.concatenate([icy, icy + 1, icy, icy + 1], axis=1)[..., None].repeat(len(return_periods), axis=2)

        iz = (
            np.arange(len(return_periods), dtype=int)[None, ...]
            .repeat(4, axis=0)[None, ...]
            .repeat(image_coords.shape[1], axis=0)
        )

        data = z.get_coordinate_selection((iz, iy, ix))  # type: ignore # index, row, column

        NAN_VALUE = -9999.0

        if interpolation == "linear":
            xf = image_coords[0, :][..., None] - icx  # type: ignore
            yf = image_coords[1, :][..., None] - icy  # type: ignore
            w0 = (1 - yf) * (1 - xf)
            w1 = yf * (1 - xf)
            w2 = (1 - yf) * xf
            w3 = yf * xf
            w = np.transpose(np.array([w0, w1, w2, w3]), (1, 0, 2))
            mask = 1 - np.isnan(np.where(data == NAN_VALUE, np.nan, data))
            w_good = w * mask
            w_good_sum = np.transpose(
                np.sum(w_good, axis=1).reshape(tuple([1]) + np.sum(w_good, axis=1).shape), axes=(1, 0, 2)
            )
            w_used = np.divide(w_good, np.where(w_good_sum == 0.0, np.nan, w_good_sum))
            return np.nan_to_num(np.sum(w_used * data, axis=1), nan=NAN_VALUE)

        elif interpolation == "max":
            data = np.where(data == NAN_VALUE, -np.inf, data)
            return np.nan_to_num(
                np.maximum.reduce([data[:, 0, :], data[:, 1, :], data[:, 2, :], data[:, 3, :]]),
                nan=NAN_VALUE,
                neginf=NAN_VALUE,
            )

        elif interpolation == "min":
            data = np.where(data == NAN_VALUE, np.inf, data)
            return np.nan_to_num(
                np.minimum.reduce([data[:, 0, :], data[:, 1, :], data[:, 2, :], data[:, 3, :]]),
                nan=NAN_VALUE,
                posinf=NAN_VALUE,
            )

        else:
            raise ValueError("interpolation must have value 'linear', 'max' or 'min")

    @staticmethod
    def _get_coordinates(longitudes, latitudes, transform: Affine):
        coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))  # type: ignore
        inv_trans = ~transform
        mat = np.array(inv_trans).reshape(3, 3)
        frac_image_coords = mat @ coords
        return frac_image_coords
