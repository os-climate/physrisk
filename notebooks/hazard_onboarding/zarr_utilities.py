import logging
import os
import pathlib
import sys
from pathlib import PurePosixPath
from typing import List, Tuple

import numpy as np
import s3fs
import zarr
from affine import Affine
from dotenv import load_dotenv


def add_logging_output_to_stdout(LOG):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOG.addHandler(handler)


def get_coordinates(longitudes, latitudes, transform):
    coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))
    inv_trans = ~transform
    mat = np.array(inv_trans).reshape(3, 3)
    frac_image_coords = mat @ coords
    image_coords = np.floor(frac_image_coords).astype(int)
    return image_coords


def get_geotiff_meta_data(path, s3):
    from tifffile.tifffile import TiffFile

    with s3.open(path) as f:
        with TiffFile(f) as tif:
            scale: Tuple[float, float, float] = tif.geotiff_metadata["ModelPixelScale"]
            tie_point: List[float] = tif.geotiff_metadata["ModelTiepoint"]
            shape: List[int] = tif.series[0].shape
            i, j, k, x, y, z = tie_point[0:6]
            sx, sy, sz = scale
            transform = Affine(sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy)
            return shape, transform


def set_credential_env_variables():
    dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.getcwd())
    dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)


def zarr_create(group_path, array_path, s3, shape, transform, return_periods):
    """
    Create zarr with given shape and affine transform
    """
    store = s3fs.S3Map(root=group_path, s3=s3, check=False)
    root = zarr.group(store=store)  # open group as root

    z = root.create_dataset(
        array_path,
        shape=(1 if return_periods is None else len(return_periods), shape[0], shape[1]),
        chunks=(1 if return_periods is None else len(return_periods), 1000, 1000),
        dtype="f4",
        overwrite=True,
    )  # array_path interpreted as path within group
    trans_members = [
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    ]
    mat3x3 = [x * 1.0 for x in trans_members] + [0.0, 0.0, 1.0]
    z.attrs["transform_mat3x3"] = mat3x3
    if return_periods is not None:
        z.attrs["index_values"] = return_periods
        z.attrs["index_name"] = "return period (years)"
    return z


def zarr_get_transform(zarr_array):
    t = zarr_array.attrs["transform_mat3x3"]  # type: ignore
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    return transform


def zarr_remove(group_path, array_path, s3):
    """
    Remove zarr array
    """
    store = s3fs.S3Map(root=group_path, s3=s3, check=False)
    root = zarr.open(store=store, mode="w")  # open group as root
    root.pop(array_path)


def zarr_read(group_path, array_path, s3, index):
    """
    Read data and transform from zarr
    """
    store = s3fs.S3Map(root=group_path, s3=s3, check=False)
    root = zarr.open(store, mode="r")
    z = root[array_path]
    t = z.attrs["transform_mat3x3"]  # type: ignore
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    return z[index, :, :], transform


def zarr_write(src_path, src_s3, dest_zarr, index):
    """
    Writes data from GeoTiff sepecified by src_path and src_s3 S3FileSystem to
    destination zarr array dest_zarr, putting data into index.
    """
    from tifffile.tifffile import TiffFile

    with src_s3.open(src_path) as f:
        with TiffFile(f) as tif:
            store = tif.series[0].aszarr()
            z_in = zarr.open(store, mode="r")
            dest_zarr[index, :, :] = z_in[:, :]
