import itertools
import logging
import os
from dataclasses import dataclass

import numpy as np
import rasterio
import s3fs
import zarr
from affine import Affine
from map_utilities import geotiff_profile, upload_geotiff, write_map_geotiff_data
from zarr_utilities import zarr_create, zarr_get_transform, zarr_read

LOG = logging.getLogger("Hazard onboarding")
LOG.setLevel(logging.INFO)

# type is "mean_degree_days"
# above_below is "above" or "below"
# ref_temp is for example "18c"
# scenario is for example "ssp585"
# {type}_{above_below}_{ref_temp}_{scenario}_{year}"
file_mapping_degree_days = {
    # degree days
    "mean_hdd32C_hist": ["mean_degree_days", "above", "32c", "historical", "1980"],
    "mean_hdd32C_ssp585_2030": ["mean_degree_days", "above", "32c", "ssp585", "2030"],
    "mean_hdd32C_ssp585_2040": ["mean_degree_days", "above", "32c", "ssp585", "2040"],
    "mean_hdd32C_ssp585_2050": ["mean_degree_days", "above", "32c", "ssp585", "2050"],
}

file_mapping_work_loss = {
    "mean_work_loss_hist": ["mean_work_loss", "high", "historical", "1980"],
    "mean_work_loss_ssp585_2030": ["mean_work_loss", "high", "ssp585", "2030"],
    "mean_work_loss_ssp585_2040": ["mean_work_loss", "high", "ssp585", "2040"],
    "mean_work_loss_ssp585_2050": ["mean_work_loss", "high", "ssp585", "2050"],
    "mean_work_loss_ssp245_2030": ["mean_work_loss", "high", "ssp245", "2030"],
    "mean_work_loss_ssp245_2040": ["mean_work_loss", "high", "ssp245", "2040"],
    "mean_work_loss_ssp245_2050": ["mean_work_loss", "high", "ssp245", "2050"],
}

longitudes = [
    69.4787,
    68.71,
    20.1047,
    19.8936,
    19.6359,
    0.5407,
    6.9366,
    6.935,
    13.7319,
    13.7319,
    14.4809,
    -68.3556,
    -68.3556,
    -68.9892,
    -70.9157,
]

latitudes = [
    34.556,
    35.9416,
    39.9116,
    41.6796,
    42.0137,
    35.7835,
    36.8789,
    36.88,
    -12.4706,
    -12.4706,
    -9.7523,
    -38.9368,
    -38.9368,
    -34.5792,
    -39.2145,
]


def filenames_degree_days(src_dir):
    for file in file_mapping_degree_days.keys():
        type, above_below, ref_temp, scenario, year = file_mapping_degree_days[file]

        src_path = os.path.join(src_dir, file + ".tiff")
        dest_filename = f"{type}_{above_below}_{ref_temp}_{scenario}_{year}"

        yield (file, src_path, dest_filename)


def filenames_work_loss(src_dir):
    for file in file_mapping_work_loss.keys():
        type, category, scenario, year = file_mapping_work_loss[file]

        src_path = os.path.join(src_dir, file + ".tiff")
        dest_filename = f"{type}_{category}_{scenario}_{year}"

        yield (file, src_path, dest_filename)


def create_map_geotiffs_chronic_heat(*, dest_bucket, dest_prefix, map_working_dir, account="osc", dest_s3):

    # array_names = [dest_filename for (file, src_path, dest_filename) in filenames_degree_days("")]
    array_names = [dest_filename for (file, src_path, dest_filename) in filenames_work_loss("")]

    profile = geotiff_profile()
    group_path = os.path.join(dest_bucket, dest_prefix, "hazard.zarr")

    store = s3fs.S3Map(root=group_path, s3=dest_s3, check=False)
    root = zarr.open(store, mode="r")
    # files = root["chronic_heat/osc/v1"].array_keys() # this seems very slow
    LOG.info("Calculating max value")
    max_value = float("-inf")
    for array_name in array_names:
        array_path = os.path.join("chronic_heat/osc/v1", array_name)
        data, transform = zarr_read(group_path=group_path, array_path=array_path, s3=dest_s3, index=0)
        max_value = max(max_value, data.max())

    # get max value for all arrays up-front

    LOG.info("Preparing and uploading files")
    for array_name in array_names:
        array_path = os.path.join("chronic_heat/osc/v1", array_name)
        data, transform = zarr_read(group_path=group_path, array_path=array_path, s3=dest_s3, index=0)

        path_out, colormap_path_out = write_map_geotiff_data(
            data,
            profile,
            data.shape[1],
            data.shape[0],
            transform,
            array_name + ".tif",
            map_working_dir,
            nodata_threshold=0,
            zero_transparent=True,
            max_intensity=max_value,  # float("inf"),
            palette="heating",
        )

        if account == "osc":
            access_token = os.environ["OSC_MAPBOX_UPLOAD_TOKEN"]
            filename = os.path.basename(path_out)
            id = filename[4:10]
            upload_geotiff(path_out, id, access_token)


def load_geotiff_rasterio(path):
    with rasterio.open(path) as dataset:
        data = dataset.read(1)
    return data, dataset.transform


def onboard_chronic_heat(
    src_dir, *, dest_bucket="redhat-osc-physical-landing-647521352890", dest_prefix="hazard_test", s3_dest
):
    LOG.info("Chronic heat")

    def onboard_inner(file, src_path, dest_filename, max_valid=None):
        data, transform = load_geotiff_rasterio(src_path)
        # useful check:
        transform = Affine(2 * 180 / data.shape[1], 0, -180.0, 0, -2 * 90 / data.shape[0], 90)
        group_path = os.path.join(dest_bucket, dest_prefix, "hazard.zarr")
        array_path = os.path.join("chronic_heat/osc/v1", dest_filename)
        LOG.info(f"Loaded file: {file} with shape {data.shape} and transform {transform}")
        LOG.info(f"Writing: {array_path} into {group_path}")

        z = zarr_create(
            group_path,
            array_path,
            s3_dest,
            data.shape,
            transform,
            None,
        )

        LOG.info(f"Created array; populating with data")
        if max_valid is not None:
            data[data > max_valid] = 0

        z[0, :, :] = data[:, :]

        with rasterio.open(src_path) as dataset:
            diff = check_rasterio_versus_zarr(dataset, z, np.array(longitudes), np.array(latitudes))

        LOG.info(f"Checking max difference is {np.max(np.abs(diff))}.")

    for (file, src_path, dest_filename) in filenames_work_loss(src_dir):
        onboard_inner(file, src_path, dest_filename, max_valid=1.0)

    # for (file, src_path, dest_filename) in filenames_degree_days(src_dir):
    #    onboard_inner(file, src_path, dest_filename)


def check_rasterio_versus_zarr(rasterio_dataset, zarr_array, longitudes, latitudes):
    """Check one dimensional zarr array versus rasterio."""
    points = [[lon, lat] for (lon, lat) in zip(longitudes, latitudes)]
    samples_rasterio = np.array(list(rasterio.sample.sample_gen(rasterio_dataset, points)))  # type:ignore
    samples_rasterio.resize([len(longitudes)])

    LOG.info(f"Rasterio " + str(samples_rasterio))

    t = zarr_array.attrs["transform_mat3x3"]  # type: ignore
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    from zarr_utilities import get_coordinates

    image_coords = get_coordinates(longitudes, latitudes, transform)
    dims = 1
    cz = np.tile(np.arange(dims), image_coords.shape[1])
    cx = np.repeat(image_coords[1, :], dims)
    cy = np.repeat(image_coords[0, :], dims)

    data = zarr_array.get_coordinate_selection((cz, cx, cy))  # type:ignore
    samples_zarr = data.reshape([len(longitudes)])

    LOG.info(f"Zarr " + str(samples_zarr))

    return samples_rasterio - samples_zarr
