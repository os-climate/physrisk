import logging
import os

import rasterio
import s3fs
from zarr_utilities import add_logging_output_to_stdout, zarr_create

LOG = logging.getLogger("Hazard onboarding")
LOG.setLevel(logging.INFO)


def load_geotiff_rasterio(path):
    with rasterio.open(path) as dataset:
        data = dataset.read(1)
    return data, dataset.transform


def onboard_chronic_heat(src_dir, dest_bucket="redhat-osc-physical-landing-647521352890"):
    LOG.info("Chronic heat")

    # destination
    dest_prefix = "hazard"

    # need credentials for target
    s3_dest = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])

    # type is "mean_degree_days"
    # above_below is "above" or "below"
    # ref_temp is for example "18c"
    # scenario is for example "ssp585"
    # {type}_{above_below}_{ref_temp}_{scenario}_{year}"

    file_mapping = {
        "mean_cdd18C_hist": ["mean_degree_days", "above", "18c", "historical", "1980"],
        "mean_hdd18C_hist": ["mean_degree_days", "below", "18c", "historical", "1980"],
        "mean_cdd18C_ssp585_2050": ["mean_degree_days", "above", "18c", "ssp585", "2050"],
        "mean_hdd18C_ssp585_2050": ["mean_degree_days", "below", "18c", "ssp585", "2050"],
    }

    for file in file_mapping.keys():
        type, above_below, ref_temp, scenario, year = file_mapping[file]

        src_path = os.path.join(src_dir, file + ".tiff")
        dest_filename = f"{type}_{above_below}_{ref_temp}_{scenario}_{year}"

        data, transform = load_geotiff_rasterio(src_path)

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
