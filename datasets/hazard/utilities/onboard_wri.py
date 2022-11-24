import logging
import os
import pathlib
import time

import numpy as np
import rasterio
import s3fs
import zarr
from affine import Affine
from botocore import UNSIGNED
from dotenv import load_dotenv
from map_utilities import write_map_geotiff
from zarr_utilities import get_geotiff_meta_data, zarr_create, zarr_write

LOG = logging.getLogger("Hazard onboarding")
LOG.setLevel(logging.INFO)

dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.environ.get("PWD", "/users/joemoorhouse/Code/physrisk/"))
dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)


def onboard_wri_riverine_inundation(
    dest_bucket="redhat-osc-physical-landing-647521352890", create_zarr=True, create_geotiff=True
):
    LOG.info("Riverine inundation")
    # http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/index.html

    src_bucket = "wri-projects"
    src_prefix = "AqueductFloodTool/download/v2"

    # destination
    dest_prefix = "hazard"

    # no authentication for source currently
    s3_source = s3fs.S3FileSystem(config_kwargs=dict(signature_version=UNSIGNED))

    # need credentials for target
    s3_dest = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])

    circ_models = ["00000NorESM1-M", "0000GFDL-ESM2M", "0000HadGEM2-ES", "00IPSL-CM5A-LR", "MIROC-ESM-CHEM"]
    years = ["2030", "2050", "2080"]
    rcps = ["rcp4p5", "rcp8p5"]

    LOG.info("Future models")
    for circ_model in circ_models:
        LOG.info("Model: " + circ_model)
        for year in years:
            LOG.info("Year: " + year)
            for rcp in rcps:
                LOG.info("RCP: " + rcp)
                try:
                    if create_zarr:
                        geotiff_to_zarr_riverine(
                            circ_model=circ_model,
                            year=year,
                            rcp=rcp,
                            src_bucket=src_bucket,
                            src_prefix=src_prefix,
                            s3_source=s3_source,
                            dest_bucket=dest_bucket,
                            dest_prefix=dest_prefix,
                            s3_dest=s3_dest,
                        )
                    if create_geotiff:
                        (colormap, map) = write_map_geotiff(input_dir, input_file, output_dir)

                except Exception as e:
                    LOG.error("Error writing zarr", exc_info=e)
                    LOG.info("Skipping...")

    LOG.info("Historical")
    if create_zarr:
        geotiff_to_zarr_riverine(
            circ_model="000000000WATCH",
            year="1980",
            rcp="historical",
            src_bucket=src_bucket,
            src_prefix=src_prefix,
            s3_source=s3_source,
            dest_bucket=dest_bucket,
            dest_prefix=dest_prefix,
            s3_dest=s3_dest,
        )


def onboard_wri_coastal_inundation(dest_bucket="redhat-osc-physical-landing-647521352890"):
    LOG.info("Coastal inundation")
    # http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/index.html

    src_bucket = "wri-projects"
    src_prefix = "AqueductFloodTool/download/v2"

    # destination
    dest_prefix = "hazard"

    # no authentication for source currently
    s3_source = s3fs.S3FileSystem(config_kwargs=dict(signature_version=UNSIGNED))

    # need credentials for target
    s3_dest = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])

    models = ["0", "0_perc_05", "0_perc_50"]
    subs = ["wtsub", "nosub"]
    years = ["2030", "2050", "2080"]
    rcps = ["rcp4p5", "rcp8p5"]

    # models = ["0"]
    # subs = ["wtsub"]
    # years = ["2080"]
    # rcps = ["rcp8p5"]

    LOG.info("Future models")
    for model in models:
        LOG.info("Model: " + model)
        for sub in subs:
            LOG.info("Subsidence: " + sub)
            for year in years:
                LOG.info("Year: " + year)
                for rcp in rcps:
                    LOG.info("RCP: " + rcp)
                    try:
                        geotiff_to_zarr_coastal(
                            model=model,
                            sub=sub,
                            year=year,
                            rcp=rcp,
                            src_bucket=src_bucket,
                            src_prefix=src_prefix,
                            s3_source=s3_source,
                            dest_bucket=dest_bucket,
                            dest_prefix=dest_prefix,
                            s3_dest=s3_dest,
                        )
                    except Exception as e:
                        LOG.error("Error writing zarr", exc_info=e)
                        LOG.info("Skipping...")

    # inuncoast_historical_nosub_hist_rp0001_5.pickle
    LOG.info("Historical")
    geotiff_to_zarr_coastal(
        model="0",
        sub="nosub",
        year="hist",
        rcp="historical",
        src_bucket=src_bucket,
        src_prefix=src_prefix,
        s3_source=s3_source,
        dest_bucket=dest_bucket,
        dest_prefix=dest_prefix,
        s3_dest=s3_dest,
    )


def create_map_geotiffs_riverine(dest_dir):
    src_bucket = "wri-projects"
    src_prefix = "AqueductFloodTool/download/v2"

    circ_models = ["00000NorESM1-M", "0000GFDL-ESM2M", "0000HadGEM2-ES", "00IPSL-CM5A-LR", "MIROC-ESM-CHEM"]
    years = ["2030", "2050", "2080"]
    rcps = ["rcp4p5", "rcp8p5"]
    src_returns = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
    circ_model = circ_models[4]
    s3_source = s3fs.S3FileSystem(config_kwargs=dict(signature_version=UNSIGNED))

    for rcp in rcps:
        for year in years:
            src_filenames = [
                "inun{0}_{1}_{2}_{3}_rp{4:05d}".format("river", rcp, circ_model, year, i) for i in src_returns
            ]
            filename = os.path.join(src_filenames[8] + ".tif")
            print(filename)
            write_map_geotiff(
                os.path.join(src_bucket, src_prefix),
                dest_dir,
                filename,
                input_s3=s3_source,
                lowest_bin_transparent=True,
            )

    circ_model = "000000000WATCH"
    rcp = "historical"
    year = "1980"
    src_filenames = ["inun{0}_{1}_{2}_{3}_rp{4:05d}".format("river", rcp, circ_model, year, i) for i in src_returns]
    filename = os.path.join(src_filenames[8] + ".tif")
    write_map_geotiff(os.path.join(src_bucket, src_prefix), dest_dir, filename, input_s3=s3_source)


def create_map_geotiffs_coastal(dest_dir):
    src_bucket = "wri-projects"
    src_prefix = "AqueductFloodTool/download/v2"

    models = ["0", "0_perc_05", "0_perc_50"]
    subs = ["wtsub", "nosub"]
    years = ["2030", "2050", "2080"]
    rcps = ["rcp4p5", "rcp8p5"]
    src_returns = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

    model = models[0]
    sub = "wtsub"
    s3_source = s3fs.S3FileSystem(config_kwargs=dict(signature_version=UNSIGNED))

    for rcp in rcps:
        for year in years:
            src_filenames = [
                "inun{0}_{1}_{2}_{3}_rp{4:04d}_{5}".format("coast", rcp, sub, year, i, model) for i in src_returns
            ]
            filename = os.path.join(src_filenames[8] + ".tif")
            print(filename)
            write_map_geotiff(os.path.join(src_bucket, src_prefix), dest_dir, filename, input_s3=s3_source)

    model = "0"
    sub = "nosub"
    year = "hist"
    rcp = "historical"
    src_filenames = ["inun{0}_{1}_{2}_{3}_rp{4:04d}_{5}".format("coast", rcp, sub, year, i, model) for i in src_returns]
    filename = os.path.join(src_filenames[8] + ".tif")
    print(filename)
    write_map_geotiff(os.path.join(src_bucket, src_prefix), dest_dir, filename, input_s3=s3_source)


def geotiff_to_zarr_riverine(
    *, circ_model, year, rcp, src_bucket, src_prefix, s3_source, dest_bucket, dest_prefix, s3_dest
):
    src_returns = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

    src_filenames = ["inun{0}_{1}_{2}_{3}_rp{4:05d}".format("river", rcp, circ_model, year, i) for i in src_returns]
    dest_filename = "inun{0}_{1}_{2}_{3}".format("river", rcp, circ_model, year)

    # get meta-data from first in list
    shape, transform = get_geotiff_meta_data(os.path.join(src_bucket, src_prefix, src_filenames[0] + ".tif"), s3_source)

    z = zarr_create(
        os.path.join(dest_bucket, dest_prefix, "hazard.zarr"),
        os.path.join("inundation/wri/v2/", dest_filename),
        s3_dest,
        shape,
        transform,
        list([r * 1.0 for r in src_returns]),
    )

    LOG.info("Destination: " + dest_filename)
    for (i, filename) in enumerate(src_filenames):
        print(f"File {i + 1}/{len(src_filenames)}", end="...")
        zarr_write(os.path.join(src_bucket, src_prefix, filename + ".tif"), s3_source, z, i)


def geotiff_to_zarr_coastal(
    *, model, sub, year, rcp, src_bucket, src_prefix, s3_source, dest_bucket, dest_prefix, s3_dest
):
    src_returns = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

    src_filenames = ["inun{0}_{1}_{2}_{3}_rp{4:04d}_{5}".format("coast", rcp, sub, year, i, model) for i in src_returns]
    dest_filename = "inun{0}_{1}_{2}_{3}_{4}".format("coast", rcp, sub, year, model)

    # get meta-data from first in list
    shape, transform = get_geotiff_meta_data(os.path.join(src_bucket, src_prefix, src_filenames[0] + ".tif"), s3_source)

    z = zarr_create(
        os.path.join(dest_bucket, dest_prefix, "hazard.zarr"),
        os.path.join("inundation/wri/v2/", dest_filename),
        s3_dest,
        shape,
        transform,
        list([r * 1.0 for r in src_returns]),
    )

    LOG.info("Destination: " + dest_filename)
    for (i, filename) in enumerate(src_filenames):
        print(f"File {i + 1}/{len(src_filenames)}", end="...")
        zarr_write(os.path.join(src_bucket, src_prefix, filename + ".tif"), s3_source, z, i)


def test_data_rasterio(coords, s3_source, src_bucket, src_prefix, dest_bucket, dest_prefix):
    circ_model = "MIROC-ESM-CHEM"
    year = "2080"
    rcp = "rcp8p5"
    src_returns = [5, 10, 25, 50, 100, 250, 500, 1000]

    src_filenames = ["inun{0}_{1}_{2}_{3}_rp{4:05d}".format("river", rcp, circ_model, year, i) for i in src_returns]

    longitudes = np.array(coords["longitudes"])[0:100]
    latitudes = np.array(coords["latitudes"])[0:100]

    res = []
    for (i, filename) in enumerate(src_filenames):
        with s3_source.open(os.path.join(src_bucket, src_prefix, filename + ".tif")) as f:
            with rasterio.open(f) as dataset:
                points = [[lon, lat] for (lon, lat) in zip(longitudes, latitudes)]
                samples_rasterio = np.array(list(rasterio.sample.sample_gen(dataset, points)))  # type:ignore
                samples_rasterio.resize([len(longitudes)])
                res.append(samples_rasterio)
    return res


def test_data_zarr(coords, s3_dest, dest_bucket, dest_prefix):
    store = s3fs.S3Map(root=os.path.join(dest_bucket, "hazard", "hazard.zarr"), s3=s3_dest, check=False)
    root = zarr.open(store, mode="r")
    z = root[os.path.join("inundation/wri/v2/", "inunriver_rcp8p5_MIROC-ESM-CHEM_2080")]

    t = z.attrs["transform_mat3x3"]  # type: ignore
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    from zarr_utilities import get_coordinates

    longitudes = np.array(coords["longitudes"])[0:100]
    latitudes = np.array(coords["latitudes"])[0:100]

    image_coords = get_coordinates(longitudes, latitudes, transform)
    cz = np.tile(np.arange(8), image_coords.shape[1])
    cx = np.repeat(image_coords[1, :], 8)
    cy = np.repeat(image_coords[0, :], 8)

    start = time.time()
    data = z.get_coordinate_selection((cz, cx, cy))  # type:ignore
    stop = time.time()
    d = data.reshape([len(longitudes), 8])
    # print(stop - start)
    return d


def check_test_data():
    import json

    with open("coords.json", "r") as f:
        coords = json.loads(f.read())

    s3_source = s3fs.S3FileSystem(config_kwargs=dict(signature_version=UNSIGNED))
    src_bucket = "wri-projects"
    src_prefix = "AqueductFloodTool/download/v2"

    # need landing credentials for target
    s3_dest = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])
    dest_bucket = "redhat-osc-physical-landing-647521352890"
    dest_prefix = "hazard"

    res = test_data_rasterio(coords, s3_source, src_bucket, src_prefix, dest_bucket, dest_prefix)
    d = test_data_zarr(coords, s3_dest, dest_bucket, dest_prefix)
    for i in range(8):
        print(np.max(np.abs(d[:, i] - res[i])))
