import base64
import hashlib
import json
import logging
import os
import sys
from os import listdir
from os.path import isfile, join
from time import sleep
from typing import List, Tuple

import numpy as np
import rasterio
import s3fs
import seaborn as sns
import zarr
from affine import Affine
from mapbox import Uploader
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from tifffile.tifffile import TiffFile


def alphanumeric(text):
    """Return alphanumeric hash from supplied string."""
    hash_int = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest(), "big")
    return base36encode(hash_int)


def base36encode(number, alphabet="0123456789abcdefghijklmnopqrstuvwxyz"):
    """Converts an integer to a base36 string."""
    if not isinstance(number, int):
        raise TypeError("number must be an integer")

    base36 = ""

    if number < 0:
        raise TypeError("number must be positive")

    if 0 <= number < len(alphabet):
        return sign + alphabet[number]

    while number != 0:
        number, i = divmod(number, len(alphabet))
        base36 = alphabet[i] + base36

    return base36


def load_s3(s3_source, src_bucket, src_prefix, filename, target_width=None):
    with s3_source.open(os.path.join(src_bucket, src_prefix, filename)) as f:
        with rasterio.open(f) as dataset:
            return load(dataset, target_width)


def load(dataset, target_width=None):
    scaling = 1 if target_width is None else float(target_width) / dataset.width
    width = int(dataset.width * scaling)
    height = int(dataset.height * scaling)

    # data = src.read(1)
    # resample data to target shape
    data = (
        dataset.read(1)
        if scaling == 1
        else dataset.read(1, out_shape=(dataset.count, height, width), resampling=Resampling.bilinear)
    )

    # scale image transform
    t = dataset.transform
    transform = Affine(t.a / scaling, t.b / scaling, t.c, t.d / scaling, t.e / scaling, t.f)
    transform = dataset.transform * dataset.transform.scale(
        (float(dataset.width) / data.shape[-1]), (float(dataset.height) / data.shape[-2])
    )

    return (data, dataset.profile, width, height, transform)


def load_fs(input_path, target_width=None):
    with rasterio.open(input_path) as dataset:
        return load(dataset, target_width)


def write_map_geotiff_s3(s3_source, src_bucket, src_prefix, filename, output_dir):
    (data, profile, width, height, transform) = load_s3(s3_source, src_bucket, src_prefix, filename)
    LOG.info("Loaded")

    write_map_geotiff(data, profile, width, height, transform, filename, output_dir)


def write_map_geotiff_fs(input_dir, filename, output_dir):
    (data, profile, width, height, transform) = load_fs(os.path.join(input_dir, filename))
    LOG.info("Loaded")

    write_map_geotiff(data, profile, width, height, transform, filename, output_dir)


def write_map_geotiff(data, profile, width, height, transform, filename, output_dir):

    max_intensity = 2.0  # 2 metres for inundation

    # we use a color map from Seaborn: looks good
    orig_map = sns.color_palette("flare", as_cmap=True)  #  plt.get_cmap("Reds") as alternative
    map = {}
    map_for_json = {}
    reds = np.zeros(256)
    greens = np.zeros(256)
    blues = np.zeros(256)
    a = np.zeros(256)
    for i in range(256):
        cols = (np.array(orig_map(i)) * 255).astype(np.uint8)
        map[i] = (cols[0], cols[1], cols[2], 200)
        map_for_json[i] = (int(cols[0]), int(cols[1]), int(cols[2]), 200)
        reds[i] = cols[0]
        greens[i] = cols[1]
        blues[i] = cols[2]
        a[i] = 200
    map[0] = map[1] = (255, 255, 255, 0)
    map_for_json[0] = map_for_json[1] = (255, 255, 255, 0)  # no data and zero are transparent
    a[0] = a[1] = 0

    filename_stub = filename.split(".")[0]
    alpha = alphanumeric(filename_stub)[0:6]
    LOG.info(f"Hashing {filename_stub} as code: {alpha}")

    colormap_path_out = os.path.join(output_dir, "colormap_" + alpha + "_" + filename_stub + ".json")
    with open(colormap_path_out, "w") as f:
        colormap_info = json.dumps(
            {
                "colormap": map_for_json,
                "nodata": {"color_index": 0},
                "min": {"data": 0.0, "color_index": 1},
                "max": {"data": 2.0, "color_index": 255},
            }
        )
        f.writelines(colormap_info)

    mask = data < 0  # == -9999.0 is alternative
    mask2 = data > max_intensity

    np.multiply(data, 254.0 / max_intensity, out=data, casting="unsafe")
    np.add(data, 1.0, out=data, casting="unsafe")  # np.clip seems a bit slow so we do not use

    # 0 is no data
    # 1 is zero
    # 255 is max intensity

    result = data.astype(np.uint8, casting="unsafe", copy=False)
    del data

    result[mask] = 0
    result[mask2] = 255
    del (mask, mask2)

    profile["dtype"] = "uint8"
    profile["nodata"] = 0
    profile["count"] = 4
    profile["width"] = width
    profile["height"] = height
    profile["transform"] = transform

    path_out = os.path.join(output_dir, "map_" + alpha + "_" + filename)

    with rasterio.open(path_out, "w", **profile) as dst:
        # dst.colorinterp = [ ColorInterp.red, ColorInterp.green, ColorInterp.blue ]
        # dst.write(result, indexes=1)
        # dst.write_colormap(1, map)
        LOG.info("Writing R 1/4")
        dst.write(reds[result], 1)
        LOG.info("Writing G 2/4")
        dst.write(greens[result], 2)
        LOG.info("Writing B 3/4")
        dst.write(blues[result], 3)
        LOG.info("Writing A 4/4")
        dst.write(a[result], 4)
    LOG.info("Complete")
    return (path_out, colormap_path_out)


def upload_geotiff(path, id):
    uploader = Uploader(
        access_token="***REMOVED***"
    )
    attempt = 0
    with open(path, "rb") as src:
        while attempt < 5:
            upload_resp = uploader.upload(src, id)
            if upload_resp.status_code == 201:
                LOG.info("Upload in progress")
                break
            if upload_resp.status_code != 422:
                raise Exception("unexpected response")
            sleep(5)

        if upload_resp.status_code != 201:
            raise Exception("could not upload")

        upload_id = upload_resp.json()["id"]
        for i in range(5):
            status_resp = uploader.status(upload_id).json()
            if status_resp["complete"]:
                LOG.info("Complete")
                break
            LOG.info("Uploading...")
            sleep(5)


# LOG = logging.getLogger("Mapbox onboarding")
# LOG.setLevel(logging.INFO)

# input_dir = r"/root/code/osc/inputs/"
# input_file = "inunriver_rcp8p5_MIROC-ESM-CHEM_2080_rp01000.tif"
# output_dir = r"/root/code/osc/inputs/"

# # print((colormap, map))
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# LOG.addHandler(handler)

# (colormap, map) = write_map_geotiff(input_dir, input_file, output_dir)
# upload_geotiff("/root/code/osc/inputs/map_inunriver_rcp8p5_MIROC-ESM-CHEM_2080_rp01000.tif", "test3")
