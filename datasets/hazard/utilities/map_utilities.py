import base64
import hashlib
import json
import logging
import os
from os import listdir
from os.path import isfile, join
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
from affine import Affine
from mapbox import Uploader
from rasterio import CRS, profiles
from rasterio.enums import Resampling

LOG = logging.getLogger("Mapbox onboarding")
LOG.setLevel(logging.INFO)


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


def get_path(bucket, prefix, filename):
    return os.path.join(bucket, prefix, filename)


def load(path, s3=None):
    if s3 is None:
        return load_fs(path)
    else:
        return load_s3(s3, path)


def load_fs(path, target_width=None):
    with rasterio.open(path) as dataset:
        return load_dataset(dataset, target_width)


def load_s3(s3_source, path, target_width=None):
    with s3_source.open(path) as f:
        with rasterio.open(f) as dataset:
            return load_dataset(dataset, target_width)


def load_dataset(dataset, target_width=None):
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


def geotiff_profile():
    crs = CRS.from_epsg(4326)
    profile = profiles.Profile(crs=crs)
    return profile


def write_map_geotiff(input_dir, output_dir, filename, input_s3=None, output_s3=None, lowest_bin_transparent=False):
    (data, profile, width, height, transform) = load(os.path.join(input_dir, filename), s3=input_s3)
    LOG.info("Loaded")

    write_map_geotiff_data(data, profile, width, height, transform, filename, output_dir, lowest_bin_transparent=lowest_bin_transparent)


def write_map_geotiff_data(
    data,
    profile,
    width,
    height,
    transform,
    filename,
    output_dir,
    s3=None,
    nodata_threshold=0,
    zero_transparent=True,
    zero_tolerance=1e-6,
    lowest_bin_transparent=False,
    max_intensity=2.0,
    palette="flare",
):

    # the Seaborn 'flare' palette is the default for representing intensity
    # perceptually uniform, use of hue and luminance, smaller values have lighter colours

    def get_colors(i):
        if palette == "heating":
            cmap = sns.color_palette("coolwarm", as_cmap=True)
            return cmap(0.5 + 0.5 * i / 256.0)
        else:
            cmap = (
                sns.color_palette(palette, as_cmap=True) if palette == "flare" else plt.get_cmap(palette)
            )  #  plt.get_cmap("Reds") as alternative
            return cmap(i)

    map = {}
    map_for_json = {}
    reds = np.zeros(256)
    greens = np.zeros(256)
    blues = np.zeros(256)
    a = np.zeros(256)
    for i in range(256):
        cols = (np.array(get_colors(i)) * 255).astype(np.uint8)
        map[i] = (cols[0], cols[1], cols[2], 200)
        map_for_json[i] = (int(cols[0]), int(cols[1]), int(cols[2]), 200)
        reds[i] = cols[0]
        greens[i] = cols[1]
        blues[i] = cols[2]
        a[i] = 200
    if lowest_bin_transparent:
        # index 1 is zero
        map[1] = (255, 255, 255, 0)
        map_for_json[1] = (255, 255, 255, 0)
        a[1] = 0

    map[0] = (255, 255, 255, 0)
    map_for_json[0] = (255, 255, 255, 0)  # index 0, no data is transparent
    a[0] = 0

    filename_stub = filename.split(".")[0]
    alpha = alphanumeric(filename_stub)[0:6]
    LOG.info(f"Hashing {filename_stub} as code: {alpha}")

    colormap_path_out = os.path.join(output_dir, "colormap_" + alpha + "_" + filename_stub + ".json")
    with (s3.open(colormap_path_out, "w") if s3 is not None else open(colormap_path_out, "w")) as f:
        json_dict = {
            "colormap": map_for_json,
            "nodata": {"color_index": 0},
            "min": {"data": 0.0},
            "max": {"data": float(max_intensity), "color_index": 255},
        }
        colormap_info = json.dumps(json_dict)
        f.writelines(colormap_info)

    mask_nodata = data < nodata_threshold  # == -9999.0 is alternative
    mask_ge_max = data > max_intensity
    if zero_transparent:
        mask_zero = data < (0.0 + zero_tolerance)

    # by zero_transparent, we mean, _exactly_ zero
    np.multiply(data, 254.0 / max_intensity, out=data, casting="unsafe")
    np.add(data, 1.0, out=data, casting="unsafe")  # np.clip seems a bit slow so we do not use

    # 0 is no data
    # 1 is zero (to max_intensity / 254)
    # 254 is max_intensity - max_intensity / 254 to max_intensity
    # 255 is >= max_intensity

    result = data.astype(np.uint8, casting="unsafe", copy=False)
    del data

    result[mask_nodata] = 0
    if zero_transparent:
        result[mask_zero] = 0
        del mask_zero

    result[mask_ge_max] = 255
    del (mask_nodata, mask_ge_max)

    profile["dtype"] = "uint8"
    profile["nodata"] = 0
    profile["count"] = 4
    profile["width"] = width
    profile["height"] = height
    profile["transform"] = transform

    bounds = rasterio.transform.array_bounds(height, width, transform)

    path_out = os.path.join(output_dir, "map_" + alpha + "_" + filename)

    def write_dataset(dst):
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

    if s3 is not None:
        with s3.open(path_out, "w") as f:
            with rasterio.open(f, "w", **profile) as dst:
                write_dataset(dst)
    else:
        with rasterio.open(path_out, "w", **profile) as dst:
            write_dataset(dst)

    LOG.info("Complete")
    return (path_out, colormap_path_out)


def upload_geotiff(path, id, access_token):
    uploader = Uploader(access_token=access_token)
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
