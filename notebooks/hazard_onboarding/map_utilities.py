import json
import logging
import os
import sys
from time import sleep

import numpy as np
import rasterio
import seaborn as sns
from affine import Affine
from mapbox import Uploader
from matplotlib import pyplot as plt
from rasterio.enums import Resampling

LOG = logging.getLogger("Mapbox onboarding")
LOG.setLevel(logging.INFO)


def load_resampled(input_path, target_width=None):

    with rasterio.open(input_path) as dataset:
        scaling = 1 if target_width is None else float(target_width) / dataset.width
        width = int(dataset.width * scaling)
        height = int(dataset.height * scaling)

        # data = src.read(1)
        # resample data to target shape
        data = dataset.read(1, out_shape=(dataset.count, height, width), resampling=Resampling.bilinear)

        # scale image transform
        t = dataset.transform
        transform = Affine(t.a / scaling, t.b / scaling, t.c, t.d / scaling, t.e / scaling, t.f)
        transform = dataset.transform * dataset.transform.scale(
            (float(dataset.width) / data.shape[-1]), (float(dataset.height) / data.shape[-2])
        )

        return (data, dataset.profile, width, height, transform)


def write_map_geotiff(input_dir, input_file, output_dir):
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

    colormap_path_out = os.path.join(output_dir, "colormap_" + input_file.split(".")[0] + ".json")
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

    (data, profile, width, height, transform) = load_resampled(os.path.join(input_dir, input_file), 30000)
    LOG.info("Resampled")

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

    path_out = os.path.join(output_dir, "map_" + input_file)

    with rasterio.open(path_out, "w", **profile) as dst:
        # dst.colorinterp = [ ColorInterp.red, ColorInterp.green, ColorInterp.blue ]
        # dst.write(result, indexes=1)
        # dst.write_colormap(1, map)
        LOG.info("Writing R 1/4")
        dst.write(reds[result], 1)
        LOG.info("Writing G 1/4")
        dst.write(greens[result], 2)
        LOG.info("Writing B 1/4")
        dst.write(blues[result], 3)
        LOG.info("Writing A 1/4")
        dst.write(a[result], 4)
    LOG.info("Complete")
    return (path_out, colormap_path_out)


def upload_geotiff(path, id):
    uploader = Uploader(
        access_token="sk.eyJ1Ijoib3NjLW1hcGJveCIsImEiOiJjbDI4NGI4eDgwNXVxM29xcmp3MjZ2cGZ0In0.xatS5xA4JhPrQe-4PemkxA"
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


input_dir = r"/root/code/osc/inputs/"
input_file = "inunriver_rcp8p5_MIROC-ESM-CHEM_2080_rp01000.tif"
output_dir = r"/root/code/osc/inputs/"

# print((colormap, map))
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)

(colormap, map) = write_map_geotiff(input_dir, input_file, output_dir)
upload_geotiff("/root/code/osc/inputs/map_inunriver_rcp8p5_MIROC-ESM-CHEM_2080_rp01000.tif", "test3")

# 2 is
