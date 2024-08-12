from itertools import chain

import numpy as np
import zarr
import zarr.core

from ..utils.lazy import lazy_import

rasterio = lazy_import("rasterio")
tifffile = lazy_import("tifffile")
affine = lazy_import("affine")


def zarr_read(path, longitudes, latitudes):
    """A version that uses Zarr rather than GDAL. Typically faster than GDAL / rasterio."""
    with tifffile.tifffile.Tifffile(path) as tif:
        scale = tif.geotiff_metadata["ModelPixelScale"]
        tie_point = tif.geotiff_metadata["ModelTiepoint"]
        store = tif.series[0].aszarr()
        zarray = zarr.open(store, mode="r")
        # shape: List[int] = tif.series[0].shape
        i, j, k, x, y, z = tie_point[0:6]
        sx, sy, sz = scale
        trans = affine.Affine(sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy)
        inv_trans = ~trans
        mat = np.array(inv_trans).reshape(3, 3)
        coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))
        frac_image_coords = mat @ coords
        image_coords = np.floor(frac_image_coords).astype(int)
        assert zarray is zarr.core.Array
        data = zarray.get_coordinate_selection((image_coords[1, :], image_coords[0, :]))  # type: ignore
        return data


def dataset_read_bounded(dataset, longitudes, latitudes, window_half_width=0.01):
    hw = window_half_width
    offsets = [[0, 0], [-hw, -hw], [-hw, hw], [hw, hw], [hw, -hw]]
    points = []
    for offset in offsets:
        points = chain(
            points,
            [
                [lon + offset[0], lat + offset[1]]
                for (lon, lat) in zip(longitudes, latitudes)
            ],
        )

    samples = np.array(list(rasterio.sample.sample_gen(dataset, points)))
    samples.resize([len(offsets), len(longitudes)])
    max_samples = np.max(samples, 0)

    return max_samples


def dataset_read_points(dataset, longitudes, latitudes, window_half_width=0.01):
    points = [[lon, lat] for (lon, lat) in zip(longitudes, latitudes)]
    samples = np.array(list(rasterio.sample.sample_gen(dataset, points)))
    return samples


def dataset_read_windows(dataset, longitudes, latitudes, window_half_width=0.01):
    # seem to need to do one window at a time: potentially slow
    hw = window_half_width
    samples = []
    for lon, lat in zip(longitudes, latitudes):
        win = rasterio.windows.from_bounds(
            lon - hw, lat - hw, lon + hw, lat + hw, dataset.transform
        )  # left, bottom, right, top
        max_intensity = np.max(dataset.read(1, window=win))
        samples.append(max_intensity[0])
    return samples


def file_read_bounded(path, longitudes, latitudes, window_half_width=0.01):
    with rasterio.open(path) as dataset:
        return dataset_read_bounded(dataset, longitudes, latitudes, window_half_width)


def file_read_points(path, longitudes, latitudes):
    with rasterio.open(path) as dataset:
        return dataset_read_points(dataset, longitudes, latitudes)
