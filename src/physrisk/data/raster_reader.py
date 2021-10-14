from itertools import chain
import numpy as np
import rasterio, rasterio.sample
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds

def dataset_read_bounded(dataset, longitudes, latitudes, window_half_width = 0.01):
    hw = window_half_width
    offsets = [[0, 0], [-hw, -hw], [-hw, hw], [hw, hw], [hw, -hw]]
    points = []
    for offset in offsets:
        points = chain(points, [[lon + offset[0], lat + offset[1]] for (lon, lat) in zip(longitudes, latitudes)])

    samples = np.array(list(rasterio.sample.sample_gen(dataset, points)))
    samples.resize([len(offsets), len(longitudes)])
    max_samples = np.max(samples, 0)
    
    return max_samples

def dataset_read_points(dataset, longitudes, latitudes, window_half_width = 0.01):
    points = [[lon, lat] for (lon, lat) in zip(longitudes, latitudes)]
    samples = np.array(list(rasterio.sample.sample_gen(dataset, points)))
    return samples

def dataset_read_windows(dataset, longitudes, latitudes, window_half_width = 0.01):
    # seem to need to do one window at a time: potentially slow
    hw = window_half_width
    samples = []
    for (lon, lat) in zip(longitudes, latitudes):
        win = from_bounds(lon - hw, lat - hw, lon + hw, lat + hw, dataset.transform) # left, bottom, right, top
        max_intensity = np.max(dataset.read(1, window = win))
        samples.append(max_intensity[0])
    return samples

def file_read_bounded(path, longitudes, latitudes, window_half_width = 0.01):
    #with MemoryFile() as memfile:
    #    with memfile.open(driver = 'GTiff', count)

    with rasterio.open(path) as dataset:
        return dataset_read_bounded(dataset, longitudes, latitudes, window_half_width)

def file_read_points(path, longitudes, latitudes):
    with rasterio.open(path) as dataset:
        return dataset_read_points(dataset, longitudes, latitudes)
