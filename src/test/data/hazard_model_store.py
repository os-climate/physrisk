import numpy as np
import zarr
import zarr.storage
from affine import Affine

from physrisk.data.hazard_data_provider import (
    get_source_path_osc_chronic_heat,
    get_source_path_wri_coastal_inundation,
    get_source_path_wri_riverine_inundation,
)


class TestData:

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

    coastal_longitudes = [12.2, 50.5919, 90.3473, 90.4295, 90.4804, 90.3429, 90.5153, 90.6007]
    coastal_latitudes = [-5.55, 26.1981, 23.6473, 23.6783, 23.5699, 23.9904, 23.59, 23.6112]


def get_mock_hazard_model_store_single_curve():
    """Create a test MemoryStore for creation of Zarr hazard model for unit testing. A single curve
    is applied at all locations."""

    return_periods = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    t = [0.008333333333333333, 0.0, -180.0, 0.0, -0.008333333333333333, 90.0, 0.0, 0.0, 1.0]
    shape = (len(return_periods), 21600, 43200)
    store = zarr.storage.MemoryStore(root="hazard.zarr")
    root = zarr.open(store=store, mode="w")
    array_path = get_source_path_wri_riverine_inundation(model="MIROC-ESM-CHEM", scenario="rcp8p5", year=2080)
    z = root.create_dataset(  # type: ignore
        array_path, shape=(shape[0], shape[1], shape[2]), chunks=(shape[0], 1000, 1000), dtype="f4"
    )
    z.attrs["transform_mat3x3"] = t
    z.attrs["index_values"] = return_periods

    longitudes = TestData.longitudes
    latitudes = TestData.latitudes
    t = z.attrs["transform_mat3x3"]
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))
    inv_trans = ~transform
    mat = np.array(inv_trans).reshape(3, 3)
    frac_image_coords = mat @ coords
    image_coords = np.floor(frac_image_coords).astype(int)
    z[:, image_coords[1, 1], image_coords[0, 1]] = np.linspace(0.1, 1.0, z.shape[0])

    return store


def mock_hazard_model_store_heat(longitudes, latitudes, curve):
    return mock_hazard_model_store_for_paths(longitudes, latitudes, curve, heat_paths)


def mock_hazard_model_store_inundation(longitudes, latitudes, curve):
    return mock_hazard_model_store_for_paths(longitudes, latitudes, curve, inundation_paths)


def mock_hazard_model_store_for_paths(longitudes, latitudes, curve, paths):
    """Create a MemoryStore for creation of Zarr hazard model to be used with unit tests,
    with the specified longitudes and latitudes set to the curve supplied."""
    if len(curve) == 1:
        return_periods = None
        shape = (1, 21600, 43200)
    else:
        return_periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        if len(curve) != len(return_periods):
            raise ValueError(f"curve must be single value or of length {len(return_periods)}")
        shape = (len(return_periods), 21600, 43200)

    t = [0.008333333333333333, 0.0, -180.0, 0.0, -0.008333333333333333, 90.0, 0.0, 0.0, 1.0]
    store = zarr.storage.MemoryStore(root="hazard.zarr")
    root = zarr.open(store=store, mode="w")

    for path in paths():
        _add_curves(root, longitudes, latitudes, path, shape, curve, return_periods, t)

    return store


def heat_paths():
    paths = []
    for model, scenario, year in [("mean_heating_degree_days", "rcp8p5", 2080)]:
        paths.append(get_source_path_osc_chronic_heat(model=model, scenario=scenario, year=year))
    return paths


def inundation_paths():
    paths = []
    for model, scenario, year in [("MIROC-ESM-CHEM", "rcp8p5", 2080), ("000000000WATCH", "historical", 1980)]:
        paths.append(get_source_path_wri_riverine_inundation(model=model, scenario=scenario, year=year))
    for model, scenario, year in [("wtsub/95", "rcp8p5", 2080), ("wtsub", "historical", 1980)]:
        paths.append(get_source_path_wri_coastal_inundation(model=model, scenario=scenario, year=year))
    return paths


def _add_curves(root, longitudes, latitudes, array_path, shape, curve, return_periods, t):
    z = root.create_dataset(  # type: ignore
        array_path, shape=(shape[0], shape[1], shape[2]), chunks=(shape[0], 1000, 1000), dtype="f4"
    )
    z.attrs["transform_mat3x3"] = t
    z.attrs["index_values"] = return_periods

    t = z.attrs["transform_mat3x3"]
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))
    inv_trans = ~transform
    mat = np.array(inv_trans).reshape(3, 3)
    frac_image_coords = mat @ coords
    image_coords = np.floor(frac_image_coords).astype(int)
    for j in range(len(longitudes)):
        z[:, image_coords[1, j], image_coords[0, j]] = curve
