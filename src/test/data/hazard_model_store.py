import numpy as np
import zarr
import zarr.storage
from affine import Affine

from physrisk.data.event_provider import get_source_path_wri_riverine_inundation


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


def get_mock_hazard_model_store(longitudes, latitudes, curve):
    return_periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    t = [0.008333333333333333, 0.0, -180.0, 0.0, -0.008333333333333333, 90.0, 0.0, 0.0, 1.0]
    shape = (len(return_periods), 21600, 43200)
    store = zarr.storage.MemoryStore(root="hazard.zarr")
    root = zarr.open(store=store, mode="w")
    for model, scenario, year in [("MIROC-ESM-CHEM", "rcp8p5", 2080), ("000000000WATCH", "historical", 1980)]:
        array_path = get_source_path_wri_riverine_inundation(model=model, scenario=scenario, year=year)
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

    return store
