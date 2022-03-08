""" Test asset impact calculations."""
import os
import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import numpy.testing
import zarr
import zarr.storage
from affine import Affine
from dotenv import load_dotenv

from physrisk import RiverineInundation, requests
from physrisk.data.event_provider import get_source_path_wri_riverine_inundation
from physrisk.data.hazard.event_provider_wri import EventProviderWri


class TestEventRetrieval(unittest.TestCase):
    """Tests asset impact calculations."""

    _longitudes = [
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
    _latitudes = [
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

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.getcwd())
        dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path, override=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_zarr_reading(self):
        request_dict = {
            "request_id": "get_hazard_data",
            "items": [
                {
                    "request_id": "get_hazard_data",
                    "request_item_id": "test_inundation",
                    "event_type": "RiverineInundation",
                    "longitudes": TestEventRetrieval._longitudes[0:3],  # coords['longitudes'][0:100],
                    "latitudes": TestEventRetrieval._latitudes[0:3],  # coords['latitudes'][0:100],
                    "year": 2080,
                    "scenario": "rcp8p5",
                    "model": "MIROC-ESM-CHEM",
                }
            ],
        }
        request = requests.HazardEventDataRequest(**request_dict)  # validate request

        store = self._get_mock_hazard_model_store()

        result = requests._get_hazard_data(
            request, {RiverineInundation: get_source_path_wri_riverine_inundation}, store=store
        )

        result.items[0].intensity_curve_set[0].intensities

        numpy.testing.assert_array_almost_equal_nulp(result.items[0].intensity_curve_set[0].intensities, np.zeros((8)))
        numpy.testing.assert_array_almost_equal_nulp(
            result.items[0].intensity_curve_set[1].intensities, np.linspace(0.1, 1.0, 8, dtype="f4")
        )
        numpy.testing.assert_array_almost_equal_nulp(result.items[0].intensity_curve_set[2].intensities, np.zeros((8)))

    @unittest.skip("requires OSC environment variables set")
    def test_zarr_reading_live(self):
        # needs valid OSC_S3_BUCKET, OSC_S3_ACCESS_KEY, OSC_S3_SECRET_KEY

        request = {
            "request_id": "get_hazard_data",
            "items": [
                {
                    "request_id": "get_hazard_data",
                    "request_item_id": "test_inundation",
                    "event_type": "RiverineInundation",
                    "longitudes": TestEventRetrieval._longitudes,
                    "latitudes": TestEventRetrieval._latitudes,
                    "year": 2080,
                    "scenario": "rcp8p5",
                    "model": "MIROC-ESM-CHEM",
                }
            ],
        }
        response = requests.get(request)
        print(response)

    def _get_mock_hazard_model_store(self):
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

        longitudes = TestEventRetrieval._longitudes
        latitudes = TestEventRetrieval._latitudes
        t = z.attrs["transform_mat3x3"]
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

        coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))
        inv_trans = ~transform
        mat = np.array(inv_trans).reshape(3, 3)
        frac_image_coords = mat @ coords
        image_coords = np.floor(frac_image_coords).astype(int)
        z[:, image_coords[1, 1], image_coords[0, 1]] = np.linspace(0.1, 1.0, z.shape[0])

        return store

    @unittest.skip("includes download of large files; deprecated")
    def test_wri_from_web(self):
        cache_folder = self.test_dir
        provider = EventProviderWri("web", cache_folder=cache_folder)
        lon = 19.885738
        lat = 45.268405
        events = provider.get_inundation_depth([lon], [lat])
        print(events)
