""" Test asset impact calculations."""
import os
import pathlib
import shutil
import tempfile
import unittest
from test.data.hazard_model_store import TestData, get_mock_hazard_model_store_single_curve

import numpy as np
import numpy.testing
from dotenv import load_dotenv

from physrisk import RiverineInundation, requests
from physrisk.data.event_provider import get_source_path_wri_riverine_inundation
from physrisk.data.hazard.event_provider_wri import EventProviderWri
from physrisk.data.inventory import Inventory


class TestEventRetrieval(unittest.TestCase):
    """Tests asset impact calculations."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.getcwd())
        dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path, override=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_hazard_data_availability(self):
        # test that validation passes:
        _ = requests.get(request_id="get_hazard_data_availability", request_dict={})

    def test_hazard_data_availability_summary(self):
        summary = Inventory().get_models_summary()
        self.assertEqual(summary["RiverineInundation"].years, [1980, 2030, 2050, 2080])
        

    def test_zarr_reading(self):
        request_dict = {
            "items": [
                {
                    "request_item_id": "test_inundation",
                    "event_type": "RiverineInundation",
                    "longitudes": TestData.longitudes[0:3],  # coords['longitudes'][0:100],
                    "latitudes": TestData.latitudes[0:3],  # coords['latitudes'][0:100],
                    "year": 2080,
                    "scenario": "rcp8p5",
                    "model": "MIROC-ESM-CHEM",
                }
            ],
        }
        # validate request
        request = requests.HazardEventDataRequest(**request_dict)  # type: ignore

        store = get_mock_hazard_model_store_single_curve()

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
            "items": [
                {
                    "request_item_id": "test_inundation",
                    "event_type": "RiverineInundation",
                    "longitudes": TestData.longitudes,
                    "latitudes": TestData.latitudes,
                    "year": 2080,
                    "scenario": "rcp8p5",
                    "model": "MIROC-ESM-CHEM",
                }
            ],
        }
        response = requests.get(request_id="get_hazard_data", request_dict=request)
        print(response)

    @unittest.skip("includes download of large files; deprecated")
    def test_wri_from_web(self):
        cache_folder = self.test_dir
        provider = EventProviderWri("web", cache_folder=cache_folder)
        lon = 19.885738
        lat = 45.268405
        events = provider.get_inundation_depth([lon], [lat])
        print(events)
