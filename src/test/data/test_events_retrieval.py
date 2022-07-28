""" Test asset impact calculations."""
import json
import os
import pathlib
import shutil
import tempfile
import unittest
from test.data.hazard_model_store import (
    TestData,
    get_mock_hazard_model_store_single_curve,
    mock_hazard_model_store_inundation,
)

import numpy as np
import numpy.testing
import scipy.interpolate
import zarr
from dotenv import load_dotenv

from physrisk import RiverineInundation, requests
from physrisk.data.hazard.event_provider_wri import EventProviderWri
from physrisk.data.hazard_data_provider import get_source_path_wri_riverine_inundation
from physrisk.data.inventory import Inventory
from physrisk.data.zarr_reader import ZarrReader


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

    def test_zarr_bilinear(self):
        # create suitable asymmetric data set and compare with scipy

        xt, yt = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
        data0 = np.exp(-(xt**2 / 25.0 + yt**2 / 16.0))
        data1 = np.exp(-(xt**2 / 36.0 + yt**2 / 25.0))

        data = np.stack([data0, data1], axis=0)

        # note that zarr array has index [z, y, x], e.g. 9, 21600, 43200 or [index, lat, lon]
        y = np.array([1.4, 2.8, 3.4])  # row indices
        x = np.array([3.2, 6.7, 7.9])  # column indices
        image_coords = np.stack([x, y])
        data_zarr = zarr.array(data)
        candidate_lin = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0, 1]), interpolation="linear"
        )
        candidate_max = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0, 1]), interpolation="max"
        )
        candidate_min = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0, 1]), interpolation="min"
        )

        image_coords_surr = np.stack(
            [
                np.concatenate([np.floor(x), np.floor(x) + 1, np.floor(x), np.floor(x) + 1]),
                np.concatenate([np.floor(y), np.floor(y), np.floor(y) + 1, np.floor(y) + 1]),
            ]
        )
        values_surr = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords_surr, np.array([0, 1]), interpolation="linear"
        ).reshape((4, 3, 2))
        interp_scipy0 = scipy.interpolate.interp2d(np.linspace(0, 9, 10), np.linspace(0, 9, 10), data0, kind="linear")
        interp_scipy1 = scipy.interpolate.interp2d(np.linspace(0, 9, 10), np.linspace(0, 9, 10), data1, kind="linear")
        expected0_lin = interp_scipy0(x, y).diagonal().reshape(len(y))
        expected1_lin = interp_scipy1(x, y).diagonal().reshape(len(y))
        expected0_max = np.max(values_surr[:, :, 0], axis=0)
        expected1_max = np.max(values_surr[:, :, 1], axis=0)
        expected0_min = np.min(values_surr[:, :, 0], axis=0)
        expected1_min = np.min(values_surr[:, :, 1], axis=0)

        numpy.testing.assert_allclose(candidate_lin[:, 0], expected0_lin, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_lin[:, 1], expected1_lin, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_max[:, 0], expected0_max, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_max[:, 1], expected1_max, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_min[:, 0], expected0_min, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_min[:, 1], expected1_min, rtol=1e-6)
        # array([0.43854423, 0.62290176, 0.50660137])
        # array([0.58346331, 0.72702827, 0.62629474])
        # array([0.60981554, 0.75222318, 0.72041193])
        # array([0.72817392, 0.82838968, 0.79717553])
        # array([0.34773063, 0.45343876, 0.45907147])
        # array([0.50555484, 0.59275348, 0.58789487])

    def test_zarr_geomax(self):
        lons_ = np.array([3.92783])
        lats_ = np.array([50.882394])
        curve = np.array(
            [0.00, 0.06997928, 0.2679602, 0.51508933, 0.69842442, 0.88040525, 1.11911115, 1.29562478, 1.47200677]
        )
        set_id = r"inundation/wri/v2\\inunriver_rcp8p5_MIROC-ESM-CHEM_2080"
        interpolation = "linear"
        delta_km = 0.100
        n_grid = 10
        store_ = mock_hazard_model_store_inundation(lons_, lats_, curve)
        zarrreader_ = ZarrReader(store_)
        curves_max_candidate, _ = zarrreader_.get_max_curves(
            set_id, lons_, lats_, interpolation=interpolation, delta_km=delta_km, n_grid=n_grid
        )
        curves_max_expected = np.array(
            [[0.00, 0.04917953, 0.1883151, 0.3619907, 0.49083358, 0.61872474, 0.78648075, 0.91052965, 1.03448614]]
        )
        numpy.testing.assert_allclose(curves_max_candidate, curves_max_expected, rtol=1e-6)

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

        request1 = {
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
        response_floor = requests.get(request_id="get_hazard_data", request_dict=request1)
        request1["interpolation"] = "linear"  # type: ignore
        response_linear = requests.get(request_id="get_hazard_data", request_dict=request1)
        print(response_linear)

        floor = json.loads(response_floor)["items"][0]["intensity_curve_set"][5]["intensities"]
        linear = json.loads(response_linear)["items"][0]["intensity_curve_set"][5]["intensities"]

        print(floor)
        print(linear)

        request2 = {
            "items": [
                {
                    "request_item_id": "test_inundation",
                    "event_type": "CoastalInundation",
                    "longitudes": TestData.coastal_longitudes,
                    "latitudes": TestData.coastal_latitudes,
                    "year": 2080,
                    "scenario": "rcp8p5",
                    "model": "wtsub/95",
                }
            ],
        }
        response = requests.get(request_id="get_hazard_data", request_dict=request2)
        print(response)

    @unittest.skip("includes download of large files; deprecated")
    def test_wri_from_web(self):
        cache_folder = self.test_dir
        provider = EventProviderWri("web", cache_folder=cache_folder)
        lon = 19.885738
        lat = 45.268405
        events = provider.get_inundation_depth([lon], [lat])
        print(events)
