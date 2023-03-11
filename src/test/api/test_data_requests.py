import json
import unittest
from test.base_test import TestWithCredentials
from test.data.hazard_model_store import (
    TestData,
    get_mock_hazard_model_store_single_curve,
    mock_hazard_model_store_heat,
)

import numpy as np
import numpy.testing

from physrisk import RiverineInundation, requests
from physrisk.data.hazard_data_provider import get_source_path_wri_riverine_inundation
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.kernel.calculation import get_source_paths_from_inventory
from physrisk.kernel.hazards import ChronicHeat


class TestDataRequests(TestWithCredentials):
    def test_hazard_data_availability(self):
        # test that validation passes:
        _ = requests.get(request_id="get_hazard_data_availability", request_dict={})

    def test_hazard_data_description(self):
        # test that validation passes:
        _ = requests.get(request_id="get_hazard_data_description", request_dict={"paths": ["test_path.md"]})

    def test_generic_source_path(self):
        inventory = Inventory(EmbeddedInventory().to_resources())
        source_paths = get_source_paths_from_inventory(inventory, None)
        result_heat = source_paths[ChronicHeat](model="mean_degree_days/above/32c", scenario="rcp8p5", year=2050)
        result_flood = source_paths[RiverineInundation](model="000000000WATCH", scenario="rcp8p5", year=2050)
        assert result_heat == "chronic_heat/osc/v1/mean_degree_days_above_32c_rcp8p5_2050"
        assert result_flood == "inundation/wri/v2/inunriver_rcp8p5_000000000WATCH_2050"

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

    def test_zarr_reading_chronic(self):
        request_dict = {
            "items": [
                {
                    "request_item_id": "test_inundation",
                    "event_type": "ChronicHeat",
                    "longitudes": TestData.longitudes[0:3],  # coords['longitudes'][0:100],
                    "latitudes": TestData.latitudes[0:3],  # coords['latitudes'][0:100],
                    "year": 2050,
                    "scenario": "ssp585",
                    "model": "mean_degree_days/above/32c",
                }
            ],
        }
        # validate request
        request = requests.HazardEventDataRequest(**request_dict)  # type: ignore

        store = mock_hazard_model_store_heat(TestData.longitudes, TestData.latitudes)

        result = requests._get_hazard_data(request, store=store)
        numpy.testing.assert_array_almost_equal_nulp(result.items[0].intensity_curve_set[0].intensities[0], 600.0)

    @unittest.skip("requires OSC environment variables set")
    def test_zarr_reading_live(self):
        # needs valid OSC_S3_BUCKET, OSC_S3_ACCESS_KEY, OSC_S3_SECRET_KEY
        request1 = {
            "items": [
                {
                    "request_item_id": "test_inundation",
                    "event_type": "ChronicHeat",
                    "longitudes": TestData.longitudes,
                    "latitudes": TestData.latitudes,
                    "year": 2030,
                    "scenario": "ssp585",
                    "model": "mean_work_loss/high",
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
        response = requests.get(request_type="get_hazard_data", request_dict=request2)
        print(response)
