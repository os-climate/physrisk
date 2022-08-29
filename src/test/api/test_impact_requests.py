import unittest

import numpy as np

from physrisk import requests
from physrisk.api.v1.common import Assets
from physrisk.kernel.assets import RealEstateAsset
from test.data.hazard_model_store import TestData, get_mock_hazard_model_store_single_curve, mock_hazard_model_store_inundation

class TestImpactRequests(unittest.TestCase):
    def test_asset_list_json(self):
        assets = {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "location": "Asia",
                    "longitude": 69.4787,
                    "latitude": 34.556,
                },
                {
                    "asset_class": "PowerGeneratingAsset",
                    "type": "Nuclear",
                    "location": "Asia",
                    "longitude": -70.9157,
                    "latitude": -39.2145},
            ],
        }
        assets_obj = Assets(**assets)
        print(assets_obj)

    def test_impact_request(self):
        assets = {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "location": "Asia",
                    "longitude": TestData.longitudes[0],
                    "latitude": TestData.latitudes[0],
                },
                {
                    "asset_class": "PowerGeneratingAsset",
                    "type": "Nuclear",
                    "location": "Asia",
                    "longitude": TestData.longitudes[1],
                    "latitude": TestData.latitudes[1],
                }
            ],
        }
        
        request_dict = {
            "assets": assets,
            "include_asset_level": True,
            "include_calc_details": True,
            "year": 2080,
            "scenario": "rcp8p5",
        }
        
        request = requests.AssetImpactRequest(**request_dict)  # type: ignore

        curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)

        result = requests._get_asset_impacts(
            request, store=store
        )
