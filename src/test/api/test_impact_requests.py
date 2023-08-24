import unittest
from test.base_test import TestWithCredentials
from test.data.hazard_model_store import TestData, mock_hazard_model_store_inundation

import numpy as np

from physrisk import requests
from physrisk.api.v1.common import Assets
from physrisk.container import Container
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths

# from physrisk.api.v1.impact_req_resp import AssetImpactResponse
# from physrisk.data.static.world import get_countries_and_continents


class TestImpactRequests(TestWithCredentials):
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
                    "latitude": -39.2145,
                },
            ],
        }
        assets_obj = Assets(**assets)
        self.assertIsNotNone(assets_obj)

    def test_impact_request(self):
        """Runs short asset-level impact request."""

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
                },
            ],
        }

        request_dict = {
            "assets": assets,
            "include_asset_level": True,
            "include_measures": False,
            "include_calc_details": True,
            "year": 2080,
            "scenario": "rcp8p5",
        }

        request = requests.AssetImpactRequest(**request_dict)  # type: ignore

        curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)

        source_paths = get_default_source_paths(EmbeddedInventory())
        response = requests._get_asset_impacts(
            request, ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store))
        )

        self.assertEqual(response.asset_impacts[0].impacts[0].hazard_type, "CoastalInundation")

    @unittest.skip("example, not test")
    def test_example_portfolios(self):
        example_portfolios = requests._get_example_portfolios()
        for assets in example_portfolios:
            request_dict = {
                "assets": assets,
                "include_asset_level": True,
                "include_calc_details": True,
                "year": 2050,
                "scenario": "ssp585",
            }
            container = Container()
            requester = container.requester()
            response = requester.get(request_id="get_asset_impact", request_dict=request_dict)
            assert response is not None
