import unittest

from physrisk.data_objects.impact_requests import Assets

# from test.data.hazard_model_store import TestData


class TestImpactRequests(unittest.TestCase):
    def test_asset_list_json(self):
        assets = {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "longitude": 69.4787,
                    "latitude": 34.556,
                },
                {"asset_class": "PowerGeneratingAsset", "type": "Nuclear", "longitude": -70.9157, "latitude": -39.2145},
            ],
        }
        assets_obj = Assets(**assets)
        print(assets_obj)
