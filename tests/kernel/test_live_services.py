import pytest
import requests

url = "https://physrisk-api-physrisk.apps.osc-cl1.apps.os-climate.org"
# url = "http://127.0.0.1:5000"


@pytest.mark.skip("only as example")
def test_live_exposure():
    request = {
        "assets": {
            "items": [
                {
                    "asset_class": "Asset",
                    "type": None,
                    "location": None,
                    "latitude": 34.556,
                    "longitude": 69.4787,
                }
            ]
        },
        "calc_settings": {"hazard_interp": "floor"},
        "scenario": "ssp585",
        "year": 2050,
    }
    result = requests.post(url + "/api/get_asset_exposure", json=request)
    print(result.json())


@pytest.mark.skip("only as example")
def test_live_impacts():  # "latitude": 34.556, "longitude": 69.4787
    request = {
        "assets": {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "location": "Asia",
                    "latitude": 23.1577,
                    "longitude": 113.8306,
                }
            ]
        },
        "include_asset_level": False,
        "include_measures": True,
        "include_calc_details": False,
        "scenarios": ["ssp585"],
        "years": [2050],
    }
    result = requests.post(url + "/api/get_asset_impact", json=request)
    print(result.json())


@pytest.mark.skip("only as example")
def test_live_hazard_data():
    request = {
        "items": [
            {
                "longitudes": [-101.731],
                "latitudes": [38.65],
                "request_item_id": "my_drought_request",
                "hazard_type": "Drought",
                "indicator_id": "months/spei12m/below/threshold",
                "scenario": "ssp585",
                "path": "drought/osc/v2/months_spei12m_below_threshold_multi_model_0_{scenario}_{year}",
                "year": 2050,
            },
        ]
    }
    from physrisk.container import Container

    container = Container()
    requester = container.requester()
    result = requester.get(request_id="get_hazard_data", request_dict=request)
    print(result)
