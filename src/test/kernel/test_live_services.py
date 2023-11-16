import pytest
import requests

# url = "https://physrisk-api2-sandbox.apps.odh-cl1.apps.os-climate.org"
url = "http://127.0.0.1:5000"


@pytest.mark.skip("only as example")
def test_live_exposure():
    request = {
        "assets": {
            "items": [
                {"asset_class": "Asset", "type": None, "location": None, "latitude": 34.556, "longitude": 69.4787}
            ]
        },
        "calc_settings": {"hazard_interp": "floor"},
        "scenario": "ssp585",
        "year": 2050,
    }
    result = requests.post(url + "/api/get_asset_exposure", json=request)
    print(result.json())


@pytest.mark.skip("only as example")
def test_live_impacts():
    request = {
        "assets": {
            "items": [
                {"asset_class": "Asset", "type": None, "location": None, "latitude": 34.556, "longitude": 69.4787}
            ]
        },
        "calc_settings": {"hazard_interp": "floor"},
        "scenario": "ssp585",
        "year": 2050,
    }
    result = requests.post(url + "/api/get_asset_exposure", json=request)
    print(result.json())
