import json
import logging
import pytest
import requests

from physrisk.container import Container
import physrisk.requests
from tests.conftest import get_result_expected

url = "https://physrisk-api-physrisk.apps.osc-cl1.apps.os-climate.org"
# url = "http://127.0.0.1:5000"

logger = logging.getLogger(__name__)


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


@pytest.mark.skip("only as example")
def test_get_image_info():
    request = {
        "resource": "inundation/river_tudelft/v2/flood_depth_unprot_{scenario}_{year}",
        "scenario_id": "historical",
        "year": 1985,
    }
    from physrisk.container import Container

    container = Container()
    requester = container.requester()
    result = requester.get(request_id="get_image_info", request_dict=request)
    print(result)


@pytest.fixture
def requester(clear_credentials):
    container = Container()
    return container.requester()


@pytest.mark.live_data("dev")
def test_live_impacts_regression(
    requester: physrisk.requests.Requester, update_expected: str
):  # "latitude": 34.556, "longitude": 69.4787
    request = {
        "assets": {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "location": "Asia",
                    "latitude": 23.0737,
                    "longitude": 113.8516,
                }
            ]
        },
        "include_asset_level": False,
        "include_measures": True,
        "include_calc_details": False,
        "scenarios": ["ssp585"],
        "years": [2050],
    }
    result = requester.get(request_id="get_asset_impact", request_dict=request)
    result_dict = json.loads(result)
    # hazards = [
    #     "CoastalInundation",
    #     "ChronicHeat",
    #     "Drought",
    #     "Fire",
    #     "Hail",
    #     "PluvialInundation",
    #     "RiverineInundation",
    #     "Wind",
    # ]
    # result_dict_cut_down = {"risk_measures": {"measure_for_assets": []}}
    # for item in result_dict["risk_measures"]["measures_for_assets"]:
    #     if item["key"]["hazard_type"] in hazards:
    #         result_dict_cut_down["risk_measures"]["measure_for_assets"].append(item)
    result_str = json.dumps(result_dict, indent=2)
    func_name = f"{__name__}.{test_live_impacts.__name__}"
    result_dict, expected_dict = get_result_expected(
        result_str, func_name, update_expected == "True"
    )
