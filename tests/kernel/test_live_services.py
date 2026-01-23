import json
import logging
from pydantic import TypeAdapter
import pytest
import requests

from physrisk.api.v1.impact_req_resp import RiskMeasures, RiskMeasuresHelper
from physrisk.container import Container
import physrisk.requests

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


# @pytest.mark.skip("only as example")
def test_example_portfolios():
    example_portfolios = physrisk.requests._get_example_portfolios_with_names()
    for name, assets in example_portfolios.items():
        if name != "mixed_small":
            continue
        logger.info(f"Running example portfolio: {name}")
        request_dict = {
            "assets": assets,
            "include_asset_level": True,
            "include_calc_details": False,
            "include_measures": True,
            "years": [2030, 2040, 2050],
            "scenarios": ["ssp585"],
        }
        container = Container()
        requester = container.requester()
        response = requester.get(
            request_id="get_asset_impact", request_dict=request_dict
        )
        # with open("result.json", "w") as f:
        #    f.write(response)
        risk_measures_dict = json.loads(response)["risk_measures"]
        helper = RiskMeasuresHelper(
            TypeAdapter(RiskMeasures).validate_python(risk_measures_dict)
        )
        for hazard_type in [
            "RiverineInundation",
            "CoastalInundation",
            "ChronicHeat",
            "Wind",
        ]:
            scores, measure_values, measure_defns = helper.get_measure(
                hazard_type, "ssp585", 2050
            )
            label, description = helper.get_score_details(scores[0], measure_defns[0])
            print(label)
