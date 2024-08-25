import pytest

import numpy as np

from physrisk.hazard_models import core_hazards
from physrisk.data.hazard_data_provider import HazardDataHint
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.hazards import ChronicHeat, RiverineInundation
from physrisk.container import Container
from physrisk import requests

from ..api.container_test import TestContainer
from ..data.hazard_model_store_test import (
    TestData,
    get_mock_hazard_model_store_single_curve,
    mock_hazard_model_store_heat,
)


def test_hazard_data_availability():
    # test that validation passes:
    container = Container()
    container.override(TestContainer())
    requester = container.requester()
    _ = requester.get(request_id="get_hazard_data_availability", request_dict={})


@pytest.mark.skip(reason="requires mocking.")
def test_hazard_data_description():
    # test that validation passes:
    container = Container()
    requester = container.requester
    _ = requester.get(
        request_id="get_hazard_data_description",
        request_dict={"paths": ["test_path.md"]},
    )


def test_generic_source_path():
    inventory = EmbeddedInventory()
    source_paths = core_hazards.get_default_source_paths(inventory)
    result_heat = source_paths[ChronicHeat](
        indicator_id="mean_degree_days/above/32c", scenario="rcp8p5", year=2050
    )
    result_flood = source_paths[RiverineInundation](
        indicator_id="flood_depth", scenario="rcp8p5", year=2050
    )
    result_flood_hist = source_paths[RiverineInundation](
        indicator_id="flood_depth", scenario="historical", year=2080
    )
    result_heat_hint = source_paths[ChronicHeat](
        indicator_id="mean_degree_days/above/32c",
        scenario="rcp8p5",
        year=2050,
        hint=HazardDataHint(
            path="chronic_heat/osc/v2/mean_degree_days_v2_above_32c_CMCC-ESM2_{scenario}_{year}"
        ),
    )

    assert (
        result_heat
        == "chronic_heat/osc/v2/mean_degree_days_v2_above_32c_ACCESS-CM2_rcp8p5_2050"
    )
    assert result_flood == "inundation/wri/v2/inunriver_rcp8p5_MIROC-ESM-CHEM_2050"
    assert result_flood_hist == "inundation/wri/v2/inunriver_rcp4p5_MIROC-ESM-CHEM_2030"
    assert (
        result_heat_hint
        == "chronic_heat/osc/v2/mean_degree_days_v2_above_32c_CMCC-ESM2_rcp8p5_2050"
    )


def test_zarr_reading():
    request_dict = {
        "items": [
            {
                "request_item_id": "test_inundation",
                "event_type": "RiverineInundation",
                "longitudes": TestData.longitudes[0:3],  # coords['longitudes'][0:100],
                "latitudes": TestData.latitudes[0:3],  # coords['latitudes'][0:100],
                "year": 2080,
                "scenario": "rcp8p5",
                "indicator_id": "flood_depth",
                "indicator_model_gcm": "MIROC-ESM-CHEM",
            }
        ],
    }
    # validate request
    request = requests.HazardDataRequest(**request_dict)  # type: ignore

    store = get_mock_hazard_model_store_single_curve()

    result = requests._get_hazard_data(
        request,
        ZarrHazardModel(
            source_paths=get_default_source_paths(EmbeddedInventory()),
            reader=ZarrReader(store=store),
        ),
    )

    np.testing.assert_array_almost_equal_nulp(
        result.items[0].intensity_curve_set[0].intensities, np.zeros((9))
    )
    np.testing.assert_array_almost_equal_nulp(
        result.items[0].intensity_curve_set[1].intensities,
        np.linspace(0.1, 1.0, 9, dtype="f4"),
    )
    np.testing.assert_array_almost_equal_nulp(
        result.items[0].intensity_curve_set[2].intensities, np.zeros((9))
    )


def test_zarr_reading_chronic():
    request_dict = {
        "group_ids": ["osc"],
        "items": [
            {
                "request_item_id": "test_inundation",
                "event_type": "ChronicHeat",
                "longitudes": TestData.longitudes[0:3],  # coords['longitudes'][0:100],
                "latitudes": TestData.latitudes[0:3],  # coords['latitudes'][0:100],
                "year": 2050,
                "scenario": "ssp585",
                "indicator_id": "mean_degree_days/above/32c",
            }
        ],
    }
    # validate request
    request = requests.HazardDataRequest(**request_dict)  # type: ignore

    store = mock_hazard_model_store_heat(TestData.longitudes, TestData.latitudes)

    source_paths = get_default_source_paths(EmbeddedInventory())
    result = requests._get_hazard_data(
        request,
        ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
    )
    np.testing.assert_array_almost_equal_nulp(
        result.items[0].intensity_curve_set[0].intensities[0], 600.0
    )


@pytest.mark.skip(reason="Requires credentials.")
def test_zarr_reading_live(load_credentials):
    # needs valid OSC_S3_BUCKET, OSC_S3_ACCESS_KEY, OSC_S3_SECRET_KEY
    container = Container()
    requester = container.requester()

    import json
    from zipfile import ZipFile

    with ZipFile("./tests/api/test_lat_lons.json.zip") as z:
        with z.open("test_lat_lons.json") as f:
            data = json.loads(f.read())

    request1 = {
        "items": [
            {
                "request_item_id": "test_inundation",
                "event_type": "ChronicHeat",
                "longitudes": data["longitudes"],
                "latitudes": data["latitudes"],
                "year": 2030,
                "scenario": "ssp585",
                "indicator_id": "mean_work_loss/high",
            }
        ],
    }

    response_floor = requester.get(request_id="get_hazard_data", request_dict=request1)
    request1["interpolation"] = "linear"  # type: ignore
    response_linear = requester.get(request_id="get_hazard_data", request_dict=request1)
    print(response_linear)

    floor = json.loads(response_floor)["items"][0]["intensity_curve_set"][5][
        "intensities"
    ]
    linear = json.loads(response_linear)["items"][0]["intensity_curve_set"][5][
        "intensities"
    ]

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
                "indicator_id": "flood_depth",
                "model_id": "wtsub/95",
            }
        ],
    }
    response = requester.get(request_id="get_hazard_data", request_dict=request2)
    print(response)
