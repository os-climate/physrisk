from pathlib import Path

import numpy as np
import pytest
from physrisk.kernel.hazard_model import HazardDataRequest
from physrisk.kernel.hazards import PluvialInundation, RiverineInundation

from physrisk.data.geocode import Geocoder
from physrisk.hazard_models.credentials_provider import EnvCredentialsProvider
from physrisk.hazard_models.hazard_cache import H3BasedCache, LMDBStore, MemoryStore
from physrisk.hazard_models.jba_hazard_model import JBAHazardModel

from tests.conftest import cache_store_tests


def lats_lons():
    latitudes = [
        22.2974609,
        22.2974609,
        22.3022371,
        22.2968475,
        22.2655762,
        22.3252006,
        22.2685972,
        22.2480085,
        22.298235,
        22.3373615,
        22.3373615,
        22.3373615,
        22.3669996,
        22.2876498,
        22.2968475,
        22.3314947,
        22.2725865,
        22.4502235,
        22.4502235,
        22.2857261,
        22.256999,
        22.256999,
        22.256999,
        22.256999,
        22.270338,
        22.2765447,
        22.3025354,
        22.3025354,
        22.2851847,
        22.2851847,
        22.3104591,
        22.2809887,
        22.2921893,
        22.2701635,
        22.2875054,
        22.3371349,
        22.3009485,
        22.2958686,
        22.2980295,
        22.3105095,
        22.3092841,
        22.3070995,
        22.3056028,
        22.3669107,
        22.301638,
        22.301638,
    ]
    longitudes = [
        114.169146,
        114.169146,
        114.1867006,
        114.1733945,
        114.1797447,
        114.2102111,
        114.1532016,
        114.1777944,
        114.1750077,
        114.1475118,
        114.1475118,
        114.1475118,
        114.1380616,
        114.2168499,
        114.1733945,
        114.1777367,
        114.176325,
        114.0288237,
        114.0288237,
        114.217443,
        114.200124,
        114.200124,
        114.200124,
        114.200124,
        114.192675,
        114.1583626,
        114.1723814,
        114.1723814,
        114.1533693,
        114.1533693,
        114.2251647,
        114.1583595,
        114.0092291,
        114.194212,
        114.1501129,
        114.148673,
        114.176113,
        114.1728713,
        114.1768655,
        114.1874283,
        114.1880432,
        114.1837226,
        114.1882695,
        114.1166138,
        114.1791412,
        114.1791412,
    ]
    return latitudes, longitudes


def test_geocoding():
    latitudes, longitudes = lats_lons()
    geocoder = Geocoder()
    countries = geocoder.get_countries(latitudes, longitudes)
    assert countries[0] == "CN"


def test_continent_from_country_code():
    continent_and_country_from_code_iso_3166 = (
        Geocoder.get_continent_and_country_from_code_iso_3166(
            country_codes=["USA", "FR", 56]
        )
    )
    assert (
        continent_and_country_from_code_iso_3166["Continent"]["USA"] == "North America"
    )
    assert continent_and_country_from_code_iso_3166["Continent"]["FR"] == "Europe"
    assert continent_and_country_from_code_iso_3166["Continent"][56] == "Europe"


@pytest.mark.skip("Exclude until cache populated")
def test_jba_hazard_model(load_credentials, hazard_dir, update_inputs):
    # this test has to be run live and will cost money!
    # run a batch of 34 lats and lons using an API batch size of 10
    # re-run the same thing individually but only get the 3 lats/lons of interest, to check the two are consistent
    latitudes, longitudes = lats_lons()
    latitudes = latitudes[0:34]
    longitudes = longitudes[0:34]

    # store = LMDBStore(str((Path(hazard_dir) / "temp" / "hazard_cache.db").absolute()))
    # cache = H3BasedCache(MemoryStore()) # in order to test this 'live', use the MemoryStore and enable API calls
    # cache = H3BasedCache(store)
    credentials = EnvCredentialsProvider(disable_api_calls=False)

    with cache_store_tests(__name__, update_inputs) as cache_store:
        model = JBAHazardModel(H3BasedCache(cache_store), credentials, max_requests=5)
        requests_riv = [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=lon,
                latitude=lat,
                indicator_id="flood_depth",
                scenario="historical",
                # scenario="rcp8p5",
                year=2050,
            )
            for lat, lon in zip(latitudes, longitudes)
        ]
        requests_pluv = [
            HazardDataRequest(
                hazard_type=PluvialInundation,
                longitude=lon,
                latitude=lat,
                indicator_id="flood_depth",
                scenario="historical",
                year=2050,
            )
            for lat, lon in zip(latitudes, longitudes)
        ]
        response = model.get_hazard_events(requests_riv + requests_pluv)
        np.testing.assert_allclose(
            response[requests_riv[2]].intensities,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.27]),
        )
        np.testing.assert_allclose(
            response[requests_riv[17]].intensities,
            np.array([0.0, 0.0, 0.0, 0.0, 0.22, 0.38]),
        )
        np.testing.assert_allclose(
            response[requests_pluv[15]].intensities,
            np.array([0.0, 0.31, 0.36, 0.43, 0.44, 0.6]),
        )
        np.testing.assert_allclose(
            response[requests_pluv[24]].intensities,
            np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.33]),
        )
        # model = JBAHazardModel(H3BasedCache(MemoryStore())) # uncomment this line to run again with a new cache (i.e. test data the same)
        # otherwise we are testing that we get the expected values if we request individually or as part of a batch
        response = model.get_hazard_events(
            [requests_riv[2], requests_pluv[15], requests_pluv[24]]
        )
        np.testing.assert_allclose(
            response[requests_riv[2]].intensities,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.27]),
        )
        np.testing.assert_allclose(
            response[requests_pluv[15]].intensities,
            np.array([0.0, 0.31, 0.36, 0.43, 0.44, 0.6]),
        )
        np.testing.assert_allclose(
            response[requests_pluv[24]].intensities,
            np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.33]),
        )
