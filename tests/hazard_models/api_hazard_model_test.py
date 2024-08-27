import pytest
from physrisk.kernel.hazard_model import HazardDataRequest
from physrisk.kernel.hazards import PluvialInundation, RiverineInundation

from physrisk.data.geocode import Geocoder
from physrisk.hazard_models.credentials_provider import EnvCredentialsProvider
from physrisk.hazard_models.hazard_cache import H3BasedCache
from physrisk.hazard_models.jba_hazard_model import JBAHazardModel

from tests.conftest import cache_store_tests


def lats_lons():
    latitudes = [
        22.3022371,
        22.2968475,
        22.3314947,
    ]
    longitudes = [
        114.1867006,
        114.1733945,
        114.1777367,
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


@pytest.mark.skip("Exclude until confirmed cache can be populated")
def test_jba_hazard_model(load_credentials, hazard_dir, update_inputs):
    """JBA test, but
    1) will incur cost
    2) requires expected results
    (need to check these can be provided)
    """
    # this test should be run both live and cached
    # run a batch of lats and lons using an API batch size of 10
    # re-run the same thing individually but only get the 3 lats/lons of interest, to check the two are consistent
    latitudes, longitudes = lats_lons()

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
                scenario="ssp585",
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
                scenario="ssp585",
                year=2050,
            )
            for lat, lon in zip(latitudes, longitudes)
        ]
        response = model.get_hazard_events(requests_riv + requests_pluv)
        assert response is not None
