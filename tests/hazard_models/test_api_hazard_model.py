import asyncio

import numpy as np
import pytest
import shapely.wkt
from physrisk.kernel.hazard_model import HazardDataRequest
from physrisk.kernel.hazards import PluvialInundation, RiverineInundation

from physrisk.data.geocode import Geocoder
from physrisk.hazard_models.credentials_provider import EnvCredentialsProvider
from physrisk.hazard_models.hazard_cache import GeometryH3BasedCache, MemoryStore
from physrisk.hazard_models.jba_hazard_model import (
    JBAAuthenticationError,
    JBACacheKey,
    JBAHazardModel,
)

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
    assert (
        countries[0] == "HK"
    )  # note Hong Kong as Special Administrative Region of China


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


def test_spatial_keys():
    latitudes, longitudes = lats_lons()
    geometries = [
        shapely.wkt.loads(f"POINT({lon} {lat})")
        for lat, lon in zip(latitudes, longitudes)
    ]
    # grid_size=0 pins the legacy exact-match hashing so these expected values are stable;
    # the default (grid_size>0) snaps coordinates first, see test_spatial_key_grid_size_tolerance.
    store = GeometryH3BasedCache(MemoryStore(), grid_size=0)
    spatial_key_lat_lon = store.spatial_key(latitudes[0], longitudes[0])
    spatial_key_geom = store.spatial_key(
        latitudes[0], longitudes[0], geometry=geometries[0]
    )
    assert spatial_key_lat_lon == "8c411c8691b0bff"
    assert spatial_key_geom == "wkbhash_C2amiZX82h4Ga8RxL7PO"


def test_spatial_key_grid_size_tolerance():
    """Tiny floating-point differences in a geometry (e.g. from a different shapely/GEOS
    version recomputing a buffer) should not change the cache key once snapped to the
    default grid, but do change it when snapping is disabled (grid_size=0)."""
    lat, lon = 22.3022371, 114.1867006
    geom_a = shapely.wkt.loads(f"POINT({lon} {lat})")
    geom_b = shapely.wkt.loads(f"POINT({lon + 1e-10} {lat - 1e-10})")

    snapping_store = GeometryH3BasedCache(MemoryStore())
    assert snapping_store.spatial_key(lat, lon, geometry=geom_a) == (
        snapping_store.spatial_key(lat, lon, geometry=geom_b)
    )

    exact_store = GeometryH3BasedCache(MemoryStore(), grid_size=0)
    assert exact_store.spatial_key(lat, lon, geometry=geom_a) != (
        exact_store.spatial_key(lat, lon, geometry=geom_b)
    )


def test_jba_hazard_model(load_credentials, hazard_dir, update_inputs):
    """JBA test, using made-up cached data."""
    # latitudes, longitudes = lats_lons()
    latitudes, longitudes = [43.264209], [5.386365]

    # store = LMDBStore(str((Path(hazard_dir) / "temp" / "hazard_cache.db").absolute()))
    # cache = H3BasedCache(MemoryStore()) # in order to test this 'live', use the MemoryStore and enable API calls
    # cache = H3BasedCache(store)
    credentials = EnvCredentialsProvider(disable_api_calls=False)

    with cache_store_tests(__name__, update_inputs) as cache_store:
        # grid_size=0: the committed hazard_cache.json fixture was generated with the
        # legacy exact-match hashing, before coordinate snapping was introduced.
        model = JBAHazardModel(
            GeometryH3BasedCache(cache_store, grid_size=0),
            credentials,
            max_requests=5,
        )
        requests_riv = [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=lon,
                latitude=lat,
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2050,
                geometry=shapely.wkt.loads(f"POINT({lon} {lat})"),
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
                geometry=shapely.wkt.loads(f"POINT({lon} {lat})"),
            )
            for lat, lon in zip(latitudes, longitudes)
        ]
        response = model.get_hazard_data(requests_riv + requests_pluv)
        assert response is not None


class FakeAPIJBAHazardModel(JBAHazardModel):
    """JBAHazardModel with the API call replaced by a fake that records requests."""

    def __init__(self, *args, fail_batches=False, fail_auth=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_batches = fail_batches
        self.fail_auth = fail_auth
        self.calls: list[list[str]] = []

    async def flood_depth(self, api_request, access_token, session):
        self.calls.append(list(api_request.spatial_keys))
        await asyncio.sleep(0.01)
        if self.fail_auth:
            raise JBAAuthenticationError("401")
        if self.fail_batches and len(api_request.spatial_keys) > 1:
            return "batch failed"
        stats = {
            "FLRF_U": {"rp_100": {"max100": 1.5}},
            "FLSW_U": {"rp_100": {"max100": 0.5}},
        }
        scenarios = ["historical"] + self._jba_scenarios_for_country(
            api_request.country_code
        )
        return {
            JBACacheKey(s, k): {"stats": stats, "geometry": "POINT(0 0)"}
            for k in api_request.spatial_keys
            for s in scenarios
        }


def _fake_model(**kwargs):
    return FakeAPIJBAHazardModel(
        GeometryH3BasedCache(MemoryStore()),
        EnvCredentialsProvider(disable_api_calls=False),
        max_requests=5,
        **kwargs,
    )


def _flood_requests():
    latitudes, longitudes = lats_lons()
    return [
        HazardDataRequest(
            hazard_type=hazard_type,
            longitude=lon,
            latitude=lat,
            indicator_id="flood_depth",
            scenario="ssp585",
            year=2050,
        )
        for hazard_type in [RiverineInundation, PluvialInundation]
        for lat, lon in zip(latitudes[:2], longitudes[:2])
    ]


def test_jba_api_requests():
    model = _fake_model()
    requests = _flood_requests()
    responses = model.get_hazard_data(requests)
    assert [len(c) for c in model.calls] == [2]  # one batch with both locations
    for req in requests:
        expected = 1.5 if req.hazard_type == RiverineInundation else 0.5
        np.testing.assert_allclose(responses[req].intensities, [expected])
    # served from the cache the second time
    model.get_hazard_data(requests)
    assert len(model.calls) == 1


def test_jba_failed_batch_rerun_singly():
    model = _fake_model(fail_batches=True)
    requests = _flood_requests()
    responses = model.get_hazard_data(requests)
    assert sorted(len(c) for c in model.calls) == [1, 1, 2]
    assert all(responses[req].intensities.size == 1 for req in requests)


def test_jba_authentication_error_raised():
    model = _fake_model(fail_auth=True)
    with pytest.raises(JBAAuthenticationError):
        model.get_hazard_data(_flood_requests())
