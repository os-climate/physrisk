import json
from unittest.mock import patch

import aiohttp
import shapely.wkt
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import HazardDataRequest, HazardEventDataResponse
from physrisk.kernel.hazards import PluvialInundation, RiverineInundation

from physrisk.data.geocode import Geocoder
from physrisk.hazard_models.credentials_provider import EnvCredentialsProvider
from physrisk.hazard_models.hazard_cache import GeometryH3BasedCache, MemoryStore
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


def test_spatial_key_vertex_order_invariant():
    """The same polygon with a different ring start vertex and orientation should give
    the same key (snap then normalize), and a buffered asset geometry should already be
    in the canonical (snapped, normalized) form so that the key-time step is a no-op."""
    lat, lon = 22.3022371, 114.1867006
    poly = Asset(latitude=lat, longitude=lon, buffer=50.0).geometry
    coords = list(poly.exterior.coords)[:-1]
    reordered = shapely.Polygon((coords[3:] + coords[:3])[::-1])
    assert reordered.wkb != poly.wkb

    store = GeometryH3BasedCache(MemoryStore())
    assert store.spatial_key(lat, lon, geometry=poly) == store.spatial_key(
        lat, lon, geometry=reordered
    )
    assert shapely.set_precision(poly, store.grid_size).normalize().wkb == poly.wkb


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


class _FakeJBAResponse:
    def __init__(self, body):
        self.status = 200
        self._body = body

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def test_jba_cache_stores_asset_lat_lon():
    """For an asset specified as latitude/longitude plus buffer, the geometry sent to JBA
    is the buffered polygon; the latitude/longitude entered should also be stored in the
    cache value, so that cache entries can be linked back to assets."""
    latitude, longitude = 43.264209, 5.386365
    asset = Asset(latitude=latitude, longitude=longitude, buffer=50.0)
    assert asset.geometry is not None and asset.geometry.geom_type == "Polygon"

    model = JBAHazardModel(
        GeometryH3BasedCache(MemoryStore()),
        EnvCredentialsProvider(disable_api_calls=False),
        max_requests=5,
    )
    sent_payloads = []
    stats = {"FLRF_U": {"rp_100": {"max100": 0.5}}}

    def fake_post(session, url, json, **kwargs):
        sent_payloads.append(json)
        body = [
            {"id": g["id"], "stats": stats, **{s: stats for s in model.jba_scenarios}}
            for g in json["geometries"]
        ]
        return _FakeJBAResponse(body)

    request = HazardDataRequest(
        hazard_type=RiverineInundation,
        longitude=asset.longitude,
        latitude=asset.latitude,
        indicator_id="flood_depth",
        scenario="ssp585",
        year=2050,
        geometry=asset.geometry,
    )
    with patch.object(aiohttp.ClientSession, "post", fake_post):
        response = model.get_hazard_data([request])
    assert isinstance(response[request], HazardEventDataResponse)

    # the buffered polygon (not the point) is sent, with no additional JBA-side buffer
    (geometry_sent,) = sent_payloads[0]["geometries"]
    assert geometry_sent["wkt_geometry"] == asset.geometry.wkt
    assert geometry_sent["buffer"] == 0

    cached = {k: json.loads(v) for k, v in model.cache_store.getall("jba/").items()}
    assert len(cached) == 1 + len(model.jba_scenarios)  # historical plus all scenarios
    for value in cached.values():
        assert value["wkt_geometry"] == asset.geometry.wkt
        assert value["buffer"] == 0
        assert value["latitude"] == latitude
        assert value["longitude"] == longitude
