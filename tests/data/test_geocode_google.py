import asyncio
import os
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from shapely.geometry import MultiPolygon, Point, Polygon

from physrisk.data.geocode_google import (
    GeocodeResult,
    GoogleGeocoder,
    Granularity,
    StructureType,
    _ADDRESS_FIELD_MASK,
    _DESTINATIONS_FIELD_MASK,
    _DESTINATIONS_URL,
    _GEOCODE_ADDRESS_URL,
    _parse_geojson,
    _parse_granularity,
    _parse_structure_type,
)

# ---------------------------------------------------------------------------
# Sample API payloads
# ---------------------------------------------------------------------------

_LOCATION = {"latitude": 37.4219999, "longitude": -122.0840575}

_POLYGON_GEOJSON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.0841, 37.4220],
            [-122.0839, 37.4220],
            [-122.0839, 37.4218],
            [-122.0841, 37.4218],
            [-122.0841, 37.4220],
        ]
    ],
}

_MULTI_POLYGON_GEOJSON = {
    "type": "MultiPolygon",
    "coordinates": [
        [
            [
                [-122.0841, 37.4220],
                [-122.0839, 37.4220],
                [-122.0839, 37.4218],
                [-122.0841, 37.4218],
                [-122.0841, 37.4220],
            ]
        ]
    ],
}

# Response from GET geocode/address/<address>
_ADDRESS_RESPONSE = {
    "results": [
        {
            "location": _LOCATION,
            "granularity": "ROOFTOP",
            "formattedAddress": "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
            "placeId": "ChIJY8sv5-i2j4AR",
        }
    ]
}

# Response from POST geocode/destinations
_DESTINATIONS_RESPONSE = {
    "destinations": [
        {
            "primary": {
                "structureType": "BUILDING",
                "displayPolygon": _POLYGON_GEOJSON,
            }
        }
    ]
}

_EMPTY_ADDRESS_RESPONSE: dict = {"results": []}
_EMPTY_DESTINATIONS_RESPONSE: dict = {"destinations": []}


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _mock_response(payload: dict, status: int = 200) -> MagicMock:
    resp = AsyncMock()
    resp.status = status
    resp.raise_for_status = MagicMock()
    resp.json = AsyncMock(return_value=payload)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _mock_session(
    address_payload: dict = _ADDRESS_RESPONSE,
    destinations_payload: dict = _DESTINATIONS_RESPONSE,
) -> MagicMock:
    """Return a mock session whose GET returns address_payload and POST returns destinations_payload."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(return_value=_mock_response(address_payload))
    session.post = MagicMock(return_value=_mock_response(destinations_payload))
    return session


@pytest.fixture
def default_session():
    return _mock_session()


# ---------------------------------------------------------------------------
# proxy / max_concurrency
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _parse_granularity
# ---------------------------------------------------------------------------


def test_parse_granularity_rooftop():
    assert _parse_granularity("ROOFTOP") is Granularity.ROOFTOP


def test_parse_granularity_approximate():
    assert _parse_granularity("APPROXIMATE") is Granularity.APPROXIMATE


def test_parse_granularity_none():
    assert _parse_granularity(None) is Granularity.UNSPECIFIED


def test_parse_granularity_unknown():
    assert _parse_granularity("NEW_VALUE") is Granularity.UNSPECIFIED


# ---------------------------------------------------------------------------
# _parse_structure_type
# ---------------------------------------------------------------------------


def test_parse_structure_type_building():
    assert _parse_structure_type("BUILDING") is StructureType.BUILDING


def test_parse_structure_type_grounds():
    assert _parse_structure_type("GROUNDS") is StructureType.GROUNDS


def test_parse_structure_type_none():
    assert _parse_structure_type(None) is StructureType.UNSPECIFIED


def test_parse_structure_type_unknown():
    assert _parse_structure_type("SOMETHING_NEW") is StructureType.UNSPECIFIED


def test_structure_type_is_building_level_true():
    assert StructureType.BUILDING.is_building_level is True
    assert StructureType.SECTION.is_building_level is True


def test_structure_type_is_building_level_false():
    assert StructureType.POINT.is_building_level is False
    assert StructureType.GROUNDS.is_building_level is False
    assert StructureType.UNSPECIFIED.is_building_level is False


# ---------------------------------------------------------------------------
# _parse_geojson
# ---------------------------------------------------------------------------


def test_parse_geojson_polygon():
    assert isinstance(_parse_geojson(_POLYGON_GEOJSON), Polygon)


def test_parse_geojson_multi_polygon():
    assert isinstance(_parse_geojson(_MULTI_POLYGON_GEOJSON), MultiPolygon)


def test_parse_geojson_none():
    assert _parse_geojson(None) is None


def test_parse_geojson_invalid():
    assert _parse_geojson({"type": "NotAShape", "coordinates": []}) is None


# ---------------------------------------------------------------------------
# GoogleGeocoder (mocked)
# ---------------------------------------------------------------------------


async def test_geocode_returns_result(default_session):
    async with GoogleGeocoder("test-key", session=default_session, fetch_building_shape=True) as geocoder:
        result = await geocoder.geocode("1600 Amphitheatre Pkwy, Mountain View")

    assert isinstance(result, GeocodeResult)
    assert result.latitude == pytest.approx(37.4219999)
    assert result.longitude == pytest.approx(-122.0840575)
    assert result.location == Point(-122.0840575, 37.4219999)
    assert result.formatted_address == "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA"
    assert result.place_id == "ChIJY8sv5-i2j4AR"
    assert result.granularity is Granularity.ROOFTOP
    assert isinstance(result.building_shape, Polygon)
    assert result.structure_type is StructureType.BUILDING
    assert result.structure_type.is_building_level is True


async def test_geocode_calls_both_endpoints(default_session):
    async with GoogleGeocoder("my-api-key", session=default_session, fetch_building_shape=True) as geocoder:
        await geocoder.geocode("Some Address")

    # geocode/address — GET
    default_session.get.assert_called_once()
    get_call = default_session.get.call_args
    assert _GEOCODE_ADDRESS_URL in get_call.args[0]
    assert get_call.kwargs["headers"]["X-Goog-Api-Key"] == "my-api-key"
    assert get_call.kwargs["headers"]["X-Goog-FieldMask"] == _ADDRESS_FIELD_MASK

    # destinations — POST
    default_session.post.assert_called_once()
    post_call = default_session.post.call_args
    assert post_call.args[0] == _DESTINATIONS_URL
    assert post_call.kwargs["headers"]["X-Goog-Api-Key"] == "my-api-key"
    assert post_call.kwargs["headers"]["X-Goog-FieldMask"] == _DESTINATIONS_FIELD_MASK
    assert post_call.kwargs["json"]["addressQuery"]["addressQuery"] == "Some Address"


async def test_geocode_returns_none_when_no_address_results():
    session = _mock_session(address_payload=_EMPTY_ADDRESS_RESPONSE)
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        result = await geocoder.geocode("Nonexistent Place XYZ")

    assert result is None


async def test_geocode_no_polygon_gives_none_shape():
    session = _mock_session(destinations_payload=_EMPTY_DESTINATIONS_RESPONSE)
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        result = await geocoder.geocode("Some Address")

    assert result is not None
    assert result.building_shape is None
    assert result.structure_type is StructureType.UNSPECIFIED


async def test_geocode_granularity_defaults_to_unspecified_when_missing():
    address_payload = {"results": [{"location": _LOCATION, "formattedAddress": "X", "placeId": "Y"}]}
    session = _mock_session(address_payload=address_payload)
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        result = await geocoder.geocode("Some Address")

    assert result is not None
    assert result.granularity is Granularity.UNSPECIFIED


async def test_geocode_many_resolves_all_addresses():
    session = _mock_session()
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        results = await geocoder.geocode_many(["Address A", "Address B", "Address C"])

    assert len(results) == 3
    assert all(isinstance(r, GeocodeResult) for r in results)
    assert session.get.call_count == 3
    session.post.assert_not_called()


async def test_geocode_many_fetches_building_shapes_when_opted_in():
    session = _mock_session()
    async with GoogleGeocoder("test-key", session=session, fetch_building_shape=True) as geocoder:
        results = await geocoder.geocode_many(["Address A", "Address B", "Address C"])

    assert len(results) == 3
    assert session.get.call_count == 3
    assert session.post.call_count == 3


async def test_geocode_passes_proxy_to_requests():
    session = _mock_session()
    async with GoogleGeocoder("k", proxy="http://proxy.corp:8080", session=session, fetch_building_shape=True) as geocoder:
        await geocoder.geocode("Some Address")

    assert session.get.call_args.kwargs["proxy"] == "http://proxy.corp:8080"
    assert session.post.call_args.kwargs["proxy"] == "http://proxy.corp:8080"


async def test_geocode_proxy_is_none_when_not_set():
    session = _mock_session()
    async with GoogleGeocoder("plain-key", session=session) as geocoder:
        await geocoder.geocode("Some Address")

    assert session.get.call_args.kwargs["proxy"] is None


async def test_geocode_many_respects_max_concurrency():
    active = 0
    peak = 0
    orig_fetch = GoogleGeocoder._fetch_geocode_address

    async def tracking_fetch(self_inner, address):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)  # yield so other tasks can interleave
        result = await orig_fetch(self_inner, address)
        active -= 1
        return result

    session = _mock_session()
    with patch.object(GoogleGeocoder, "_fetch_geocode_address", tracking_fetch):
        async with GoogleGeocoder("k", session=session, max_concurrency=2) as geocoder:
            await geocoder.geocode_many([f"Address {i}" for i in range(6)])

    assert peak <= 2


async def test_geocode_building_shape_not_fetched_by_default():
    session = _mock_session()
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        result = await geocoder.geocode("Some Address")

    session.post.assert_not_called()
    assert result is not None
    assert result.building_shape is None
    assert result.structure_type is StructureType.UNSPECIFIED


async def test_geocode_building_shape_fetched_when_opted_in():
    session = _mock_session()
    async with GoogleGeocoder("test-key", session=session, fetch_building_shape=True) as geocoder:
        result = await geocoder.geocode("Some Address")

    session.post.assert_called_once()
    assert isinstance(result.building_shape, Polygon)


async def test_geocode_region_code_included_in_params():
    session = _mock_session()
    async with GoogleGeocoder("test-key", session=session, region_code="GB") as geocoder:
        await geocoder.geocode("10 Downing Street, London")

    assert session.get.call_args.kwargs["params"]["regionCode"] == "GB"


async def test_geocode_no_region_code_omitted_from_params():
    session = _mock_session()
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        await geocoder.geocode("Some Address")

    assert "regionCode" not in session.get.call_args.kwargs["params"]


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------


@pytest.mark.live_data("dev")
async def test_geocode_googleplex_live():
    dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.getcwd())
    dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=True)

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_MAPS_API_KEY not set")

    async with GoogleGeocoder(api_key) as geocoder:
        result = await geocoder.geocode("1600 Amphitheatre Pkwy, Mountain View, CA")

    assert result is not None
    assert result.latitude == pytest.approx(37.422, abs=0.05)
    assert result.longitude == pytest.approx(-122.084, abs=0.05)
    assert isinstance(result.location, Point)
    assert "Mountain View" in result.formatted_address
    assert result.granularity in (Granularity.ROOFTOP, Granularity.RANGE_INTERPOLATED)
    assert isinstance(result.building_shape, (Polygon, MultiPolygon))
