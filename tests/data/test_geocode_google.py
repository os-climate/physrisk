import os
import pathlib

import pytest
from dotenv import load_dotenv
from shapely.geometry import MultiPolygon, Point, Polygon

from physrisk.data.geocode_google import (
    GeocodeResult,
    GoogleGeocoder,
    StructureType,
    _DESTINATIONS_URL,
    _FIELD_MASK,
    _parse_geojson,
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

_API_RESPONSE = {
    "destinations": [
        {
            "primary": {
                "location": _LOCATION,
                "formattedAddress": "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
                "place": "places/ChIJY8sv5-i2j4AR",
                "structureType": "BUILDING",
                "displayPolygon": _POLYGON_GEOJSON,
            }
        }
    ]
}

_EMPTY_RESPONSE: dict = {"destinations": []}


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


from unittest.mock import AsyncMock, MagicMock


def _mock_response(payload: dict, status: int = 200) -> MagicMock:
    resp = AsyncMock()
    resp.status = status
    resp.raise_for_status = MagicMock()
    resp.json = AsyncMock(return_value=payload)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _mock_session(payload: dict, status: int = 200) -> MagicMock:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=_mock_response(payload, status))
    return session


@pytest.fixture
def api_response_session():
    return _mock_session(_API_RESPONSE)


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


async def test_geocode_returns_result(api_response_session):
    async with GoogleGeocoder("test-key", session=api_response_session) as geocoder:
        result = await geocoder.geocode("1600 Amphitheatre Pkwy, Mountain View")

    assert isinstance(result, GeocodeResult)
    assert result.latitude == pytest.approx(37.4219999)
    assert result.longitude == pytest.approx(-122.0840575)
    assert result.location == Point(-122.0840575, 37.4219999)
    assert result.formatted_address == "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA"
    assert result.place_id == "ChIJY8sv5-i2j4AR"
    assert isinstance(result.building_shape, Polygon)
    assert result.structure_type is StructureType.BUILDING
    assert result.structure_type.is_building_level is True


async def test_geocode_sends_correct_request():
    session = _mock_session(_API_RESPONSE)
    async with GoogleGeocoder("my-api-key", session=session) as geocoder:
        await geocoder.geocode("Some Address")

    session.post.assert_called_once()
    call_kwargs = session.post.call_args

    assert call_kwargs.args[0] == _DESTINATIONS_URL
    assert call_kwargs.kwargs["headers"]["X-Goog-Api-Key"] == "my-api-key"
    assert call_kwargs.kwargs["headers"]["X-Goog-FieldMask"] == _FIELD_MASK
    assert call_kwargs.kwargs["json"]["addressQuery"]["addressQuery"] == "Some Address"


async def test_geocode_returns_none_when_no_destinations():
    session = _mock_session(_EMPTY_RESPONSE)
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        result = await geocoder.geocode("Nonexistent Place XYZ")

    assert result is None


async def test_geocode_no_polygon_gives_none_shape():
    payload = {
        "destinations": [
            {
                "primary": {
                    "location": _LOCATION,
                    "formattedAddress": "Some Address",
                    "place": "places/abc123",
                }
            }
        ]
    }
    session = _mock_session(payload)
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        result = await geocoder.geocode("Some Address")

    assert result is not None
    assert result.building_shape is None


async def test_geocode_many_resolves_all_addresses():
    session = _mock_session(_API_RESPONSE)
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        results = await geocoder.geocode_many(["Address A", "Address B", "Address C"])

    assert len(results) == 3
    assert all(isinstance(r, GeocodeResult) for r in results)
    assert session.post.call_count == 3


async def test_geocode_region_code_included_in_body():
    session = _mock_session(_API_RESPONSE)
    async with GoogleGeocoder("test-key", session=session, region_code="GB") as geocoder:
        await geocoder.geocode("10 Downing Street, London")

    assert session.post.call_args.kwargs["json"]["regionCode"] == "GB"


async def test_geocode_no_region_code_omitted_from_body():
    session = _mock_session(_API_RESPONSE)
    async with GoogleGeocoder("test-key", session=session) as geocoder:
        await geocoder.geocode("Some Address")

    assert "regionCode" not in session.post.call_args.kwargs["json"]


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
    assert isinstance(result.building_shape, (Polygon, MultiPolygon))
