"""Async geocoder using the Google Geocoding API v4 destinations endpoint.

POST https://geocode.googleapis.com/v4/geocode/destinations

Each request resolves one address to a lat/lon ``Point`` and a building
``displayPolygon`` (GeoJSON → shapely geometry).  ``geocode_many`` submits
all addresses concurrently via ``asyncio.gather``.

Authentication uses the ``X-Goog-Api-Key`` header.  OAuth 2.0 (scope
``maps-platform.destinations``) can be substituted by passing a session
with the appropriate ``Authorization`` header already set.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import aiohttp
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry import shape as shapely_shape

_DESTINATIONS_URL = "https://geocode.googleapis.com/v4/geocode/destinations"

_FIELD_MASK = ",".join(
    [
        "destinations.primary.location",
        "destinations.primary.formattedAddress",
        "destinations.primary.place",
        "destinations.primary.structureType",
        "destinations.primary.displayPolygon",
    ]
)

BuildingShape = Union[Polygon, MultiPolygon]


class StructureType(str, Enum):
    """Precision level returned by the API in ``primary.structureType``.

    Values reflect how narrowly the address was resolved:
    ``BUILDING`` is the most precise (rooftop/plot level); ``POINT`` and
    ``GROUNDS`` indicate progressively coarser matches.
    """

    UNSPECIFIED = "STRUCTURE_TYPE_UNSPECIFIED"
    POINT = "POINT"
    SECTION = "SECTION"
    BUILDING = "BUILDING"
    GROUNDS = "GROUNDS"

    @property
    def is_building_level(self) -> bool:
        """True when the geocode is precise enough to identify a single building."""
        return self in (StructureType.BUILDING, StructureType.SECTION)


@dataclass
class GeocodeResult:
    latitude: float
    longitude: float
    location: Point
    building_shape: Optional[BuildingShape]
    formatted_address: str
    place_id: str
    structure_type: StructureType


class GoogleGeocoder:
    """Async geocoder backed by the Google Geocoding API v4 destinations endpoint.

    Use as an async context manager so the underlying ``aiohttp`` session is
    closed automatically::

        async with GoogleGeocoder(api_key) as geocoder:
            result = await geocoder.geocode("1600 Amphitheatre Pkwy, Mountain View")
            results = await geocoder.geocode_many(["address 1", "address 2"])
    """

    def __init__(
        self,
        api_key: str,
        *,
        session: Optional[aiohttp.ClientSession] = None,
        language_code: str = "en",
        region_code: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._external_session = session
        self._session: Optional[aiohttp.ClientSession] = session
        self._language_code = language_code
        self._region_code = region_code

    async def __aenter__(self) -> "GoogleGeocoder":
        if self._external_session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._external_session is None and self._session is not None:
            await self._session.close()
            self._session = None

    async def geocode(self, address: str) -> Optional[GeocodeResult]:
        """Resolve one address to lat/lon and building polygon."""
        assert self._session is not None, "Use GoogleGeocoder as an async context manager"

        body: Dict[str, Any] = {
            "addressQuery": {"addressQuery": address},
            "languageCode": self._language_code,
        }
        if self._region_code:
            body["regionCode"] = self._region_code

        headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": _FIELD_MASK,
            "Content-Type": "application/json",
        }

        async with self._session.post(
            _DESTINATIONS_URL, json=body, headers=headers
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        destinations = data.get("destinations")
        if not destinations:
            return None

        primary = destinations[0].get("primary", {})
        loc = primary.get("location", {})
        lat = float(loc.get("latitude", 0))
        lng = float(loc.get("longitude", 0))
        place = primary.get("place", "")

        return GeocodeResult(
            latitude=lat,
            longitude=lng,
            location=Point(lng, lat),
            building_shape=_parse_geojson(primary.get("displayPolygon")),
            formatted_address=primary.get("formattedAddress", ""),
            place_id=place[len("places/"):] if place.startswith("places/") else place,
            structure_type=_parse_structure_type(primary.get("structureType")),
        )

    async def geocode_many(
        self, addresses: Sequence[str]
    ) -> List[Optional[GeocodeResult]]:
        """Geocode a batch of addresses concurrently (one destinations request each)."""
        return list(await asyncio.gather(*[self.geocode(a) for a in addresses]))


def _parse_structure_type(value: Optional[str]) -> StructureType:
    try:
        return StructureType(value) if value else StructureType.UNSPECIFIED
    except ValueError:
        return StructureType.UNSPECIFIED


def _parse_geojson(
    geojson: Optional[Dict[str, Any]]
) -> Optional[BuildingShape]:
    """Convert a GeoJSON dict returned by ``displayPolygon`` to a shapely geometry."""
    if not geojson:
        return None
    try:
        geom = shapely_shape(geojson)
    except Exception:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    return None
