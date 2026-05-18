"""Async geocoder using the Google Geocoding API v4.

Two endpoints are called concurrently for each address:

- ``GET  https://geocode.googleapis.com/v4/geocode/address/<address>``
  → lat/lon, granularity (ROOFTOP / RANGE_INTERPOLATED / …), formatted address, place ID.

- ``POST https://geocode.googleapis.com/v4/geocode/destinations``
  → building ``displayPolygon`` (GeoJSON → shapely) and ``structureType``.

``geocode_many`` resolves all addresses concurrently via ``asyncio.gather``,
bounded by ``max_concurrency`` to avoid exhausting API rate limits.
Authentication uses the ``X-Goog-Api-Key`` header.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import quote

import aiohttp
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry import shape as shapely_shape

_GEOCODE_ADDRESS_URL = "https://geocode.googleapis.com/v4/geocode/address"
_DESTINATIONS_URL = "https://geocode.googleapis.com/v4/geocode/destinations"

_ADDRESS_FIELD_MASK = ",".join(
    [
        "results.location",
        "results.granularity",
        "results.formattedAddress",
        "results.placeId",
    ]
)

_DESTINATIONS_FIELD_MASK = ",".join(
    [
        "destinations.primary.structureType",
        "destinations.primary.displayPolygon",
    ]
)

BuildingShape = Union[Polygon, MultiPolygon]


class Granularity(str, Enum):
    """Coordinate precision returned by the geocode/address endpoint."""

    UNSPECIFIED = "GRANULARITY_UNSPECIFIED"
    ROOFTOP = "ROOFTOP"
    RANGE_INTERPOLATED = "RANGE_INTERPOLATED"
    GEOMETRIC_CENTER = "GEOMETRIC_CENTER"
    APPROXIMATE = "APPROXIMATE"


class StructureType(str, Enum):
    """Place structure returned by the destinations endpoint in ``primary.structureType``.

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
    granularity: Granularity
    structure_type: StructureType


class GoogleGeocoder:
    """Async geocoder backed by the Google Geocoding API v4.

    Use as an async context manager so the underlying ``aiohttp`` session is
    closed automatically::

        async with GoogleGeocoder(api_key) as geocoder:
            result = await geocoder.geocode("1600 Amphitheatre Pkwy, Mountain View")
            results = await geocoder.geocode_many(["address 1", "address 2"])

    Args:
        api_key: Google Maps Platform API key.
        proxy: Optional HTTP/HTTPS proxy URL (e.g. ``"http://proxy.corp:8080"``).
        max_concurrency: Maximum number of in-flight API requests at once.
            Applies across both endpoints when ``fetch_building_shape`` is
            enabled (each address then makes two requests).
        fetch_building_shape: When ``True`` also calls the destinations
            endpoint to populate ``building_shape`` and ``structure_type``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        proxy: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        language_code: str = "en",
        region_code: Optional[str] = None,
        fetch_building_shape: bool = False,
        max_concurrency: int = 50,
    ) -> None:
        self._api_key = api_key
        self._proxy = proxy
        self._external_session = session
        self._session: Optional[aiohttp.ClientSession] = session
        self._language_code = language_code
        self._region_code = region_code
        self._fetch_building_shape = fetch_building_shape
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self) -> "GoogleGeocoder":
        if self._external_session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._external_session is None and self._session is not None:
            await self._session.close()
            self._session = None

    async def geocode(self, address: str) -> Optional[GeocodeResult]:
        """Resolve one address; calls geocode/address and destinations concurrently."""
        assert self._session is not None, "Use GoogleGeocoder as an async context manager"

        async with self._semaphore:
            if self._fetch_building_shape:
                geocode_data, destinations_data = await asyncio.gather(
                    self._fetch_geocode_address(address),
                    self._fetch_destinations(address),
                )
            else:
                geocode_data = await self._fetch_geocode_address(address)
                destinations_data = None

        if geocode_data is None:
            return None

        building_shape: Optional[BuildingShape] = None
        structure_type = StructureType.UNSPECIFIED
        if destinations_data:
            primary = destinations_data[0].get("primary", {})
            building_shape = _parse_geojson(primary.get("displayPolygon"))
            structure_type = _parse_structure_type(primary.get("structureType"))

        loc = geocode_data.get("location", {})
        return GeocodeResult(
            latitude=float(loc.get("latitude", 0)),
            longitude=float(loc.get("longitude", 0)),
            location=Point(loc.get("longitude", 0), loc.get("latitude", 0)),
            building_shape=building_shape,
            formatted_address=geocode_data.get("formattedAddress", ""),
            place_id=geocode_data.get("placeId", ""),
            granularity=_parse_granularity(geocode_data.get("granularity")),
            structure_type=structure_type,
        )

    async def geocode_many(
        self, addresses: Sequence[str]
    ) -> List[Optional[GeocodeResult]]:
        """Geocode a batch of addresses, bounded by ``max_concurrency``."""
        return list(await asyncio.gather(*[self.geocode(a) for a in addresses]))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _common_headers(self, field_mask: str) -> Dict[str, str]:
        return {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": field_mask,
        }

    async def _fetch_geocode_address(self, address: str) -> Optional[Dict[str, Any]]:
        """Call geocode/address → returns the first GeocodeResult dict or None."""
        assert self._session is not None
        params: Dict[str, str] = {"languageCode": self._language_code}
        if self._region_code:
            params["regionCode"] = self._region_code

        url = f"{_GEOCODE_ADDRESS_URL}/{quote(address, safe='')}"
        async with self._session.get(
            url, params=params, headers=self._common_headers(_ADDRESS_FIELD_MASK),
            proxy=self._proxy,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        results = data.get("results")
        return results[0] if results else None

    async def _fetch_destinations(self, address: str) -> Optional[List[Dict[str, Any]]]:
        """Call destinations → returns the destinations list or None."""
        assert self._session is not None
        body: Dict[str, Any] = {
            "addressQuery": {"addressQuery": address},
            "languageCode": self._language_code,
        }
        if self._region_code:
            body["regionCode"] = self._region_code

        headers = {**self._common_headers(_DESTINATIONS_FIELD_MASK), "Content-Type": "application/json"}
        async with self._session.post(
            _DESTINATIONS_URL, json=body, headers=headers, proxy=self._proxy,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return data.get("destinations") or None


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------


def _parse_granularity(value: Optional[str]) -> Granularity:
    try:
        return Granularity(value) if value else Granularity.UNSPECIFIED
    except ValueError:
        return Granularity.UNSPECIFIED


def _parse_structure_type(value: Optional[str]) -> StructureType:
    try:
        return StructureType(value) if value else StructureType.UNSPECIFIED
    except ValueError:
        return StructureType.UNSPECIFIED


def _parse_geojson(geojson: Optional[Dict[str, Any]]) -> Optional[BuildingShape]:
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
