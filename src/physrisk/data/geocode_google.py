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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import quote

import aiohttp
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import shape as shapely_shape

_GEOCODE_ADDRESS_URL = "https://geocode.googleapis.com/v4/geocode/address"
_DESTINATIONS_URL = "https://geocode.googleapis.com/v4/geocode/destinations"

_ADDRESS_FIELD_MASK = ",".join(
    [
        "results.location",
        "results.granularity",
        "results.formattedAddress",
        "results.placeId",
        "results.types",
    ]
)

_DESTINATIONS_FIELD_MASK = ",".join(
    [
        "destinations.primary.structureType",
        "destinations.primary.displayPolygon",
    ]
)

BuildingShape = Union[Polygon, MultiPolygon]


class _RateLimiter:
    """Async token-bucket rate limiter.

    Tokens accumulate at ``rate`` per second up to a burst cap of ``rate``.
    The lock is released before sleeping so that multiple waiters can proceed
    as soon as tokens become available rather than queuing strictly.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens: float = rate
        self._updated_at: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                loop = asyncio.get_running_loop()
                now = loop.time()
                if self._updated_at == 0.0:
                    self._updated_at = now
                elapsed = now - self._updated_at
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                self._updated_at = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)


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
    building_shape: Optional[BuildingShape]
    formatted_address: str
    place_id: str
    granularity: Granularity
    structure_type: StructureType
    types: List[str] = field(default_factory=list)
    candidate_count: int = field(default=1)


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
        region_code: Default CLDR region code applied to every request (e.g. ``"GB"``).
            Can be overridden per-call via the ``region_code`` argument on
            ``geocode`` / ``geocode_many``.
        requests_per_second: Maximum ``geocode`` calls dispatched per second
            across the whole instance (token-bucket algorithm).  Each call
            counts once regardless of whether ``fetch_building_shape`` is
            enabled.  Set to ``None`` to disable throttling.
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
        requests_per_second: Optional[float] = 25.0,
    ) -> None:
        self._api_key = api_key
        self._proxy = proxy
        self._external_session = session
        self._session: Optional[aiohttp.ClientSession] = session
        self._language_code = language_code
        self._region_code = region_code
        self._fetch_building_shape = fetch_building_shape
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_limiter = (
            _RateLimiter(requests_per_second) if requests_per_second else None
        )

    async def __aenter__(self) -> "GoogleGeocoder":
        if self._external_session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._external_session is None and self._session is not None:
            await self._session.close()
            self._session = None

    async def geocode(
        self, address: str, *, region_code: Optional[str] = None
    ) -> Optional[GeocodeResult]:
        """Resolve one address; calls geocode/address and destinations concurrently.

        Args:
            address: Free-form address string.
            region_code: CLDR region code for this request; overrides the
                instance-level default when provided.
        """
        assert self._session is not None, (
            "Use GoogleGeocoder as an async context manager"
        )

        effective_region = region_code if region_code is not None else self._region_code

        if self._rate_limiter:
            await self._rate_limiter.acquire()
        async with self._semaphore:
            if self._fetch_building_shape:
                address_results, destinations_data = await asyncio.gather(
                    self._fetch_geocode_address(address, effective_region),
                    self._fetch_destinations(address, effective_region),
                )
            else:
                address_results = await self._fetch_geocode_address(
                    address, effective_region
                )
                destinations_data = None

        if not address_results:
            return None

        geocode_data = address_results[0]

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
            building_shape=building_shape,
            formatted_address=geocode_data.get("formattedAddress", ""),
            place_id=geocode_data.get("placeId", ""),
            granularity=_parse_granularity(geocode_data.get("granularity")),
            structure_type=structure_type,
            types=geocode_data.get("types") or [],
            candidate_count=len(address_results),
        )

    async def geocode_many(
        self,
        addresses: Sequence[str],
        *,
        region_codes: Optional[Sequence[Optional[str]]] = None,
    ) -> List[Optional[GeocodeResult]]:
        """Geocode a batch of addresses, bounded by ``max_concurrency``.

        Args:
            addresses: Sequence of free-form address strings.
            region_codes: Per-address CLDR region codes, aligned with
                ``addresses``.  Each entry overrides the instance-level default
                for that address; use ``None`` in a slot to fall back to the
                instance default.  Omit the argument entirely to use the
                instance default for every address.
        """
        if region_codes is not None and len(region_codes) != len(addresses):
            raise ValueError(
                f"region_codes length ({len(region_codes)}) must match addresses length ({len(addresses)})"
            )
        codes: Sequence[Optional[str]] = (
            region_codes if region_codes is not None else [None] * len(addresses)
        )
        return list(
            await asyncio.gather(
                *[self.geocode(a, region_code=rc) for a, rc in zip(addresses, codes)]
            )
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _common_headers(self, field_mask: str) -> Dict[str, str]:
        return {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": field_mask,
        }

    async def _fetch_geocode_address(
        self, address: str, region_code: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Call geocode/address → returns the full results list or None."""
        assert self._session is not None
        params: Dict[str, str] = {"languageCode": self._language_code}
        if region_code:
            params["regionCode"] = region_code

        url = f"{_GEOCODE_ADDRESS_URL}/{quote(address, safe='')}"
        async with self._session.get(
            url,
            params=params,
            headers=self._common_headers(_ADDRESS_FIELD_MASK),
            proxy=self._proxy,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return data.get("results") or None

    async def _fetch_destinations(
        self, address: str, region_code: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Call destinations → returns the destinations list or None."""
        assert self._session is not None
        body: Dict[str, Any] = {
            "addressQuery": {"addressQuery": address},
            "languageCode": self._language_code,
        }
        if region_code:
            body["regionCode"] = region_code

        headers = {
            **self._common_headers(_DESTINATIONS_FIELD_MASK),
            "Content-Type": "application/json",
        }
        async with self._session.post(
            _DESTINATIONS_URL,
            json=body,
            headers=headers,
            proxy=self._proxy,
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
