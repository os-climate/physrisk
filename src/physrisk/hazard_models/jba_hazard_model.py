import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import aiohttp
import numpy as np
from physrisk.data.hazard_data_provider import HazardDataProvider, ScenarioYear
from physrisk.kernel.hazard_model import (
    HazardDataFailedResponse,
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import (
    CoastalInundation,
    PluvialInundation,
    RiverineInundation,
)

from physrisk.data.geocode import Geocoder
from physrisk.utils.event_loop import get_loop, run
from physrisk.hazard_models.credentials_provider import (
    CredentialsProvider,
    EnvCredentialsProvider,
)
from physrisk.hazard_models.hazard_cache import H3BasedCache

logger = logging.getLogger(__name__)


class Indicator(NamedTuple):
    hazard_type: str
    indicator_id: str


class ItemType(str, Enum):
    request = "request"
    response = "response"


class JBACacheKey(NamedTuple):
    # JBA responses for a given lat/lon contain all hazards but a single scenario, hence the key comprises:
    jba_scenario: str  # the JBA combination of scenario and year
    spatial_key: str


class RequestKey(NamedTuple):
    country_code: str  # 2 letter code
    # jba_scenario: str


@dataclass
class APIRequest:
    spatial_keys: Sequence[str]  # spatial keys for latitudes and longitudes in order
    latitudes: Sequence[float]
    longitudes: Sequence[float]
    country_code: str
    location_cache_keys: Dict[
        str, List[JBACacheKey]
    ]  # for each spatial_keys list of cache keys that is requested

    # this is requested for all jba_scenarios systematically
    def request_count(self):
        return len(self.cache_keys)


@dataclass
class RequestWeights:
    request: HazardDataRequest
    weights: List[Tuple[JBACacheKey, float]]


class JBAHazardModel(HazardModel):
    def __init__(
        self,
        cache_store: H3BasedCache,
        credentials: Optional[CredentialsProvider] = None,
        geocoder: Optional[Geocoder] = None,
        max_requests: int = 5,
        cmip: int = 6,
        batch_size: int = 100,
        restrict_coverage: bool = False,
        only_request_required: bool = False,
    ):
        """JBAHazardModel retrieves data via the JBA API.
        https://api.jbarisk.com/docs/index.html
        Note that JBA requests are changed per location, therefore a good policy is to request all scenarios (e.g. SSPs)
        and years likely to be needed and cache against future need. This is the default but can be turned off via:
        only_request_required.

        Args:
            cache_store (H3BasedCache): Results caching store.
            credentials (Optional[CredentialsProvider], optional): Credentials provider. Defaults to None.
            geocoder (Optional[Geocoder], optional): Geocoder (needed to identify country in current JBA API version).
                Defaults to None.
            max_requests (int, optional): Maximum number of requests permitted; if exceeded an exception is raised.
                Defaults to 5.
            cmip (int, optional): Can take values 5 (CMIP5, i.e. RCPs) or 6 (CMIP6, i.e. SSPs).
                Defaults to 6.
            batch_size (int, optional): Number of spatial locations in each JBA API call.
                Defaults to 100.
            restrict_coverage (bool, optional): If True, restrict the number of scenarios for performance reasons.
            only_request_required (bool, optional): If True, request only scenarios and years required (i.e. not extra ones to cache)
        """
        self.cache_store = cache_store
        self.credentials = (
            credentials if credentials is not None else EnvCredentialsProvider()
        )
        self.geocoder = geocoder if geocoder is not None else Geocoder()
        self.indicators = set(
            [
                Indicator(hazard_type="RiverineInundation", indicator_id="flood_depth"),
                Indicator(hazard_type="PluvialInundation", indicator_id="flood_depth"),
                Indicator(hazard_type="CoastalInundation", indicator_id="flood_depth"),
                Indicator(hazard_type="RiverineInundation", indicator_id="flood_sop"),
                Indicator(hazard_type="PluvialInundation", indicator_id="flood_sop"),
                Indicator(hazard_type="CoastalInundation", indicator_id="flood_sop"),
            ]
        )
        self.year_ranges = {
            2030: "2016-2045",
            2040: "2026-2055",
            2050: "2036-2065",
            2080: "2066-2095",
            2100: "2086-2115",
        }
        # for any API call we request these 15 scenarios
        self.pillar_years = [
            2030,
            2050,
            2080,
        ]  # we could have included 2040 and 2100, but removed for better performance
        self.historical_year = 2025
        self.lock = Lock()
        self.max_requests = max_requests
        self.cmip = cmip
        self.batch_size = batch_size
        self.jba_scenarios = [
            self.jba_scenario(s, y)
            for y in self.pillar_years
            for s in (
                ["ssp245", "ssp585"]
                if restrict_coverage
                else ["ssp126", "ssp245", "ssp585"]
            )
        ]
        self.restrict_coverage = restrict_coverage
        self.only_request_required = only_request_required  # here in case needed in future. If true, only scenarios that are explicitly asked for are
        # included in API request; improves performance.

    def check_requests(self, requests: Sequence[HazardDataRequest]):
        if any(
            r
            for r in requests
            if Indicator(
                hazard_type=r.hazard_type.__name__, indicator_id=r.indicator_id
            )
            not in self.indicators
        ):
            raise ValueError("invalid request")

    def get_hazard_data(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        # noqa:C90
        with self.lock:
            # why don't we want this to be accessed by more than one thread at the same time?
            # 1) JBA API returns Riverine and Pluvial hazards at the same time. We want to make sure that
            # we get these and cache just once: otherwise a risk that we request the same data multiple times!
            # 2) We are already maxing out the number of requests to JBA using async for a single thread.
            if not self.geocoder:
                self.geocoder = Geocoder()
            cache_key_country: Dict[JBACacheKey, str] = {}  # cache item to requests
            request_groups: Dict[RequestKey, List[JBACacheKey]] = defaultdict(
                list
            )  # request to cache items
            self.check_requests(requests)
            # some deviations from 2-letter country codes:
            country_mapping = {
                "ES-ML": "ES",  # Melilla as ES
                "GG": "GB",  # Guernsey uses GB map
                "HK": "CN",
                "IE": "IE30",
                "JE": "FR5C",  # Jersey uses France map
                "MC": "FR5C",  # Monaco uses France map
                "NI": "NIC",  # Nicaragua uses NIC
                "FR": "FR5C",  # France 5m model including coastal inundation
                "AU": "AUC",
            }  # Australian model including coastal inundation
            result: MutableMapping[HazardDataRequest, HazardDataResponse] = {}
            # group requests by common location
            requests_by_location: Dict[str, List[HazardDataRequest]] = defaultdict(list)
            all_years: set[int] = set()
            for item in requests:
                spatial_key = self.cache_store.spatial_key(
                    item.latitude, item.longitude
                )
                requests_by_location[spatial_key].append(item)
                if item.scenario != "historical":
                    all_years.add(item.year)
            # JBA requires a 2-letter country code per request (at time of writing),
            # so it is necessary to geocode.
            lats, lons = (
                [r[0].latitude for r in requests_by_location.values()],
                [r[0].longitude for r in requests_by_location.values()],
            )
            countries = [
                country_mapping.get(c, c)
                for c in self.geocoder.get_countries(lats, lons)
            ]
            # note a single cache entry can provide information for multiple requests, because each entry contains
            # information about different hazards, or because points are close.
            # for interpolation, the list of pillar years for different requested years is calculated
            # ahead of time: e.g. 2036 needs 2030 and 2050 pillars.
            requested_years = sorted(list(all_years))
            weights = HazardDataProvider._weights(
                "ssp", self.pillar_years, requested_years, self.historical_year
            )
            weights_histo = HazardDataProvider._weights(
                "historical", self.pillar_years, requested_years, self.historical_year
            )
            pillar_years_lookup = {k.year: v for k, v in weights.items()}
            pillar_years_lookup[-1] = weights_histo[ScenarioYear("historical", -1)]
            cache_keys: set[JBACacheKey] = (
                set()
            )  # the set of cache keys to be requested (spatial key, year and scenario)
            req_weights_set: List[
                RequestWeights
            ] = []  # for each request the linear combination of cache keys required (interpolating)
            for reqs, country in zip(requests_by_location.values(), countries):
                for req in reqs:
                    req_weights: List[Tuple[JBACacheKey, float]] = []
                    for weight in pillar_years_lookup[
                        -1 if req.scenario == "historical" else req.year
                    ].weights:
                        cache_key = JBACacheKey(
                            jba_scenario=self.jba_scenario(
                                req.scenario, weight[0].year, country
                            ),
                            spatial_key=self.cache_store.spatial_key(
                                req.latitude, req.longitude
                            ),
                        )
                        cache_key_country[cache_key] = country
                        cache_keys.add(cache_key)
                        req_weights.append((cache_key, weight[1]))
                    req_weights_set.append(RequestWeights(req, req_weights))
            # requests are grouped by country
            for cache_key in cache_keys:
                country_code = cache_key_country[cache_key]
                request_groups[RequestKey(country_code=country_code)].append(cache_key)
            access_token = self.credentials.jba_access_key()
            api_requests: List[APIRequest] = []
            # contains the raw results for all cache keys, first populated by looking in cache
            # and then by making API calls if needed.
            cached_responses: Dict[JBACacheKey, Dict] = {}
            for request_key, group_cache_keys in request_groups.items():
                # process anything that can be sourced from the cache and identify extra API requests needed
                group_cached_responses, api_requests_batch = (
                    self._identify_api_requests(
                        request_key, group_cache_keys, requests_by_location
                    )
                )
                cached_responses.update(group_cached_responses)
                api_requests.extend(api_requests_batch)
            # if there are extra API requests, make these (in parallel) and process to get results
            n_requests = len([k for r in api_requests for k in r.spatial_keys])
            logger.info(f"{n_requests} API requests total")
            batches = [len(r.spatial_keys) for r in api_requests]
            logger.info(f"{len(api_requests)} batches of requests of size ({batches})")
            if n_requests > self.max_requests:
                raise ValueError(
                    f"would make {n_requests} requests to JBA API, more than {self.max_requests} maximum. "
                    "provider_max_requests may be set incorrectly."
                )
            cached_responses.update(
                self._process_api_requests(api_requests, access_token)
            )
            for req_weight in req_weights_set:
                resps = [
                    self._process_response(
                        req_weight.request, cached_responses.get(key, {"stats": None})
                    )
                    for key, _ in req_weight.weights
                ]
                if len(resps) == 1:
                    result[req_weight.request] = resps[0]
                elif len(resps) == 2:
                    if not isinstance(
                        resps[0], HazardEventDataResponse
                    ) or not isinstance(resps[1], HazardEventDataResponse):
                        result[req_weight.request] = HazardDataFailedResponse(
                            ValueError("no data returned")
                        )
                        continue
                    result[req_weight.request] = HazardEventDataResponse(
                        resps[0].return_periods,
                        resps[0].intensities * req_weight.weights[0][1]
                        + resps[1].intensities * req_weight.weights[1][1],
                        units="m",
                        path="jba",
                    )
            failures = [
                r for r in result.values() if isinstance(r, HazardDataFailedResponse)
            ]
            if any(failures):
                logger.error(
                    f"{len(failures)} errors in JBA batch (logs limited to first 3)"
                )
                errors = (str(i.error) for i in failures)
                for _ in range(min(len(failures), 3)):
                    logger.error(next(errors))
            return result

    def jba_cache_id(self, key: JBACacheKey):
        # for dealing with buffer > 10 m, we have two options
        # 1) Change the spatial key resolution to match the buffer size
        # 2) Add a buffer part to the key
        return f"jba/{key.jba_scenario}/{key.spatial_key}"

    def jba_request_id(self, spatial_key: str):
        # requests to JBA API are made for all scenarios
        return f"jba/{spatial_key}"

    def jba_scenario(self, scenario: str, year: int, country: str = ""):
        if scenario == "historical":
            return "historical"
        if self.cmip == 5:
            if scenario == "rcp2p6" or scenario == "ssp126":
                prefix = "rcp26"
            elif scenario == "rcp8p5" or scenario == "ssp585":
                prefix = "rcp85"
            elif scenario == "rcp45" or scenario == "ssp245":
                prefix = "rcp45"
            else:
                raise ValueError(
                    f"scenario {scenario} not supported by JBA Risk Management API"
                )
        else:
            if scenario not in ["ssp126", "ssp245", "ssp585"]:
                raise ValueError(
                    f"scenario {scenario} not supported by JBA Risk Management API"
                )
            prefix = scenario
        if self.cmip == 6:
            try:
                range = self.year_ranges[year]
            except Exception:
                raise ValueError(
                    f"scenario {scenario} not supported by JBA Risk Management API"
                )
        else:
            gb_code = country in ["GB", "NI", "ROI"] and "rcp" in scenario
            if year == 2030:
                range = "2031-2035" if gb_code else "2016-2045"
            elif year == 2040:
                range = "2041-2045" if gb_code else "2026-2055"
            elif year == 2050:
                range = "2051-2055" if gb_code else "2036-2065"
            elif year == 2080:
                range = "2081-2085" if gb_code else "2066-2095"
            else:
                raise ValueError(
                    f"scenario {scenario} not supported by JBA Risk Management API"
                )
        return prefix + "_" + range

    async def flood_depth(
        self, api_request: APIRequest, access_token: str, session: aiohttp.ClientSession
    ):
        if len(api_request.spatial_keys) == 0:
            return {}
        if self.credentials.jba_api_disabled():
            logger.error("JBA requests made but API calls disabled")
            raise ValueError("JBA requests made but API calls disabled")
        req_id_to_keys: Dict[str, List[JBACacheKey]] = defaultdict(list)
        for k in api_request.spatial_keys:
            req_id_to_keys[self.jba_request_id(k)] = [
                JBACacheKey(s, k) for s in (["historical"] + self.jba_scenarios)
            ]
        req_ids = list(req_id_to_keys.keys())
        country_code = api_request.country_code  # e.g. CN or FR
        # https://api.jbarisk.com/docs/1.2/index.html
        url = "https://api.jbarisk.com/flooddepths/" + country_code
        request = {
            "country_code": country_code,
            "geometries": [
                {"id": id, "wkt_geometry": f"POINT({lon} {lat})", "buffer": 10}
                for id, lat, lon in zip(
                    req_ids, api_request.latitudes, api_request.longitudes
                )
            ],
        }
        params = {
            "CSTHs": ",".join(
                self.jba_scenarios
                if not self.only_request_required
                else list(
                    set(
                        k.jba_scenario
                        for v in api_request.location_cache_keys.values()
                        for k in v
                    )
                )
            ),
            "baseline": "true",
        }
        logger.debug("JBA request URL: " + url)
        logger.debug("JBA request payload: " + json.dumps(request))
        headers = {"Authorization": f"Basic {access_token}"}
        proxies = self.credentials.proxies()
        try:
            response_dict = None
            async with session.post(
                url=url,
                json=request,
                params=params,
                proxy=proxies["https"],
                headers=headers,  # , ssl=False can be used *in dev* if SSL verify issue
            ) as response:
                response_dict = await response.json()
                logger.debug("JBA response: " + json.dumps(response_dict))
                if response.status == 200:
                    try:
                        # we expect results for all self.jba_scenarios and "stats"
                        result = {}
                        for item in response_dict:
                            for cache_key in req_id_to_keys[item["id"]]:
                                key = (
                                    "stats"
                                    if cache_key.jba_scenario == "historical"
                                    else cache_key.jba_scenario
                                )
                                result[cache_key] = {"stats": item[key]}
                        return result
                    except Exception:
                        ids = ",".join(id for id in req_ids)
                        logging.error(
                            f"Unexpected response for URL {url} (request IDs: {ids})"
                        )
                        # logging.exception("") # do not include exception info
                        # as we assume useful info is in response.
                        return str(response_dict)
                else:
                    # Request failed
                    logging.error(f"Response status {response.status}")
                    return str(response_dict)
        except Exception:
            # proxy or authentication errors would be expected to come here, hence
            # use of logging.exception to ensure exception info is included.
            logging.exception("JBA API raised exception")
            return (
                "JBA API request failed"
                if response_dict is None
                else str(response_dict)
            )

    def _identify_api_requests(
        self,
        request_key: RequestKey,
        cache_keys: Iterable[JBACacheKey],
        requests_by_location: Dict[str, List[HazardDataRequest]],
    ):
        """Process any results that can be sourced from the cache and identify the
        requests to the API that are needed."""
        batches: List[APIRequest] = []
        cache_ids = [self.jba_cache_id(k) for k in cache_keys]
        # first checks cache
        cached_responses: Dict[JBACacheKey, Dict] = {}
        for cache_key, item in zip(cache_keys, self.cache_store.getitems(cache_ids)):
            if item is not None:
                value = json.loads(item)
                if value["stats"] is not None:
                    cached_responses[cache_key] = value
        # we need to create requests for anything not in cache
        # req_keys_all = [k for k in cache_keys if k not in cached_responses]
        location_cache_keys: Dict[str, List[JBACacheKey]] = defaultdict(list)
        missing_cache_keys = [k for k in cache_keys if k not in cached_responses]
        for k in missing_cache_keys:
            location_cache_keys[k.spatial_key].append(k)
        req_keys_all = list(location_cache_keys.keys())
        # but batch up for requesting
        batch_size = self.batch_size
        req_key_batches = [
            req_keys_all[i : min(i + batch_size, len(req_keys_all))]
            for i in range(0, len(req_keys_all), batch_size)
        ]
        for req_keys in req_key_batches:
            first_req = [requests_by_location[k][0] for k in req_keys]
            lats = [r.latitude for r in first_req]
            lons = [r.longitude for r in first_req]
            batches.append(
                APIRequest(
                    location_cache_keys=location_cache_keys,
                    spatial_keys=req_keys,
                    latitudes=lats,
                    longitudes=lons,
                    country_code=request_key.country_code,
                )
            )
        return cached_responses, batches

    def _process_api_requests(
        self,
        api_requests: Sequence[APIRequest],
        access_token: str,
        concurrent_requests: int = 8,
    ):
        """Make required requests to the API, updating cache and process responses."""
        check_total = 0
        # the code is async in order to run a large number of API requests in parallel
        # and may be called from within a thread-pool
        # we could create a new event loop, e.g. via (later closing loop):
        # loop = asyncio.new_event_loop()
        # loop.run_until_complete(gather_requests(api_requests))
        # but we prefer to use a single loop in away analagous to accessing Zarr data
        # which uses the AsyncFileSystem of fsspec.
        loop = get_loop()
        cached_responses = {}
        with aiohttp.TCPConnector(
            limit_per_host=concurrent_requests, loop=loop
        ) as conn:
            reruns: List[APIRequest] = []

            async def gather_requests(api_requests: Sequence[APIRequest]):
                semaphore = asyncio.Semaphore(concurrent_requests)
                async with aiohttp.ClientSession(
                    connector=conn, connector_owner=False
                ) as session:

                    async def request_single(request: APIRequest):
                        nonlocal check_total
                        async with semaphore:
                            responses = await self.flood_depth(
                                request, access_token, session
                            )
                            check_total += len(request.spatial_keys)
                            if isinstance(responses, str):
                                # a string indicates an error
                                reruns.append(request)
                            else:
                                self.cache_store.setitems(
                                    {
                                        self.jba_cache_id(k): json.dumps(v)
                                        for k, v in responses.items()
                                    }
                                )
                                cached_responses.update(responses)
                            # if check_total // 500 != (check_total - len(request.spatial_keys)) // 500:
                            #    logger.info(
                            #        f"Total of {check_total} spatial location requests made"
                            #    )

                    await asyncio.gather(*(request_single(req) for req in api_requests))

            run(gather_requests(api_requests), loop)
            # for failed batches we run again, but as single requests
            # this is needed because the JBA API will fail for all locations
            # if a single one is out of bounds (e.g. off-shore wind farm)
            single_api_requests = [
                APIRequest(
                    spatial_keys=[spatial_key],
                    latitudes=[lat],
                    longitudes=[lon],
                    country_code=rerun.country_code,
                    location_cache_keys=rerun.location_cache_keys,
                )
                for rerun in reruns
                for spatial_key, lat, lon in zip(
                    rerun.spatial_keys, rerun.latitudes, rerun.longitudes
                )
            ]
            if len(single_api_requests) > 0:
                run(gather_requests(single_api_requests), loop)
            logger.info(f"Check: {check_total} requests made")
            logger.info(f"Check: {len(single_api_requests)} reruns")
        return cached_responses

    def _process_response(self, request: HazardDataRequest, response: Dict):
        if request.hazard_type == RiverineInundation:
            tag = "FLRF_U"
        elif request.hazard_type == PluvialInundation:
            tag = "FLSW_U"
        elif request.hazard_type == CoastalInundation:
            tag = "STSU_U"
        else:
            raise ValueError("unexpected hazard type")
        if response["stats"] is None:
            return HazardDataFailedResponse(ValueError("no data returned"))
        elif request.indicator_id == "flood_sop":
            sop = response["stats"].get("FLRF_U", {}).get("sop", 0)
            return HazardParameterDataResponse(
                [sop, sop], units="years", path="jba"
            )  # min and max: in this case just a single value
        elif request.indicator_id == "flood_depth":
            return_periods: List[float] = []
            intens: List[float] = []
            for key, value in response["stats"].get(tag, {}).items():
                assert isinstance(key, str)
                if key.startswith("rp_"):
                    return_periods.append(float(key[3:]))
                    intens.append(value["max" + key[3:]])
            return HazardEventDataResponse(
                np.array(return_periods),
                np.array(intens),
                units="m",
                path="jba",
            )
        else:
            raise NotImplementedError()

    # scenarios = self.climate_change_scenarios("CN", access_token)
    # def climate_change_scenarios(self, country_code: str, access_token: str):
    #     url = f"https://api.jbarisk.com/flooddepths/ccscenarios/{country_code}"
    # ...
