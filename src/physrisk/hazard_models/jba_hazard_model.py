import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Dict, List, Mapping, MutableMapping, NamedTuple, Optional, Sequence

import aiohttp
import numpy as np
from physrisk.kernel.hazard_model import (
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import PluvialInundation, RiverineInundation

from physrisk.data.geocode import Geocoder
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
    jba_scenario: str


@dataclass
class APIRequest:
    cache_keys: Sequence[JBACacheKey]
    latitudes: Sequence[float]
    longitudes: Sequence[float]
    country_code: str
    jba_scenario: str

    def request_count(self):
        return len(self.cache_keys)


class JBAHazardModel(HazardModel):
    def __init__(
        self,
        cache_store: H3BasedCache,
        credentials: Optional[CredentialsProvider] = None,
        geocoder: Optional[Geocoder] = None,
        max_requests=5,
    ):
        self.cache_store = cache_store
        self.credentials = (
            credentials if credentials is not None else EnvCredentialsProvider()
        )
        self.geocoder = geocoder if geocoder is not None else Geocoder()
        self.indicators = set(
            [
                Indicator(hazard_type="RiverineInundation", indicator_id="flood_depth"),
                Indicator(hazard_type="PluvialInundation", indicator_id="flood_depth"),
                Indicator(hazard_type="RiverineInundation", indicator_id="flood_sop"),
                Indicator(hazard_type="PluvialInundation", indicator_id="flood_sop"),
            ]
        )
        self.lock = Lock()
        self.max_requests = max_requests

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
            # 1) JBA aPI returns Riverine and Pluvial hazards at the same time. We want to make sure that
            # we get these and cache just once: otherwise a risk that we request the same data multiple times!
            # 2) We are already maxing out the number of requests to JBA using async for a single thread.

            if not self.geocoder:
                self.geocoder = Geocoder()
            groups: Dict[JBACacheKey, List[HazardDataRequest]] = defaultdict(
                list
            )  # cache item to requests
            request_groups: Dict[RequestKey, List[JBACacheKey]] = defaultdict(
                list
            )  # request to cache items

            self.check_requests(requests)

            # a single cache entry can provide information for multiple requests, because each entry contains
            # information about different hazards, or points are close.
            for item in requests:
                jba_scenario = self.jba_scenario(item.scenario, item.year)
                cache_key = JBACacheKey(
                    jba_scenario=jba_scenario,
                    spatial_key=self.cache_store.spatial_key(
                        item.latitude, item.longitude
                    ),
                )
                groups[cache_key].append(item)

            # JBA requires a 2-letter country code per request (at time of writing),
            # so it is necessary to geocode the country and group.
            group_lats, group_lons = (
                [v[0].latitude for v in groups.values()],
                [v[0].longitude for v in groups.values()],
            )
            group_country_codes = self.geocoder.get_countries(group_lats, group_lons)

            for country_code, cache_key in zip(group_country_codes, groups.keys()):
                request_groups[
                    RequestKey(
                        country_code=country_code, jba_scenario=cache_key.jba_scenario
                    )
                ].append(cache_key)

            access_token = self.credentials.jba_access_key()
            api_requests: List[APIRequest] = []
            # for each group, process any cached data and get batches to run for non-cached
            result: MutableMapping[HazardDataRequest, HazardDataResponse] = {}
            for request_key, cache_keys in request_groups.items():
                # process anything that can be sourced from the cache and identify extra API requests needed
                results_batch, api_requests_batch = self._identify_api_requests(
                    request_key, cache_keys, groups
                )
                result.update(results_batch)
                api_requests.extend(api_requests_batch)

            # if there are extra API requests, make these (in parallel) and process to get results
            n_requests = len([k for r in api_requests for k in r.cache_keys])
            if n_requests > self.max_requests:
                raise ValueError(
                    f"would make {n_requests} requests to JBA API, more than {self.max_requests} maximum. \
                        provider_max_requests may be set incorrectly."
                )

            results_batch = self._process_api_requests(
                api_requests, groups, access_token
            )
            result.update(results_batch)

            return result

    def jba_cache_id(self, key: JBACacheKey):
        # for dealing with buffer > 10 m, we have two options
        # 1) Change the spatial key resolution to match the buffer size
        # 2) Add a buffer part to the key
        return f"jba/{key.jba_scenario}/{key.spatial_key}"

    def jba_scenario(self, scenario: str, year: int):
        if scenario == "historical":
            return "historical"

        if (
            scenario == "rcp8p5" or scenario == "ssp585"
        ):  # for now proxy pending availability true scnearios
            prefix = "rcp85"
        elif (
            scenario == "rcp45" or scenario == "ssp245"
        ):  # for now proxy pending availability true scnearios
            prefix = "rcp45"
        else:
            raise ValueError(
                f"scenario {scenario} not supported by JBA Risk Management API"
            )
        if year == 2030:
            range = "2016-2045"
        elif year == 2050:
            range = "2036-2065"
        elif year == 2080:
            range = "2066-2095"
        else:
            raise ValueError(
                f"scenario {scenario} not supported by JBA Risk Management API"
            )
        return prefix + "_" + range

    async def flood_depth(
        self, api_request: APIRequest, access_token: str, session: aiohttp.ClientSession
    ):
        if len(api_request.cache_keys) == 0:
            return {}

        if self.credentials.jba_api_disabled():
            logger.info("JBA requests made but API calls disabled")

        id_to_key = {self.jba_cache_id(k): k for k in api_request.cache_keys}
        req_ids = list(id_to_key.keys())

        country_code = api_request.country_code  # e.g. CN or FR
        url = (
            "https://api.jbarisk.com/flooddepths/" + country_code
            if api_request.jba_scenario == "historical"
            else f"https://api.jbarisk.com/flooddepths/{country_code}/scenario/{api_request.jba_scenario}"
        )

        request = {
            "country_code": country_code,
            "geometries": [
                {"id": id, "wkt_geometry": f"POINT({lon} {lat})", "buffer": 10}
                for id, lat, lon in zip(
                    req_ids, api_request.latitudes, api_request.longitudes
                )
            ],
        }
        logger.info("JBA request: " + json.dumps(request))
        headers = {"Authorization": f"Bearer {access_token}"}
        proxies = self.credentials.proxies()
        async with session.post(
            url=url, json=request, proxy=proxies["https"], headers=headers
        ) as response:  # verify=False
            response_dict = await response.json()
            logger.info("JBA response: " + json.dumps(response_dict))
            if response.status == 200:
                return {id_to_key[item["id"]]: item for item in response_dict}
            else:
                # Request failed
                return None

    def _identify_api_requests(
        self,
        request_key: RequestKey,
        cache_keys: Sequence[JBACacheKey],
        groups: Dict[JBACacheKey, List[HazardDataRequest]],
    ):
        """Process any results that can be sourced from the cache and identify the
        requests to the API that are needed."""
        batches: List[APIRequest] = []

        cache_ids = [self.jba_cache_id(k) for k in cache_keys]
        # first checks cache
        cached_responses: Dict[JBACacheKey, Dict] = {
            cache_key: json.loads(item)
            for cache_key, item in zip(cache_keys, self.cache_store.getitems(cache_ids))
            if item is not None
        }
        # we need to create requests for anything not in cache
        req_keys_all = [k for k in cache_keys if k not in cached_responses]
        # but batch up for requesting
        batch_size = 100  # 10
        req_key_batches = [
            req_keys_all[i : min(i + batch_size, len(req_keys_all))]
            for i in range(0, len(req_keys_all), batch_size)
        ]
        for req_keys in req_key_batches:
            lats = [groups[k][0].latitude for k in req_keys]
            lons = [groups[k][0].longitude for k in req_keys]
            batches.append(
                APIRequest(
                    cache_keys=req_keys,
                    latitudes=lats,
                    longitudes=lons,
                    country_code=request_key.country_code,
                    jba_scenario=request_key.jba_scenario,
                )
            )

        request_results = self._process_responses(groups, cached_responses)
        return request_results, batches

    def _process_api_requests(
        self,
        api_requests: Sequence[APIRequest],
        groups: Dict[JBACacheKey, List[HazardDataRequest]],
        access_token: str,
        concurrent_requests: int = 10,
    ):
        """Make required requests to the API, updating cache and process responses."""
        try:
            loop = asyncio.get_event_loop()
        except Exception:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        conn = aiohttp.TCPConnector(limit_per_host=concurrent_requests)
        results: Dict[HazardDataRequest, HazardDataResponse] = {}

        async def gather_requests(api_requests: Sequence[APIRequest]):
            semaphore = asyncio.Semaphore(concurrent_requests)
            session = aiohttp.ClientSession(connector=conn)

            async def request_single(request: APIRequest):
                async with semaphore:
                    responses = await self.flood_depth(request, access_token, session)
                    self.cache_store.setitems(
                        {
                            self.jba_cache_id(k): json.dumps(v)
                            for k, v in responses.items()
                        }
                    )
                    request_results = self._process_responses(groups, responses)
                    results.update(request_results)

            await asyncio.gather(*(request_single(req) for req in api_requests))

        # loop = asyncio.get_event_loop()
        loop.run_until_complete(gather_requests(api_requests))
        conn.close()
        return results

    def _process_responses(
        self,
        groups: Dict[JBACacheKey, List[HazardDataRequest]],
        cached_responses: Dict[JBACacheKey, Dict],
    ):
        result: MutableMapping[HazardDataRequest, HazardDataResponse] = {}
        for cache_key in cached_responses.keys():
            response = cached_responses[cache_key]
            for req in groups[cache_key]:
                if req.hazard_type == RiverineInundation:
                    tag = "FLRF_U"
                elif req.hazard_type == PluvialInundation:
                    tag = "FLSW_U"
                else:
                    raise ValueError("unexpected hazard type")
                if req.indicator_id == "flood_sop":
                    sop = response["stats"]["FLRF_U"].get("sop", 0)
                    resp: HazardDataResponse = HazardParameterDataResponse(
                        np.array([sop, sop])
                    )  # min and max: in this case just a single value
                elif req.indicator_id == "flood_depth":
                    return_periods: List[float] = []
                    intens: List[float] = []
                    for key, value in response["stats"][tag].items():
                        assert isinstance(key, str)
                        if key.startswith("rp_"):
                            return_periods.append(float(key[3:]))
                            intens.append(value["max" + key[3:]])
                    resp = HazardEventDataResponse(
                        np.array(return_periods), np.array(intens)
                    )
                result[req] = resp
        return result

    # scenarios = self.climate_change_scenarios("CN", access_token)
    # def climate_change_scenarios(self, country_code: str, access_token: str):
    #     url = f"https://api.jbarisk.com/flooddepths/ccscenarios/{country_code}"
    # ...
