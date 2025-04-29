import asyncio
import concurrent.futures
from collections import defaultdict
import logging
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazards import Hazard, IndicatorData, indicator_data

from ..kernel.hazard_model import (
    HazardDataFailedResponse,
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from .hazard_data_provider import HazardDataProvider, ScenarioYear, SourcePath, SourcePaths

logger = logging.getLogger(__name__)


class PregeneratedHazardModel(HazardModel):
    """Hazard event model that processes requests using EventProviders."""

    def __init__(
        self,
        hazard_data_providers: Dict[Type[Hazard], HazardDataProvider],
    ):
        self.hazard_data_providers = hazard_data_providers

    def get_hazard_data(  # noqa: C901
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        # A note on concurrency.
        # The requests will be batched up with batches accessing the same data set
        # (e.g. same Zarr array in case of Zarr data).
        # Within each batch if there are multiple lats/lons, the necessary chunks are
        # accessed asynchronously (thanks to async chunk stores in case of Zarr).
        # Across batches we could
        # 1) make async and use event loop executor for CPU-bound parts
        # e.g. asyncio.get_event_loop().run_in_executor
        # 2) use thread pool
        # for now we do 2; 1 might be preferred if the number of threads needed to download
        # data in parallel becomes large (probably not, given use of async in Zarr).

        return self._get_cascading_hazard_data_batches(requests)
        #return self._get_hazard_data(requests)

    def _get_hazard_data(  # noqa: C901
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        batches = defaultdict(list)
        for request in requests:
            batches[request.group_key()].append(request)

        responses: MutableMapping[HazardDataRequest, HazardDataResponse] = {}
        # can change max_workers=1 for debugging
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self._get_hazard_data_batch, batches[key], responses)
                for key in batches.keys()
            ]
            concurrent.futures.wait(futures)
        return responses

    def _get_cascading_hazard_data_batches(self, requests: Sequence[HazardDataRequest]):
        #try:
        #    loop = asyncio.get_event_loop()
        #except Exception:
        #    loop = asyncio.new_event_loop()
        #    asyncio.set_event_loop(loop)
        
        responses: MutableMapping[HazardDataRequest, HazardDataResponse] = {}
        batches: Dict[Tuple[str, str], List[HazardDataRequest]] = defaultdict(list)
        # find the requests for the same indicator, but different scenarios and years
        
        async def all_requests(): 
            async def single_indicator(hazard_type: Type[Hazard], indicator_id: str, hint: str, batch: List[HazardDataRequest]):
                lat_lon_index = {}
                is_event = (indicator_data(hazard_type, indicator_id) == IndicatorData.EVENT)
                for req in batch:
                    lat_lon_index.setdefault((req.latitude, req.longitude, req.buffer), len(lat_lon_index))
                # get the list of scenarios and years needed
                scenarios = set(req.scenario for req in batch)
                years = set(req.year for req in batch if req.scenario != "historical")

                latitudes = [lat_lon[0] for lat_lon in lat_lon_index]
                longitudes = [lat_lon[1] for lat_lon in lat_lon_index]
                # all scenarios and all years for the latitudes and longitudes are obtained

                try:
                    hazard_data_provider = self.hazard_data_providers[hazard_type]
                except Exception:
                    no_provider_err = Exception(
                        f"no hazard data provider for hazard type {hazard_type.__name__}."
                    )
                    for req in batch:
                        responses[req] = HazardDataFailedResponse(err=no_provider_err)
                    return

                results = await hazard_data_provider.get_data_cascading(longitudes, latitudes,
                                                                            indicator_id=indicator_id,
                                                                            scenarios=scenarios,
                                                                            years=years,
                                                                            hint=hint)
                # finally, unpack
                for scenario in scenarios:
                    for year in ([-1] if scenario == "historical" else years):
                        res = results.get(ScenarioYear(scenario, year), None)
                        for req in batch:
                            if req.scenario != scenario or (scenario != "historical" and req.year != year):
                                continue
                            if res is None:
                                responses[req] = HazardDataFailedResponse(reason="no match")
                                continue
                            index = lat_lon_index[(req.latitude, req.longitude, req.buffer)]
                            if res.mask_unprocessed[index]:
                                # item remains unprocessed, presumably because out of bounds of all paths
                                responses[req] = HazardDataFailedResponse(reason="out of bounds")
                                continue
                            if is_event:
                                # if event data contains NaNs, this is taken to be zero    
                                responses[req] = res.values[index]
                                valid = ~np.isnan(res.values[index, :])
                                valid_periods, valid_intensities = (
                                    res.indices[valid],
                                    res.values[index, :][valid],
                                )
                                if len(valid_periods) == 0:
                                    valid_periods, valid_intensities = (
                                        np.array([100]),
                                        np.array([0]),
                                    )
                                responses[req] = HazardEventDataResponse(
                                    valid_periods,
                                    valid_intensities.astype(dtype="float64"),
                                    res.units,
                                    res.paths[index],
                                )
                            else:
                                valid = ~np.isnan(res.values[index, :])
                                responses[req] = HazardParameterDataResponse(
                                    res.values[index, :][valid].astype(dtype="float64"),
                                    res.indices[valid],
                                    res.units,
                                    res.paths[index],
                                )

            #asyncio.get_event_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=64))
            for request in requests:
                batches[(request.hazard_type, request.indicator_id, request.hint.group_key() if request.hint is not None else None)].append(request)
            
            await asyncio.gather(*(single_indicator(hazard_type, indicator_id, batch[0].hint, batch) for (hazard_type, indicator_id, hint), batch in batches.items()))
                
        #loop.run_until_complete(all_requests())
        asyncio.run(all_requests())
        return responses

    def _get_hazard_data_batch(
        self,
        batch: List[HazardDataRequest],
        responses: MutableMapping[HazardDataRequest, HazardDataResponse],
    ):
        failures = []
        try:
            hazard_type, indicator_id, scenario, year, hint, buffer = (
                batch[0].hazard_type,
                batch[0].indicator_id,
                batch[0].scenario,
                batch[0].year,
                batch[0].hint,
                batch[0].buffer,
            )
            longitudes = [req.longitude for req in batch]
            latitudes = [req.latitude for req in batch]

            try:
                hazard_data_provider = self.hazard_data_providers[hazard_type]
            except Exception:
                no_provider_err = Exception(
                    f"no hazard data provider for hazard type {hazard_type.__name__}."
                )
                for req in batch:
                    responses[req] = HazardDataFailedResponse(err=no_provider_err)
                return

            if indicator_data(hazard_type, indicator_id) == IndicatorData.EVENT:
                intensities, return_periods, units, path = (
                    hazard_data_provider.get_data(
                        longitudes,
                        latitudes,
                        indicator_id=indicator_id,
                        scenario=scenario,
                        year=year,
                        hint=hint,
                        buffer=buffer,
                    )
                )

                for i, req in enumerate(batch):
                    valid = ~np.isnan(intensities[i, :])
                    valid_periods, valid_intensities = (
                        return_periods[valid],
                        intensities[i, :][valid],
                    )
                    if len(valid_periods) == 0:
                        valid_periods, valid_intensities = (
                            np.array([100]),
                            np.array([0]),
                        )
                    responses[req] = HazardEventDataResponse(
                        valid_periods,
                        valid_intensities.astype(dtype="float64"),
                        units,
                        path,
                    )
            else:  # type: ignore
                parameters, defns, units, path = hazard_data_provider.get_data(
                    longitudes,
                    latitudes,
                    indicator_id=indicator_id,
                    scenario=scenario,
                    year=year,
                    hint=hint,
                )

                for i, req in enumerate(batch):
                    valid = ~np.isnan(parameters[i, :])
                    responses[req] = HazardParameterDataResponse(
                        parameters[i, :][valid].astype(dtype="float64"),
                        defns[valid],
                        units,
                        path,
                    )
        except Exception as err:
            # e.g. the requested data is unavailable
            for req in batch:
                failed_response = HazardDataFailedResponse(err)
                responses[req] = failed_response
                failures.append(failed_response)

        if any(failures):
            # only a warning: perhaps the caller does not expect data to be present for all
            # year/scenario combinations.
            logger.warning(
                f"{len(failures)} requests failed in batch (hazard_type={hazard_type.__name__}, indicator_id={indicator_id}, "
                f"scenario={scenario}, year={year}): (logs limited to first 3)"
            )
            errors = (str(i.error) for i in failures)
            for _ in range(min(len(failures), 3)):
                logger.warning(next(errors))
        return


class ZarrHazardModel(PregeneratedHazardModel):
    def __init__(
        self,
        *,
        source_paths: SourcePaths,
        reader: Optional[ZarrReader] = None,
        store=None,
        interpolation="floor",
    ):
        # share ZarrReaders across HazardDataProviders
        zarr_reader = ZarrReader(store=store) if reader is None else reader
        hazard_types = source_paths.hazard_types()
        super().__init__(
            {
                t: HazardDataProvider(
                    t, source_paths, zarr_reader=zarr_reader, interpolation=interpolation
                )
                for t in hazard_types
            }
        )
