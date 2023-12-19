import asyncio
import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Mapping, MutableMapping, Optional, cast

import numpy as np

from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazards import Hazard, HazardKind

from ..kernel.hazard_model import (
    HazardDataFailedResponse,
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from .hazard_data_provider import AcuteHazardDataProvider, ChronicHazardDataProvider, HazardDataProvider, SourcePath


class PregeneratedHazardModel(HazardModel):
    """Hazard event model that processes requests using EventProviders."""

    def __init__(
        self,
        hazard_data_providers: Dict[type, HazardDataProvider],
    ):
        self.acute_hazard_data_providers = dict(
            (k, cast(AcuteHazardDataProvider, v))
            for (k, v) in hazard_data_providers.items()
            if Hazard.kind(k) == HazardKind.acute
        )
        self.chronic_hazard_data_providers = dict(
            (k, cast(ChronicHazardDataProvider, v))
            for (k, v) in hazard_data_providers.items()
            if Hazard.kind(k) == HazardKind.chronic
        )

    def get_hazard_events(  # noqa: C901
        self, requests: List[HazardDataRequest]
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
        # data in parallel becomes large (probably not given use of async in Zarr).

        # result = asyncio.gather(self._get_hazard_events(requests))
        # return result
        return asyncio.run(self._get_hazard_events(requests))

    async def _get_hazard_events(  # noqa: C901
        self, requests: List[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        batches = defaultdict(list)
        for request in requests:
            batches[request.group_key()].append(request)

        responses: MutableMapping[HazardDataRequest, HazardDataResponse] = {}
        # can change max_workers=1 for debugging
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self._get_hazard_events_batch, batches[key], responses) for key in batches.keys()
            ]
            concurrent.futures.wait(futures)
        return responses

    def _get_hazard_events_batch(
        self, batch: List[HazardDataRequest], responses: MutableMapping[HazardDataRequest, HazardDataResponse]
    ):
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
            if hazard_type.kind == HazardKind.acute:  # type: ignore
                intensities, return_periods = self.acute_hazard_data_providers[hazard_type].get_intensity_curves(
                    longitudes,
                    latitudes,
                    indicator_id=indicator_id,
                    scenario=scenario,
                    year=year,
                    hint=hint,
                    buffer=buffer,
                )

                for i, req in enumerate(batch):
                    valid = ~np.isnan(intensities[i, :])
                    valid_periods, valid_intensities = return_periods[valid], intensities[i, :][valid]
                    if len(valid_periods) == 0:
                        valid_periods, valid_intensities = np.array([100]), np.array([0])
                    responses[req] = HazardEventDataResponse(valid_periods, valid_intensities)
            elif hazard_type.kind == HazardKind.chronic:  # type: ignore
                parameters, defns = self.chronic_hazard_data_providers[hazard_type].get_parameters(
                    longitudes, latitudes, indicator_id=indicator_id, scenario=scenario, year=year, hint=hint
                )

                for i, req in enumerate(batch):
                    responses[req] = HazardParameterDataResponse(parameters[i, :], defns)
        except Exception as err:
            # e.g. the requested data is unavailable
            for i, req in enumerate(batch):
                responses[req] = HazardDataFailedResponse(err)
        return


class ZarrHazardModel(PregeneratedHazardModel):
    def __init__(
        self,
        *,
        source_paths: Dict[type, SourcePath],
        reader: Optional[ZarrReader] = None,
        store=None,
        interpolation="floor",
    ):
        # share ZarrReaders across HazardDataProviders
        zarr_reader = ZarrReader(store=store) if reader is None else reader

        super().__init__(
            dict(
                (
                    t,
                    (
                        AcuteHazardDataProvider(sp, zarr_reader=zarr_reader, interpolation=interpolation)
                        if Hazard.kind(t) == HazardKind.acute
                        else ChronicHazardDataProvider(sp, zarr_reader=zarr_reader, interpolation=interpolation)
                    ),
                )
                for t, sp in source_paths.items()
            )
        )
