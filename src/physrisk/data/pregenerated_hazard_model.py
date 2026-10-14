import asyncio
import concurrent.futures
from collections import defaultdict
import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np

from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazards import (
    Drought,
    Fire,
    Hail,
    Hazard,
    IndicatorData,
    Landslide,
    Subsidence,
    indicator_data,
)

from ..kernel.hazard_model import (
    HazardDataFailedResponse,
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from .hazard_data_provider import (
    CascadingHazardDataProvider,
    HazardDataHint,
    HazardDataProvider,
    ScenarioYear,
    SourcePaths,
)

logger = logging.getLogger(__name__)


class PregeneratedHazardModel(HazardModel):
    """Batches hazard data requests and delegates them to hazard data providers."""

    def __init__(
        self,
        hazard_data_providers: Dict[Type[Hazard], HazardDataProvider],
        zarr_max_workers: int = 32,
        nan_is_zero: Optional[set[tuple[type[Hazard], str]]] = None,
        nan_is_no_data: Optional[set[tuple[type[Hazard], str]]] = None,
    ):
        """
        Args:
            hazard_data_providers: Map from hazard type to its data provider.
            zarr_max_workers: Max threads for concurrent Zarr chunk reads.
            nan_is_zero: (hazard_type, indicator_id) pairs where NaN is treated as 0. Defaults to common indicators (fire, drought, hail, subsidence, landslide).
            nan_is_no_data: (hazard_type, indicator_id) pairs where NaN causes a failed response. Mutually exclusive with nan_is_zero.
        """
        self.hazard_data_providers = hazard_data_providers
        self.zarr_max_workers = zarr_max_workers
        self._nan_is_zero: set[tuple[type[Hazard], str]] = (
            nan_is_zero
            if nan_is_zero is not None
            else PregeneratedHazardModel._default_nan_is_zero()
        )
        self._nan_is_no_data: set[tuple[type[Hazard], str]] = (
            nan_is_no_data if nan_is_no_data is not None else set()
        )
        if any(self._nan_is_zero & self._nan_is_no_data):
            raise ValueError(
                f"element {next(iter(self._nan_is_zero & self._nan_is_no_data))} appears in nan_is_zero and nan_is_no_data"
            )

    def get_hazard_data(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        """Return one hazard data response for each request.

        Args:
            requests: Hazard data requests to process.

        Returns:
            Responses keyed by their corresponding requests.
        """
        if len(requests) == 0:
            return {}

        # A note on concurrency.
        # The requests will be batched up with batches accessing the same data set
        # (e.g. same Zarr array in case of Zarr data).
        # Within each batch if there are multiple lats/lons, the necessary chunks are
        # accessed asynchronously (thanks to async chunk stores in case of Zarr).
        # Across batches we also call asynchronously.
        logger.info(f"Retrieving data for {len(requests)} hazard data requests")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            responses = asyncio.run(self._get_hazard_data_in_batches(requests))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                responses = executor.submit(
                    lambda: asyncio.run(self._get_hazard_data_in_batches(requests))
                ).result()
        logger.info("Data retrieval complete")
        self._log_response_issues(responses)
        return responses

    async def _get_hazard_data_in_batches(
        self, requests: Sequence[HazardDataRequest]
    ) -> dict[HazardDataRequest, HazardDataResponse]:
        """Group compatible requests and retrieve their data concurrently."""
        batches: dict[tuple[type[Hazard], str, str | None], list[HazardDataRequest]] = (
            defaultdict(list)
        )
        for request in requests:
            batches[
                (
                    request.hazard_type,
                    request.indicator_id,
                    request.hint.group_key() if request.hint is not None else None,
                )
            ].append(request)

        asyncio.get_running_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self.zarr_max_workers)
        )
        batch_results = await asyncio.gather(
            *(
                self._get_batch_data(hazard_type, indicator_id, batch[0].hint, batch)
                for (hazard_type, indicator_id, _), batch in batches.items()
            )
        )
        return {
            request: response
            for batch_result in batch_results
            for request, response in batch_result.items()
        }

    async def _get_batch_data(
        self,
        hazard_type: type[Hazard],
        indicator_id: str,
        hint: HazardDataHint | None,
        batch: list[HazardDataRequest],
    ) -> dict[HazardDataRequest, HazardDataResponse]:
        """Retrieve and unpack one batch of compatible hazard data requests."""
        responses: dict[HazardDataRequest, HazardDataResponse] = {}
        sampling_point_index: dict[tuple[float, float, int | None], int] = {}
        is_event = indicator_data(hazard_type, indicator_id) == IndicatorData.EVENT
        nan_is_zero = (hazard_type, indicator_id) in self._nan_is_zero
        nan_is_no_data = (hazard_type, indicator_id) in self._nan_is_no_data
        non_negative = nan_is_zero
        for request in batch:
            sampling_point = (request.latitude, request.longitude, request.buffer)
            if sampling_point not in sampling_point_index:
                sampling_point_index[sampling_point] = len(sampling_point_index)

        scenarios = list(dict.fromkeys(request.scenario for request in batch))
        years = list(
            dict.fromkeys(
                request.year for request in batch if request.scenario != "historical"
            )
        )
        latitudes = np.array([latitude for latitude, _, _ in sampling_point_index])
        longitudes = np.array([longitude for _, longitude, _ in sampling_point_index])
        buffers = [buffer for _, _, buffer in sampling_point_index]

        hazard_data_provider = self.hazard_data_providers.get(hazard_type)
        if hazard_data_provider is None:
            no_provider_err = Exception(
                f"no hazard data provider for hazard type {hazard_type.__name__}"
            )
            return {
                request: HazardDataFailedResponse(err=no_provider_err)
                for request in batch
            }

        results = await hazard_data_provider.get_data(
            longitudes,
            latitudes,
            indicator_id=indicator_id,
            scenarios=scenarios,
            years=years,
            hint=hint,
            buffer=buffers[0],
        )

        for request in batch:
            key = ScenarioYear(
                request.scenario,
                -1 if request.scenario == "historical" else request.year,
            )
            result = results.get(key)
            if result is None:
                responses[request] = HazardDataFailedResponse(
                    reason="no data available"
                )
                continue

            sampling_point = (request.latitude, request.longitude, request.buffer)
            index = sampling_point_index[sampling_point]
            if not bool(result.coverage_mask[index]):
                responses[request] = HazardDataFailedResponse(reason="out of bounds")
                continue

            indices_length = int(result.indices_length[index])
            values = result.values[index, :indices_length]
            indices = result.indices[index, :indices_length]
            if is_event:
                valid = ~np.isnan(values)
                valid_periods, valid_intensities = indices[valid], values[valid]
                if len(valid_periods) == 0:
                    valid_periods, valid_intensities = np.array([100]), np.array([0])
                responses[request] = HazardEventDataResponse(
                    valid_periods,
                    valid_intensities.astype(dtype="float64"),
                    result.units,
                    result.paths[index],
                )
            else:
                if nan_is_no_data and np.any(np.isnan(values)):
                    responses[request] = HazardDataFailedResponse(
                        reason="unexpected nan"
                    )
                    continue
                if nan_is_zero:
                    values[np.isnan(values)] = 0.0
                if non_negative:
                    values[values < 0.0] = 0.0
                responses[request] = HazardParameterDataResponse(
                    values.astype(dtype="float64"),
                    indices,
                    result.units,
                    result.paths[index],
                )

        return responses

    @staticmethod
    def _log_response_issues(
        responses: Dict[HazardDataRequest, HazardDataResponse],
    ) -> None:
        # in some cases requested data cannot be retrieved, leading to a HazardDataFailedResponse
        # this may be handled by the vulnerability model or might result in missing results.
        grouped_failures: Dict[
            str, List[Tuple[HazardDataRequest, HazardDataFailedResponse]]
        ] = defaultdict(list)
        for req, resp in responses.items():
            if isinstance(resp, HazardDataFailedResponse):
                key = resp.reason if resp.reason else str(resp.error)
                grouped_failures[key].append((req, resp))
        for key, values in grouped_failures.items():
            logger.info(
                f"{len(values)} {'failures' if len(values) > 1 else 'failure'}: {key}. Limited to first 5:"
            )
            for v in values[0:5]:
                logger.info(str(v[0]))

    @staticmethod
    def _default_nan_is_zero():
        return {
            (Drought, "months/spei3m/below/-2"),
            (Hail, "days/above/5cm"),
            (Fire, "fire_probability"),
            (Subsidence, "subsidence_probability"),
            (Landslide, "landslide_probability"),
        }


class ZarrHazardModel(PregeneratedHazardModel):
    def __init__(
        self,
        *,
        source_paths: SourcePaths,
        reader: Optional[ZarrReader] = None,
        store=None,
        interpolation="floor",
        interpolate_years: bool = False,
        zarr_max_workers: int = 32,
        nan_is_zero: Optional[set[tuple[type[Hazard], str]]] = None,
        nan_is_no_data: Optional[set[tuple[type[Hazard], str]]] = None,
    ):
        """Hazard model backed by Zarr arrays.

        Args:
            source_paths: Provides paths to Zarr arrays for each hazard type and indicator.
            reader: Shared ZarrReader; created from store if not provided.
            store: Zarr store (local, remote, or in-memory); used when reader is None.
            interpolation: Spatial interpolation method ("floor" or "linear").
            interpolate_years: Whether to interpolate hazard data between available years.
            zarr_max_workers: Max threads for concurrent Zarr chunk reads.
            nan_is_zero: (hazard_type, indicator_id) pairs where NaN is treated as 0. Defaults to common indicators (fire, drought, hail, subsidence, landslide).
            nan_is_no_data: (hazard_type, indicator_id) pairs where NaN causes a failed response. Mutually exclusive with nan_is_zero.
        """
        # share ZarrReaders across hazard data providers
        zarr_reader = ZarrReader(store=store) if reader is None else reader
        hazard_types = source_paths.hazard_types()
        super().__init__(
            {
                t: CascadingHazardDataProvider(
                    t,
                    source_paths,
                    zarr_reader=zarr_reader,
                    interpolation=interpolation,
                    interpolate_years=interpolate_years,
                )
                for t in hazard_types
            },
            zarr_max_workers=zarr_max_workers,
            nan_is_zero=nan_is_zero,
            nan_is_no_data=nan_is_no_data,
        )
