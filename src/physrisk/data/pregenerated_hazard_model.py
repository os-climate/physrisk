from collections import defaultdict
from typing import Dict, List, Mapping, MutableMapping, Optional

from physrisk.kernel import calculation
from physrisk.kernel.events import HazardKind

from ..kernel.hazard_model import (
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from .hazard_data_provider import AcuteHazardDataProvider, ChronicHazardDataProvider, SourcePath


class PregeneratedHazardModel(HazardModel):
    """Hazard event model that processes requests using EventProviders."""

    def __init__(
        self,
        acute_hazard_data_providers: Dict[type, AcuteHazardDataProvider],
        chronic_hazard_data_providers: Dict[type, ChronicHazardDataProvider],
    ):
        self.acute_hazard_data_providers = acute_hazard_data_providers
        self.chronic_hazard_data_providers = chronic_hazard_data_providers

    def get_hazard_events(self, requests: List[HazardDataRequest]) -> Mapping[HazardDataRequest, HazardDataResponse]:

        batches = defaultdict(list)
        for request in requests:
            batches[request.group_key()].append(request)

        responses: MutableMapping[HazardDataRequest, HazardDataResponse] = {}
        for key in batches.keys():
            batch: List[HazardDataRequest] = batches[key]
            event_type, model, scenario, year = batch[0].event_type, batch[0].model, batch[0].scenario, batch[0].year
            longitudes = [req.longitude for req in batch]
            latitudes = [req.latitude for req in batch]
            if event_type.kind == HazardKind.acute:  # type: ignore
                intensities, return_periods = self.acute_hazard_data_providers[event_type].get_intensity_curves(
                    longitudes, latitudes, model=model, scenario=scenario, year=year
                )

                for i, req in enumerate(batch):
                    responses[req] = HazardEventDataResponse(return_periods, intensities[i, :])
            elif event_type.kind == HazardKind.chronic:  # type: ignore
                parameters = self.chronic_hazard_data_providers[event_type].get_parameters(
                    longitudes, latitudes, model=model, scenario=scenario, year=year
                )

                for i, req in enumerate(batch):
                    responses[req] = HazardParameterDataResponse(parameters[i])
        return responses


class ZarrHazardModel(PregeneratedHazardModel):
    def __init__(
        self,
        acute_source_paths: Optional[Dict[type, SourcePath]] = None,
        chronic_source_paths: Optional[Dict[type, SourcePath]] = None,
        store=None,
        interpolation="floor",
    ):
        if acute_source_paths is None:
            acute_source_paths = calculation.get_default_accute_zarr_source_paths()
        super().__init__(
            dict(
                (t, AcuteHazardDataProvider(sp, store=store, interpolation=interpolation))
                for t, sp in acute_source_paths.items()
            ),
            dict(
                (t, ChronicHazardDataProvider(sp, store=store, interpolation=interpolation))
                for t, sp in acute_source_paths.items()
            ),
        )
