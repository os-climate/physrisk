from collections import defaultdict
from typing import Dict, List, Optional

from physrisk.kernel import calculation

from ..kernel.hazard_model import EventDataRequest, EventDataResponse, HazardModel
from .event_provider import EventProvider, SourcePath


class PregeneratedHazardModel(HazardModel):
    """Hazard event model that processes requests using EventProviders."""

    def __init__(self, event_providers: Dict[type, EventProvider]):
        self.event_providers = event_providers

    def get_hazard_events(
        self,
        requests: List[EventDataRequest],
    ) -> Dict[EventDataRequest, EventDataResponse]:

        batches = defaultdict(list)
        for request in requests:
            batches[request.group_key()].append(request)

        responses = {}
        for key in batches.keys():
            batch = batches[key]
            event_type, model, scenario, year = batch[0].event_type, batch[0].model, batch[0].scenario, batch[0].year
            longitudes = [req.longitude for req in batch]
            latitudes = [req.latitude for req in batch]
            intensities, return_periods = self.event_providers[event_type].get_intensity_curves(
                longitudes, latitudes, model=model, scenario=scenario, year=year
            )

            for i, req in enumerate(batch):
                responses[req] = EventDataResponse(return_periods, intensities[i, :])

        return responses


class ZarrHazardModel(PregeneratedHazardModel):
    def __init__(self, source_paths: Optional[Dict[type, SourcePath]] = None, store=None):
        if source_paths is None:
            source_paths = calculation.get_default_zarr_source_paths()
        super().__init__(dict((t, EventProvider(sp, store=store)) for t, sp in source_paths.items()))
