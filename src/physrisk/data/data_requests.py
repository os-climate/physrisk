from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from typing_extensions import Protocol


class EventDataRequest:
    """Request for a hazard event intensity curve."""

    def __init__(self, event_type: type, longitude: float, latitude: float, *, model: str, scenario: str, year: int):
        """Create EventDataRequest.

        Args:
            event_type: type of hazard event.
            longitude: required longitude.
            latitude: required latitude.
            model: model identifier.
            scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
            year: projection year, e.g. 2080.
        """
        self.event_type = event_type
        self.longitude = longitude
        self.latitude = latitude
        self.model = model
        self.scenario = scenario
        self.year = year

    def group_key(self):
        """Key used to group EventDataRequests into batches."""
        return tuple((self.event_type, self.model, self.scenario, self.year))


class EventDataResponse:
    """Response to EventDataRequest."""

    def __init__(self, return_periods: np.ndarray, intensities: np.ndarray):
        """Create ReturnPeriodEvDataResp.

        Args:
            return_periods: return periods in years.
            intensities: hazard event intensity for each return period.
        """
        self.return_periods = return_periods
        self.intensities = intensities


class DataSource(Protocol):
    def __call__(self, longitudes, latitudes, *, model: str, scenario: str, year: int) -> Tuple[np.ndarray, np.ndarray]:
        ...


def process_requests(
    requests: List[EventDataRequest], data_sources: Dict[type, DataSource]
) -> Dict[EventDataRequest, EventDataResponse]:
    """Create ReturnPeriodEvDataResp.

    Args:
        return_periods: return periods in years.
        intensities: hazard event intensity for each return period.
    """

    batches = defaultdict(list)
    for request in requests:
        batches[request.group_key()].append(request)

    responses = {}
    for key in batches.keys():
        batch = batches[key]
        event_type, model, scenario, year = batch[0].event_type, batch[0].model, batch[0].scenario, batch[0].year
        longitudes = [req.longitude for req in batch]
        latitudes = [req.latitude for req in batch]
        intensities, return_periods = data_sources[event_type](
            longitudes, latitudes, model=model, scenario=scenario, year=year
        )

        for i, req in enumerate(batch):
            responses[req] = EventDataResponse(return_periods, intensities[i, :])

    return responses
