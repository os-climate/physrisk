from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Protocol, Tuple

import numpy as np


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


class HazardModel(ABC):
    """Hazard event model. The model accepts a set of EventDataRequests and returns the corresponding
    EventDataResponses."""

    @abstractmethod
    def get_hazard_events(self, requests: List[EventDataRequest]) -> Dict[EventDataRequest, EventDataResponse]:
        """Process the hazard event requests and return responses."""
        ...


class DataSource(Protocol):
    def __call__(self, longitudes, latitudes, *, model: str, scenario: str, year: int) -> Tuple[np.ndarray, np.ndarray]:
        ...


class CompositeHazardModel(HazardModel):
    """Hazard Model that uses other models to process EventDataRequests."""

    def __init__(self, hazard_models: Dict[type, HazardModel]):
        self.hazard_models = hazard_models

    def get_hazard_events(self, requests: List[EventDataRequest]) -> Dict[EventDataRequest, EventDataResponse]:
        requests_by_event_type = defaultdict(list)

        for request in requests:
            requests_by_event_type[request.event_type].append(request)

        responses = {}
        for event_type, reqs in requests_by_event_type.items():
            responses.update(self.hazard_models[event_type].get_hazard_events(reqs))

        return responses
