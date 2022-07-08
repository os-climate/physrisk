from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Mapping, Protocol, Tuple

import numpy as np


class HazardDataRequest:
    """Request for hazard data. The event_type determines whether the hazard is acute or chronic.
    An acute hazard is an event and the response will therefore comprise hazard intensities for the
    different event return periods. A chronic hazard on the other hand is a shift in a climate parameter
    and the parameter value is returned."""

    def __init__(self, hazard_type: type, longitude: float, latitude: float, *, model: str, scenario: str, year: int):
        """Create HazardDataRequest.

        Args:
            event_type: type of hazard.
            longitude: required longitude.
            latitude: required latitude.
            model: model identifier.
            scenario: identifier of scenario, e.g. rcp8p5 (RCP 8.5).
            year: projection year, e.g. 2080.
        """
        self.hazard_type = hazard_type
        self.longitude = longitude
        self.latitude = latitude
        self.model = model
        self.scenario = scenario
        self.year = year

    def group_key(self):
        """Key used to group EventDataRequests into batches."""
        return tuple((self.hazard_type, self.model, self.scenario, self.year))


class HazardDataResponse:
    pass


class HazardEventDataResponse(HazardDataResponse):
    """Response to HazardDataRequest for acute hazards."""

    def __init__(self, return_periods: np.ndarray, intensities: np.ndarray):
        """Create HazardEventDataResponse.

        Args:
            return_periods: return periods in years.
            intensities: hazard event intensity for each return period.
        """
        self.return_periods = return_periods
        self.intensities = intensities


class HazardParameterDataResponse(HazardDataResponse):
    """Response to HazardDataRequest."""

    def __init__(self, parameter: np.ndarray):
        """Create HazardParameterDataResponse.

        Args:
            parameter: the chronic hazard parameter value.
        """
        self.parameter = parameter


class HazardModel(ABC):
    """Hazard event model. The model accepts a set of EventDataRequests and returns the corresponding
    EventDataResponses."""

    @abstractmethod
    def get_hazard_events(self, requests: List[HazardDataRequest]) -> Mapping[HazardDataRequest, HazardDataResponse]:
        """Process the hazard data requests and return responses."""
        ...


class DataSource(Protocol):
    def __call__(self, longitudes, latitudes, *, model: str, scenario: str, year: int) -> Tuple[np.ndarray, np.ndarray]:
        ...


class CompositeHazardModel(HazardModel):
    """Hazard Model that uses other models to process EventDataRequests."""

    def __init__(self, hazard_models: Dict[type, HazardModel]):
        self.hazard_models = hazard_models

    def get_hazard_events(self, requests: List[HazardDataRequest]) -> Mapping[HazardDataRequest, HazardDataResponse]:
        requests_by_event_type = defaultdict(list)

        for request in requests:
            requests_by_event_type[request.hazard_type].append(request)

        responses: Dict[HazardDataRequest, HazardDataResponse] = {}
        for event_type, reqs in requests_by_event_type.items():
            events_reponses = self.hazard_models[event_type].get_hazard_events(reqs)
            responses.update(events_reponses)

        return responses
