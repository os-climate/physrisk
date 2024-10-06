import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple, Type

import numpy as np

from physrisk.data.hazard_data_provider import HazardDataHint
from physrisk.kernel.hazards import Hazard


class HazardDataRequest:
    """Request for hazard data. The event_type determines whether the hazard is acute or chronic.
    An acute hazard is an event and the response will therefore comprise hazard intensities for the
    different event return periods. A chronic hazard on the other hand is a shift in a climate parameter
    and the parameter value is returned."""

    def __init__(
        self,
        hazard_type: Type[Hazard],
        longitude: float,
        latitude: float,
        *,
        indicator_id: str,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
        buffer: Optional[int] = None,
    ):
        """Create HazardDataRequest.

        Args:
            hazard_type (Type[Hazard]): Type of hazard.
            longitude (float): Longitude.
            latitude (float): Latitude.
            indicator_id (str): Hazard indicator identifier, e.g. "flood_depth".
            scenario (str): Scenario identifier, e.g. "SSP585", "RCP8p5".
            year (int): Year for which data required.
            hint (Optional[HazardDataHint], optional): Hint, typically providing the data set path.
                Defaults to None.
            buffer (Optional[int], optional): If not None applies a buffer around the point in metres, within [0, 1000m]. Defaults to None.
        """
        self.hazard_type = hazard_type
        self.longitude = longitude
        self.latitude = latitude
        self.indicator_id = indicator_id
        self.scenario = scenario
        self.year = year
        self.hint = hint
        self.buffer = buffer

    def group_key(self):
        """Key used to group EventDataRequests into batches."""
        return tuple(
            (
                self.hazard_type,
                self.indicator_id,
                self.scenario,
                self.year,
                None if self.hint is None else self.hint.group_key(),
            )
        )


class HazardDataResponse:
    pass


class HazardDataFailedResponse(HazardDataResponse):
    def __init__(self, err: Exception):
        self.error = err


class HazardEventDataResponse(HazardDataResponse):
    """Response to HazardDataRequest for acute hazards."""

    def __init__(
        self,
        return_periods: np.ndarray,
        intensities: np.ndarray,
        units: str = "default",
        path: str = "unknown",
    ):
        """Create HazardEventDataResponse.

        Args:
            return_periods: return periods in years.
            intensities: hazard event intensity for each return period, or set of hazard event intensities corresponding to different events. # noqa: E501
            path: path to the hazard indicator data source.
        """

        self.return_periods = return_periods
        self.intensities = intensities
        self.units = sys.intern(units)
        self.path = sys.intern(path)


class HazardParameterDataResponse(HazardDataResponse):
    """Response to HazardDataRequest."""

    def __init__(
        self,
        parameters: np.ndarray,
        param_defns: np.ndarray = np.empty([]),
        units: str = "default",
        path: str = "unknown",
    ):
        """Create HazardParameterDataResponse. In general the chronic parameters are an array of values.
        For example, a chronic hazard may be the number of days per year with average temperature
        above :math:`x' degrees for :math:`x' in [25, 30, 35, 40]°C. In this case the param_defns would
        contain np.array([25, 30, 35, 40]). In some cases the hazard may be a scalar value.
        Parameters will typically be a (1D) array of values where vulnerability models
        require a number of parameters (e.g. to model decrease of efficiency as temperature increases).

        Args:
            parameters (np.ndarray): Chronic hazard parameter values.
            param_defns (np.ndarray): Chronic hazard parameter definitions.
            path: path to the hazard indicator data source.
        """
        self.parameters = parameters
        self.param_defns = param_defns
        self.units = sys.intern(units)
        self.path = sys.intern(path)

    @property
    def parameter(self) -> float:
        """Convenience function to return single parameter.

        Returns:
            float: Single parameter.
        """
        return self.parameters[0]


class HazardModelFactory(Protocol):
    def hazard_model(
        self, interpolation: str = "floor", provider_max_requests: Dict[str, int] = {}
    ):
        """Create a HazardModel instance based on a number of options.

        Args:
            interpolation (str): Interpolation type to use for sub-pixel raster interpolation (where
            this is supported by hazard models).
            provider_max_requests (Dict[str, int]): The maximum permitted number of permitted
            requests to external providers.
        """
        ...


class HazardModel(ABC):
    """Hazard model. The model accepts a set of HazardDataRequests and returns the corresponding
    HazardDataResponses."""

    @abstractmethod
    def get_hazard_data(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        """Process the hazard indicator data requests and return responses.

        Args:
            requests (Sequence[HazardDataRequest]): Hazard indicator data requests.

        Returns:
            Mapping[HazardDataRequest, HazardDataResponse]: Responses for all hazard indicator data requests.
        """
        ...

    def get_hazard_events(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        """Deprecated: this has been renamed to get_hazard_data."""
        return self.get_hazard_data(requests)


class DataSource(Protocol):
    def __call__(
        self, longitudes, latitudes, *, model: str, scenario: str, year: int
    ) -> Tuple[np.ndarray, np.ndarray]: ...


class CompositeHazardModel(HazardModel):
    """Hazard Model that uses other models to process EventDataRequests."""

    def __init__(self, hazard_models: Dict[type, HazardModel]):
        self.hazard_models = hazard_models

    def get_hazard_data(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        requests_by_event_type = defaultdict(list)

        for request in requests:
            requests_by_event_type[request.hazard_type].append(request)

        responses: Dict[HazardDataRequest, HazardDataResponse] = {}
        for event_type, reqs in requests_by_event_type.items():
            events_reponses = self.hazard_models[event_type].get_hazard_data(reqs)
            responses.update(events_reponses)

        return responses
