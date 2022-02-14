from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, Union

from physrisk.data.data_requests import EventDataRequest, EventDataResponse
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_event_distrib import HazardEventDistrib
from physrisk.kernel.vulnerability_distrib import VulnerabilityDistrib

PLUGINS = dict()


def repeat(num_times):
    def decorator_repeat(func):
        ...  # Create and return a wrapper function

    return decorator_repeat


def applies_to_events(event_types):
    def decorator_events(func):
        PLUGINS[func.__name__] = func
        return func

    return decorator_events


def applies_to_assets(asset_types):
    def decorator_events(func):
        PLUGINS[func.__name__] = func
        return func

    return decorator_events


class Model(ABC):
    """Models generate the VulnerabilityDistrib and HazardEventDistrib of an
    Asset.
    """

    def __init__(self, model: str, event_type: type, year: int, scenario: str):
        self.model = model
        self.event_type = event_type
        self.year = year
        self.scenario = scenario
        self._event_types: List[type] = []
        self._asset_types: List[type] = []

    @abstractmethod
    def get_event_data_requests(
        self, asset: Asset, *, scenario: str, year: int
    ) -> Union[EventDataRequest, Iterable[EventDataRequest]]:
        """Provide the one or more hazard event data requests required in order to calculate
        the VulnerabilityDistrib and HazardEventDistrib for the asset."""
        ...

    @abstractmethod
    def get_distributions(
        self, asset: Asset, event_data_responses: List[EventDataResponse]
    ) -> Tuple[VulnerabilityDistrib, HazardEventDistrib]:
        """Return distributions for asset, VulnerabilityDistrib and HazardEventDistrib.
        The hazard event data is used to do this.

        Args:
            asset: the asset.
            event_data_responses: the responses to the requests made by get_event_data_requests, in the same order.
        """
        ...

    def _check_event_type(self):
        if self.event_type not in self._event_types:
            raise NotImplementedError(f"model does not support events of type {self.event_type.__name__}")
