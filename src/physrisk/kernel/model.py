from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, Union

from physrisk.data.data_requests import EventDataRequest, EventDataResponse
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_event_distrib import HazardEventDistrib
from physrisk.kernel.vulnerability_distrib import VulnerabilityDistrib


class Model(ABC):
    """Models generate the VulnerabilityDistrib and HazardEventDistrib of an
    Asset.
    """

    @abstractmethod
    def get_event_data_requests(self, asset: Asset) -> Union[EventDataRequest, Iterable[EventDataRequest]]:
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
