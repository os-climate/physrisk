from abc import ABC, abstractmethod


class Model(ABC):
    """Exposure/vulnerability model that generates the vulnerability
    and asset event distributions of an assets for different types of
    hazard event.
    """

    @abstractmethod
    def get_event_data_requests(self, event):
        pass

    @abstractmethod
    def get_distributions(self, event_data_responses):
        pass
