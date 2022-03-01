# import numpy.typing as npt
from abc import ABC, abstractmethod


class AssetImpact:
    """Calculates the impacts associated with a portfolio of assets."""

    def __init__(self, assets, vulnerabilities):
        pass


class AssetEventProvider(ABC):
    @abstractmethod
    def get_asset_events(self, assets, event_types):
        """Source event distributions in the locale of each asset for events of certain types"""


class ModelsBuilder(ABC):
    """Provides VulnerabilityModels and EventProviders for a type of aset."""

    @abstractmethod
    def get_vulnerability_model(self, asset_type):
        pass

    @abstractmethod
    def get_event_data_provider(self, asset_type):
        """Return a list of backends matching the specified filtering.
        Args:
            asset_type (AssetType): type of asset.
        Returns:
            dict[EventType, AssetEvents]: a list of Backends that match the filtering
                criteria.
        """
        pass
