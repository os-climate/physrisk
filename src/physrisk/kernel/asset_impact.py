import numpy as np
#import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Optional, Any
from .asset_event_distribution import AssetEventDistribution
from .impact_distribution import ImpactDistribution
from .vulnerability_distrib import VulnerabilityDistrib

def get_impact_distribution(event_dist : AssetEventDistribution, vulnerability_dist : VulnerabilityDistrib) -> ImpactDistribution:
    impact_prob = vulnerability_dist.prob_matrix.T @ event_dist.prob
    return ImpactDistribution(vulnerability_dist.event_type, vulnerability_dist.impact_bins, impact_prob)

class AssetImpact:
    """Calculates the impacts associated with a portfolio of assets."""
    
    def __init__(self, assets, vulnerabilities):
        pass

class AssetEventProvider(ABC):
    
    @abstractmethod
    def get_asset_events(assets, eventTypes):
        """Source event distributions in the locale of each asset for events of certain types """

class ModelsBuilder(ABC):
    """Provides VulnerabilityModels and EventProviders for a type of aset."""
    
    @abstractmethod
    def get_vulnerability_model(assetType):
        pass

    @abstractmethod
    def get_event_data_provider(assetType):
        """Return a list of backends matching the specified filtering.
        Args:
            assetType (AssetType): type of asset.
        Returns:
            dict[EventType, AssetEvents]: a list of Backends that match the filtering
                criteria.
        """
        pass
 