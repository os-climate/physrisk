import numpy as np
from typing import List, Union

class ImpactDistribution:
    """Impact distributions specific to an asset."""
    
    __slots__ = ["_event_type", "_impact_bins", "_prob"]
    
    def __init__(self,
        event_type: type,
        impact_bins: Union[List[float], np.ndarray],   
        prob: Union[List[float], np.ndarray]):
        """Create a new asset event distribution.
        Args:
            event_type: type of event
            impact_bins: non-decreasing impact bin bounds
            prob: probabilities with size [len(intensity_bins) - 1] 
        """
        self._event_type = event_type
        self._impact_bins = np.array(impact_bins) 
        self._prob = np.array(prob)

    def impact_bins(self):
        return zip(self._impact_bins[0:-1], self._impact_bins[1:])

    def mean_impact(self):
        return np.sum((self._impact_bins[:-1] + self._impact_bins[1:]) * self._prob / 2)

    @property
    def prob(self) -> np.ndarray:
        return self._prob