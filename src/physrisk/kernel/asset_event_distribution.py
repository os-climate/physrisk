import numpy as np
from typing import List

class AssetEventDistribution:
    """Event intensity distributions specific to an asset."""
    
    __slots__ = ["_event_type", "_intensity_bins", "_prob"]
    
    def __init__(self,
        event_type: type,
        intensity_bins: List[float],   
        prob: List[float]):
        """Create a new asset event distribution.
        Args:
            event_type: type of event
            intensity_bins: non-decreasing intensity bin bounds
            prob: probabilities with size [len(intensity_bins) - 1] 
        """
        self._event_type = event_type
        self._intensity_bins = np.array(intensity_bins) 
        self._prob = np.array(prob)

    def intensity_bins(self):
        return zip(self._intensity_bins[0:-1], self._intensity_bins[1:])

    @property
    def prob(self) -> np.ndarray:
        return self._prob