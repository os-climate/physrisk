import physrisk.kernel.curve as curve
import numpy as np
from typing import Optional, List, Union

from physrisk.kernel.exceedance_curve import ExceedanceCurve

class AssetEventDistrib:
    """Event intensity distributions specific to an asset."""
    
    __slots__ = ["__event_type", "__intensity_bins", "__prob", "__exceedance"]
    
    def __init__(self,
        event_type: type,
        intensity_bins: Union[List[float], np.ndarray],   
        prob: Union[List[float], np.ndarray],
        exceedance: Optional[ExceedanceCurve] = None):
        """Create a new asset event distribution.
        Args:
            event_type: type of event
            intensity_bins: non-decreasing intensity bin bounds
            prob: probabilities with size [len(intensity_bins) - 1] 
            exceedence: exceedence curve, for reference
        """
        self.__event_type = event_type
        self.__intensity_bins = np.array(intensity_bins) 
        self.__prob = np.array(prob)
        self.__exceedance = exceedance

    def intensity_bins_explicit(self):
        return zip(self.__intensity_bins[0:-1], self.__intensity_bins[1:])

    #def to_exceedance_curve(self):
    #    return curve.to_exceedance_curve(self.__intensity_bins, self.__prob) 

    @property
    def intensity_bins(self) -> np.ndarray:
        return self.__intensity_bins

    @property
    def prob(self) -> np.ndarray:
        return self.__prob

    @property
    def exceedance(self):
        return self.__exceedance