from enum import Enum
from typing import List, Union

import numpy as np

from .curve import to_exceedance_curve


class ImpactType(Enum):
    damage = 1
    disruption = 2


class ImpactDistrib:
    """Impact distributions specific to an asset."""

    __slots__ = ["__hazard_type", "__impact_bins", "__prob", "impact_type"]

    def __init__(
        self,
        hazard_type: type,
        impact_bins: Union[List[float], np.ndarray],
        prob: Union[List[float], np.ndarray],
        impact_type: ImpactType = ImpactType.damage,
    ):
        """Create a new asset event distribution.
        Args:
            event_type: type of event
            impact_bins: non-decreasing impact bin bounds
            prob: probabilities with size [len(intensity_bins) - 1]
        """
        self.__hazard_type = hazard_type
        self.__impact_bins = np.array(impact_bins)
        self.impact_type = impact_type
        self.__prob = np.array(prob)

    def impact_bins_explicit(self):
        return zip(self.__impact_bins[0:-1], self.__impact_bins[1:])

    def mean_impact(self):
        return np.sum((self.__impact_bins[:-1] + self.__impact_bins[1:]) * self.__prob / 2)

    def to_exceedance_curve(self):
        return to_exceedance_curve(self.__impact_bins, self.__prob)

    @property
    def hazard_type(self) -> type:
        return self.__hazard_type

    @property
    def impact_bins(self) -> np.ndarray:
        return self.__impact_bins

    @property
    def prob(self) -> np.ndarray:
        return self.__prob
