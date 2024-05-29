from enum import Enum
from typing import List, Union, Optional

import numpy as np

from physrisk.kernel.curve import to_exceedance_curve


class ImpactType(Enum):
    damage = 1
    disruption = 2


class ImpactDistrib:
    """Impact distributions specific to an asset."""

    __slots__ = ["__hazard_type", "__impact_bins", "__prob", "impact_type", "__path"]

    def __init__(
        self,
        hazard_type: type,
        impact_bins: Union[List[float], np.ndarray],
        prob: Union[List[float], np.ndarray],
        impact_type: ImpactType = ImpactType.damage,
        path: Optional[List[str]] = None,
    ):
        """Create a new asset event distribution.
        Args:
            event_type: type of event
            impact_bins: non-decreasing impact bin bounds
            prob: probabilities with size [len(intensity_bins) - 1]
            path: path to the hazard event data source
        """
        self.__hazard_type = hazard_type
        self.__impact_bins = np.array(impact_bins)
        self.impact_type = impact_type
        self.__prob = np.array(prob)
        self.__path = path

    def impact_bins_explicit(self):
        return zip(self.__impact_bins[0:-1], self.__impact_bins[1:])

    def mean_impact(self):
        return np.sum((self.__impact_bins[:-1] + self.__impact_bins[1:]) * self.__prob / 2)

    def stddev_impact(self):
        mean = self.mean_impact()
        bin_mids = (self.__impact_bins[:-1] + self.__impact_bins[1:]) / 2
        return np.sqrt(np.sum(self.__prob * (bin_mids - mean) * (bin_mids - mean)))

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

    @property
    def path(self) -> List[str]:
        return self.__path


class EmptyImpactDistrib(ImpactDistrib):
    def __init__(self):
        pass
