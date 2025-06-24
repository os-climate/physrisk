from enum import Enum
from typing import List, Type, Union

import numpy as np

from physrisk.kernel.curve import to_exceedance_curve
from physrisk.kernel.hazards import Hazard


class ImpactType(Enum):
    damage = 1
    disruption = 2


class ImpactDistrib:
    """Impact distributions specific to an asset."""

    __slots__ = ["__hazard_type", "__impact_bins", "__prob", "impact_type", "__path"]

    def __init__(
        self,
        hazard_type: Type[Hazard],
        impact_bins: Union[List[float], np.ndarray],
        prob: Union[List[float], np.ndarray],
        path: List[str],
        impact_type: ImpactType = ImpactType.damage,
    ):
        """Create a new impact distribution.
        Args:
            hazard_type: Type of hazard.
            impact_bins: Non-decreasing impact bin bounds.
            prob: Probabilities with size [len(impact_bins) - 1].
            path: Path to the hazard indicator data source.
            impact_type: Type of impact: damage or disruption.
        """
        self.__hazard_type = hazard_type
        self.__impact_bins = np.array(impact_bins)
        self.impact_type = impact_type
        self.__prob = np.array(prob)
        self.__path = path

    def impact_bins_explicit(self):
        return zip(self.__impact_bins[0:-1], self.__impact_bins[1:])

    def mean_impact(self):
        return np.sum(
            (self.__impact_bins[:-1] + self.__impact_bins[1:]) * self.__prob / 2
        )

    def stddev_impact(self):
        mean = self.mean_impact()
        bin_mids = (self.__impact_bins[:-1] + self.__impact_bins[1:]) / 2
        return np.sqrt(np.sum(self.__prob * (bin_mids - mean) * (bin_mids - mean)))

    def above_mean_stddev_impact(self):
        mean = self.mean_impact()
        bin_mids = (self.__impact_bins[:-1] + self.__impact_bins[1:]) / 2
        above_mean_bins = bin_mids[mean <= bin_mids]
        if len(above_mean_bins) == 0:
            return 0.0
        if len(above_mean_bins) == len(self.__prob):
            return self.stddev_impact()
        above_mean_probs = self.__prob[-len(above_mean_bins) :] / np.sum(
            self.__prob[-len(above_mean_bins) :]
        )
        above_mean_mean = np.sum(above_mean_bins * above_mean_probs / 2)
        return np.sqrt(
            np.sum(
                above_mean_probs
                * (above_mean_bins - above_mean_mean)
                * (above_mean_bins - above_mean_mean)
            )
        )

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
