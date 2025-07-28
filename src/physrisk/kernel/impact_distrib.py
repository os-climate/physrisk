from enum import Enum
from typing import Sequence, Type, Union

import numpy as np

from physrisk.kernel.curve import to_exceedance_curve
from physrisk.kernel.hazards import Hazard


class ImpactType(Enum):
    damage = 1
    disruption = 2


class ImpactDistrib:
    """Impact distributions specific to an asset."""

    __slots__ = ["_hazard_type", "_impact_bins", "_prob", "impact_type", "_path"]

    def __init__(
        self,
        hazard_type: Type[Hazard],
        impact_bins: Union[Sequence[float], np.ndarray],
        prob: Union[Sequence[float], np.ndarray],
        path: Sequence[str],
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
        self._hazard_type = hazard_type
        self._impact_bins = np.array(impact_bins)
        self.impact_type = impact_type
        self._prob = np.array(prob)
        self._path = path

    def impact_bins_explicit(self):
        return zip(self.__impact_bins[0:-1], self._impact_bins[1:])

    def mean_impact(self):
        return np.sum((self._impact_bins[:-1] + self._impact_bins[1:]) * self._prob / 2)

    def stddev_impact(self):
        mean = self.mean_impact()
        bin_mids = (self._impact_bins[:-1] + self._impact_bins[1:]) / 2
        return np.sqrt(np.sum(self._prob * (bin_mids - mean) * (bin_mids - mean)))

    def above_mean_stddev_impact(self):
        mean = self.mean_impact()
        bin_mids = (self._impact_bins[:-1] + self._impact_bins[1:]) / 2
        above_mean_bins = bin_mids[mean <= bin_mids]
        if len(above_mean_bins) == 0:
            return 0.0
        if len(above_mean_bins) == len(self._prob):
            return self.stddev_impact()
        above_mean_probs = self._prob[-len(above_mean_bins) :] / np.sum(
            self._prob[-len(above_mean_bins) :]
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
        return to_exceedance_curve(self._impact_bins, self._prob)

    @property
    def hazard_type(self) -> type:
        return self._hazard_type

    @property
    def impact_bins(self) -> np.ndarray:
        return self._impact_bins

    @property
    def prob(self) -> np.ndarray:
        return self._prob

    @property
    def path(self) -> Sequence[str]:
        return self._path


class EmptyImpactDistrib(ImpactDistrib):
    def __init__(self):
        pass


class PlaceholderImpactDistrib(ImpactDistrib):
    def __init__(self):
        pass
