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

    def standard_deviation(self):
        return self._standard_deviation(
            self._impact_bins, self._prob, include_zero=True
        )

    def semi_standard_deviation(self):
        mean = self.mean_impact()
        # we may have one bin that straddles the mean
        index = np.searchsorted(self._impact_bins, mean, "right")
        # straddling bin is the one from index-1 to index
        # if this bin has zero width, we only keep the bins above the mean
        # if not, we split the bin
        if index == 0:
            bins = self._impact_bins
            prob = self._prob
        elif self._impact_bins[index - 1] == mean:
            bins = self._impact_bins[index:]
            prob = self._prob[index:]
        else:
            bins = np.insert(self._impact_bins[index:], 0, mean)
            prob_frac = (self._impact_bins[index] - mean) / (
                self._impact_bins[index] - self._impact_bins[index - 1]
            )
            prob = np.insert(self._prob[index:], 0, self._prob[index - 1] * prob_frac)
        return self._standard_deviation(bins, prob, include_zero=False)

    def _standard_deviation(
        self, impact_bins: np.ndarray, prob: np.ndarray, include_zero: bool = True
    ):
        """The impact bins only give the probabilities for impacts > 0.
        If include_zero is True, the zero impact is included.
        """
        bins_lower = impact_bins[:-1]
        bins_upper = impact_bins[1:]
        zero_width = bins_lower == bins_upper
        std_contrib = np.zeros_like(bins_lower)
        mean = self.mean_impact()
        std_contrib[~zero_width] = (
            (
                (bins_upper[~zero_width] - mean) ** 3
                - (bins_lower[~zero_width] - mean) ** 3
            )
            * prob[~zero_width]
            / (3.0 * (bins_upper[~zero_width] - bins_lower[~zero_width]))
        )
        std_contrib[zero_width] = (bins_lower[zero_width] - mean) ** 2 * prob[
            zero_width
        ]
        zero_contrib = 0.0
        if include_zero:
            zero_contrib = (1.0 - sum(prob)) * (mean**2)
        return np.sqrt(np.sum(std_contrib) + zero_contrib)

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
