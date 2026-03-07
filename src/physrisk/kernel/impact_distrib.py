from enum import Enum
import sys
from typing import Optional, Sequence, Type, Union

import numpy as np

from physrisk.kernel.curve import to_exceedance_curve
from physrisk.kernel.hazards import Hazard


class ImpactType(Enum):
    damage = 1
    disruption = 2


class ImpactDistrib:
    """Impact distributions specific to an asset."""

    __slots__ = [
        "_hazard_type",
        "_impact_bin_edges",
        "_probabilities",
        "_hazard_indicator_id",
        "_impact_type",
        "_path",
    ]

    def __init__(
        self,
        hazard_type: Type[Hazard],
        impact_bin_edges: Union[Sequence[float], np.ndarray],
        probabilities: Union[Sequence[float], np.ndarray],
        hazard_indicator_id: Optional[str] = None,
        impact_type: ImpactType = ImpactType.damage,
        path: Sequence[str] = [],
    ):
        """Create a new impact distribution.
        Args:
            hazard_type: Type of hazard.
            impact_bins: Non-decreasing impact bin edges/bounds. e.g. bin edges [0.0, 0.1, 0.5, 1.0] implies three bins: 0.0 < i <= 0.1, 0.1 < i <= 0.5, 0.5 < i <= 1.0.
            probabilities: Probabilities of each bin; has size [len(impact_bins) - 1].
            hazard_indicator_id: Hazard indicator ID, used when multiple impacts are calculated from different hazard indicators of the same hazard type.
            impact_type: Type of impact: damage or disruption.
            path: Paths (unique identifiers) of the hazard indicator data sources used to calculate the impact distribution. Provides the main hazard indicator (specified by the ID), but also any additional hazard indicators. For example, for flood, 'flood_depth' but also standard of protection data sources.
        """
        self._hazard_type = hazard_type
        self._hazard_indicator_id = (
            sys.intern(hazard_indicator_id) if hazard_indicator_id is not None else None
        )
        self._impact_bin_edges = np.array(impact_bin_edges)
        self._impact_type = impact_type
        self._probabilities = np.array(probabilities)
        self._path = path

    def impact_bins_explicit(self):
        return zip(self._impact_bin_edges[0:-1], self._impact_bin_edges[1:])

    def mean_impact(self):
        return np.sum(
            (self._impact_bin_edges[:-1] + self._impact_bin_edges[1:])
            * self._probabilities
            / 2
        )

    def standard_deviation(self):
        return self._standard_deviation(
            self._impact_bin_edges, self._probabilities, include_zero=True
        )

    def semi_standard_deviation(self):
        mean = self.mean_impact()
        # we may have one bin that straddles the mean
        index = np.searchsorted(self._impact_bin_edges, mean, "right")
        # straddling bin is the one from index-1 to index
        # if this bin has zero width, we only keep the bins above the mean
        # if not, we split the bin
        if index == 0:
            bins = self._impact_bin_edges
            prob = self._probabilities
        elif self._impact_bin_edges[index - 1] == mean:
            bins = self._impact_bin_edges[index:]
            prob = self._probabilities[index:]
        else:
            bins = np.insert(self._impact_bin_edges[index:], 0, mean)
            prob_frac = (self._impact_bin_edges[index] - mean) / (
                self._impact_bin_edges[index] - self._impact_bin_edges[index - 1]
            )
            prob = np.insert(
                self._probabilities[index:],
                0,
                self._probabilities[index - 1] * prob_frac,
            )
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
        return to_exceedance_curve(self._impact_bin_edges, self._probabilities)

    @property
    def hazard_type(self) -> type:
        return self._hazard_type

    @property
    def hazard_indicator_id(self) -> str | None:
        return self._hazard_indicator_id

    @property
    def impact_bin_edges(self) -> np.ndarray:
        return self._impact_bin_edges

    @property
    def impact_type(self):
        return self._impact_type

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities

    @property
    def path(self) -> Sequence[str]:
        return self._path


class EmptyReason(int, Enum):
    NO_VULNERABILITY = (
        2  # models/config yielded no vulnerability for asset/hazard combination
    )
    NO_DATA = 1  # some hazard indicator data could not be sourced
    EXCEPTION = 3  # some exception was raised during impact calculation


class EmptyImpactDistrib(ImpactDistrib):
    def __init__(self, empty_reason: EmptyReason = EmptyReason.NO_VULNERABILITY):
        self.empty_reason = empty_reason


class PlaceholderImpactDistrib(ImpactDistrib):
    def __init__(self):
        pass
