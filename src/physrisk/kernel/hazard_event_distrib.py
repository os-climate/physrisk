from typing import List, Union

import numpy as np

from . import curve


class HazardEventDistrib:
    """Intensity distribution of a hazard event (e.g. inundation depth, wind speed etc),
    specific to an asset -- that is, at the location of the asset."""

    __slots__ = ["__event_type", "__intensity_bins", "__prob", "__path", "__exceedance"]

    def __init__(
        self,
        event_type: type,
        intensity_bins: Union[List[float], np.ndarray],
        prob: Union[List[float], np.ndarray],
        path: List[str],
    ):
        """Create a new asset event distribution.
        Args:
            event_type: type of event
            intensity_bins: non-decreasing intensity bin edgess.
            e.g. bin edges [1.0, 1.5, 2.0] imply two bins: 1.0 < i <= 1.5, 1.5 < i <= 2.0
            prob: (annual) probability of occurrence for each intensity bin with size [len(intensity_bins) - 1].
            path: path to the hazard indicator data source.
        """
        self.__event_type = event_type
        self.__intensity_bins = np.array(intensity_bins)
        self.__prob = np.array(prob)
        self.__path = path

    def intensity_bins(self):
        return zip(self.__intensity_bins[0:-1], self.__intensity_bins[1:])

    def to_exceedance_curve(self):
        return curve.to_exceedance_curve(self.__intensity_bins, self.__prob)

    @property
    def intensity_bin_edges(self) -> np.ndarray:
        return self.__intensity_bins

    @property
    def prob(self) -> np.ndarray:
        return self.__prob

    @property
    def path(self) -> List[str]:
        return self.__path


class EmptyHazardEventDistrib(HazardEventDistrib):
    def __init__(self):
        pass
