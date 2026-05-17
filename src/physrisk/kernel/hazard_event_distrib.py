from typing import List, Union

import numpy as np

from physrisk.kernel.hazards import Hazard

from . import curve


class HazardEventDistrib:
    """Intensity distribution of a hazard event (e.g. inundation depth, wind speed etc),
    specific to an asset -- that is, at the location of the asset."""

    __slots__ = [
        "__event_type",
        "__intensity_bins",
        "__prob",
        "__path",
        "__exceedance",
        "__units",
    ]

    def __init__(
        self,
        event_type: type[Hazard],
        intensity_bins: Union[List[float], np.ndarray],
        prob: Union[List[float], np.ndarray],
        path: List[str],
        units: str,
    ):
        """Create a new asset event distribution.

        Args:
            event_type: type of hazard event.
            intensity_bins: non-decreasing intensity bin edges.
                e.g. bin edges [1.0, 1.5, 2.0] imply two bins: 1.0 < i <= 1.5, 1.5 < i <= 2.0.
            prob: annual probability of occurrence for each intensity bin;
                length must be ``len(intensity_bins) - 1``.
            path: path(s) to the hazard indicator data source.
            units: physical units of the intensity values (e.g. ``"m"`` for inundation depth).
        """
        self.__event_type = event_type
        self.__intensity_bins = np.array(intensity_bins)
        self.__prob = np.array(prob)
        self.__path = path
        self.__units = units

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

    @property
    def units(self) -> str:
        return self.__units


class EmptyHazardEventDistrib(HazardEventDistrib):
    def __init__(self):
        pass
