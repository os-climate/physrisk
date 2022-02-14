from typing import List, Union
import physrisk.kernel.curve as curve

import numpy as np


class Distribution:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev


class ImpactCurve:
    """Provides the impact on an asset, an impact being either a fractional damage or disruption,
    that occurs as a result of a hazard event of a given intensity."""

    __slots__ = ["intensities", "impacts", "impact_distribs"]

    def __init__(
        self,
        intensities: Union[List[float], np.ndarray],
        impacts: Union[List[float], np.ndarray] = None,
        impact_distribs=None,
    ):
        """Create a new asset event distribution.
        Args:
            intensities: possible intensities of hazard event.
            impacts: fractional damage or fractional average disruption occurring as a result of hazard event of given intensity.
            distributions: provides the pdf and optiononally the cdf of the impact distribution
        """

        # probabilities must be sorted and decreasing
        # values must be sorted and non-decreasing (intens[i + 1] >= intens[i])
        if impacts is not None and len(intensities) != len(impacts):
            raise ValueError("same number of intensities and impacts expected")
        if not np.all(np.diff(intensities) > 0):
            raise ValueError("intensities must be sorted and increasing")

        self.intensities = np.array(intensities)
        self.impacts = np.array(impacts)
        self.impact_distribs = impact_distribs

    def to_prob_matrix(self, impact_bin_edges):
        # construct a cdf probability matrix at each intensity point
        # the probability is the prob that the impact is greater than the specified
        cdf_matrix = np.empty([len(impact_bin_edges), len(self.intensities)])

        for i, _ in enumerate(self.intensities):
            cdf_matrix[:, i] = self.impact_distribs[i].cdf(impact_bin_edges)

        prob_matrix = cdf_matrix[1:, :] - cdf_matrix[:-1, :]

        return prob_matrix

        # first we calculate pdf at centre of impact bins for curve intensities
        # from scipy import interpolate
        # impact_bin_centres = (impact_bin_edges[1:] + impact_bin_edges[:-1]) / 2
        # pdf_matrix = np.empty([len(impact_bin_centres), len(self.intensities)])

        # next create interpolation function
        # f = interpolate.interp2d(impact_bin_centres, self.intensities, pdf_matrix, kind='linear')

        # and interpolate the intensities
        # intensity_bin_centres = (intensity_bin_edges[1:] + intensity_bin_edges[:-1]) / 2
        # prob_matrix = f(impact_bin_centres, intensity_bin_centres)
        # prob_matrix = prob_matrix / np.sum(prob_matrix, 0)

        # return prob_matrix
