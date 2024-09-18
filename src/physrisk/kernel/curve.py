from typing import List, Union

import numpy as np


def add_x_value_to_curve(x, curve_x, curve_y):
    """Add an x value to a curve, interpolated from the existing curve.
    curve_x and curve_y are the curve x and y values.
    curve_x is sorted non-decreasing. This function may be used to align curves and bins.
    """
    # note some care needed as multiple identical curve_x are permitted: cannot simply use np.interp
    # this might be a candidate for numba optimization

    i = np.searchsorted(curve_x, x)

    if i == len(curve_y):
        curve_y = np.insert(curve_y, i, curve_y[i - 1])  # flat extrapolation
        curve_x = np.insert(curve_x, i, x)
    elif x == curve_x[i]:
        # point already exists; nothing to do
        return curve_x, curve_y, i
    elif i == 0:
        curve_y = np.insert(curve_y, 0, curve_y[0])  # flat extrapolation
        curve_x = np.insert(curve_x, 0, x)
    else:
        pl, pu = curve_y[i - 1], curve_y[i]
        il, iu = curve_x[i - 1], curve_x[i]
        # linear interpolation; quadratic interpolation (linear in probability density) may also be of interest
        prob = pl + (x - il) * (pu - pl) / (iu - il)
        curve_y = np.insert(curve_y, i, prob)
        curve_x = np.insert(curve_x, i, x)

    return curve_x, curve_y, i


def to_exceedance_curve(bin_edges, probs):
    """An exceedance curve gives the probability that the random variable is greater than the value,
    a type of cumulative probability.
    """
    nz = np.asarray(probs > 0).nonzero()
    fnz = nz[0][0] if len(nz[0]) > 0 else 0
    nz_values = bin_edges[fnz:]
    nz_probs = probs[fnz:]
    cum_prob = np.insert(np.cumsum(nz_probs[::-1]), 0, 0.0)[::-1]
    return ExceedanceCurve(cum_prob, nz_values)


def process_bin_edges_and_probs(bin_edges, probs, range_fraction=0.05):
    r = bin_edges[-1] - bin_edges[0]
    r = bin_edges[0] if r == 0 else r
    new_edges = []
    new_probs = []
    # say we have edges
    # 0, 1, 2, 3, 3, 3, 5, 7
    # we want to convert to
    # 0, 1, 2, 3 + d, 5, 7
    # 2d = (7 - 0) * 0.01
    i = 0
    while i < len(bin_edges):
        j = __next_non_equal_index(bin_edges, i)
        if j == i + 1:
            i = i + 1
            new_edges.append(bin_edges[i])
            new_probs.append(probs[i])
            continue
        if j >= len(bin_edges):
            delta = r * range_fraction
        else:
            delta = min(r * range_fraction, 0.25 * (bin_edges[j] - bin_edges[i]))
        new_edges.append(bin_edges[i])
        new_edges.append(bin_edges[i] + delta)
        new_probs.append(np.sum(probs[i:j]))
        i = j
    return new_edges, new_probs


def process_bin_edges_for_graph(bin_edges, range_fraction=0.05):
    """Process infinitessimal (zero width) bins for graph display.
    We make width 5% of range or 1/4 the width to the next bin edge, whichever is smaller
    """
    r = bin_edges[-1] - bin_edges[0]
    r = bin_edges[0] if r == 0 else r
    new_edges = np.copy(bin_edges)
    # say we have edges
    # 0, 1, 2, 3, 3, 3, 5, 7
    # we want to convert to
    # 0, 1, 2, 3, 3 + d, 3 + 2d, 5, 7
    # 2d = (7 - 0) * 0.01
    i = 0
    while i < len(bin_edges):
        j = __next_non_equal_index(bin_edges, i)
        if j == i + 1:
            i = i + 1
            continue
        if j >= len(bin_edges):
            delta = r * range_fraction / (j - i - 1)
        else:
            delta = min(r * range_fraction, 0.25 * (bin_edges[j] - bin_edges[i])) / (
                j - i - 1
            )
        offset = delta
        for k in range(i + 1, j):
            new_edges[k] = new_edges[k] + offset
            offset += delta
        i = j
    return new_edges


def __next_non_equal_index(ndarray, i):
    j = i + 1
    c = ndarray[i]
    while j < len(ndarray) and ndarray[j] == c:
        j = j + 1
    return j


class ExceedanceCurve:
    """A point on an exceedance curve comprises an value, v, and a probability, p.
    p is the probability that the random variable >= v, e.g. an event occurs with event value (e.g. intensity) >= v.
    """

    __slots__ = ["probs", "values"]

    def __init__(
        self,
        probs: Union[List[float], np.ndarray],
        values: Union[List[float], np.ndarray],
    ):
        """Create a new asset event distribution.
        Args:
            probs: exceedance probabilities (must be sorted and decreasing).
            values: values (must be sorted and non-decreasing).
        """

        # probabilities must be sorted and decreasing
        # values must be sorted and non-decreasing (intens[i + 1] >= intens[i])
        if len(probs) != len(values):
            raise ValueError("same number of probabilities and values expected")
        if not np.all(np.diff(probs) <= 0):
            raise ValueError("probs must be sorted and decreasing")
        if not np.all(np.diff(values) >= 0):
            raise ValueError("values must be sorted and non-decreasing")

        self.probs = np.array(probs)
        self.values = np.array(values)

    def add_value_point(self, value):
        """Add a point to the curve with specified value and exceedance
        probability determined from existing curve by linear interpolation.
        """
        values, probs, _ = add_x_value_to_curve(value, self.values, self.probs)
        return ExceedanceCurve(probs, values)

    def get_value(self, prob):
        return np.interp(prob, self.probs[::-1], self.values[::-1])

    def get_probability_bins(self, include_last: bool = False):
        r"""Convert from exceedance (cumulative) probability to bins of constant probability density.
        This is equivalent to the assumption of linear interpolation of exceedance points.

        .. math::
            p^\text{b}_i = p^\text{e}_{i + 1} - p^\text{e}_i

        Returns:
            value_bins (ndarray), probs: The contiguous bin lower and upper values, probabilities of each bin.
            If value_bins is of length n then there are n-1 bins and n-1 probabilities

        """
        value_bins = self.values[:]
        probs = self.probs[:-1] - self.probs[1:]  # type: ignore
        if include_last or len(self.values) == 1:
            value_bins = np.append(
                value_bins, value_bins[-1]
            )  # last bin has zero width
            probs = np.append(probs, self.probs[-1])
        return value_bins, probs

    def get_samples(self, uniforms):
        """Return value, v, for each probability p in uniforms such that p is the probability that the random variable
        < v."""
        return np.where(
            uniforms > (1.0 - self.probs[0]),
            np.interp(uniforms, 1.0 - self.probs, self.values),
            0.0,
        )
