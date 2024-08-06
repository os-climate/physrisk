from typing import List, Protocol

import numpy as np
from numba import float64, njit
from numba.experimental import jitclass


# @njit(cache=True)
def calculate_cumulative_probs(
    bins_lower: np.ndarray, bins_upper: np.ndarray, probs: np.ndarray
):
    # note: in some circumstances we could exclude the two extreme points and rely on flat extrapolation
    # this implementation retains points for clarity, sacrificing some performance
    assert bins_lower.size == bins_upper.size
    assert probs.shape[1] == bins_lower.size
    nb_bins = bins_lower.size  # aka M: number of bins
    n = probs.shape[0]
    nb_points = bins_lower.size * 2 - np.count_nonzero(
        bins_lower[1:] == bins_upper[:-1]
    )
    cum_prob = np.zeros(n)
    values = np.zeros(nb_points)
    cum_probs = np.zeros(shape=(n, nb_points))
    index = 0
    values[0] = bins_lower[0]
    cum_probs[:, 0] = cum_prob
    for i in range(nb_bins):
        # index is index of last point in cumulative curve
        if bins_lower[i] == values[index]:
            # bin is contiguous with previous: add just upper bound
            cum_prob += probs[:, i]
            values[index + 1] = bins_upper[i]
            cum_probs[:, index + 1] = cum_prob
            index += 1
        else:
            # bin not contiguous: add lower and upper bounds
            values[index + 1] = bins_lower[i]
            cum_probs[:, index + 1] = cum_prob
            cum_prob += probs[:, i]
            values[index + 2] = bins_upper[i]
            cum_probs[:, index + 2] = cum_prob
            index += 2
    return values, cum_probs


@njit(cache=True)
def sample_from_cumulative_probs(
    values: np.ndarray, cum_probs: np.ndarray, uniforms: np.ndarray
):
    n = cum_probs.shape[0]
    nb_samples = uniforms.shape[1]
    assert uniforms.shape[0] == n
    samples = np.zeros(shape=(n, nb_samples))
    for i in range(n):
        samples[i, :] = np.interp(uniforms[i, :], cum_probs[i, :], values)
    return samples


class MultivariateDistribution(Protocol):
    def inv_cumulative_marginal_probs(self, cum_probs: np.ndarray): ...


class EmpiricalMultivariateDistribution(MultivariateDistribution):
    """Stores an N dimensional empirical probability density function."""

    def __init__(
        self, bins_lower: np.ndarray, bins_upper: np.ndarray, probs: np.ndarray
    ):
        """N marginal probability distributions are each represented as a set of bins of
        uniform probability density.

        Args:
            bins_lower (np.ndarray): Lower bounds of M probability bins (M,).
            bins_upper (np.ndarray): Upper bounds of M probability bins (M,).
            probs (np.ndarray): Probabilities of bins (N, M).

        Raises:
            ValueError: _description_
        """
        if (
            bins_lower.ndim > 1
            or bins_upper.ndim > 1
            or bins_lower.size != bins_upper.size
        ):
            raise ValueError("bin upper and lower bounds must be 1-D and same size.")
        if probs.ndim != 2 or probs.shape[1] != bins_upper.size:
            raise ValueError("probabilities must be (N, M).")
        if np.any(bins_lower[1:] < bins_lower[:-1]) or np.any(
            bins_upper[1:] < bins_upper[:-1]
        ):
            raise ValueError("bins must be non-decreasing.")
        if np.any(bins_upper[1:] < bins_lower[:-1]):
            raise ValueError("bins may not overlap.")
        self.bins_lower = bins_lower
        self.bins_upper = bins_upper
        self.probs = probs

    def inv_cumulative_marginal_probs(self, cum_probs: np.ndarray):
        """Calculate inverse cumulative probabilities for each of the N
        marginal probability distributions. By definition, this is the
        vectorized form of get_inv_cumulative_marginal_prob, vectorized
        as a performance optimization.

        Args:
            cum_probs (np.ndarray): Cumulative probabilities (N, P), P being number of samples.
            axis (int): Specifies the axis of the N events.
        """
        values_dist, cum_probs_dist = calculate_cumulative_probs(
            self.bins_lower, self.bins_upper, self.probs
        )
        return sample_from_cumulative_probs(values_dist, cum_probs_dist, cum_probs)


def event_samples(
    impacts_bins: np.ndarray, probs: List[np.ndarray], nb_events: int, nb_samples: int
):
    if any([p.size != 1 and p.size != nb_events for p in probs]):
        raise ValueError(
            f"probabilities must be scalar or vector or length {nb_events}."
        )

    return event_samples_numba(impacts_bins, probs, nb_events, nb_samples)


def find(elements: np.ndarray, value):
    """In case we need a specific formulation..."""
    current: int = 0
    lower: int = 0
    upper: int = elements.size - 1
    while lower != upper - 1:
        current = (lower + upper) // 2
        if elements[current] >= value:
            upper = current
        else:
            lower = current
    return current


@njit(cache=True)
def event_samples_numba(
    impacts_bins: np.ndarray, probs: List[np.ndarray], nb_events: int, nb_samples: int
):
    samples = np.zeros(shape=(nb_samples, nb_events))
    np.random.seed(111)
    cum_probs = np.zeros(len(probs))
    for i in range(nb_events):
        # for each event calculate cumulative probability distribution
        sum = 0.0
        for j in range(len(probs)):
            sum += probs[j][i]
            cum_probs[j] = sum
        cum_probs[-1] = np.minimum(cum_probs[-1], 1.0)
        u = np.random.rand(nb_samples)
        samples[:, i] = np.interp(u, cum_probs, impacts_bins[1:])
    return samples


spec = [
    ("values", float64[:]),
    ("cum_probs", float64[:]),
]


@jitclass(spec)
class CumulativeProb(object):
    def __init__(self, values: np.ndarray, cum_probs: np.ndarray):
        self.values = values
        self.cum_probs = cum_probs

    @property
    def size(self):
        return self.values.size
