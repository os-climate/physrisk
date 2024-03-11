import typing
import unittest

import numpy as np

from physrisk.kernel.assets import WindTurbine
from physrisk.kernel.events import (
    EmpiricalMultivariateDistribution,
    MultivariateDistribution,
    calculate_cumulative_probs,
)
from physrisk.kernel.hazard_model import HazardEventDataResponse

TAsset = typing.TypeVar("TAsset", contravariant=True)


class SupportsEventImpact(typing.Protocol[TAsset]):
    def get_impact(self, asset: TAsset, event_data: HazardEventDataResponse) -> MultivariateDistribution:
        pass


class WindTurbineModel(SupportsEventImpact[WindTurbine]):
    """Placeholder wind turbine model to be populated."""

    def prob_collapse(self, turbine: WindTurbine, wind_speed_hub: np.ndarray):
        """Calculates probability of turbine collapse for a num ber of events given the wind speed
        at the hub per event and characteristics of the turbine.

        Args:
            turbine (WindTurbine): Wind turbine asset.
            wind_speed_hub (np.ndarray): 1-D array of wind speeds at hub height for each event (length of array is number of events). # noqa: E501

        Returns:
            np.ndarray: 1-D array of probabilities of collapse for each event (length of array is number of events).
        """
        # just a placeholder model that returns a probability of 0.3 for all events, regardless of wind speed!
        return np.ones_like(wind_speed_hub) * 0.3

    def get_impact(self, asset: WindTurbine, event_data: HazardEventDataResponse):
        """Returns the probability distributions of fractional damage to the turbine for each event.

        Args:
            asset (WindTurbine): Wind turbine asset.
            event_data (HazardEventDataResponse): Provides wind speeds for different events with different probabilities. # noqa: E501

        Raises:
            NotImplementedError: Supports only the case of single probability bin: i.e. for each event the wind speed is deterministic. # noqa: E501

        Returns:
            MultivariateDistribution: Probability distribution of impacts associated with events.
        """
        intens = event_data.intensities
        # shape is (nb_events, nb_prob_bins)
        if intens.ndim > 1 and intens.shape[0] != 1:
            # only the single probability bin case implemented
            raise NotImplementedError()
        wind_speed = intens.squeeze(axis=0)  # vector of wind speeds
        pc = self.prob_collapse(asset, wind_speed)
        bins_lower = np.array([0.0, 1.0])
        bins_upper = np.array([0.0, 1.0])
        probs = np.stack([1.0 - pc, pc], axis=1)
        return EmpiricalMultivariateDistribution(bins_lower, bins_upper, probs)


class TestWindTurbineModels:
    def test_cumulative_probs(self):
        """Test calculation of cumulative probability from a combination of lumped probability and uniform probability
        density bins.
        """
        bins_lower = np.array([2.0, 3.0])
        bins_upper = np.array([2.5, 3.5])
        probs = np.array([[0.5, 0.5], [0.2, 0.8]])
        values, cum_probs = calculate_cumulative_probs(bins_lower, bins_upper, probs)
        np.testing.assert_almost_equal(values, [2.0, 2.5, 3.0, 3.5])
        np.testing.assert_almost_equal(cum_probs[0, :], [0, 0.5, 0.5, 1.0])
        np.testing.assert_almost_equal(cum_probs[1, :], [0, 0.2, 0.2, 1.0])

        bins_lower = np.array([2.0])
        bins_upper = np.array([2.0])
        probs = np.array([[1.0], [1.0]])
        values, cum_probs = calculate_cumulative_probs(bins_lower, bins_upper, probs)
        np.testing.assert_almost_equal(values, [2.0, 2.0])
        np.testing.assert_almost_equal(cum_probs[0, :], [0, 1.0])
        np.testing.assert_almost_equal(cum_probs[1, :], [0, 1.0])

        bins_lower = np.array([2.0, 2.0])
        bins_upper = np.array([2.0, 3.0])
        probs = np.array([[0.5, 0.5], [0.1, 0.9]])
        values, cum_probs = calculate_cumulative_probs(bins_lower, bins_upper, probs)
        np.testing.assert_almost_equal(values, [2.0, 2.0, 3.0])
        np.testing.assert_almost_equal(cum_probs[0, :], [0.0, 0.5, 1.0])
        np.testing.assert_almost_equal(cum_probs[1, :], [0, 0.1, 1.0])

        bins_lower = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        bins_upper = np.array([2.0, 3.0, 5.0, 5.5, 6.0])
        probs = np.array([[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1, 0.6]])
        values, cum_probs = calculate_cumulative_probs(bins_lower, bins_upper, probs)
        np.testing.assert_almost_equal(values, [2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.0])
        np.testing.assert_almost_equal(cum_probs[0, :], [0, 0.1, 0.1, 0.3, 0.3, 0.6, 0.8, 0.8, 1.0])
        np.testing.assert_almost_equal(cum_probs[1, :], [0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 1.0])

    def test_sampling(self):
        """Test sampling from probability distributions comprising two lumped probabilities."""
        bins_lower = np.array([2.0, 3.0])
        bins_upper = np.array([2.0, 3.0])
        probs = np.array([[0.3, 0.7], [0.4, 0.6]])
        pdf = EmpiricalMultivariateDistribution(bins_lower, bins_upper, probs)
        gen = np.random.Generator(np.random.MT19937(111))
        samples = pdf.inv_cumulative_marginal_probs(gen.random(size=(2, 10000)))
        check_2 = np.count_nonzero(samples[0, :] == 2.0)
        check_3 = np.count_nonzero(samples[0, :] == 3.0)
        assert check_2 == 3062
        assert check_3 == 6938
        check_2 = np.count_nonzero(samples[1, :] == 2.0)
        check_3 = np.count_nonzero(samples[1, :] == 3.0)
        assert check_2 == 3924
        assert check_3 == 6076

    @unittest.skip("Performance test: slow.")
    def test_performance(self):
        nb_events = 10000
        nb_samples = 10
        bins_lower = np.array([0.0, 1.0])
        bins_upper = np.array([0.0, 1.0])
        probs = np.tile(np.array([[0.3, 0.7]]), reps=(nb_events, 1))
        pdf = EmpiricalMultivariateDistribution(bins_lower, bins_upper, probs)
        gen = np.random.Generator(np.random.MT19937(111))
        uniforms = gen.random(size=(nb_events, nb_samples))
        samples = pdf.inv_cumulative_marginal_probs(uniforms)
        for i in range(1000):
            uniforms = gen.random(size=(nb_events, nb_samples))
            samples = pdf.inv_cumulative_marginal_probs(uniforms)
        print(samples)

    def test_rotor_damage_event_based(self):
        """Test demonstrating how WindTurbineModel can be used in an event-based calculation."""
        # The data response for a single asset will be a HazardEventDataResponse.
        # The format supports a distribution of hazard intensities *per event*
        # but here we simply have a single realisation per event.

        # The hazard model is responsible for sourcing the events. Here we provide output from
        # that model directly.

        rng = np.random.Generator(np.random.MT19937(111))
        asset1, asset2 = WindTurbine(), WindTurbine()
        nb_events = 20
        response_asset1 = HazardEventDataResponse(np.array([1.0]), np.array(rng.weibull(a=4, size=[1, nb_events])))
        response_asset2 = HazardEventDataResponse(np.array([1.0]), np.array(rng.weibull(a=4, size=[1, nb_events])))

        turbine_model = WindTurbineModel()

        # provides the impact distributions for each asset
        impacts_asset1 = turbine_model.get_impact(asset1, response_asset1)
        impacts_asset2 = turbine_model.get_impact(asset2, response_asset2)

        # we sample 10 times for each event for each asset
        uniforms = rng.random(size=(nb_events, 10))
        samples_asset1 = impacts_asset1.inv_cumulative_marginal_probs(uniforms)
        samples_asset2 = impacts_asset2.inv_cumulative_marginal_probs(uniforms)

        # we can then combine samples and calculate measures...
        # for now just sanity-check that we get the approx. 0.3 of total loss in events from placeholder.
        np.testing.assert_almost_equal(np.count_nonzero(samples_asset1 == 1.0) / samples_asset1.size, 0.31)
        np.testing.assert_almost_equal(np.count_nonzero(samples_asset2 == 1.0) / samples_asset2.size, 0.31)
