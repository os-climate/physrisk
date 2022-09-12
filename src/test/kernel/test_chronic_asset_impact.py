import unittest
from test.data.hazard_model_store import TestData, mock_hazard_model_store_heat
from typing import Iterable, Union

import numpy as np
from scipy.stats import norm

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.kernel import calculation
from physrisk.kernel.assets import Asset, IndustrialActivity
from physrisk.kernel.hazard_model import HazardDataRequest, HazardDataResponse, HazardParameterDataResponse
from physrisk.kernel.hazards import ChronicHeat
from physrisk.kernel.impact_distrib import ImpactDistrib, ImpactType
from physrisk.kernel.vulnerability_model import VulnerabilityModelBase


class ExampleChronicHeatModel(VulnerabilityModelBase):
    """Example chronic vulnerability model for extreme heat (summary should fit on one line).

    More decription below as per
    https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html
    """

    def __init__(self, model: str = "mean_degree_days_above_32c"):
        super().__init__(model, ChronicHeat)  # opportunity to give a model hint, but blank here
        annual_time_loss_mins_per_degree_day_mean = 8  # from paper...:
        annual_time_loss_mins_per_degree_day_std = 2
        MINS_IN_YEAR = 60 * 8 * 240
        self.annual_fraction_loss_per_degree_day_mean = annual_time_loss_mins_per_degree_day_mean / MINS_IN_YEAR
        self.annual_fraction_loss_per_degree_day_std = annual_time_loss_mins_per_degree_day_std / MINS_IN_YEAR
        # load any data needed by the model here in the constructor

    def get_data_requests(
        self, asset: Asset, *, scenario: str, year: int
    ) -> Union[HazardDataRequest, Iterable[HazardDataRequest]]:
        """Request the hazard data needed by the vulnerability model for a specific asset
        (this is a Google-style doc string)

        Args:
            asset: Asset for which data is requested.
            scenario: Climate scenario of calculation.
            year: Projection year of calculation.

        Returns:
            Single or multiple data requests.
        """

        # specify hazard data needed. Model string is hierarchical and '/' separated.
        model = "mean_degree_days/above/32c"

        return [
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario="historical",
                year=1980,
                model=model,
            ),
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                model=model,
            ),
        ]

    def get_impact(self, asset: Asset, data_responses: Iterable[HazardDataResponse]) -> ImpactDistrib:
        """Calcaulate impact (disruption) of asset based on the hazard data returned.

        Args:
            asset: Asset for which impact is calculated.
            data_responses: responses to the hazard data requests generated in get_data_requests.

        Returns:
            Probability distribution of impacts.
        """
        baseline_dd_above_mean, scenario_dd_above_mean = data_responses

        # check expected type; can maybe do this more nicely
        assert isinstance(baseline_dd_above_mean, HazardParameterDataResponse)
        assert isinstance(scenario_dd_above_mean, HazardParameterDataResponse)

        # TODO: add model here
        # use hazard data requests via:
        delta_dd_above_mean = scenario_dd_above_mean.parameter - baseline_dd_above_mean.parameter
        delta_dd_above_std = 0.1 * delta_dd_above_mean
        fraction_loss_mean = self.annual_fraction_loss_per_degree_day_mean * delta_dd_above_mean
        fraction_loss_std = np.sqrt(
            (self.annual_fraction_loss_per_degree_day_std * delta_dd_above_std) ** 2
            + (self.annual_fraction_loss_per_degree_day_std * delta_dd_above_mean) ** 2
            + (self.annual_fraction_loss_per_degree_day_mean * delta_dd_above_std) ** 2
        )

        return get_impact_distrib(fraction_loss_mean, fraction_loss_std, ChronicHeat, ImpactType.disruption)


class TestChronicAssetImpact(unittest.TestCase):
    """Tests the impact on an asset of a chronic hazard model."""

    def test_chronic_vulnerability_model(self):
        """Testing the generation of an asset when only an impact curve (e.g. damage curve is available)"""

        store = mock_hazard_model_store_heat(TestData.longitudes, TestData.latitudes)
        hazard_model = ZarrHazardModel(source_paths=calculation.get_default_zarr_source_paths(), store=store)
        # to run a live calculation, we omit the store parameter

        scenario = "ssp585"
        year = 2050

        vulnerability_models = {IndustrialActivity: [ExampleChronicHeatModel()]}

        assets = [
            IndustrialActivity(lat, lon, type="Construction")
            for lon, lat in zip(TestData.longitudes, TestData.latitudes)
        ][:1]

        results = calculation.calculate_impacts(
            assets, hazard_model, vulnerability_models, scenario=scenario, year=year
        )

        value_test = list(results.values())[0].impact.mean_impact()
        value_test = list(results.values())[0].impact.prob
        value_exp = np.array(
            [
                2.63052640e-04,
                4.74274374e-04,
                1.09631192e-03,
                2.36170792e-03,
                4.74140169e-03,
                8.87108408e-03,
                1.54680940e-02,
                2.51355139e-02,
                3.80653836e-02,
                5.37235439e-02,
                7.63018177e-01,
                8.66955283e-02,
                8.59258945e-05,
                3.79032547e-10,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        )
        value_diff = np.sum(np.abs(value_test - value_exp))
        self.assertAlmostEqual(value_diff, 0.0, places=8)


def get_impact_distrib(
    fraction_loss_mean: float, fraction_loss_std: float, hazard_type: type, impact_type: ImpactType
) -> ImpactDistrib:
    """Calcaulate impact (disruption) of asset based on the hazard data returned.

    Args:
        fraction_loss_mean: mean of the impact distribution
        fraction_loss_std: standard deviation of the impact distribution
        hazard_type: Hazard Type.
        impact_type: Impact Type.

    Returns:
        Probability distribution of impacts.
    """
    impact_bins = np.concatenate(
        [
            np.linspace(-0.001, 0.001, 1, endpoint=False),
            np.linspace(0.001, 0.01, 9, endpoint=False),
            np.linspace(0.01, 0.1, 10, endpoint=False),
            np.linspace(0.1, 0.999, 10, endpoint=False),
            np.linspace(0.999, 1.001, 2),
        ]
    )
    probs = np.diff(
        np.vectorize(lambda x: norm.cdf(x, loc=fraction_loss_mean, scale=max(1e-12, fraction_loss_std)))(impact_bins)
    )
    probs_norm = np.sum(probs)
    if probs_norm < 1e-8:
        if fraction_loss_mean <= 0.0:
            probs = np.concatenate((np.array([1.0]), np.zeros(len(impact_bins) - 2)))
        elif fraction_loss_mean >= 1.0:
            probs = np.concatenate((np.zeros(len(impact_bins) - 2), np.array([1.0])))
    else:
        probs = probs / probs_norm

    return ImpactDistrib(hazard_type, impact_bins, probs, impact_type)
