import unittest
from typing import Iterable, List, Union, cast

import numpy as np
from scipy.stats import norm

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import Asset, IndustrialActivity
from physrisk.kernel.hazard_model import HazardDataRequest, HazardDataResponse, HazardParameterDataResponse
from physrisk.kernel.hazards import ChronicHeat
from physrisk.kernel.impact import calculate_impacts
from physrisk.kernel.impact_distrib import ImpactDistrib, ImpactType
from physrisk.kernel.vulnerability_model import DictBasedVulnerabilityModels, VulnerabilityModelBase
from physrisk.vulnerability_models.chronic_heat_models import ChronicHeatGZNModel

from ..data.hazard_model_store_test import TestData, mock_hazard_model_store_heat


class ExampleChronicHeatModel(VulnerabilityModelBase):
    """Example chronic vulnerability model for extreme heat (summary should fit on one line).

    More decription below as per
    https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html
    """

    def __init__(self, indicator_id: str = "mean_degree_days_above_32c", delta: bool = True):
        super().__init__(
            indicator_id=indicator_id, hazard_type=ChronicHeat, impact_type=ImpactType.disruption
        )  # opportunity to give a model hint, but blank here

        self.time_lost_per_degree_day = 4.671  # This comes from the paper converted to celsius
        self.time_lost_per_degree_day_se = 2.2302  # This comes from the paper converted to celsius
        self.total_labour_hours = 107460  # OECD Average hours worked within the USA in one year.
        self.delta = delta

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
        indicator_id = "mean_degree_days/above/32c"

        return [
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario="historical",
                year=1980,
                indicator_id=indicator_id,
            ),
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                indicator_id=indicator_id,
            ),
        ]

    def get_impact(self, asset: Asset, data_responses: List[HazardDataResponse]) -> ImpactDistrib:
        """Calcaulate impact (disruption) of asset based on the hazard data returned.

        Args:
            asset: Asset for which impact is calculated.
            data_responses: responses to the hazard data requests generated in get_data_requests.

        Returns:
            Probability distribution of impacts.
        """
        assert isinstance(asset, IndustrialActivity)
        baseline_dd_above_mean, scenario_dd_above_mean = data_responses

        # check expected type; can maybe do this more nicely
        assert isinstance(baseline_dd_above_mean, HazardParameterDataResponse)
        assert isinstance(scenario_dd_above_mean, HazardParameterDataResponse)
        # Ensuring that the values are greater than zero. Should be by defition.
        assert scenario_dd_above_mean.parameter >= 0
        assert baseline_dd_above_mean.parameter >= 0

        # use hazard data requests via:

        hazard_paths = [cast(HazardParameterDataResponse, r).path for r in data_responses]

        # Allow for either a delta approach or a level estimate.
        delta_dd_above_mean: float = scenario_dd_above_mean.parameter - baseline_dd_above_mean.parameter * self.delta
        hours_worked = self.total_labour_hours
        fraction_loss_mean = (delta_dd_above_mean * self.time_lost_per_degree_day) / hours_worked
        fraction_loss_std = (delta_dd_above_mean * self.time_lost_per_degree_day_se) / hours_worked

        return get_impact_distrib(
            fraction_loss_mean, fraction_loss_std, ChronicHeat, hazard_paths, ImpactType.disruption
        )


def get_impact_distrib(
    fraction_loss_mean: float,
    fraction_loss_std: float,
    hazard_type: type,
    hazard_paths: List[str],
    impact_type: ImpactType,
) -> ImpactDistrib:
    """Calculate impact (disruption) of asset based on the hazard data returned.

    Args:
        fraction_loss_mean: mean of the impact distribution
        fraction_loss_std: standard deviation of the impact distribution
        hazard_type: Hazard Type.
        hazard_paths: Path to the hazard event data source.
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

    probs_cumulative = np.vectorize(lambda x: norm.cdf(x, loc=fraction_loss_mean, scale=max(1e-12, fraction_loss_std)))(
        impact_bins
    )
    probs_cumulative[-1] = np.maximum(probs_cumulative[-1], 1.0)
    probs = np.diff(probs_cumulative)

    probs_norm = np.sum(probs)
    prob_differential = 1 - probs_norm
    if probs_norm < 1e-8:
        if fraction_loss_mean <= 0.0:
            probs = np.concatenate((np.array([1.0]), np.zeros(len(impact_bins) - 2)))
        elif fraction_loss_mean >= 1.0:
            probs = np.concatenate((np.zeros(len(impact_bins) - 2), np.array([1.0])))
    else:
        probs[0] = probs[0] + prob_differential

    return ImpactDistrib(hazard_type, impact_bins, probs, hazard_paths, impact_type)


class TestChronicAssetImpact(unittest.TestCase):
    """Tests the impact on an asset of a chronic hazard model."""

    def test_chronic_vulnerability_model(self):
        """Testing the generation of an asset when only an impact curve (e.g. damage curve is available)"""

        store = mock_hazard_model_store_heat(TestData.longitudes, TestData.latitudes)
        hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)
        # to run a live calculation, we omit the store parameter

        scenario = "ssp585"
        year = 2050

        vulnerability_models = DictBasedVulnerabilityModels({IndustrialActivity: [ChronicHeatGZNModel()]})

        assets = [
            IndustrialActivity(lat, lon, type="Construction")
            for lon, lat in zip(TestData.longitudes, TestData.latitudes)
        ][:1]

        results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)

        value_test = list(results.values())[0].impact.mean_impact()
        value_test = list(results.values())[0].impact.prob
        value_exp = np.array(
            [
                0.02656777935,
                0.01152965908,
                0.01531928095,
                0.01983722513,
                0.02503479879,
                0.03079129430,
                0.03690901485,
                0.04311790414,
                0.04909118572,
                0.05447159590,
                0.51810304973,
                0.16109092806,
                0.00807680527,
                0.00005941883,
                0.00000005990,
                0.00000000001,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
                0.00000000000,
            ]
        )
        value_diff = np.sum(np.abs(value_test - value_exp))
        self.assertAlmostEqual(value_diff, 0.0, places=8)
