import unittest
from test.data.hazard_model_store import TestData, mock_hazard_model_store_heat_WBGT
from typing import Iterable, Union

import numpy as np

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.kernel import calculation
from physrisk.kernel.assets import Asset, IndustrialActivity
from physrisk.kernel.hazard_model import HazardDataRequest, HazardDataResponse, HazardParameterDataResponse
from physrisk.kernel.hazards import ChronicHeat
from physrisk.kernel.impact_distrib import ImpactDistrib, ImpactType
from physrisk.models.chronic_heat_models import ChronicHeatGznModel, get_impact_distrib


class ExampleWbgtGzJointModel(ChronicHeatGznModel):

    """Example implementation of the wbgt chronic heat model. This model
    inherits attributes from the ChronicHeatGZN model and estimate the
    results based on applying both GZN and WBGT"""

    def __init__(self, model: str = "mean_work_loss_high"):
        super().__init__(model, ChronicHeat)  # opportunity to give a model hint, but blank here

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
        model_gzn = "mean_degree_days/above/32c"
        model_wbgt = "mean_work_loss/high"

        return [
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario="historical",
                year=1980,
                model=model_gzn,
            ),
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                model=model_gzn,
            ),
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario="historical",
                year=1980,
                model=model_wbgt,
            ),
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                model=model_wbgt,
            ),
        ]

    def get_impact(self, asset: Asset, data_responses: Iterable[HazardDataResponse]) -> ImpactDistrib:
        """
        Function to return the impact distribution of the wbgt model.
        """

        (
            baseline_dd_above_mean,
            scenario_dd_above_mean,
            baseline_work_loss_wbgt,
            scenario_work_loss_wbgt,
        ) = data_responses

        assert isinstance(baseline_dd_above_mean, HazardParameterDataResponse)
        assert isinstance(scenario_dd_above_mean, HazardParameterDataResponse)
        assert isinstance(baseline_work_loss_wbgt, HazardParameterDataResponse)
        assert isinstance(scenario_work_loss_wbgt, HazardParameterDataResponse)

        hours_worked = self.total_labour_hours
        fraction_loss_mean_base_gzn = (baseline_dd_above_mean.parameter * self.time_lost_per_degree_day) / hours_worked

        fraction_loss_mean_scenario_gzn = (
            scenario_dd_above_mean.parameter * self.time_lost_per_degree_day
        ) / hours_worked

        fraction_loss_std_scenario_delta = (
            (scenario_dd_above_mean.parameter - baseline_dd_above_mean.parameter) * self.time_lost_per_degree_day_se
        ) / hours_worked

        baseline_work_ability = (1 - fraction_loss_mean_base_gzn) * (1 - baseline_work_loss_wbgt.parameter)
        scenario_work_ability = (1 - fraction_loss_mean_scenario_gzn) * (1 - scenario_work_loss_wbgt.parameter)

        total_work_loss_delta: float = baseline_work_ability - scenario_work_ability

        return get_impact_distrib(
            total_work_loss_delta, fraction_loss_std_scenario_delta, ChronicHeat, ImpactType.disruption
        )


class TestChronicAssetImpact(unittest.TestCase):
    """Tests the impact on an asset of a chronic hazard model."""

    def test_wbgt_vulnerability(self):
        store = mock_hazard_model_store_heat_WBGT(TestData.longitudes, TestData.latitudes)
        hazard_model = ZarrHazardModel(source_paths=calculation.get_default_zarr_source_paths(), store=store)

        scenario = "ssp585"
        year = 2050

        vulnerability_models = {IndustrialActivity: [ExampleWbgtGzJointModel()]}

        assets = [
            IndustrialActivity(lat, lon, type="Construction")
            for lon, lat in zip(TestData.longitudes, TestData.latitudes)
        ][:1]

        results = calculation.calculate_impacts(
            assets, hazard_model, vulnerability_models, scenario=scenario, year=year
        )

        value_test = list(results.values())[0].impact.prob

        value_exp = np.array(
            [
                0.00866966045,
                0.00457985836,
                0.00653567945,
                0.00908970969,
                0.01232054378,
                0.01627535057,
                0.02095325305,
                0.02629015878,
                0.03214812288,
                0.03831234212,
                0.52036287903,
                0.27929630752,
                0.02483245562,
                0.00033305568,
                0.00000062287,
                0.00000000015,
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
        np.testing.assert_almost_equal(value_test, value_exp, decimal=8)
