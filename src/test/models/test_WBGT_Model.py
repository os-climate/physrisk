import unittest
from test.data.hazard_model_store import TestData, mock_hazard_model_store_heat_WBGT
from typing import Iterable, List, Union, cast

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

    def work_type_mapping(self):
        return {"low": ["low", "medium"], "medium": ["medium", "low", "high"], "high": ["high", "medium"]}

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

        work_type_mapping = self.work_type_mapping()
        assert isinstance(asset, IndustrialActivity)
        # specify hazard data needed. Model string is hierarchical and '/' separated.
        model_gzn = "mean_degree_days/above/32c"
        model_wbgt = "mean_work_loss/"

        asset_types = [type_asset for type_asset in work_type_mapping[asset.type]]
        wbgt_data_requests = []
        for i_asset_types in asset_types:
            wbgt_data_requests.append(
                HazardDataRequest(
                    self.hazard_type,
                    asset.longitude,
                    asset.latitude,
                    scenario="historical",
                    year=2010,
                    model=model_wbgt + i_asset_types,
                )
            )

            wbgt_data_requests.append(
                HazardDataRequest(
                    self.hazard_type,
                    asset.longitude,
                    asset.latitude,
                    scenario=scenario,
                    year=year,
                    model=model_wbgt + i_asset_types,
                )
            )

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
        ] + wbgt_data_requests

    def get_impact(self, asset: Asset, data_responses: List[HazardDataResponse]) -> ImpactDistrib:
        """
        Function to return the impact distribution of the wbgt model.
        """

        assert isinstance(asset, IndustrialActivity)
        wbgt_responses = [cast(HazardParameterDataResponse, r) for r in data_responses[2:]]

        baseline_dd_above_mean = cast(HazardParameterDataResponse, data_responses[0])
        scenario_dd_above_mean = cast(HazardParameterDataResponse, data_responses[1])

        hours_worked = self.total_labour_hours
        fraction_loss_mean_base_gzn = (baseline_dd_above_mean.parameter * self.time_lost_per_degree_day) / hours_worked

        fraction_loss_mean_scenario_gzn = (
            scenario_dd_above_mean.parameter * self.time_lost_per_degree_day
        ) / hours_worked

        fraction_loss_std_base = (baseline_dd_above_mean.parameter * self.time_lost_per_degree_day_se) / hours_worked

        fraction_loss_std_scenario = (
            scenario_dd_above_mean.parameter * self.time_lost_per_degree_day_se
        ) / hours_worked

        baseline_work_ability = (1 - fraction_loss_mean_base_gzn) * (1 - wbgt_responses[0].parameter)
        scenario_work_ability = (1 - fraction_loss_mean_scenario_gzn) * (1 - wbgt_responses[1].parameter)

        # Getting the parameters required for the uniform distribution.
        if asset.type in ["low", "high"]:
            a_historical = (
                wbgt_responses[0].parameter - abs((wbgt_responses[2].parameter - wbgt_responses[0].parameter)) / 2
            )
            b_historical = (
                wbgt_responses[0].parameter + abs((wbgt_responses[2].parameter - wbgt_responses[0].parameter)) / 2
            )
            a_scenario = (
                wbgt_responses[1].parameter - abs((wbgt_responses[3].parameter - wbgt_responses[1].parameter)) / 2
            )
            b_scenario = (
                wbgt_responses[1].parameter + abs((wbgt_responses[3].parameter - wbgt_responses[1].parameter)) / 2
            )
        elif asset.type == "medium":
            a_historical = wbgt_responses[0].parameter - (wbgt_responses[2].parameter - wbgt_responses[0].parameter) / 2
            b_historical = wbgt_responses[0].parameter + (wbgt_responses[4].parameter - wbgt_responses[0].parameter) / 2
            a_scenario = (
                wbgt_responses[1].parameter - abs((wbgt_responses[3].parameter - wbgt_responses[1].parameter)) / 2
            )
            b_scenario = (
                wbgt_responses[1].parameter + abs((wbgt_responses[5].parameter - wbgt_responses[1].parameter)) / 2
            )

        # Estimation of the variance
        variance_historical_uni = ((b_historical - a_historical) ** 2) / 12
        variance_scenario_uni = ((b_scenario - a_scenario) ** 2) / 12

        variance_historical = two_variable_joint_variance(
            (1 - fraction_loss_mean_base_gzn),
            fraction_loss_std_base**2,
            (1 - wbgt_responses[0].parameter),
            variance_historical_uni,
        )
        variance_scenario = two_variable_joint_variance(
            (1 - fraction_loss_mean_scenario_gzn),
            fraction_loss_std_scenario**2,
            (1 - wbgt_responses[1].parameter),
            variance_scenario_uni,
        )

        std_delta = variance_scenario ** (0.5) - variance_historical ** (0.5)

        total_work_loss_delta: float = baseline_work_ability - scenario_work_ability

        return get_impact_distrib(total_work_loss_delta, std_delta, ChronicHeat, ImpactType.disruption)


def two_variable_joint_variance(ex, varx, ey, vary):
    """
    Function to estimate the variance of two uncorrelated variables.
    """
    return varx * vary + varx * (ey**2) + vary * (ex**2)


class TestChronicAssetImpact(unittest.TestCase):
    """Tests the impact on an asset of a chronic hazard model."""

    def test_wbgt_vulnerability(self):
        store = mock_hazard_model_store_heat_WBGT(TestData.longitudes, TestData.latitudes)
        hazard_model = ZarrHazardModel(source_paths=calculation.get_default_zarr_source_paths(), store=store)

        scenario = "ssp585"
        year = 2050

        vulnerability_models = {IndustrialActivity: [ExampleWbgtGzJointModel()]}

        assets = [
            IndustrialActivity(lat, lon, type="high") for lon, lat in zip(TestData.longitudes, TestData.latitudes)
        ][:1]

        results = calculation.calculate_impacts(
            assets, hazard_model, vulnerability_models, scenario=scenario, year=year
        )

        value_test = list(results.values())[0].impact.prob

        value_exp = np.array(
            [
                0.00000119194,
                0.00000046573,
                0.00000063758,
                0.00000086889,
                0.00000117871,
                0.00000159172,
                0.00000213966,
                0.00000286314,
                0.00000381379,
                0.00000505696,
                0.00021143251,
                0.00167372506,
                0.00924050344,
                0.03560011430,
                0.09575512509,
                0.17988407024,
                0.23607703667,
                0.21646814108,
                0.13867487025,
                0.06205630207,
                0.02433887116,
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
