import unittest
from test.data.hazard_model_store import TestData, mock_hazard_model_store_heat
from typing import Iterable, Union

import numpy as np

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

    def __init__(self):
        super().__init__("", ChronicHeat)  # opportunity to give a model hint, but blank here
        annual_time_loss_mins_per_degree_day = 8  # from paper...:
        self.annual_fraction_loss_per_degree_day = annual_time_loss_mins_per_degree_day / (60 * 8 * 240)
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
        baseline_model = "mean_degree_days/above/3c"
        delta_cooling_model = "mean_delta_degree_days/above/32c"

        return [
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario="historical",
                year=1980,
                model=baseline_model,
            ),
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                model=delta_cooling_model,
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
        baseline_dd_above_resp, delta_dd_above_resp = data_responses

        # check expected type; can maybe do this more nicely
        assert isinstance(baseline_dd_above_resp, HazardParameterDataResponse)
        assert isinstance(delta_dd_above_resp, HazardParameterDataResponse)

        # TODO: add model here
        # use hazard data requests via:
        delta_dd_above = delta_dd_above_resp.parameter
        fraction_loss = self.annual_fraction_loss_per_degree_day * delta_dd_above
        assert fraction_loss is not None  # to remove (just to keep tox happy that we are doing something with result!)

        impact_bins = np.array(
            [0.0, 0.01, 0.02]
        )  # bins defining fractional disruption of business activity revenue (0 to 0.01; 0.01 to 0.02)
        prob = np.array([0.1, 0.05])  # fractional disruption of business activity revenue for each bin

        return ImpactDistrib(ChronicHeat, impact_bins, prob, ImpactType.disruption)


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
        ]

        results = calculation.calculate_impacts(
            assets, hazard_model, vulnerability_models, scenario=scenario, year=year
        )

        self.assertIsNot(results, None)
