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
    def __init__(self):
        super().__init__("", ChronicHeat)  # opportunity to give a model hint, but blank here
        # load any data needed by the model here in the constructor

    def get_data_requests(
        self, asset: Asset, *, scenario: str, year: int
    ) -> Union[HazardDataRequest, Iterable[HazardDataRequest]]:

        # specify hazard data needed
        baseline_model = "mean_degree_days/cooling/18C"
        delta_cooling_model = "mean_delta_degree_days/cooling/18C"

        return [
            HazardDataRequest(
                self.event_type, asset.longitude, asset.latitude, scenario="historical", year=1980, model=baseline_model
            ),
            HazardDataRequest(
                self.event_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                model=delta_cooling_model,
            ),
        ]

    def get_impact(self, asset: Asset, data_responses: Iterable[HazardDataResponse]) -> ImpactDistrib:

        baseline_cooling_dd_resp, delta_cooling_dd_resp = data_responses

        # check expected type; can maybe do this more nicely
        assert isinstance(baseline_cooling_dd_resp, HazardParameterDataResponse)
        assert isinstance(delta_cooling_dd_resp, HazardParameterDataResponse)

        # TODO: add model here
        # use hazard data requests via:
        delta_cooling_dd = delta_cooling_dd_resp.parameter  # typing: ignore
        assert delta_cooling_dd is not None  # to remove: just to keep tox happy

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
