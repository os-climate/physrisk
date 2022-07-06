""" Test asset impact calculations."""
import unittest
from test.data.hazard_model_store import TestData, mock_hazard_model_store_heat
from typing import Iterable, Union

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.kernel import calculation
from physrisk.kernel.assets import Asset, IndustrialActivity
from physrisk.kernel.events import ChronicHeat
from physrisk.kernel.hazard_model import HazardDataRequest, HazardDataResponse
from physrisk.kernel.impact_distrib import ImpactDistrib, ImpactType
from physrisk.kernel.vulnerability_model import VulnerabilityModelBase


class ExampleChronicHeatModel(VulnerabilityModelBase):
    def __init__(self, model: str = "mean_heating_degree_days"):
        super().__init__(model, ChronicHeat)

    def get_data_requests(
        self, asset: Asset, *, scenario: str, year: int
    ) -> Union[HazardDataRequest, Iterable[HazardDataRequest]]:
        return HazardDataRequest(
            self.event_type, asset.longitude, asset.latitude, scenario=scenario, year=year, model=self.model
        )

    def get_impact(self, asset: Asset, data_responses: Iterable[HazardDataResponse]) -> ImpactDistrib:
        return ImpactDistrib(ChronicHeat, [0], [0], ImpactType.disruption)


class TestChronicAssetImpact(unittest.TestCase):
    """Tests the impact on an asset of a chronic hazard model."""

    def test_chronic_vulnerability_model(self):
        """Testing the generation of an asset when only an impact curve (e.g. damage curve is available)"""
        parameter = 250.0

        store = mock_hazard_model_store_heat(TestData.longitudes, TestData.latitudes, [parameter])
        hazard_model = ZarrHazardModel(
            acute_source_paths=calculation.get_default_accute_zarr_source_paths(), store=store
        )

        scenario = "rcp8p5"
        year = 2080

        vulnerability_models = {IndustrialActivity: [ExampleChronicHeatModel()]}

        assets = [
            IndustrialActivity(lat, lon, type="Construction")
            for lon, lat in zip(TestData.longitudes, TestData.latitudes)
        ]

        results = calculation.calculate_impacts(
            assets, hazard_model, vulnerability_models, scenario=scenario, year=year
        )

        self.assertIsNot(results, None)
