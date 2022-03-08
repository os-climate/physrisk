import unittest
from datetime import datetime

import numpy as np

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.kernel import calculation
from physrisk.kernel.assets import Asset, PowerGeneratingAsset
from physrisk.kernel.financial_model import FinancialDataProvider, FinancialModel

from ..data.hazard_model_store import TestData, get_mock_hazard_model_store


class MockFinancialDataProvider(FinancialDataProvider):
    def get_asset_value(self, asset: Asset, currency: str) -> float:
        return 1000

    def get_asset_aggregate_cashflows(self, asset: Asset, start: datetime, end: datetime, currency: str) -> float:
        return 1000


class TestAssetImpact(unittest.TestCase):
    """Tests asset impact calculations."""

    def test_financial_model(self):
        curve = np.array(
            [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        )
        store = get_mock_hazard_model_store(TestData.longitudes, TestData.latitudes, curve)
        hazard_model = ZarrHazardModel(source_paths=calculation.get_default_zarr_source_paths(), store=store)

        model = FinancialModel(hazard_model=hazard_model)

        data_provider = MockFinancialDataProvider()

        assets = [PowerGeneratingAsset(lat, lon) for lon, lat in zip(TestData.longitudes, TestData.latitudes)]
        model.get_financial_impacts(assets, data_provider=data_provider, scenario="rcp8p5", year=2050)

        self.assertTrue(True)
