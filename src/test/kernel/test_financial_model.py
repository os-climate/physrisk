import unittest
from datetime import datetime

import numpy as np

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.embedded import get_default_source_paths
from physrisk.kernel.assets import Asset, PowerGeneratingAsset
from physrisk.kernel.financial_model import FinancialDataProvider, FinancialModel
from physrisk.kernel.loss_model import LossModel

from ..data.hazard_model_store import TestData, mock_hazard_model_store_inundation


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
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)

        # we need to define
        # 1) The hazard models
        # 2) The vulnerability models
        # 3) The financial models

        hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)

        model = LossModel(hazard_model=hazard_model)

        data_provider = MockFinancialDataProvider()
        financial_model = FinancialModel(data_provider)

        assets = [PowerGeneratingAsset(lat, lon) for lon, lat in zip(TestData.longitudes, TestData.latitudes)]
        measures = model.get_financial_impacts(
            assets, financial_model=financial_model, scenario="ssp585", year=2080, sims=100000
        )

        np.testing.assert_array_almost_equal_nulp(
            measures["RiverineInundation"]["percentile_values"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0, 2000.0]),
        )
