""" Test asset impact calculations."""
import unittest
from test.data.hazard_model_store import TestData, get_mock_hazard_model_store

import numpy as np

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.kernel import calculation
from physrisk.kernel.assets import RealEstateAsset
from physrisk.models.real_estate_models import RealEstateInundationModel


class TestRealEstateModels(unittest.TestCase):
    """Tests RealEstateInundationModel."""

    def test_real_estate_model(self):

        curve = np.array(
            [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        )
        store = get_mock_hazard_model_store(TestData.longitudes, TestData.latitudes, curve)
        hazard_model = ZarrHazardModel(source_paths=calculation.get_default_zarr_source_paths(), store=store)

        assets = [
            RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
            for lon, lat in zip(TestData.longitudes[0:1], TestData.latitudes[0:1])
        ]

        scenario = "rcp8p5"
        year = 2080

        vulnerability_models = {RealEstateAsset: [RealEstateInundationModel()]}

        results = calculation.calculate_impacts(
            assets, hazard_model, vulnerability_models, scenario=scenario, year=year
        )

        np.testing.assert_allclose(results[assets[0]].impact.prob,
            np.array([0.02816851, 0.19360632, 0.11700387, 0.06039094, 0.03344832,
                0.02109813, 0.01503788, 0.01139472, 0.00864163, 0.00626335,
                0.00394632]), rtol=1e-6)
