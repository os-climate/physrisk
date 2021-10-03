""" Test asset impact calculations."""
import unittest
import numpy as np
from physrisk import Asset, AssetEventDistrib, ExceedanceCurve, VulnerabilityDistrib
from physrisk import Drought, Inundation
from physrisk import get_impact_distrib
from physrisk.models import InnundationModel
from src.physrisk.models.power_generating_asset_model import InnundationModel

class EventResponseMock:

    def __init__(self, return_periods, intensities):
        self.return_periods = return_periods
        self.intensities = intensities

class TestPowerGeneratingAssetWri(unittest.TestCase):
    """Tests World Resource Institute (WRI) models for power generating assets."""
        
    def test_innundation(self):
        # exceedance curve
        return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
        base_depth = np.array([0.0, 0.22372675, 0.3654859, 0.5393629, 0.6642473, 0.78564394, 0.9406518, 1.0539534, 1.1634114])
        future_depth = np.array([0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114])
        
        # we mock the response of the data request
        responses_mock = EventResponseMock(return_periods, base_depth), EventResponseMock(return_periods, future_depth)

        latitude, longitude = 45.268405, 19.885738
        assets = [Asset(latitude, longitude)]
        model = InnundationModel(assets)

        vul, event = model.get_distributions(responses_mock)

        impact = get_impact_distrib(event, vul)
        mean = impact.mean_impact()

        self.assertAlmostEqual(mean, 4.8453897)










        