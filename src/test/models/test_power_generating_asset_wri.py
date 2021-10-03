""" Test asset impact calculations."""
import unittest
import numpy as np
from physrisk import Asset, AssetEventDistrib, ExceedanceCurve, VulnerabilityDistrib
from physrisk import Drought, Inundation
from physrisk import get_impact_distrib
from physrisk.models import InnundationModel
from physrisk.data.hazard.event_provider_wri import EventProviderWri

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

    @unittest.skip("example, not test")
    def test_with_data_sourcing(self):
        cache_folder = r"<cache file>"
        provider = EventProviderWri('web', cache_folder = cache_folder)
        lon = 19.885738
        lat = 45.268405

        provider.get_inundation_file_name_stub_river(5, "rcp8p5", "river", "MIROC-ESM-CHEM", 2080)

        lat2, lon2 = 45.258405, 19.895738

        events_river = provider.get_inundation_depth([lon], [lat], type = "river", scenario="rcp8p5", model = "MIROC-ESM-CHEM", year = "2080")
        #events_coast = provider.get_inundation_depth([lon], [lat], type = "coast", scenario="rcp8p5", subsidence = True, year = "2080", sea_level = 0)
        
        self.assertAlmostEqual(1, 1)








        