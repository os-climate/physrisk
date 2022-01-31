""" Test asset impact calculations."""
import unittest, os
import numpy as np
from physrisk.utils.lazy_importing import lazy_import
import physrisk
from physrisk import Asset, AssetEventDistrib, ExceedanceCurve, VulnerabilityDistrib
from physrisk.kernel import Drought, RiverineInundation
from physrisk.kernel import calculate_impacts, get_impact_distrib
import physrisk.data.data_requests as dr
from physrisk.data import ReturnPeriodEvDataResp
from physrisk.models import InundationModel
from physrisk.data.hazard.event_provider_wri import EventProviderWri
import physrisk.data.geotiff_reader as rr
import time
from physrisk.kernel.assets import PowerGeneratingAsset
pd = lazy_import('pandas')

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
        model = InundationModel(assets)

        vul, event = model.get_distributions(assets[0], responses_mock)

        impact = get_impact_distrib(event, vul)
        mean = impact.mean_impact()

        self.assertAlmostEqual(mean, 4.8453897)

    @unittest.skip("example, not test")
    def test_with_data_sourcing(self):
                
        cache_folder = r"<cache folder>"

        asset_list = pd.read_csv(os.path.join(cache_folder, "wri-all.csv"))

        types = asset_list["primary_fuel"].unique()

        filtered = asset_list.loc[asset_list['primary_fuel'] == 'Gas'] # Nuclear

        interest =  [3,   8,  13,  14,  22,  25,  27,  28,  33,  40,  51,  64,  65, 66,  71,  72,  80,  88,  92, 109]

        filtered = filtered[22:23]

        longitudes = np.array(filtered['longitude'])
        latitudes = np.array(filtered['latitude'])
        generation = np.array(filtered['estimated_generation_gwh'])

        assets = [PowerGeneratingAsset(lat, lon, generation = gen, primary_fuel = 'gas') for lon, lat, gen in zip(longitudes, latitudes, generation)]

        detailed_results = calculate_impacts(assets, cache_folder = cache_folder)
        detailed_results[assets[0]].impact.to_exceedance_curve()
        means = np.array([detailed_results[asset].mean_impact for asset in assets])

        self.assertAlmostEqual(1, 1)








        