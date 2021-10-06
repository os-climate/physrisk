""" Test asset impact calculations."""
import unittest, os
import numpy as np
import pandas as pd
from physrisk import Asset, AssetEventDistrib, ExceedanceCurve, VulnerabilityDistrib
from physrisk import Drought, Inundation
from physrisk import get_impact_distrib
import physrisk.data.data_requests as dr
from physrisk.data import ReturnPeriodEvDataResp
from physrisk.models import InnundationModel
from physrisk.data.hazard.event_provider_wri import EventProviderWri
import physrisk.data.raster_reader as rr
from src import physrisk

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

        vul, event = model.get_distributions(assets[0], responses_mock)

        impact = get_impact_distrib(event, vul)
        mean = impact.mean_impact()

        self.assertAlmostEqual(mean, 4.8453897)

    @unittest.skip("example, not test")
    def test_with_data_sourcing(self):
                
        cache_folder = r"C:/Users/joemo/Code/Repos/WRI-EBRD-Flood-Module/data_1"

        asset_list = pd.read_csv(os.path.join(cache_folder, "wri-all.csv"))

        types = asset_list["primary_fuel"].unique()

        gas = asset_list.loc[asset_list['primary_fuel'] == 'Gas'] # Nuclear

        gas = gas[0:10]

        longitudes = np.array(gas['longitude'])
        latitudes = np.array(gas['latitude'])
        generation = np.array(gas['estimated_generation_gwh'])

        assets = [Asset(lon, lat, generation = gen, primary_fuel = 'gas') for lon, lat, gen in zip(longitudes, latitudes, generation)]

        model = InnundationModel()

        event_requests_by_asset = [model.get_event_data_requests(asset) for asset in assets]
        event_requests = [req for event_request_by_asset in event_requests_by_asset for req in event_request_by_asset]

        # select hazard data source
        events_innundation_provider = EventProviderWri('web', cache_folder = cache_folder).get_inundation_depth
        event_provider = { Inundation : events_innundation_provider }

        responses = dr.process_requests(event_requests, event_provider)
        
        for asset, requests in zip(assets, event_requests_by_asset):
            vul, event = model.get_distributions(asset, [responses[req] for req in requests])
            impact = get_impact_distrib(event, vul)
            mean = impact.mean_impact()

        self.assertAlmostEqual(1, 1)








        