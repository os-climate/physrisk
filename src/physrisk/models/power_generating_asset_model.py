import numpy as np
from physrisk.kernel.events import HighTemperature
from typing import List
from physrisk.kernel import Asset, PowerGeneratingAsset, Inundation, VulnerabilityModel
from physrisk.kernel import AssetEventDistrib, VulnerabilityDistrib
from physrisk.data import EventDataRequest
from physrisk import ExceedanceCurve

class InnundationModel(VulnerabilityModel):
    __asset_types = [PowerGeneratingAsset]
    __event_types = [Inundation]
    
    def __init__(self, assets: List[Asset]):
        # default impact curve
        self.__assets = assets
        self.__curve_depth = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1])
        self.__curve_impact = np.array([0, 1, 2, 7, 14, 30, 60, 180, 365])
        pass

    @staticmethod
    def event_data_requests(asset: Asset):
        # assuming here that other specific look-ups wold be needed
        histo =  EventDataRequest(Inundation, latitude = asset.latitude, longitude = asset.longitude, 
            scenario = "historical", sea_level = 0, type = "river", subsidence = True, year = 2080)
        
        future = EventDataRequest(Inundation, latitude = asset.latitude, longitude = asset.longitude, 
            scenario = "rcp8p5", sea_level = 0, type = "river", subsidence = True, year = 2080)
        
        return histo, future

    def get_distributions(self, event_data_responses):
        """Return vulnerability and asset event distributions"""

        histo, future = event_data_responses
        
        protection_return_period = 250.0 
        curve_histo = ExceedanceCurve(1.0 / histo.return_periods, histo.intensities)
        protection_depth = curve_histo.get_value(1.0 / protection_return_period)
        
        curve_future = ExceedanceCurve(1.0 / future.return_periods, future.intensities)
        curve_future = curve_future.add_value_point(protection_depth)

        depth_bins, probs = curve_future.get_probability_bins()

        impact_bins = np.interp(depth_bins, self.__curve_depth, self.__curve_impact)
        
        # keep all bins, but make use of vulnerability matrix to apply protection level
        # for improved performance we could truncate (and treat identify matrix as a special case)
        # but this general version allows model uncertainties to be added
        probs_protected = np.where(depth_bins[1:] <= protection_depth, 0.0, 1.0)
        n_bins = len(probs)
        vul = VulnerabilityDistrib(type(Inundation), depth_bins, impact_bins, np.diag(probs_protected)) 
        event = AssetEventDistrib(type(Inundation), depth_bins, probs) 

        return vul, event


