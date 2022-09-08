from typing import Iterable, Union

import numpy as np

from ..kernel.assets import Asset, PowerGeneratingAsset
from ..kernel.curve import ExceedanceCurve
from ..kernel.hazard_event_distrib import HazardEventDistrib
from ..kernel.hazard_model import HazardDataRequest, HazardDataResponse, HazardEventDataResponse
from ..kernel.hazards import ChronicHeat, RiverineInundation
from ..kernel.vulnerability_distrib import VulnerabilityDistrib
from ..kernel.vulnerability_model import (
    DeterministicVulnerabilityModel,
    VulnerabilityModelAcuteBase,
    applies_to_assets,
    applies_to_events,
)


@applies_to_events([RiverineInundation])
@applies_to_assets([PowerGeneratingAsset])
class InundationModel(VulnerabilityModelAcuteBase):
    def __init__(self, model="MIROC-ESM-CHEM"):
        # default impact curve
        self.__curve_depth = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1])
        self.__curve_impact = np.array([0, 1, 2, 7, 14, 30, 60, 180, 365])
        self.__model = model
        self.__base_model = "000000000WATCH"
        super().__init__(model, RiverineInundation)
        pass

    def get_data_requests(
        self, asset: Asset, *, scenario: str, year: int
    ) -> Union[HazardDataRequest, Iterable[HazardDataRequest]]:
        """Provide the list of hazard event data requests required in order to calculate
        the VulnerabilityDistrib and HazardEventDistrib for the asset."""

        histo = HazardDataRequest(
            RiverineInundation,
            asset.longitude,
            asset.latitude,
            scenario="historical",
            year=1980,
            model=self.__base_model,
        )

        future = HazardDataRequest(
            RiverineInundation, asset.longitude, asset.latitude, scenario="rcp8p5", year=2080, model=self.__model
        )

        return histo, future

    def get_distributions(self, asset: Asset, event_data_responses: Iterable[HazardDataResponse]):
        """Return distributions for asset, based on hazard event date:
        VulnerabilityDistrib and HazardEventDistrib."""

        histo, future = event_data_responses
        assert isinstance(histo, HazardEventDataResponse)
        assert isinstance(future, HazardEventDataResponse)

        protection_return_period = 250.0
        curve_histo = ExceedanceCurve(1.0 / histo.return_periods, histo.intensities)
        # the protection depth is the 250-year-return-period inundation depth at the asset location
        protection_depth = curve_histo.get_value(1.0 / protection_return_period)

        curve_future = ExceedanceCurve(1.0 / future.return_periods, future.intensities)
        curve_future = curve_future.add_value_point(protection_depth)

        depth_bins, probs = curve_future.get_probability_bins()

        impact_bins = np.interp(depth_bins, self.__curve_depth, self.__curve_impact) / 365.0

        # keep all bins, but make use of vulnerability matrix to apply protection level
        # for improved performance we could truncate (and treat identify matrix as a special case)
        # but this general version allows model uncertainties to be added
        probs_protected = np.where(depth_bins[1:] <= protection_depth, 0.0, 1.0)

        vul = VulnerabilityDistrib(RiverineInundation, depth_bins, impact_bins, np.diag(probs_protected))
        event = HazardEventDistrib(RiverineInundation, depth_bins, probs)

        return vul, event


@applies_to_events([ChronicHeat])
@applies_to_assets([PowerGeneratingAsset])
class TemperatureModel(DeterministicVulnerabilityModel):
    def __init__(self):
        # does nothing
        pass
