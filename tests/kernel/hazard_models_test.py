from dataclasses import dataclass
from typing import Dict, List, Mapping, NamedTuple, Sequence, Tuple

import numpy as np

import tests.data.hazard_model_store as hms
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.hazard_model import (
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import ChronicHeat, Wind
from physrisk.kernel.impact import calculate_impacts
from physrisk.vulnerability_models.real_estate_models import GenericTropicalCycloneModel


@dataclass
class SinglePointData:
    latitude: float
    longitude: float
    scenario: str
    year: int
    wind_return_periods: np.ndarray  # years
    wind_intensities: np.ndarray  # m/s
    chronic_heat_intensity: float  # days over 35C
    # etc


class PointsKey(NamedTuple):
    latitude: float
    longitude: float
    scenario: str
    year: int


class PointBasedHazardModel(HazardModel):
    def __init__(self, points: Sequence[SinglePointData]):
        """HazardModel suitable for storing relatively small number (<~ million say) of individual hazard
        data points.

        Args:
            points (Sequence[SinglePointData]): List of points.
        """
        self.points: Dict[Tuple[PointsKey, float, float], SinglePointData] = {
            self._get_key(p.latitude, p.longitude, p.scenario, p.year): p for p in points
        }

    def _get_key(self, latitude: float, longitude: float, scenario: str, year: int):
        return PointsKey(latitude=round(latitude, 3), longitude=round(longitude, 3), scenario=scenario, year=year)

    def get_hazard_events(self, requests: List[HazardDataRequest]) -> Mapping[HazardDataRequest, HazardDataResponse]:
        response: Dict[HazardDataRequest, HazardDataResponse] = {}
        for request in requests:
            point = self.points[self._get_key(request.latitude, request.longitude, request.scenario, request.year)]
            if request.hazard_type == Wind and request.indicator_id == "max_speed":
                response[request] = HazardEventDataResponse(
                    return_periods=point.wind_return_periods, intensities=point.wind_intensities
                )
            elif request.hazard_type == ChronicHeat and request.indicator_id == "days/above/35c":
                response[request] = HazardParameterDataResponse(np.array(point.chronic_heat_intensity))
            # etc
        return response


def test_using_point_based_hazard_model():
    # test that shows how data already present for a number of points can be used in a HazardModel
    scenario = "rcp8p5"
    year = 2080
    assets = [
        RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
        for lon, lat in zip(hms.TestData.longitudes[0:1], hms.TestData.latitudes[0:1])
    ]
    # fmt: off
    wind_return_periods = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]) # noqa
    wind_intensities = np.array([37.279999, 44.756248, 48.712502, 51.685001, 53.520000, 55.230000, 56.302502, 57.336250, 58.452499, 59.283749, 63.312500, 65.482498, 66.352501, 67.220001, 67.767502, 68.117500, 68.372498, 69.127502, 70.897499 ]) # noqa
    # fmt: on
    point = SinglePointData(
        hms.TestData.latitudes[0],
        hms.TestData.longitudes[0],
        scenario=scenario,
        year=year,
        wind_return_periods=wind_return_periods,
        wind_intensities=wind_intensities,
        chronic_heat_intensity=0,
    )

    hazard_model = PointBasedHazardModel([point])
    vulnerability_models = {RealEstateAsset: [GenericTropicalCycloneModel()]}
    results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)
    impact_distrib = results[(assets[0], Wind, None, None)].impact
    mean_impact = impact_distrib.mean_impact()
    np.testing.assert_almost_equal(mean_impact, 0.009909858317497338)
