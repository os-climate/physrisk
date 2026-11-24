import asyncio
from dataclasses import dataclass
from typing import Dict, Mapping, NamedTuple, Sequence, Tuple

import numpy as np
import pytest

from tests.data.test_hazard_model_store import TestData
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.hazard_model import (
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import ChronicHeat, Wind
from physrisk.kernel.impact import ImpactKey, calculate_impacts
from physrisk.kernel.vulnerability_model import DictBasedVulnerabilityModels
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
            self._get_key(p.latitude, p.longitude, p.scenario, p.year): p
            for p in points
        }

    def _get_key(self, latitude: float, longitude: float, scenario: str, year: int):
        return PointsKey(
            latitude=round(latitude, 3),
            longitude=round(longitude, 3),
            scenario=scenario,
            year=year,
        )

    def get_hazard_data(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        response: Dict[HazardDataRequest, HazardDataResponse] = {}
        for request in requests:
            point = self.points[
                self._get_key(
                    request.latitude, request.longitude, request.scenario, request.year
                )
            ]
            if request.hazard_type == Wind and request.indicator_id == "max_speed":
                response[request] = HazardEventDataResponse(
                    return_periods=point.wind_return_periods,
                    intensities=point.wind_intensities,
                )
            elif (
                request.hazard_type == ChronicHeat
                and request.indicator_id == "days/above/35c"
            ):
                response[request] = HazardParameterDataResponse(
                    parameters=np.array(point.chronic_heat_intensity),
                )
            # etc
        return response


def test_using_point_based_hazard_model():
    # test that shows how data already present for a number of points can be used in a HazardModel
    scenario = "rcp8p5"
    year = 2080
    assets = [
        RealEstateAsset(
            latitude=lat, longitude=lon, location="Asia", type="Buildings/Industrial"
        )
        for lon, lat in zip(TestData.longitudes[0:1], TestData.latitudes[0:1])
    ]
    # fmt: off
    wind_return_periods = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]) # noqa
    wind_intensities = np.array([37.279999, 44.756248, 48.712502, 51.685001, 53.520000, 55.230000, 56.302502, 57.336250, 58.452499, 59.283749, 63.312500, 65.482498, 66.352501, 67.220001, 67.767502, 68.117500, 68.372498, 69.127502, 70.897499 ]) # noqa
    # fmt: on
    point = SinglePointData(
        TestData.latitudes[0],
        TestData.longitudes[0],
        scenario=scenario,
        year=year,
        wind_return_periods=wind_return_periods,
        wind_intensities=wind_intensities,
        chronic_heat_intensity=0,
    )

    hazard_model = PointBasedHazardModel([point])
    vulnerability_models = DictBasedVulnerabilityModels(
        {RealEstateAsset: [GenericTropicalCycloneModel()]}
    )
    results = calculate_impacts(
        assets, hazard_model, vulnerability_models, scenarios=[scenario], years=[year]
    )
    key_result = ImpactKey(assets[0], Wind, scenario, year)
    impact_distrib = results[key_result][0].impact
    mean_impact = impact_distrib.mean_impact()
    np.testing.assert_almost_equal(mean_impact, 0.009909858317497338)


class SyncOnlyHazardModel(HazardModel):
    def get_hazard_data(self, requests):
        return {
            request: HazardParameterDataResponse(np.array([1.0]))
            for request in requests
        }


class AsyncOnlyHazardModel(HazardModel):
    async def get_hazard_data_async(self, requests):
        await asyncio.sleep(0)
        return {
            request: HazardParameterDataResponse(np.array([2.0]))
            for request in requests
        }


def _heat_request():
    return HazardDataRequest(
        ChronicHeat,
        0.0,
        51.5,
        indicator_id="days/above/35c",
        scenario="ssp585",
        year=2050,
    )


def test_sync_only_hazard_model_can_be_awaited():
    request = _heat_request()
    response = asyncio.run(SyncOnlyHazardModel().get_hazard_data_async([request]))
    assert response[request].parameter == 1.0


def test_async_only_hazard_model_can_be_called_synchronously():
    request = _heat_request()
    response = AsyncOnlyHazardModel().get_hazard_data([request])
    assert response[request].parameter == 2.0


def test_async_only_hazard_model_can_be_called_synchronously_inside_event_loop():
    request = _heat_request()

    async def call_sync_from_loop():
        return AsyncOnlyHazardModel().get_hazard_data([request])

    response = asyncio.run(call_sync_from_loop())
    assert response[request].parameter == 2.0


def test_hazard_model_must_implement_sync_or_async():
    class IncompleteHazardModel(HazardModel):
        pass

    with pytest.raises(NotImplementedError, match="IncompleteHazardModel"):
        IncompleteHazardModel().get_hazard_data([_heat_request()])
    with pytest.raises(NotImplementedError, match="IncompleteHazardModel"):
        asyncio.run(IncompleteHazardModel().get_hazard_data_async([_heat_request()]))
