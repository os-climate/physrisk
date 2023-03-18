from test.base_test import TestWithCredentials
from test.data.hazard_model_store import TestData, mock_hazard_model_store_for_paths

import numpy as np

from physrisk.api.v1.hazard_data import HazardResource, Scenario
from physrisk.data.inventory import Inventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.kernel.assets import Asset
from physrisk.kernel.calculation import get_source_paths_from_inventory
from physrisk.kernel.exposure import Category, JupterExposureMeasure
from physrisk.kernel.hazards import CombinedInundation


class TestExposureMeasures(TestWithCredentials):
    def test_jupiter_exposure(self):
        resources = [
            HazardResource(
                type="CombinedInundation",
                path="",
                id="flooded_fraction",
                display_name="",
                description="",
                array_name="fraction_{scenario}_{year}",
                scenarios=[Scenario(id="ssp585", years=[2030])],
                units="",
            ),
            HazardResource(
                type="Fire",
                path="",
                id="fire_probability",
                display_name="",
                description="",
                array_name="fire_probability_{scenario}_{year}",
                scenarios=[Scenario(id="ssp585", years=[2030])],
                units="",
            ),
            HazardResource(
                type="Drought",
                path="",
                id="months/spei3m/below/-2",
                display_name="",
                description="",
                array_name="months_spei3m_below_-2_{scenario}_{year}",
                scenarios=[Scenario(id="ssp585", years=[2030])],
                units="",
            ),
            HazardResource(
                type="Hail",
                path="",
                id="days/above/5cm",
                display_name="",
                description="",
                array_name="days_above_5cm_{scenario}_{year}",
                scenarios=[Scenario(id="ssp585", years=[2030])],
                units="",
            ),
            HazardResource(
                type="ChronicHeat",
                path="",
                id="days/above/35c",
                display_name="",
                description="",
                array_name="days_above_35c_{scenario}_{year}",
                scenarios=[Scenario(id="ssp585", years=[2030])],
                units="",
            ),
            HazardResource(
                type="Wind",
                path="",
                id="max/1min",
                display_name="",
                description="",
                array_name="max_1min_{scenario}_{year}",
                scenarios=[Scenario(id="ssp585", years=[2030])],
                units="",
            ),
        ]

        def paths():
            return [r.array_name.format(scenario="ssp585", year=2030) for r in resources]

        assets = [Asset(lat, lon) for (lat, lon) in zip(TestData.latitudes, TestData.longitudes)]
        store = mock_hazard_model_store_for_paths(TestData.longitudes, TestData.latitudes, np.array([0.1]), paths)
        measure = JupterExposureMeasure()
        hazard_model = ZarrHazardModel(source_paths=get_source_paths_from_inventory(Inventory(resources)), store=store)
        requests = measure.get_data_requests(assets[0], scenario="ssp585", year=2030)
        responses = hazard_model.get_hazard_events(requests)
        results = measure.get_exposures(assets[0], (responses[req] for req in requests))
        assert results[CombinedInundation] == Category.HIGH
