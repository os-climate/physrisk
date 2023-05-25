import json
from test.base_test import TestWithCredentials
from test.data.hazard_model_store import TestData, mock_hazard_model_store_path_curves

import fsspec.implementations.local as local
import numpy as np

import physrisk.api.v1.common
from physrisk.api.v1.exposure_req_resp import AssetExposureRequest, AssetExposureResponse
from physrisk.api.v1.hazard_data import HazardResource, Scenario
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.assets import Asset
from physrisk.kernel.calculation import calculate_exposures, get_source_paths_from_inventory
from physrisk.kernel.exposure import Category, JupterExposureMeasure
from physrisk.kernel.hazards import ChronicHeat, CombinedInundation, Drought, Fire, Hail, Wind
from physrisk.requests import Requester


class TestExposureMeasures(TestWithCredentials):
    def test_jupiter_exposure_service(self):
        assets, store, hazard_model, expected = self._get_components()
        requester = Requester(
            hazard_model=hazard_model,
            inventory=Inventory(EmbeddedInventory().to_resources()),
            inventory_reader=InventoryReader(fs=local.LocalFileSystem(), base_path=""),
            reader=ZarrReader(store=store),
            colormaps=EmbeddedInventory().colormaps(),
        )
        assets_api = physrisk.api.v1.common.Assets(
            items=[
                physrisk.api.v1.common.Asset(asset_class="Asset", latitude=a.latitude, longitude=a.longitude)
                for a in assets[0:1]
            ]
        )
        request = AssetExposureRequest(assets=assets_api, scenario="ssp585", year=2050)
        response = requester.get(request_id="get_asset_exposure", request_dict=request.dict())
        result = AssetExposureResponse(**json.loads(response)).items[0]
        expected = dict((k.__name__, v) for (k, v) in expected.items())
        for key in result.exposures.keys():
            assert result.exposures[key].category == expected[key].name

    def test_jupiter_exposure(self):
        assets, _, hazard_model, expected = self._get_components()
        asset = assets[0]
        measure = JupterExposureMeasure()

        results = calculate_exposures([asset], hazard_model, measure, scenario="ssp585", year=2030)
        categories = results[asset].hazard_categories
        for k, v in expected.items():
            assert categories[k][0] == v

    def _get_components(self):
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
                type="Fire",
                path="",
                id="fire_probability",
                display_name="",
                description="",
                array_name="fire_probability_{scenario}_{year}",
                scenarios=[Scenario(id="ssp585", years=[2030])],
                units="",
            ),
        ]

        values = [np.array([v]) for v in [0.02, 15, 100, 0.7, 0.1, 0.9]]

        expected = {
            CombinedInundation: Category.LOW,
            ChronicHeat: Category.MEDIUM,
            Wind: Category.MEDIUM,
            Drought: Category.HIGH,
            Hail: Category.LOWEST,
            Fire: Category.HIGHEST,
        }

        def path_curves():
            return dict((r.array_name.format(scenario="ssp585", year=2030), v) for (r, v) in zip(resources, values))

        assets = [Asset(lat, lon) for (lat, lon) in zip(TestData.latitudes, TestData.longitudes)]

        store = mock_hazard_model_store_path_curves(TestData.longitudes, TestData.latitudes, path_curves())
        hazard_model = ZarrHazardModel(source_paths=get_source_paths_from_inventory(Inventory(resources)), store=store)

        return assets, store, hazard_model, expected
