import json

import fsspec.implementations.local as local
import numpy as np

import physrisk.api.v1.common
from physrisk.api.v1.exposure_req_resp import (
    AssetExposureRequest,
    AssetExposureResponse,
)
from physrisk.container import ZarrHazardModelFactory
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import Asset
from physrisk.kernel.calculation import DefaultMeasuresFactory
from physrisk.kernel.exposure import (
    Category,
    JupterExposureMeasure,
    calculate_exposures,
)
from physrisk.kernel.hazards import (
    ChronicHeat,
    CombinedInundation,
    Drought,
    Fire,
    Hail,
    Wind,
)
from physrisk.requests import Requester

from ..test_base import TestWithCredentials
from ..data.test_hazard_model_store import TestData, mock_hazard_model_store_path_curves


class TestExposureMeasures(TestWithCredentials):
    def test_jupiter_exposure_service(self):
        assets, store, hazard_model_factory, expected = self._get_components()
        inventory = EmbeddedInventory()
        requester = Requester(
            hazard_model_factory=hazard_model_factory,
            vulnerability_models_factory=None,
            inventory=inventory,
            inventory_reader=InventoryReader(fs=local.LocalFileSystem(), base_path=""),
            reader=ZarrReader(store=store),
            colormaps=inventory.colormaps(),
            measures_factory=DefaultMeasuresFactory,
        )
        assets_api = physrisk.api.v1.common.Assets(
            items=[
                physrisk.api.v1.common.Asset(
                    asset_class="Asset", latitude=a.latitude, longitude=a.longitude
                )
                for a in assets[0:1]
            ]
        )
        request = AssetExposureRequest(assets=assets_api, scenario="ssp585", year=2050)
        response = requester.get(
            request_id="get_asset_exposure", request_dict=request.model_dump()
        )
        result = AssetExposureResponse(**json.loads(response)).items[0]
        expected = dict((k.__name__, v) for (k, v) in expected.items())
        for key in result.exposures.keys():
            assert result.exposures[key].category == expected[key].name

    def test_jupiter_exposure(self):
        assets, _, hazard_model_factory, expected = self._get_components()
        asset = assets[0]
        measure = JupterExposureMeasure()

        results = calculate_exposures(
            [asset],
            hazard_model_factory.hazard_model(),
            measure,
            scenario="ssp585",
            year=2030,
        )
        categories = results[asset].hazard_categories
        for k, v in expected.items():
            assert categories[k][0] == v

    def _get_components(self):
        # "precipitation/jupiter/v1/max_daily_water_equivalent_{scenario}_{year}"
        paths = [
            "combined_flood/jupiter/v1/fraction_{scenario}_{year}",
            "chronic_heat/jupiter/v1/days_above_35c_{scenario}_{year}",
            "wind/jupiter/v1/max_1min_{scenario}_{year}",
            "drought/jupiter/v1/months_spei3m_below_-2_{scenario}_{year}",
            "hail/jupiter/v1/days_above_5cm_{scenario}_{year}",
            "fire/jupiter/v1/fire_probability_{scenario}_{year}",
        ]

        all_resources = EmbeddedInventory().resources
        resources = [all_resources[p] for p in paths]

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
            return dict(
                (r.path.format(scenario="ssp585", year=2030), v)
                for (r, v) in zip(resources, values)
            )

        assets = [
            Asset(lat, lon)
            for (lat, lon) in zip(TestData.latitudes, TestData.longitudes)
        ]

        store = mock_hazard_model_store_path_curves(
            TestData.longitudes, TestData.latitudes, path_curves()
        )

        hazard_model_factory = ZarrHazardModelFactory(
            source_paths=get_default_source_paths(EmbeddedInventory()), store=store
        )

        return assets, store, hazard_model_factory, expected
