import unittest
from test.base_test import TestWithCredentials
from test.data.hazard_model_store import (
    TestData,
    add_curves,
    mock_hazard_model_store_inundation,
    shape_transform_21600_43200,
    zarr_memory_store,
)

import numpy as np

from physrisk import requests
from physrisk.api.v1.common import Assets
from physrisk.container import Container
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import PowerGeneratingAsset, RealEstateAsset, ThermalPowerGeneratingAsset
from physrisk.vulnerability_models.power_generating_asset_models import InundationModel
from physrisk.vulnerability_models.real_estate_models import (
    RealEstateCoastalInundationModel,
    RealEstateRiverineInundationModel,
)
from physrisk.vulnerability_models.thermal_power_generation_models import (
    ThermalPowerGenerationAirTemperatureModel,
    ThermalPowerGenerationDroughtModel,
    ThermalPowerGenerationRiverineInundationModel,
)

# from physrisk.api.v1.impact_req_resp import AssetImpactResponse
# from physrisk.data.static.world import get_countries_and_continents


class TestImpactRequests(TestWithCredentials):
    def test_asset_list_json(self):
        assets = {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "location": "Asia",
                    "longitude": 69.4787,
                    "latitude": 34.556,
                },
                {
                    "asset_class": "PowerGeneratingAsset",
                    "type": "Nuclear",
                    "location": "Asia",
                    "longitude": -70.9157,
                    "latitude": -39.2145,
                },
            ],
        }
        assets_obj = Assets(**assets)
        self.assertIsNotNone(assets_obj)

    def test_impact_request(self):
        """Runs short asset-level impact request."""

        assets = {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "location": "Asia",
                    "longitude": TestData.longitudes[0],
                    "latitude": TestData.latitudes[0],
                },
                {
                    "asset_class": "PowerGeneratingAsset",
                    "type": "Nuclear",
                    "location": "Asia",
                    "longitude": TestData.longitudes[1],
                    "latitude": TestData.latitudes[1],
                },
            ],
        }

        request_dict = {
            "assets": assets,
            "include_asset_level": True,
            "include_measures": False,
            "include_calc_details": True,
            "years": [2080],
            "scenarios": ["rcp8p5"],
        }

        request = requests.AssetImpactRequest(**request_dict)  # type: ignore

        curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)

        source_paths = get_default_source_paths(EmbeddedInventory())
        vulnerability_models = {
            PowerGeneratingAsset: [InundationModel()],
            RealEstateAsset: [RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
        }

        response = requests._get_asset_impacts(
            request,
            ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
            vulnerability_models=vulnerability_models,
        )

        self.assertEqual(response.asset_impacts[0].impacts[0].hazard_type, "CoastalInundation")

    def test_risk_model_impact_request(self):
        """Tests the risk model functionality of the impact request."""

        assets = {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "location": "Asia",
                    "longitude": TestData.longitudes[0],
                    "latitude": TestData.latitudes[0],
                },
                {
                    "asset_class": "PowerGeneratingAsset",
                    "type": "Nuclear",
                    "location": "Asia",
                    "longitude": TestData.longitudes[1],
                    "latitude": TestData.latitudes[1],
                },
            ],
        }

        request_dict = {
            "assets": assets,
            "include_asset_level": True,
            "include_measures": False,
            "include_calc_details": True,
            "years": [2080],
            "scenarios": ["rcp8p5"],
        }

        request = requests.AssetImpactRequest(**request_dict)  # type: ignore

        curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)

        source_paths = get_default_source_paths(EmbeddedInventory())
        vulnerability_models = {
            PowerGeneratingAsset: [InundationModel()],
            RealEstateAsset: [RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
        }

        response = requests._get_asset_impacts(
            request,
            ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
            vulnerability_models=vulnerability_models,
        )

        self.assertEqual(response.asset_impacts[0].impacts[0].hazard_type, "CoastalInundation")

    def test_thermal_power_generation(self):
        latitudes = np.array([32.6017])
        longitudes = np.array([-87.7811])

        assets = {
            "items": [
                {
                    "asset_class": "ThermalPowerGeneratingAsset",
                    "type": "Gas",
                    "capacity": 1288.4,
                    "location": "North America",
                    "latitude": latitudes[0],
                    "longitude": longitudes[0],
                }
            ]
        }

        request_dict = {
            "assets": assets,
            "include_asset_level": True,
            "include_calc_details": True,
            "years": [2050],
            "scenarios": ["ssp585"],
        }

        request = requests.AssetImpactRequest(**request_dict)  # type: ignore

        store, root = zarr_memory_store()

        # Add mock riverine inundation data:
        return_periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        shape, t = shape_transform_21600_43200(return_periods=return_periods)
        add_curves(
            root,
            longitudes,
            latitudes,
            "inundation/wri/v2/inunriver_rcp8p5_MIROC-ESM-CHEM_2030",
            shape,
            np.array(
                [
                    8.378922939300537e-05,
                    0.3319014310836792,
                    0.7859689593315125,
                    1.30947744846344,
                    1.6689927577972412,
                    2.002290964126587,
                    2.416414737701416,
                    2.7177860736846924,
                    3.008821725845337,
                ]
            ),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "inundation/wri/v2/inunriver_rcp8p5_MIROC-ESM-CHEM_2050",
            shape,
            np.array(
                [
                    0.001158079132437706,
                    0.3938717246055603,
                    0.8549619913101196,
                    1.3880255222320557,
                    1.7519289255142212,
                    2.0910017490386963,
                    2.5129663944244385,
                    2.8202412128448486,
                    3.115604877471924,
                ]
            ),
            return_periods,
            t,
        )

        return_periods = [0.0]
        shape, t = shape_transform_21600_43200(return_periods=return_periods)

        # Add mock drought data:
        add_curves(
            root,
            longitudes,
            latitudes,
            "drought/jupiter/v1/months_spei3m_below_-2_ssp585_2050",
            shape,
            np.array([0.16958899796009064]),
            return_periods,
            t,
        )

        # Add mock chronic heat data:
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_25c_ACCESS-CM2_ssp585_2050",
            shape,
            np.array([148.55369567871094]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_30c_ACCESS-CM2_ssp585_2050",
            shape,
            np.array([65.30751037597656]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_35c_ACCESS-CM2_ssp585_2050",
            shape,
            np.array([0.6000000238418579]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_40c_ACCESS-CM2_ssp585_2050",
            shape,
            np.array([0.0]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_45c_ACCESS-CM2_ssp585_2050",
            shape,
            np.array([0.0]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_50c_ACCESS-CM2_ssp585_2050",
            shape,
            np.array([0.0]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_55c_ACCESS-CM2_ssp585_2050",
            shape,
            np.array([0.0]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_25c_ACCESS-CM2_historical_2005",
            shape,
            np.array([120.51940155029297]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_30c_ACCESS-CM2_historical_2005",
            shape,
            np.array([14.839207649230957]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_35c_ACCESS-CM2_historical_2005",
            shape,
            np.array([0.049863386899232864]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_40c_ACCESS-CM2_historical_2005",
            shape,
            np.array([0.0]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_45c_ACCESS-CM2_historical_2005",
            shape,
            np.array([0.0]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_50c_ACCESS-CM2_historical_2005",
            shape,
            np.array([0.0]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_tas_above_55c_ACCESS-CM2_historical_2005",
            shape,
            np.array([0.0]),
            return_periods,
            t,
        )

        source_paths = get_default_source_paths(EmbeddedInventory())
        vulnerability_models = {
            ThermalPowerGeneratingAsset: [
                ThermalPowerGenerationAirTemperatureModel(),
                ThermalPowerGenerationDroughtModel(),
                ThermalPowerGenerationRiverineInundationModel(),
            ]
        }

        response = requests._get_asset_impacts(
            request,
            ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
            vulnerability_models=vulnerability_models,
        )

        self.assertEqual(response.asset_impacts[0].impacts[0].impact_mean, 0.009028426082166906)
        self.assertEqual(response.asset_impacts[0].impacts[1].impact_mean, 0.0005486720213255343)
        self.assertEqual(response.asset_impacts[0].impacts[2].impact_mean, 0.005372887389199415)

    @unittest.skip("example, not test")
    def test_example_portfolios(self):
        example_portfolios = requests._get_example_portfolios()
        for assets in example_portfolios:
            request_dict = {
                "assets": assets,
                "include_asset_level": True,
                "include_calc_details": True,
                "years": [2050],
                "scenarios": ["ssp585"],
            }
            container = Container()
            requester = container.requester()
            response = requester.get(request_id="get_asset_impact", request_dict=request_dict)
            with open("out.json", "w") as f:
                f.write(response)
            assert response is not None
