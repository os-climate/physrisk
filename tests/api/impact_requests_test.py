import json
import unittest

import numpy as np
from pydantic import TypeAdapter

from physrisk import requests
from physrisk.api.v1.common import Asset, Assets
from physrisk.api.v1.impact_req_resp import RiskMeasures, RiskMeasuresHelper
from physrisk.container import Container
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import (
    PowerGeneratingAsset,
    RealEstateAsset,
    ThermalPowerGeneratingAsset,
)
from physrisk.kernel.vulnerability_model import DictBasedVulnerabilityModels
from physrisk.vulnerability_models.power_generating_asset_models import InundationModel
from physrisk.vulnerability_models.real_estate_models import (
    RealEstateCoastalInundationModel,
    RealEstateRiverineInundationModel,
)
from physrisk.vulnerability_models.thermal_power_generation_models import (
    ThermalPowerGenerationAirTemperatureModel,
    ThermalPowerGenerationDroughtModel,
    ThermalPowerGenerationRiverineInundationModel,
    ThermalPowerGenerationWaterStressModel,
    ThermalPowerGenerationWaterTemperatureModel,
)

from ..base_test import TestWithCredentials
from ..data.hazard_model_store_test import (
    TestData,
    add_curves,
    mock_hazard_model_store_inundation,
    shape_transform_21600_43200,
    zarr_memory_store,
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

    def test_extra_fields(self):
        assets = {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
                    "location": "Asia",
                    "longitude": 69.4787,
                    "latitude": 34.556,
                    "extra_field": 2.0,
                    "capacity": 1000.0,
                }
            ],
        }
        assets = requests.create_assets(Assets(**assets))
        # in the case of RealEstateAsset, extra fields are allowed, including those not in the Pydantic Asset object
        assert assets[0].capacity == 1000.0
        assert assets[0].extra_field == 2.0

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

        curve = np.array(
            [0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163]
        )
        store = mock_hazard_model_store_inundation(
            TestData.longitudes, TestData.latitudes, curve
        )

        source_paths = get_default_source_paths(EmbeddedInventory())
        vulnerability_models = DictBasedVulnerabilityModels(
            {
                PowerGeneratingAsset: [InundationModel()],
                RealEstateAsset: [
                    RealEstateCoastalInundationModel(),
                    RealEstateRiverineInundationModel(),
                ],
            }
        )

        response = requests._get_asset_impacts(
            request,
            ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
            vulnerability_models=vulnerability_models,
        )

        self.assertEqual(
            response.asset_impacts[0].impacts[0].hazard_type, "CoastalInundation"
        )

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

        curve = np.array(
            [0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163]
        )
        store = mock_hazard_model_store_inundation(
            TestData.longitudes, TestData.latitudes, curve
        )

        source_paths = get_default_source_paths(EmbeddedInventory())
        vulnerability_models = DictBasedVulnerabilityModels(
            {
                PowerGeneratingAsset: [InundationModel()],
                RealEstateAsset: [
                    RealEstateCoastalInundationModel(),
                    RealEstateRiverineInundationModel(),
                ],
            }
        )

        response = requests._get_asset_impacts(
            request,
            ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
            vulnerability_models=vulnerability_models,
        )

        self.assertEqual(
            response.asset_impacts[0].impacts[0].hazard_type, "CoastalInundation"
        )

    def test_thermal_power_generation(self):
        latitudes = np.array([32.6017])
        longitudes = np.array([-87.7811])

        assets = [
            ThermalPowerGeneratingAsset(
                latitude=latitudes[0],
                longitude=longitudes[0],
                location="North America",
                capacity=1288.4,
                type=archetype,
            )
            for archetype in [
                "Gas",
                "Gas/Gas",
                "Gas/Steam",
                "Gas/Steam/Dry",
                "Gas/Steam/OnceThrough",
                "Gas/Steam/Recirculating",
            ]
        ]

        assets_provided_in_the_request = False

        request_dict = {
            "assets": Assets(
                items=(
                    [
                        Asset(
                            asset_class=asset.__class__.__name__,
                            latitude=asset.latitude,
                            longitude=asset.longitude,
                            type=asset.type,
                            capacity=asset.capacity,
                            location=asset.location,
                        )
                        for asset in assets
                    ]
                    if assets_provided_in_the_request
                    else []
                )
            ),
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
            "inundation/wri/v2/inunriver_rcp4p5_MIROC-ESM-CHEM_2030",
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

        # Add mock drought data:
        return_periods = [0.0, -1.0, -1.5, -2.0, -2.5, -3.0, -3.6]
        shape, t = shape_transform_21600_43200(return_periods=return_periods)
        add_curves(
            root,
            longitudes,
            latitudes,
            "drought/osc/v1/months_spei12m_below_index_MIROC6_ssp585_2050",
            shape,
            np.array(
                [
                    6.900000095367432,
                    1.7999999523162842,
                    0.44999998807907104,
                    0.06584064255906408,
                    0.06584064255906408,
                    0.0,
                    0.0,
                ]
            ),
            return_periods,
            t,
        )

        return_periods = [0.0]
        shape, t = shape_transform_21600_43200(return_periods=return_periods)

        # Add mock drought (Jupiter) data:
        add_curves(
            root,
            longitudes,
            latitudes,
            "drought/jupiter/v1/months_spei3m_below_-2_ssp585_2050",
            shape,
            np.array([0.06584064255906408]),
            return_periods,
            t,
        )

        # Add mock water-related risk data:
        add_curves(
            root,
            longitudes,
            latitudes,
            "water_risk/wri/v2/water_stress_ssp585_2050",
            shape,
            np.array([0.14204320311546326]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "water_risk/wri/v2/water_supply_ssp585_2050",
            shape,
            np.array([76.09415435791016]),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "water_risk/wri/v2/water_supply_historical_1999",
            shape,
            np.array([88.62285614013672]),
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

        # Add mock water temperature data:
        return_periods = [
            5,
            7.5,
            10,
            12.5,
            15,
            17.5,
            20,
            22.5,
            25,
            27.5,
            30,
            32.5,
            35,
            37.5,
            40,
        ]
        shape, t = shape_transform_21600_43200(return_periods=return_periods)
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/nluu/v2/weeks_water_temp_above_GFDL_historical_1991",
            shape,
            np.array(
                [
                    52.0,
                    51.9,
                    49.666668,
                    45.066666,
                    38.0,
                    31.1,
                    26.0,
                    21.066668,
                    14.233334,
                    8.0333338,
                    5.0999999,
                    2.3666666,
                    6.6666669,
                    3.3333335,
                    0.0,
                ]
            ),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/nluu/v2/weeks_water_temp_above_GFDL_rcp8p5_2050",
            shape,
            np.array(
                [
                    51.85,
                    51.5,
                    50.25,
                    46.75,
                    41.95,
                    35.35,
                    29.4,
                    24.55,
                    20.15,
                    13.85,
                    6.75,
                    3.5,
                    1.3,
                    0.25,
                    0.1,
                ]
            ),
            return_periods,
            t,
        )

        # Add mock WBGT data:
        return_periods = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        shape, t = shape_transform_21600_43200(return_periods=return_periods)
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_wbgt_above_ACCESS-CM2_ssp585_2050",
            shape,
            np.array(
                [
                    363.65054,
                    350.21094,
                    303.6388,
                    240.48442,
                    181.82924,
                    128.46844,
                    74.400276,
                    1.3997267,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            return_periods,
            t,
        )
        add_curves(
            root,
            longitudes,
            latitudes,
            "chronic_heat/osc/v2/days_wbgt_above_ACCESS-CM2_historical_2005",
            shape,
            np.array(
                [
                    361.95273,
                    342.51804,
                    278.8146,
                    213.5123,
                    157.4511,
                    101.78238,
                    12.6897545,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            return_periods,
            t,
        )

        source_paths = get_default_source_paths(EmbeddedInventory())
        vulnerability_models = DictBasedVulnerabilityModels(
            {
                ThermalPowerGeneratingAsset: [
                    ThermalPowerGenerationAirTemperatureModel(),
                    ThermalPowerGenerationDroughtModel(),
                    ThermalPowerGenerationRiverineInundationModel(),
                    ThermalPowerGenerationWaterStressModel(),
                    ThermalPowerGenerationWaterTemperatureModel(),
                ]
            }
        )

        response = requests._get_asset_impacts(
            request,
            ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
            vulnerability_models=vulnerability_models,
            assets=None if assets_provided_in_the_request else assets,
        )

        # Air Temperature
        self.assertAlmostEqual(
            response.asset_impacts[0].impacts[0].impact_mean, 0.0075618606988512764
        )
        self.assertAlmostEqual(
            response.asset_impacts[1].impacts[0].impact_mean, 0.0075618606988512764
        )
        self.assertAlmostEqual(
            response.asset_impacts[2].impacts[0].impact_mean, 0.0025192163596997963
        )
        self.assertAlmostEqual(
            response.asset_impacts[3].impacts[0].impact_mean, 0.0025192163596997963
        )
        self.assertAlmostEqual(response.asset_impacts[4].impacts[0].impact_mean, 0.0)
        self.assertAlmostEqual(response.asset_impacts[5].impacts[0].impact_mean, 0.0)

        # Drought
        self.assertAlmostEqual(
            response.asset_impacts[0].impacts[1].impact_mean, 0.0008230079663917424
        )
        self.assertAlmostEqual(response.asset_impacts[1].impacts[1].impact_mean, 0.0)
        self.assertAlmostEqual(
            response.asset_impacts[2].impacts[1].impact_mean, 0.0008230079663917424
        )
        self.assertAlmostEqual(response.asset_impacts[3].impacts[1].impact_mean, 0.0)
        self.assertAlmostEqual(
            response.asset_impacts[4].impacts[1].impact_mean, 0.0008230079663917424
        )
        self.assertAlmostEqual(
            response.asset_impacts[5].impacts[1].impact_mean, 0.0008230079663917424
        )

        # Riverine Inundation
        self.assertAlmostEqual(
            response.asset_impacts[0].impacts[2].impact_mean, 0.0046864436945997625
        )
        self.assertAlmostEqual(
            response.asset_impacts[1].impacts[2].impact_mean, 0.0046864436945997625
        )
        self.assertAlmostEqual(
            response.asset_impacts[2].impacts[2].impact_mean, 0.0046864436945997625
        )
        self.assertAlmostEqual(
            response.asset_impacts[3].impacts[2].impact_mean, 0.0046864436945997625
        )
        self.assertAlmostEqual(
            response.asset_impacts[4].impacts[2].impact_mean, 0.0046864436945997625
        )
        self.assertAlmostEqual(
            response.asset_impacts[5].impacts[2].impact_mean, 0.0046864436945997625
        )

        # Water Stress
        self.assertAlmostEqual(
            response.asset_impacts[0].impacts[3].impact_mean, 0.010181435900296947
        )
        self.assertAlmostEqual(response.asset_impacts[1].impacts[3].impact_mean, 0.0)
        self.assertAlmostEqual(
            response.asset_impacts[2].impacts[3].impact_mean, 0.010181435900296947
        )
        self.assertAlmostEqual(response.asset_impacts[3].impacts[3].impact_mean, 0.0)
        self.assertAlmostEqual(
            response.asset_impacts[4].impacts[3].impact_mean, 0.010181435900296947
        )
        self.assertAlmostEqual(
            response.asset_impacts[5].impacts[3].impact_mean, 0.010181435900296947
        )

        # Water Temperature
        self.assertAlmostEqual(
            response.asset_impacts[0].impacts[4].impact_mean, 0.1448076958069578
        )
        self.assertAlmostEqual(response.asset_impacts[1].impacts[4].impact_mean, 0.0)
        self.assertAlmostEqual(
            response.asset_impacts[2].impacts[4].impact_mean, 0.1448076958069578
        )
        self.assertAlmostEqual(response.asset_impacts[3].impacts[4].impact_mean, 0.0)
        self.assertAlmostEqual(
            response.asset_impacts[4].impacts[4].impact_mean, 0.1448076958069578
        )
        self.assertAlmostEqual(
            response.asset_impacts[5].impacts[4].impact_mean, 0.005896707722257193
        )

        vulnerability_models = DictBasedVulnerabilityModels(
            {
                ThermalPowerGeneratingAsset: [
                    ThermalPowerGenerationDroughtModel(
                        impact_based_on_a_single_point=True
                    ),
                ]
            }
        )

        response = requests._get_asset_impacts(
            request,
            ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
            vulnerability_models=vulnerability_models,
            assets=None if assets_provided_in_the_request else assets,
        )

        # Drought (Jupiter)
        self.assertAlmostEqual(
            response.asset_impacts[0].impacts[0].impact_mean, 0.0005859470850072303
        )
        self.assertAlmostEqual(response.asset_impacts[1].impacts[0].impact_mean, 0.0)
        self.assertAlmostEqual(
            response.asset_impacts[2].impacts[0].impact_mean, 0.0005859470850072303
        )
        self.assertAlmostEqual(response.asset_impacts[3].impacts[0].impact_mean, 0.0)
        self.assertAlmostEqual(
            response.asset_impacts[4].impacts[0].impact_mean, 0.0005859470850072303
        )
        self.assertAlmostEqual(
            response.asset_impacts[5].impacts[0].impact_mean, 0.0005859470850072303
        )

    @unittest.skip("example, not test")
    def test_example_portfolios(self):
        example_portfolios = requests._get_example_portfolios()
        for assets in example_portfolios:
            request_dict = {
                "assets": assets,
                "include_asset_level": True,
                "include_calc_details": False,
                "years": [2030, 2040, 2050],
                "scenarios": ["ssp585"],
            }
            container = Container()
            requester = container.requester()
            response = requester.get(
                request_id="get_asset_impact", request_dict=request_dict
            )
            with open("out.json", "w") as f:
                f.write(response)
            assert response is not None

    @unittest.skip("example, not test")
    def test_example_portfolios_risk_measures(self):
        assets = {
            "items": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Commercial",
                    "location": "Europe",
                    "longitude": 11.5391,
                    "latitude": 48.1485,
                }
            ],
        }
        # 48.1485°, 11.5391°
        # 48.1537°, 11.5852°
        request_dict = {
            "assets": assets,
            "include_asset_level": True,
            "include_calc_details": True,
            "include_measures": True,
            "years": [2030, 2040, 2050],
            "scenarios": ["ssp245", "ssp585"],  # ["ssp126", "ssp245", "ssp585"],
        }
        container = Container()
        requester = container.requester()
        response = requester.get(
            request_id="get_asset_impact", request_dict=request_dict
        )
        response = requester.get(
            request_id="get_asset_impact", request_dict=request_dict
        )
        risk_measures_dict = json.loads(response)["risk_measures"]
        helper = RiskMeasuresHelper(
            TypeAdapter(RiskMeasures).validate_python(risk_measures_dict)
        )
        for hazard_type in [
            "RiverineInundation",
            "CoastalInundation",
            "ChronicHeat",
            "Wind",
        ]:
            scores, measure_values, measure_defns = helper.get_measure(
                hazard_type, "ssp585", 2050
            )
            label, description = helper.get_score_details(scores[0], measure_defns[0])
            print(label)
