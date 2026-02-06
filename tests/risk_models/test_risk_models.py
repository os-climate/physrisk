"""Test asset impact calculations."""

from typing import Dict, Optional, Sequence, Set, Type

import numpy as np
from dependency_injector import providers

from physrisk.kernel.hazards import Hazard
from physrisk.api.v1.impact_req_resp import (
    AssetImpactResponse,
    Category,
    RiskMeasureKey,
    RiskMeasuresHelper,
)
from physrisk.container import Container
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import Asset, RealEstateAsset
from physrisk.kernel.calculation import alternate_default_vulnerability_models_scores
from physrisk.kernel.hazard_model import HazardModelFactory
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    Drought,
    Fire,
    Hail,
    Precipitation,
    RiverineInundation,
    Wind,
)
from physrisk.kernel.impact_distrib import ImpactType
from physrisk.kernel.risk import (
    AssetLevelRiskModel,
    MeasureKey,
    NullAssetBasedPortfolioRiskMeasureCalculator,
    RiskMeasuresFactory,
)
from physrisk.risk_models.portfolio_risk_model import (
    AveragingAssetBasedPortfolioRiskMeasureCalculator,
)
from physrisk.kernel.vulnerability_model import (
    DictBasedVulnerabilityModels,
    VulnerabilityModels,
    VulnerabilityModelsFactory as PVulnerabilityModelsFactory,
)
from physrisk.requests import _create_risk_measures
from physrisk.risk_models.generic_risk_model import GenericScoreBasedRiskMeasures
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures
from physrisk.vulnerability_models.config_based_impact_curves import (
    VulnerabilityConfigItem,
)
from physrisk.vulnerability_models.example_models import PlaceholderVulnerabilityModel
from physrisk.vulnerability_models.real_estate_models import (
    GenericTropicalCycloneModel,
    RealEstateCoastalInundationModel,
    RealEstatePluvialInundationModel,
    RealEstateRiverineInundationModel,
)
from physrisk.vulnerability_models.vulnerability import VulnerabilityModelsFactory

from ..test_base import TestWithCredentials
from ..data.test_hazard_model_store import (
    TestData,
    ZarrStoreMocker,
    inundation_return_periods,
)


def _config_based_vulnerability_models():
    config = [
        VulnerabilityConfigItem(
            hazard_class="ChronicHeat",
            asset_class="Asset",
            asset_identifier="type=Generic",
            indicator_id="days_wbgt_above",
            indicator_units="days/year",
            impact_id="disruption",
            curve_type="threshold/piecewise_linear",
            points_x=np.array([20, 24, 28, 32, 36, 40, 44, 48]),
            points_y=np.array([0, 0.01, 0.125, 0.45, 0.775, 0.975, 1.0, 1.0]),
        ),
        VulnerabilityConfigItem(
            hazard_class="Drought",
            asset_class="Asset",
            asset_identifier="type=Generic",
            indicator_id="months/spei12m/below/threshold",
            indicator_units="months/year",
            impact_id="disruption",
            curve_type="threshold/piecewise_linear",
            points_x=np.array([-3.5, -3.0, -2.5, -2]),
            points_y=np.array([1.0, 0.2, 0.1, 0.0]),
        ),
        VulnerabilityConfigItem(
            hazard_class="Fire",
            asset_class="Asset",
            asset_identifier="type=Generic",
            indicator_id="fire_probability",
            indicator_units="",
            impact_id="damage",
            curve_type="threshold/piecewise_linear",
            points_x=np.array([0.0]),
            points_y=np.array([0.7]),
        ),
        VulnerabilityConfigItem(
            hazard_class="Hail",
            asset_class="Asset",
            asset_identifier="type=Generic",
            indicator_id="days/above/5cm",
            indicator_units="days/year",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=np.array([0.0, 10.0]),
            points_y=np.array([0.0, 0.01]),
        ),
    ]
    return config


def _vulnerability_models():
    model_set = [
        RealEstateCoastalInundationModel(),
        RealEstateRiverineInundationModel(),
        RealEstatePluvialInundationModel(),
        GenericTropicalCycloneModel(),
        PlaceholderVulnerabilityModel(
            "max/daily/water_equivalent", Precipitation, ImpactType.damage
        ),
    ]
    programmatic_models = {Asset: model_set, RealEstateAsset: model_set}
    factory = VulnerabilityModelsFactory(
        config=_config_based_vulnerability_models(),
        programmatic_models=programmatic_models,
    )
    models = factory.vulnerability_models(disable_api_calls=True)
    return models


class TestRiskModels(TestWithCredentials):
    def test_risk_indicator_model(self):
        scenarios = ["ssp585"]
        years = [2050]

        assets = self._create_assets()
        hazard_model = self._create_hazard_model(scenarios, years)

        model = AssetLevelRiskModel(
            hazard_model,
            DictBasedVulnerabilityModels(
                alternate_default_vulnerability_models_scores()
            ),
            {RealEstateAsset: RealEstateToyRiskMeasures()},
            NullAssetBasedPortfolioRiskMeasureCalculator(),
        )
        measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
        _, measures = model.calculate_risk_measures(
            assets, scenarios=scenarios, years=years
        )

        # how to get a score using the MeasureKey
        measure = measures[
            MeasureKey(assets[0], scenarios[0], years[0], RiverineInundation)
        ]
        score = measure.score
        measure_0 = measure.measure_0
        np.testing.assert_allclose([measure_0], [0.89306593179])

        # packing up the risk measures, e.g. for JSON transmission:
        risk_measures = _create_risk_measures(
            measures, measure_ids_for_asset, definitions, assets, scenarios, years
        )
        # we still have a key, but no asset:
        key = RiskMeasureKey(
            hazard_type="RiverineInundation",
            scenario_id=scenarios[0],
            year=str(years[0]),
            measure_id=risk_measures.score_based_measure_set_defn.measure_set_id,
        )
        item = next(m for m in risk_measures.measures_for_assets if m.key == key)
        score2 = item.scores[0]
        measure_0_2 = item.measures_0[0]
        assert score == score2
        assert measure_0 == measure_0_2

        helper = RiskMeasuresHelper(risk_measures)
        asset_scores, measures, definitions = helper.get_measure(
            "CoastalInundation", scenarios[0], years[0]
        )
        label, description = helper.get_score_details(asset_scores[0], definitions[0])
        assert asset_scores[0] == 4

    def _create_assets(self):
        assets = [
            RealEstateAsset(
                latitude=TestData.latitudes[0],
                longitude=TestData.longitudes[0],
                location="Asia",
                type="Buildings/Industrial",
                id=f"unique_asset_{i}",
            )
            for i in range(2)
        ]
        return assets

    def _create_assets_json(self, assets: Sequence[RealEstateAsset]):
        assets_dict = {
            "items": [
                {
                    "asset_class": type(asset).__name__,
                    "type": asset.type,
                    "location": asset.location,
                    "longitude": asset.longitude,
                    "latitude": asset.latitude,
                    "number_of_storeys": 2,
                }
                for asset in assets
            ],
        }
        return assets_dict

    def _create_hazard_model(self, scenarios, years):
        source_paths = get_default_source_paths()

        def sp_riverine(scenario, year):
            return (
                source_paths.resource_paths(
                    RiverineInundation, indicator_id="flood_depth", scenarios=[scenario]
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        def sp_coastal(scenario, year):
            return (
                source_paths.resource_paths(
                    CoastalInundation, indicator_id="flood_depth", scenarios=[scenario]
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        def sp_wind(scenario, year):
            return (
                source_paths.resource_paths(
                    Wind, indicator_id="max_speed", scenarios=[scenario]
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        def sp_heat(scenario, year):
            return (
                source_paths.resource_paths(
                    ChronicHeat, indicator_id="days_wbgt_above", scenarios=[scenario]
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        def sp_heat2(scenario, year):
            return (
                source_paths.resource_paths(
                    ChronicHeat,
                    indicator_id="mean_degree_days/above/index",
                    scenarios=[scenario],
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        def sp_fire(scenario, year):
            return (
                source_paths.resource_paths(
                    Fire, indicator_id="fire_probability", scenarios=[scenario]
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        def sp_hail(scenario, year):
            return (
                source_paths.resource_paths(
                    Hail, indicator_id="days/above/5cm", scenarios=[scenario]
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        def sp_drought(scenario, year):
            return (
                source_paths.resource_paths(
                    Drought,
                    indicator_id="months/spei12m/below/threshold",
                    scenarios=[scenario],
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        def sp_precipitation(scenario, year):
            return (
                source_paths.resource_paths(
                    Precipitation,
                    indicator_id="max/daily/water_equivalent",
                    scenarios=[scenario],
                )[0]
                .scenarios[scenario]
                .path(year)
            )

        mocker = ZarrStoreMocker()
        return_periods = inundation_return_periods()
        flood_histo_curve = np.array(
            [0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163]
        )
        flood_projected_curve = np.array(
            [0.0596, 0.333, 0.605, 0.915, 1.164, 1.503, 1.649, 1.763, 1.963]
        )

        for path in [sp_riverine("historical", 1980), sp_coastal("historical", 1980)]:
            mocker.add_curves_global(
                path,
                TestData.longitudes,
                TestData.latitudes,
                return_periods,
                flood_histo_curve,
            )

        for path in [sp_riverine("ssp585", 2050), sp_coastal("ssp585", 2050)]:
            mocker.add_curves_global(
                path,
                TestData.longitudes,
                TestData.latitudes,
                return_periods,
                flood_projected_curve,
            )

        mocker.add_curves_global(
            sp_wind("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            TestData.wind_return_periods,
            TestData.wind_intensities_1,
            units="m/s",
        )
        mocker.add_curves_global(
            sp_wind("ssp585", 2050),
            TestData.longitudes,
            TestData.latitudes,
            TestData.wind_return_periods,
            TestData.wind_intensities_2,
            units="m/s",
        )
        mocker.add_curves_global(
            sp_heat("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            [5.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            np.array(
                [
                    327.28,
                    315.19,
                    273.28,
                    216.43,
                    163.65,
                    115.62,
                    66.96,
                    1.26,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        mocker.add_curves_global(
            sp_heat("ssp585", 2050),
            TestData.longitudes,
            TestData.latitudes,
            [5.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            np.array(
                [
                    363.65,
                    350.21,
                    303.64,
                    240.48,
                    181.83,
                    128.47,
                    74.40,
                    1.40,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        mocker.add_curves_global(
            sp_heat2("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            TestData.temperature_thresholds,
            TestData.degree_days_above_index_1,
        )
        mocker.add_curves_global(
            sp_heat2("ssp585", 2050),
            TestData.longitudes,
            TestData.latitudes,
            TestData.temperature_thresholds,
            TestData.degree_days_above_index_2,
        )
        mocker.add_curves_global(
            sp_fire("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            [0],
            [0.15],
        )
        mocker.add_curves_global(
            sp_fire("ssp585", 2050),
            TestData.longitudes,
            TestData.latitudes,
            [0],
            [0.2],
        )
        mocker.add_curves_global(
            sp_hail("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            [0],
            [2.15],
        )
        mocker.add_curves_global(
            sp_hail("ssp585", 2050),
            TestData.longitudes,
            TestData.latitudes,
            [0],
            [4],
        )
        mocker.add_curves_global(
            sp_drought("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            # data should be this way around, consistent with temperature type threshold curves
            [0.0, -1.0, -1.5, -2.0, -2.5, -3.0, -3.6],
            np.array([6.5, 2.1, 0.86, 0.29, 0.058, 0.0, 0.0]),
        )
        mocker.add_curves_global(
            sp_drought("ssp585", 2050),
            TestData.longitudes,
            TestData.latitudes,
            [0.0, -1.0, -1.5, -2.0, -2.5, -3.0, -3.6],
            np.array([7.3, 3.7, 2.4, 1.5, 0.56, 0.075, 0.0]),
        )
        mocker.add_curves_global(
            sp_precipitation("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            [0],
            [10],
        )
        mocker.add_curves_global(
            sp_precipitation("ssp585", 2050),
            TestData.longitudes,
            TestData.latitudes,
            [0],
            [70],
        )

        return ZarrHazardModel(
            source_paths=get_default_source_paths(), store=mocker.store
        )

    def test_generic_model_via_requests_default_vulnerability(self):
        scenarios = ["ssp585", "historical"]
        years = [2050]

        assets = self._create_assets()
        # hazard_model = ZarrHazardModel(source_paths=get_default_source_paths())
        hazard_model = self._create_hazard_model(scenarios, years)

        request_dict = {
            "assets": self._create_assets_json(assets),
            "include_asset_level": False,
            "include_measures": True,
            "include_calc_details": True,
            "years": years,
            "scenarios": scenarios,
        }

        container = Container()

        class TestHazardModelFactory(HazardModelFactory):
            def hazard_model(
                self,
                interpolation: Optional[str] = "floor",
                provider_max_requests: Dict[str, int] = {},
                interpolate_years: bool = False,
            ):
                return hazard_model

        container.override_providers(
            hazard_model_factory=providers.Factory(TestHazardModelFactory)
        )
        container.override_providers(
            config=providers.Configuration(default={"zarr_sources": ["embedded"]})
        )
        container.override_providers(inventory_reader=None)
        container.override_providers(zarr_reader=None)
        container.override_providers(sig_figures=6)

        requester = container.requester()
        res = requester.get(request_id="get_asset_impact", request_dict=request_dict)
        response = AssetImpactResponse.model_validate_json(res)

        res = next(
            ma
            for ma in response.risk_measures.measures_for_assets
            if ma.key.hazard_type == "RiverineInundation"
        )
        np.testing.assert_allclose(res.measures_0, [0.0959039, 0.0959039])

    def test_generic_model(self):
        scenarios = ["ssp585"]
        years = [2050]

        assets = [
            Asset(TestData.latitudes[0], TestData.longitudes[0], id=f"unique_id_{i}")
            for i in range(2)
        ]
        # assets = [RealEstateAsset(TestData.latitudes[0], TestData.longitudes[0], location="Asia", type="Buildings/Industrial") for i in range(2)]
        hazard_model = self._create_hazard_model(scenarios, years)

        generic_measures = GenericScoreBasedRiskMeasures()

        model = AssetLevelRiskModel(
            hazard_model,
            _vulnerability_models(),
            {Asset: generic_measures, RealEstateAsset: generic_measures},
            NullAssetBasedPortfolioRiskMeasureCalculator(),
        )
        measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
        _, measures = model.calculate_risk_measures(
            assets, scenarios=scenarios, years=years
        )
        np.testing.assert_approx_equal(
            measures[MeasureKey(assets[0], scenarios[0], years[0], Wind)].measure_0,
            0.0101490442129,
        )
        # another check on drought.
        # thresholds:
        # [0.0, -1.0, -1.5, -2.0, -2.5, -3.0, -3.6],
        # months/year above threshold:
        # [6.5, 2.1, 0.86, 0.29, 0.058, 0.0, 0.0]
        # [7.3, 3.7, 2.4, 1.5, 0.56, 0.075, 0.0]
        # points_x = (np.array([-2, -2.5, -3, -3.5]),)
        # points_y = (np.array([0, 0.1, 0.2, 1.0]),)
        expected_hist = 0.058 * 0.15 / 12 + (0.29 - 0.058) * 0.05 / 12
        expected_fut = (
            0.075 * 0.6 / 12 + (0.56 - 0.075) * 0.15 / 12 + (1.5 - 0.56) * 0.05 / 12
        )

        np.testing.assert_approx_equal(
            measures[MeasureKey(assets[0], scenarios[0], years[0], Drought)].measure_0,
            expected_fut - expected_hist,
        )
        np.testing.assert_equal(
            measures[MeasureKey(assets[0], scenarios[0], years[0], Drought)].score,
            Category.MEDIUM,
        )

    def test_generic_model_via_requests_custom(self):
        scenarios = ["ssp585", "historical"]
        years = [2050]

        assets = self._create_assets()
        # hazard_model = ZarrHazardModel(source_paths=get_default_source_paths())
        hazard_model = self._create_hazard_model(scenarios, years)

        request_dict = {
            "assets": self._create_assets_json(assets),
            "include_asset_level": True,
            "include_measures": True,
            "include_calc_details": True,
            "years": years,
            "scenarios": scenarios,
        }

        container = Container()

        class TestHazardModelFactory(HazardModelFactory):
            def hazard_model(
                self,
                interpolation: Optional[str] = "floor",
                provider_max_requests: Dict[str, int] = {},
                interpolate_years: bool = False,
            ):
                return hazard_model

        class TestVulnerabilityModelsFactory(PVulnerabilityModelsFactory):
            def vulnerability_models(
                self, hazard_scope: Optional[Set[Type[Hazard]]] = None
            ) -> VulnerabilityModels:
                return _vulnerability_models()

        class TestMeasuresFactory(RiskMeasuresFactory):
            def asset_calculators(self, use_case_id: str):
                return {RealEstateAsset: GenericScoreBasedRiskMeasures()}

            def portfolio_calculator(self, use_case_id: str):
                return AveragingAssetBasedPortfolioRiskMeasureCalculator()

        container.override_providers(
            hazard_model_factory=providers.Factory(TestHazardModelFactory)
        )
        container.override_providers(
            vulnerability_models_factory=providers.Factory(
                TestVulnerabilityModelsFactory
            )
        )
        container.override_providers(
            measures_factory=providers.Factory(TestMeasuresFactory)
        )
        container.override_providers(
            config=providers.Configuration(default={"zarr_sources": ["embedded"]})
        )
        container.override_providers(inventory_reader=None)
        container.override_providers(zarr_reader=None)
        container.override_providers(sig_figures=6)

        requester = container.requester()
        res = requester.get(request_id="get_asset_impact", request_dict=request_dict)
        # check 'round-trip' validation:
        response = AssetImpactResponse.model_validate_json(res, strict=False)
        # check that when there is a placeholder vulnerability model with no impact, the calculation details are still returned.
        impact_for_placeholder = next(
            i for i in response.asset_impacts[0].impacts if i.key.hazard_type == "Fire"
        )
        assert impact_for_placeholder.calc_details.hazard_path is not None
