from typing import Dict, Sequence

import numpy as np
import pytest
from dependency_injector import providers

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
from physrisk.kernel.risk import AssetLevelRiskModel, MeasureKey
from physrisk.kernel.vulnerability_model import (
    DictBasedVulnerabilityModels,
)
from physrisk.requests import _create_risk_measures
from physrisk.risk_models.generic_risk_model import GenericScoreBasedRiskMeasures
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures
from physrisk.vulnerability_models.example_models import PlaceholderVulnerabilityModel
from physrisk.vulnerability_models.real_estate_models import (
    GenericTropicalCycloneModel,
    RealEstateCoastalInundationModel,
    RealEstatePluvialInundationModel,
    RealEstateRiverineInundationModel,
)

from ..data.hazard_model_store_test import (
    TestData,
    ZarrStoreMocker,
    inundation_return_periods,
)


@pytest.fixture
def create_assets():
    assets = [
        RealEstateAsset(
            TestData.latitudes[0],
            TestData.longitudes[0],
            location="Asia",
            type="Buildings/Industrial",
            id=f"unique_asset_{i}",
        )
        for i in range(2)
    ]
    return assets


def create_assets_json(assets: Sequence[RealEstateAsset]):
    assets_dict = {
        "items": [
            {
                "asset_class": type(asset).__name__,
                "type": asset.type,
                "location": asset.location,
                "longitude": asset.longitude,
                "latitude": asset.latitude,
                "attributes": {
                    "number_of_storeys": "2",
                    "structure_type": "concrete",
                },
            }
            for asset in assets
        ],
    }
    return assets_dict


def create_hazard_model(scenarios, years):
    source_paths = get_default_source_paths()

    def sp_riverine(scenario, year):
        return source_paths[RiverineInundation](
            indicator_id="flood_depth", scenario=scenario, year=year
        )

    def sp_coastal(scenario, year):
        return source_paths[CoastalInundation](
            indicator_id="flood_depth", scenario=scenario, year=year
        )

    def sp_wind(scenario, year):
        return source_paths[Wind](
            indicator_id="max_speed", scenario=scenario, year=year
        )

    def sp_heat(scenario, year):
        return source_paths[ChronicHeat](
            indicator_id="days/above/35c", scenario=scenario, year=year
        )

    def sp_heat_mean_degrees(scenario, year):
        return source_paths[ChronicHeat](
            indicator_id="mean_degree_days/above/index", scenario=scenario, year=year
        )

    def sp_fire(scenario, year):
        return source_paths[Fire](
            indicator_id="fire_probability", scenario=scenario, year=year
        )

    def sp_hail(scenario, year):
        return source_paths[Hail](
            indicator_id="days/above/5cm", scenario=scenario, year=year
        )

    def sp_drought(scenario, year):
        return source_paths[Drought](
            indicator_id="months/spei3m/below/-2", scenario=scenario, year=year
        )

    def sp_precipitation(scenario, year):
        return source_paths[Precipitation](
            indicator_id="max/daily/water_equivalent", scenario=scenario, year=year
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

    for path in [sp_riverine("rcp8p5", 2050), sp_coastal("rcp8p5", 2050)]:
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
        sp_wind("rcp8p5", 2050),
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
        TestData.temperature_thresholds,
        TestData.degree_days_above_index_1,
    )
    mocker.add_curves_global(
        sp_heat("rcp8p5", 2050),
        TestData.longitudes,
        TestData.latitudes,
        TestData.temperature_thresholds,
        TestData.degree_days_above_index_2,
    )
    mocker.add_curves_global(
        sp_heat_mean_degrees("historical", -1),
        TestData.longitudes,
        TestData.latitudes,
        TestData.temperature_thresholds,
        TestData.degree_days_above_index_1,
    )
    mocker.add_curves_global(
        sp_heat_mean_degrees("rcp8p5", 2050),
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
        sp_fire("rcp8p5", 2050),
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
        sp_hail("rcp8p5", 2050),
        TestData.longitudes,
        TestData.latitudes,
        [0],
        [4],
    )
    mocker.add_curves_global(
        sp_drought("historical", -1),
        TestData.longitudes,
        TestData.latitudes,
        [0],
        [3],
    )
    mocker.add_curves_global(
        sp_drought("rcp8p5", 2050),
        TestData.longitudes,
        TestData.latitudes,
        [0],
        [0.8],
    )
    mocker.add_curves_global(
        sp_precipitation("historical", -1),
        TestData.longitudes,
        TestData.latitudes,
        [0],
        [10],
    )
    mocker.add_curves_global(
        sp_precipitation("rcp8p5", 2050),
        TestData.longitudes,
        TestData.latitudes,
        [0],
        [70],
    )

    return ZarrHazardModel(source_paths=get_default_source_paths(), store=mocker.store)


@pytest.fixture
def hazard_model():
    scenarios = ["rcp8p5"]
    years = [2050]
    return create_hazard_model(scenarios, years)


def test_risk_indicator_model(hazard_model, create_assets):
    scenarios = ["rcp8p5"]
    years = [2050]

    assets = create_assets

    model = AssetLevelRiskModel(
        hazard_model,
        vulnerability_models=None,
        measure_calculators={RealEstateAsset: RealEstateToyRiskMeasures()},
        use_case_id="DEFAULT",
    )
    measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
    _, measures = model.calculate_risk_measures(
        assets, prosp_scens=scenarios, years=years
    )

    measure = measures[
        MeasureKey(assets[0], scenarios[0], years[0], RiverineInundation)
    ]
    score = measure.score
    measure_0 = measure.measure_0
    np.testing.assert_allclose([measure_0], [0.89306593179])

    risk_measures = _create_risk_measures(
        measures, measure_ids_for_asset, definitions, assets, scenarios, years
    )
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
        "ChronicHeat", scenarios[0], years[0]
    )
    label, description = helper.get_score_details(asset_scores[0], definitions[0])
    assert asset_scores[0] == 4


def test_via_requests(hazard_model, create_assets):
    scenarios = ["rcp8p5"]
    years = [2050]

    assets = create_assets
    request_dict = {
        "assets": create_assets_json(assets),
        "include_asset_level": False,
        "include_measures": True,
        "include_calc_details": False,
        "years": years,
        "scenarios": scenarios,
    }

    container = Container()

    class TestHazardModelFactory(HazardModelFactory):
        def hazard_model(
            self,
            interpolation: str = "floor",
            provider_max_requests: Dict[str, int] = {},
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

    requester = container.requester()
    res = requester.get(request_id="get_asset_impact", request_dict=request_dict)
    response = AssetImpactResponse.model_validate_json(res)

    res = next(
        ma
        for ma in response.risk_measures.measures_for_assets
        if ma.key.hazard_type == "RiverineInundation"
    )
    np.testing.assert_allclose(res.measures_0, [0.89306593179, 0.89306593179])


def test_generic_model(hazard_model):
    scenarios = ["rcp8p5"]
    years = [2050]

    assets = [
        Asset(TestData.latitudes[0], TestData.longitudes[0], id=f"unique_id_{i}")
        for i in range(2)
    ]

    model_set = [
        RealEstateCoastalInundationModel(),
        RealEstateRiverineInundationModel(),
        RealEstatePluvialInundationModel(),
        GenericTropicalCycloneModel(),
        PlaceholderVulnerabilityModel("fire_probability", Fire, ImpactType.damage),
        PlaceholderVulnerabilityModel("days/above/35c", ChronicHeat, ImpactType.damage),
        PlaceholderVulnerabilityModel("days/above/5cm", Hail, ImpactType.damage),
        PlaceholderVulnerabilityModel(
            "months/spei3m/below/-2", Drought, ImpactType.damage
        ),
        PlaceholderVulnerabilityModel(
            "max/daily/water_equivalent", Precipitation, ImpactType.damage
        ),
    ]

    vulnerability_models = {Asset: model_set, RealEstateAsset: model_set}

    generic_measures = GenericScoreBasedRiskMeasures()
    model = AssetLevelRiskModel(
        hazard_model=hazard_model,
        vulnerability_models=DictBasedVulnerabilityModels(vulnerability_models),
        measure_calculators={
            Asset: generic_measures,
            RealEstateAsset: generic_measures,
        },
        use_case_id="GENERIC",
    )
    measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
    _, measures = model.calculate_risk_measures(
        assets, prosp_scens=scenarios, years=years
    )
    np.testing.assert_approx_equal(
        measures[MeasureKey(assets[0], scenarios[0], years[0], Wind)].measure_0,
        214.01549835205077,
    )
    np.testing.assert_equal(
        measures[MeasureKey(assets[0], scenarios[0], years[0], Drought)].score,
        Category.HIGH,
    )
