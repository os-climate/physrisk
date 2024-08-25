from typing import Dict, Sequence

import numpy as np
import pytest
from dependency_injector import providers

import physrisk.data.static.world as wd
from physrisk.api.v1.impact_req_resp import (
    AssetImpactResponse,
    RiskMeasureKey,
    RiskMeasuresHelper,
)
from physrisk.container import Container
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel import calculation  # noqa: F401 ## Avoid circular imports
from physrisk.kernel.assets import ThermalPowerGeneratingAsset
from physrisk.kernel.hazard_model import HazardModelFactory
from physrisk.kernel.hazards import (
    WaterRisk,
)
from physrisk.kernel.risk import AssetLevelRiskModel, MeasureKey
from physrisk.kernel.vulnerability_model import (
    DictBasedVulnerabilityModels,
    VulnerabilityModelsFactory,
)
from physrisk.requests import _create_risk_measures
from physrisk.vulnerability_models.thermal_power_generation_models import (
    ThermalPowerGenerationAqueductWaterRiskModel,
)


@pytest.fixture
def create_assets(wri_power_plant_assets):
    asset_list = wri_power_plant_assets
    filtered = asset_list.loc[
        asset_list["primary_fuel"].isin(["Coal", "Gas", "Nuclear", "Oil"])
    ]
    filtered = filtered[-60 < filtered["latitude"]]

    longitudes = np.array(filtered["longitude"])
    latitudes = np.array(filtered["latitude"])

    primary_fuels = np.array(
        [
            primary_fuel.replace(" and ", "And").replace(" ", "")
            for primary_fuel in filtered["primary_fuel"]
        ]
    )

    # Capacity describes a maximum electric power rate.
    # Generation describes the actual electricity output of the plant over a period of time.
    capacities = np.array(filtered["capacity_mw"])

    countries, continents = wd.get_countries_and_continents(
        latitudes=latitudes, longitudes=longitudes
    )

    assets = []
    for latitude, longitude, capacity, primary_fuel, country in zip(
        latitudes,
        longitudes,
        capacities,
        primary_fuels,
        countries,
    ):
        if country in ["Spain"]:
            assets.append(
                ThermalPowerGeneratingAsset(
                    latitude,
                    longitude,
                    type=primary_fuel,
                    location=country,
                    capacity=capacity,
                )
            )

    return assets


def create_assets_json(assets: Sequence[ThermalPowerGeneratingAsset]):
    assets_dict = {
        "items": [
            {
                "asset_class": type(asset).__name__,
                "type": asset.type,
                "location": asset.location,
                "longitude": asset.longitude,
                "latitude": asset.latitude,
            }
            for asset in assets
        ],
    }
    return assets_dict


@pytest.mark.skip(reason="Requires credentials.")
def test_risk_indicator_model(load_credentials, create_assets):
    scenarios = ["ssp585"]
    years = [2030]
    reader = ZarrReader()
    hazard_model = ZarrHazardModel(
        source_paths=get_default_source_paths(), reader=reader
    )
    assets = create_assets

    vulnerability_models = DictBasedVulnerabilityModels(
        {ThermalPowerGeneratingAsset: [ThermalPowerGenerationAqueductWaterRiskModel()]}
    )

    model = AssetLevelRiskModel(
        hazard_model=hazard_model,
        vulnerability_models=vulnerability_models,
        use_case_id="STRESS_TEST",
    )

    measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
    _, measures = model.calculate_risk_measures(
        assets, prosp_scens=scenarios, years=years
    )

    measure = measures[MeasureKey(assets[0], scenarios[0], years[0], WaterRisk)]
    score = measure.score
    measure_0 = measure.measure_0
    np.testing.assert_allclose([measure_0], [0.30079105])

    risk_measures = _create_risk_measures(
        measures, measure_ids_for_asset, definitions, assets, scenarios, years
    )
    key = RiskMeasureKey(
        hazard_type="WaterRisk",
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
        "WaterRisk", scenarios[0], years[0]
    )
    label, description = helper.get_score_details(asset_scores[0], definitions[0])
    assert asset_scores[0] == pytest.approx(0)  # TODO check if asserts are correct


@pytest.mark.skip(reason="Requires credentials.")
def test_via_requests(load_credentials, create_assets):
    scenarios = ["ssp585"]
    years = [2030]
    reader = ZarrReader()
    hazard_model = ZarrHazardModel(
        source_paths=get_default_source_paths(), reader=reader
    )

    assets = create_assets
    request_dict = {
        "assets": create_assets_json(assets=assets),
        "include_asset_level": False,
        "include_measures": True,
        "include_calc_details": False,
        "use_case_id": "STRESS_TEST",
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

    class TestVulnerabilityModelFactory(VulnerabilityModelsFactory):
        def vulnerability_models(self):
            vulnerability_models = DictBasedVulnerabilityModels(
                {
                    ThermalPowerGeneratingAsset: [
                        ThermalPowerGenerationAqueductWaterRiskModel()
                    ]
                }
            )
            return vulnerability_models

    container.override_providers(
        hazard_model_factory=providers.Factory(TestHazardModelFactory)
    )

    container.override_providers(
        config=providers.Configuration(default={"zarr_sources": ["embedded"]})
    )
    container.override_providers(inventory_reader=reader)
    container.override_providers(zarr_reader=reader)

    container.override_providers(
        vulnerability_models_factory=providers.Factory(TestVulnerabilityModelFactory)
    )

    requester = container.requester()
    res = requester.get(request_id="get_asset_impact", request_dict=request_dict)
    response = AssetImpactResponse.model_validate_json(res)

    res = next(
        ma
        for ma in response.risk_measures.measures_for_assets
        if ma.key.hazard_type == "WaterRisk"
    )
    np.testing.assert_allclose(res.measures_0[1], 1.067627)
