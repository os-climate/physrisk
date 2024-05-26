""" Test asset impact calculations."""

from typing import Sequence

import numpy as np

from physrisk import requests
from physrisk.api.v1.impact_req_resp import RiskMeasureKey, RiskMeasuresHelper
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.calculation import get_default_vulnerability_models
from physrisk.kernel.hazards import ChronicHeat, CoastalInundation, RiverineInundation, Wind
from physrisk.kernel.risk import AssetLevelRiskModel, MeasureKey
from physrisk.kernel.vulnerability_model import DictBasedVulnerabilityModels
from physrisk.requests import _create_risk_measures
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures

from ..base_test import TestWithCredentials
from ..data.hazard_model_store_test import TestData, ZarrStoreMocker, inundation_return_periods


class TestRiskModels(TestWithCredentials):
    def test_risk_indicator_model(self):
        scenarios = ["rcp8p5"]
        years = [2050]

        assets = self._create_assets()
        hazard_model = self._create_hazard_model(scenarios, years)

        model = AssetLevelRiskModel(
            hazard_model,
            DictBasedVulnerabilityModels(get_default_vulnerability_models()),
            {RealEstateAsset: RealEstateToyRiskMeasures()},
        )
        measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
        _, measures = model.calculate_risk_measures(assets, prosp_scens=scenarios, years=years)

        # how to get a score using the MeasureKey
        measure = measures[MeasureKey(assets[0], scenarios[0], years[0], RiverineInundation)]
        score = measure.score
        measure_0 = measure.measure_0
        np.testing.assert_allclose([measure_0], [0.89306593179])

        # packing up the risk measures, e.g. for JSON transmission:
        risk_measures = _create_risk_measures(measures, measure_ids_for_asset, definitions, assets, scenarios, years)
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
        asset_scores, measures, definitions = helper.get_measure("ChronicHeat", scenarios[0], years[0])
        label, description = helper.get_score_details(asset_scores[0], definitions[0])
        assert asset_scores[0] == 4


    def _create_assets(self):
        assets = [
            RealEstateAsset(TestData.latitudes[0], TestData.longitudes[0], location="Asia", type="Buildings/Industrial")
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
                    "attributes": {
                        "number_of_storeys": "2",
                        "structure_type": "concrete",
                    },
                }
                for asset in assets
            ],
        }
        return assets_dict

    def _create_hazard_model(self, scenarios, years):
        source_paths = get_default_source_paths()

        def sp_riverine(scenario, year):
            return source_paths[RiverineInundation](indicator_id="flood_depth", scenario=scenario, year=year)

        def sp_coastal(scenario, year):
            return source_paths[CoastalInundation](indicator_id="flood_depth", scenario=scenario, year=year)

        def sp_wind(scenario, year):
            return source_paths[Wind](indicator_id="max_speed", scenario=scenario, year=year)

        def sp_heat(scenario, year):
            return source_paths[ChronicHeat](indicator_id="mean_degree_days/above/index", scenario=scenario, year=year)

        mocker = ZarrStoreMocker()
        return_periods = inundation_return_periods()
        flood_histo_curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        flood_projected_curve = np.array([0.0596, 0.333, 0.605, 0.915, 1.164, 1.503, 1.649, 1.763, 1.963])

        for path in [sp_riverine("historical", 1980), sp_coastal("historical", 1980)]:
            mocker.add_curves_global(path, TestData.longitudes, TestData.latitudes, return_periods, flood_histo_curve)

        for path in [sp_riverine("rcp8p5", 2050), sp_coastal("rcp8p5", 2050)]:
            mocker.add_curves_global(
                path, TestData.longitudes, TestData.latitudes, return_periods, flood_projected_curve
            )

        mocker.add_curves_global(
            sp_wind("historical", -1),
            TestData.longitudes,
            TestData.latitudes,
            TestData.wind_return_periods,
            TestData.wind_intensities_1,
        )
        mocker.add_curves_global(
            sp_wind("rcp8p5", 2050),
            TestData.longitudes,
            TestData.latitudes,
            TestData.wind_return_periods,
            TestData.wind_intensities_2,
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

        return ZarrHazardModel(source_paths=get_default_source_paths(), store=mocker.store)

    def test_via_requests(self):
        scenarios = ["ssp585"]
        years = [2050]

        assets = self._create_assets()
        # hazard_model = ZarrHazardModel(source_paths=get_default_source_paths())
        hazard_model = self._create_hazard_model(scenarios, years)

        request_dict = {
            "assets": self._create_assets_json(assets),
            "include_asset_level": False,
            "include_measures": True,
            "include_calc_details": False,
            "years": years,
            "scenarios": scenarios,
        }

        request = requests.AssetImpactRequest(**request_dict)
        response = requests._get_asset_impacts(
            request,
            hazard_model,
            vulnerability_models=DictBasedVulnerabilityModels(get_default_vulnerability_models()),
        )
        res = next(
            ma for ma in response.risk_measures.measures_for_assets if ma.key.hazard_type == "RiverineInundation"
        )
        np.testing.assert_allclose(res.measures_0, [0.89306593179, 0.89306593179])
        # json_str = json.dumps(response.model_dump(), cls=NumpyArrayEncoder)
