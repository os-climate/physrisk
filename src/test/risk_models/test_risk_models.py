""" Test asset impact calculations."""
import test.data.hazard_model_store as hms
from test.base_test import TestWithCredentials
from test.data.hazard_model_store import TestData, ZarrStoreMocker
from typing import NamedTuple, Sequence

import numpy as np

from physrisk import requests
from physrisk.api.v1.impact_req_resp import RiskMeasureKey, RiskMeasures
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.calculation import get_default_vulnerability_models
from physrisk.kernel.hazards import CoastalInundation, RiverineInundation, Wind
from physrisk.kernel.risk import AssetLevelRiskModel, MeasureKey
from physrisk.requests import _create_risk_measures
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures


class TestRiskModels(TestWithCredentials):
    def test_risk_indicator_model(self):
        scenarios = ["rcp8p5"]
        years = [2050]

        assets = self._create_assets()
        hazard_model = self._create_hazard_model(scenarios, years)

        model = AssetLevelRiskModel(
            hazard_model, get_default_vulnerability_models(), {RealEstateAsset: RealEstateToyRiskMeasures()}
        )
        measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
        _, measures = model.calculate_risk_measures(assets, prosp_scens=scenarios, years=years)

        # how to get a score using the MeasureKey
        measure = measures[MeasureKey(assets[0], scenarios[0], years[0], RiverineInundation)]
        score = measure.score
        measure_0 = measure.measure_0
        np.testing.assert_allclose([measure_0], [0.0896857])

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
        self.interpret_risk_measures(risk_measures)

        # example_json = risk_measures.model_dump_json()

    def _create_assets(self):
        # assets = [
        #    RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
        #    for lon, lat in zip(TestData.longitudes[0:1], TestData.latitudes[0:1])
        # ]

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

        mocker = ZarrStoreMocker()
        return_periods = hms.inundation_return_periods()
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
            vulnerability_models=get_default_vulnerability_models(),
        )
        res = next(
            ma for ma in response.risk_measures.measures_for_assets if ma.key.hazard_type == "RiverineInundation"
        )
        np.testing.assert_allclose(res.measures_0, [0.0896856974, 0.0896856974])
        # json_str = json.dumps(response.model_dump(), cls=NumpyArrayEncoder)

    def interpret_risk_measures(self, risk_measure: RiskMeasures):
        class Key(NamedTuple):  # hashable key for looking up measures
            hazard_type: str
            scenario_id: str
            year: str
            measure_id: str

        def key(key: RiskMeasureKey):
            return Key(
                hazard_type=key.hazard_type, scenario_id=key.scenario_id, year=key.year, measure_id=key.measure_id
            )

        # this is called a measure set, since each type of asset can have its own measure that defines the score.
        measure_set_id = risk_measure.score_based_measure_set_defn.measure_set_id
        measures = {key(m.key): m for m in risk_measure.measures_for_assets}

        # interested in asset with index 1
        asset_index = 1
        for hazard_type in [
            "ChronicHeat",
            "ChronicWind",
            "CoastalInundation",
            "CombinedInundation",
            "Drought",
            "Fire",
            "Hail",
            "Hazard",
            "Inundation",
            "Precipitation",
            "RiverineInundation",
            "Wind",
        ]:
            # for each type of hazard and each asset, the definition of the score can be different.
            # The definition is given by:
            measure_id = risk_measure.score_based_measure_set_defn.asset_measure_ids_for_hazard[hazard_type][
                asset_index
            ]
            # note that one of the aims of the schema is to keep the JSON size small for large numbers of assets;
            # hence arrays of length number of assets are used.
            # the definition of the measure:
            measure_definition = (
                risk_measure.score_based_measure_set_defn.score_definitions[measure_id] if measure_id != "na" else None
            )
            for scenario in risk_measure.scenarios:
                for year in scenario.years:
                    measure_key = Key(
                        hazard_type=hazard_type, scenario_id=scenario.id, year=str(year), measure_id=measure_set_id
                    )
                    if measure_key in measures:
                        measure = measures[measure_key]
                        asset_score = measure.scores[asset_index]
                        if asset_score != -1:
                            assert measure_definition is not None
                            asset_measure = measure.measures_0[asset_index]
                            print(f"For key {measure_key}, asset score is {asset_score}.")
                            print(f"The measure ID is {measure_id}.")
                            values = measure_definition.values
                            description = next(v for v in values if v.value == asset_score).description
                            print(
                                f"The description for measure ID {measure_id} of score {asset_score} is: {description}"
                            )
                            print(f"The underlying measure value is {asset_measure}.")
                            print(
                                f"The definition of the underlying measure is: \
                                {measure_definition.underlying_measures[0].description}"
                            )
