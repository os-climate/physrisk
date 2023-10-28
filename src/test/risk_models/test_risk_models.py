""" Test asset impact calculations."""
import test.data.hazard_model_store as hms
import unittest
from test.data.hazard_model_store import TestData, ZarrStoreMocker

import numpy as np

from physrisk.api.v1.impact_req_resp import RiskMeasureKey
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.calculation import get_default_vulnerability_models
from physrisk.kernel.hazards import CoastalInundation, RiverineInundation, Wind
from physrisk.kernel.risk import AssetLevelRiskModel, MeasureKey
from physrisk.requests import _create_risk_measures
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures


class TestRiskModels(unittest.TestCase):
    def test_risk_indicator_model(self):
        source_paths = get_default_source_paths()
        scenarios = ["rcp8p5"]
        years = [2050]

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
            mocker.add_curves_global(path, TestData.longitudes, TestData.latitudes, return_periods, flood_projected_curve)

        mocker.add_curves_global(sp_wind("historical", -1), TestData.longitudes, TestData.latitudes, return_periods, TestData.wind_intensities_1)
        mocker.add_curves_global(sp_wind("rcp8p5", -1), TestData.longitudes, TestData.latitudes, return_periods, TestData.wind_intensities_2)

        hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=mocker.store)

        assets = [
            RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
            for lon, lat in zip(TestData.longitudes[0:1], TestData.latitudes[0:1])
        ]

        assets = [
            RealEstateAsset(TestData.latitudes[0], TestData.longitudes[0], location="Asia", type="Buildings/Industrial")
            for i in range(2)
        ]

        model = AssetLevelRiskModel(
            hazard_model, get_default_vulnerability_models(), {RealEstateAsset: RealEstateToyRiskMeasures()}
        )
        measure_ids_for_asset, definitions = model.populate_measure_definitions(assets)
        _, measures = model.calculate_risk_measures(assets, prosp_scens=scenarios, years=years)

        # how to get a score using the MeasureKey
        measure = measures[MeasureKey(assets[0], scenarios[0], years[0], RiverineInundation)]
        score = measure.score
        measure_0 = measure.measure_0

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
