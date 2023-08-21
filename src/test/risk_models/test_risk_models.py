""" Test asset impact calculations."""
import test.data.hazard_model_store as hms
import unittest
from test.data.hazard_model_store import TestData

import numpy as np

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.calculation import get_default_vulnerability_models
from physrisk.kernel.hazards import CoastalInundation, RiverineInundation
from physrisk.kernel.risk import AssetLevelRiskModel, MeasureKey
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures


class TestRiskModels(unittest.TestCase):
    """Tests RealEstateInundationModel."""

    def test_risk_indicator_model(self):
        source_paths = get_default_source_paths()
        scenarios = ["rcp8p5"]
        years = [2050]

        def sp_riverine(scenario, year):
            return source_paths[RiverineInundation](indicator_id="flood_depth", scenario=scenario, year=year)

        def sp_coastal(scenario, year):
            return source_paths[CoastalInundation](indicator_id="flood_depth", scenario=scenario, year=year)

        store, root = hms.zarr_memory_store()
        return_periods = hms.inundation_return_periods()
        shape, transform = hms.shape_transform_21600_43200(return_periods=return_periods)

        histo_curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        projected_curve = np.array([0.0596, 0.333, 0.605, 0.915, 1.164, 1.503, 1.649, 1.763, 1.963])
        for path in [sp_riverine("historical", 1980), sp_coastal("historical", 1980)]:
            hms.add_curves(
                root, TestData.longitudes, TestData.latitudes, path, shape, histo_curve, return_periods, transform
            )
        for path in [sp_riverine("rcp8p5", 2050), sp_coastal("rcp8p5", 2050)]:
            hms.add_curves(
                root, TestData.longitudes, TestData.latitudes, path, shape, projected_curve, return_periods, transform
            )

        hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)

        assets = [
            RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
            for lon, lat in zip(TestData.longitudes[0:1], TestData.latitudes[0:1])
        ]

        model = AssetLevelRiskModel(
            hazard_model, get_default_vulnerability_models(), {RealEstateAsset: RealEstateToyRiskMeasures()}
        )
        impacts, measures = model.calculate_risk_measures(assets, prosp_scens=scenarios, years=years)
        print(measures[MeasureKey(assets[0], scenarios[0], years[0], RiverineInundation)])
