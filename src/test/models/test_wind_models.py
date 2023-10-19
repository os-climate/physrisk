import test.data.hazard_model_store as hms

import numpy as np

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import ResourceSubset, get_default_source_path_provider
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.hazards import Wind
from physrisk.kernel.impact import calculate_impacts
from physrisk.vulnerability_models.real_estate_models import GenericTropicalCycloneModel


def test_wind_real_estate_model():
    scenario = "rcp8p5"
    year = 2080
    # mock some IRIS data for the calculation:
    store, root = hms.zarr_memory_store()
    return_periods = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    intensity = np.array([37.279999, 44.756248, 48.712502, 51.685001, 53.520000, 55.230000, 56.302502, 57.336250, 58.452499, 59.283749, 63.312500, 65.482498, 66.352501, 67.220001, 67.767502, 68.117500, 68.372498, 69.127502, 70.897499 ])
    shape, transform = hms.shape_transform_21600_43200(return_periods=return_periods)
    path = f"wind/iris/v1/max_speed_{scenario}_{year}".format(scenario=scenario, year=year)
    hms.add_curves(
        root, hms.TestData.longitudes, hms.TestData.latitudes, path, shape, intensity, return_periods, transform
    )

    provider = get_default_source_path_provider()

    def select_iris_osc(candidates: ResourceSubset, scenario: str, year: int, hint=None):
        return candidates.with_group_id("iris_osc").first()

    # specify use of IRIS (OSC contribution)
    provider.add_selector(Wind, "max_speed", select_iris_osc)

    hazard_model = ZarrHazardModel(source_paths=provider.source_paths(), store=store)
    assets = [
        RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
        for lon, lat in zip(hms.TestData.longitudes[0:1], hms.TestData.latitudes[0:1])
    ]
    vulnerability_models = {RealEstateAsset: [GenericTropicalCycloneModel()]}
    results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)
    # check calculation
    cum_probs = 1.0 / np.array(return_periods)
    probs = cum_probs[:-1] - cum_probs[1:]
    model = GenericTropicalCycloneModel()
    edges = np.interp(intensity, model.damage_curve_intensities, model.damage_curve_impacts)
    centres = (edges[1:] + edges[:-1]) / 2
    mean_check = np.sum(probs * centres)

    impact_distrib = results[(assets[0], Wind)].impact
    mean_impact = impact_distrib.mean_impact()
    np.testing.assert_allclose(mean_impact, mean_check)
