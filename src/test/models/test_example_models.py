import unittest
from test.data.hazard_model_store import TestData, mock_hazard_model_store_inundation

import numpy as np
from scipy import stats

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import Asset, RealEstateAsset
from physrisk.kernel.hazard_model import HazardEventDataResponse
from physrisk.kernel.hazards import Inundation, RiverineInundation
from physrisk.kernel.impact import calculate_impacts
from physrisk.kernel.impact_distrib import ImpactType
from physrisk.kernel.vulnerability_matrix_provider import VulnMatrixProvider
from physrisk.kernel.vulnerability_model import VulnerabilityModel
from physrisk.vulnerability_models.example_models import ExampleCdfBasedVulnerabilityModel


class ExampleRealEstateInundationModel(VulnerabilityModel):
    def __init__(self):
        self.intensities = np.array([0, 0.01, 0.5, 1.0, 1.5, 2, 3, 4, 5, 6])
        self.impact_means = np.array([0, 0.2, 0.44, 0.58, 0.68, 0.78, 0.85, 0.92, 0.96, 1.0])
        self.impact_stddevs = np.array([0, 0.17, 0.14, 0.14, 0.17, 0.14, 0.13, 0.10, 0.06, 0])
        impact_bin_edges = np.array([0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        super().__init__(
            indicator_id="flood_depth",
            hazard_type=RiverineInundation,
            impact_bin_edges=impact_bin_edges,
            impact_type=ImpactType.damage,
        )

    def get_impact_curve(self, intensities, asset):
        # we interpolate the mean and standard deviation and use this to construct distributions
        impact_means = np.interp(intensities, self.intensities, self.impact_means)
        impact_stddevs = np.interp(intensities, self.intensities, self.impact_stddevs)
        return VulnMatrixProvider(
            intensities, impact_cdfs=[checked_beta_distrib(m, s) for m, s in zip(impact_means, impact_stddevs)]
        )


def delta_cdf(y):
    return lambda x: np.where(x >= y, 1, 0)


def checked_beta_distrib(mean, std):
    if mean == 0:
        return delta_cdf(0)
    if mean == 1.0:
        return delta_cdf(1)
    else:
        return beta_distrib(mean, std)


def beta_distrib(mean, std):
    cv = std / mean
    a = (1 - mean) / (cv * cv) - mean
    b = a * (1 - mean) / mean
    return lambda x, a=a, b=b: stats.beta.cdf(x, a, b)


class TestExampleModels(unittest.TestCase):
    def test_pdf_based_vulnerability_model(self):
        model = ExampleCdfBasedVulnerabilityModel(indicator_id="", hazard_type=Inundation)

        latitude, longitude = 45.268405, 19.885738
        asset = Asset(latitude, longitude)

        return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
        intensities = np.array(
            [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        )

        mock_response = HazardEventDataResponse(return_periods, intensities)

        vul, event = model.get_distributions(asset, [mock_response])

    def test_user_supplied_model(self):
        curve = np.array(
            [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        )
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)
        hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)

        scenario = "rcp8p5"
        year = 2080

        vulnerability_models = {RealEstateAsset: [ExampleRealEstateInundationModel()]}

        assets = [
            RealEstateAsset(lat, lon, location="Asia", type="Building/Industrial")
            for lon, lat in zip(TestData.longitudes, TestData.latitudes)
        ]

        results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)

        self.assertAlmostEqual(results[assets[0], RiverineInundation].impact.to_exceedance_curve().probs[0], 0.499)
