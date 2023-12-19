""" Test asset impact calculations."""
import unittest
from test.data.hazard_model_store import TestData, mock_hazard_model_store_inundation

import numpy as np

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.hazards import CoastalInundation, RiverineInundation
from physrisk.kernel.impact import ImpactKey, calculate_impacts
from physrisk.vulnerability_models.real_estate_models import (
    RealEstateCoastalInundationModel,
    RealEstateRiverineInundationModel,
)


class TestRealEstateModels(unittest.TestCase):
    """Tests RealEstateInundationModel."""

    def test_real_estate_model_details(self):
        curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)
        hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)

        # location="Europe", type="Buildings/Residential"
        assets = [
            RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
            for lon, lat in zip(TestData.longitudes[0:1], TestData.latitudes[0:1])
        ]

        scenario = "rcp8p5"
        year = 2080

        vulnerability_models = {RealEstateAsset: [RealEstateRiverineInundationModel()]}

        results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)

        hazard_bin_edges = results[ImpactKey(assets[0], RiverineInundation)].event.intensity_bin_edges
        hazard_bin_probs = results[ImpactKey(assets[0], RiverineInundation)].event.prob

        # check one:
        # the probability of inundation greater than 0.505m in a year is 1/10.0
        # the probability of inundation greater than 0.333m in a year is 1/5.0
        # therefore the probability of an inundation between 0.333 and 0.505 in a year is 1/5.0 - 1/10.0
        np.testing.assert_almost_equal(hazard_bin_edges[1:3], np.array([0.333, 0.505]))
        np.testing.assert_almost_equal(hazard_bin_probs[1], 0.1)

        # check that intensity bin edges for vulnerability matrix are same as for hazard
        vulnerability_intensity_bin_edges = results[
            ImpactKey(assets[0], RiverineInundation)
        ].vulnerability.intensity_bins
        np.testing.assert_almost_equal(vulnerability_intensity_bin_edges, hazard_bin_edges)

        # check the impact distribution the matrix is size [len(intensity_bins) - 1, len(impact_bins) - 1]
        cond_probs = results[ImpactKey(assets[0], RiverineInundation)].vulnerability.prob_matrix[1, :]
        # check conditional prob for inundation intensity 0.333..0.505
        mean, std = np.mean(cond_probs), np.std(cond_probs)
        np.testing.assert_almost_equal(cond_probs.sum(), 1)
        np.testing.assert_allclose([mean, std], [0.09090909, 0.08184968], rtol=1e-6)

        # probability that impact occurs between impact bin edge 1 and impact bin edge 2
        prob_impact = np.dot(
            hazard_bin_probs, results[ImpactKey(assets[0], RiverineInundation)].vulnerability.prob_matrix[:, 1]
        )
        np.testing.assert_almost_equal(prob_impact, 0.19350789547968042)

        # no check with pre-calculated values for others:
        np.testing.assert_allclose(
            results[ImpactKey(assets[0], RiverineInundation)].impact.prob,
            np.array(
                [
                    0.02815762,
                    0.1935079,
                    0.11701139,
                    0.06043065,
                    0.03347816,
                    0.02111368,
                    0.01504522,
                    0.01139892,
                    0.00864469,
                    0.00626535,
                    0.00394643,
                ]
            ),
            rtol=2e-6,
        )

    def test_coastal_real_estate_model(self):
        curve = np.array([0.223, 0.267, 0.29, 0.332, 0.359, 0.386, 0.422, 0.449, 0.476])

        store = mock_hazard_model_store_inundation(TestData.coastal_longitudes, TestData.coastal_latitudes, curve)
        hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)

        # location="Europe", type="Buildings/Residential"
        assets = [
            RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
            for lon, lat in zip(TestData.coastal_longitudes[0:1], TestData.coastal_latitudes[0:1])
        ]

        scenario = "rcp8p5"
        year = 2080

        vulnerability_models = {RealEstateAsset: [RealEstateCoastalInundationModel()]}

        results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)

        np.testing.assert_allclose(
            results[ImpactKey(assets[0], CoastalInundation)].impact.prob,
            np.array(
                [
                    2.78081230e-02,
                    1.96296619e-01,
                    1.32234770e-01,
                    7.36581177e-02,
                    3.83434609e-02,
                    1.83916914e-02,
                    7.97401009e-03,
                    3.04271878e-03,
                    9.79400125e-04,
                    2.41250436e-04,
                    2.98387241e-05,
                ]
            ),
            rtol=2e-6,
        )

    def test_commercial_real_estate_model_details(self):
        curve = np.array(
            [2.8302893e-06, 0.09990284, 0.21215445, 0.531271, 0.7655724, 0.99438345, 1.2871761, 1.502281, 1.7134278]
        )
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)
        hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)

        # location="South America", type="Buildings/Commercial"
        assets = [
            RealEstateAsset(lat, lon, location="South America", type="Buildings/Commercial")
            for lon, lat in zip(TestData.longitudes[-4:-3], TestData.latitudes[-4:-3])
        ]

        scenario = "rcp8p5"
        year = 2080

        # impact bin edges are calibrated so that hazard_bin_probs == impact_bin_probs
        # when the impact standard deviation is negligible:
        vulnerability_models = {
            RealEstateAsset: [
                RealEstateRiverineInundationModel(
                    impact_bin_edges=np.array(
                        [
                            0,
                            0.030545039098059,
                            0.125953058445539,
                            0.322702019487674,
                            0.566880882840096,
                            0.731980974578735,
                            0.823993215529066,
                            0.884544511664047,
                            0.922115133960502,
                            0.969169745946688,
                            1.0,
                        ]
                    )
                )
            ]
        }

        results = calculate_impacts(assets, hazard_model, vulnerability_models, scenario=scenario, year=year)

        hazard_bin_edges = results[ImpactKey(assets[0], RiverineInundation)].event.intensity_bin_edges
        hazard_bin_probs = results[ImpactKey(assets[0], RiverineInundation)].event.prob

        # check one:
        # the probability of inundation greater than 0.531271m in a year is 1/25
        # the probability of inundation greater than 0.21215445m in a year is 1/10
        # therefore the probability of an inundation between 0.21215445 and 0.531271 in a year is 1/10 - 1/25
        np.testing.assert_almost_equal(hazard_bin_edges[2:4], np.array([0.21215445, 0.531271]))
        np.testing.assert_almost_equal(hazard_bin_probs[2], 0.06)

        # check that intensity bin edges for vulnerability matrix are same as for hazard
        vulnerability_intensity_bin_edges = results[
            ImpactKey(assets[0], RiverineInundation)
        ].vulnerability.intensity_bins
        np.testing.assert_almost_equal(vulnerability_intensity_bin_edges, hazard_bin_edges)

        # check the impact distribution the matrix is size [len(intensity_bins) - 1, len(impact_bins) - 1]
        cond_probs = results[ImpactKey(assets[0], RiverineInundation)].vulnerability.prob_matrix[2, :]
        # check conditional prob for inundation intensity at 0.371712725m
        mean, std = np.mean(cond_probs), np.std(cond_probs)
        np.testing.assert_almost_equal(cond_probs.sum(), 1)
        np.testing.assert_allclose([mean, std], [0.1, 0.2884275164878624], rtol=1e-6)

        # probability that impact occurs between impact bin edge 2 and impact bin edge 3
        prob_impact = np.dot(
            hazard_bin_probs, results[ImpactKey(assets[0], RiverineInundation)].vulnerability.prob_matrix[:, 2]
        )
        np.testing.assert_almost_equal(prob_impact, 0.10040196672295522)

        # no check with pre-calculated values for others:
        np.testing.assert_allclose(
            results[ImpactKey(assets[0], RiverineInundation)].impact.prob,
            np.array(
                [
                    2.009085e-07,
                    3.001528e-01,
                    1.004020e-01,
                    5.885136e-02,
                    1.760415e-02,
                    1.159864e-02,
                    6.130639e-03,
                    2.729225e-03,
                    1.446537e-03,
                    8.450993e-05,
                ]
            ),
            rtol=2e-6,
        )
