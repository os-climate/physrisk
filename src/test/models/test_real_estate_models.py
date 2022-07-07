""" Test asset impact calculations."""
import unittest
from test.data.hazard_model_store import TestData, mock_hazard_model_store_inundation

import numpy as np

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.kernel import calculation
from physrisk.kernel.assets import RealEstateAsset
from physrisk.models.real_estate_models import RealEstateCoastalInundationModel, RealEstateRiverineInundationModel


class TestRealEstateModels(unittest.TestCase):
    """Tests RealEstateInundationModel."""

    def test_real_estate_model_details(self):

        curve = np.array([0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163])
        store = mock_hazard_model_store_inundation(TestData.longitudes, TestData.latitudes, curve)
        hazard_model = ZarrHazardModel(source_paths=calculation.get_default_zarr_source_paths(), store=store)

        # location="Europe", type="Buildings/Residential"
        assets = [
            RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
            for lon, lat in zip(TestData.longitudes[0:1], TestData.latitudes[0:1])
        ]

        scenario = "rcp8p5"
        year = 2080

        vulnerability_models = {RealEstateAsset: [RealEstateRiverineInundationModel()]}

        results = calculation.calculate_impacts(
            assets, hazard_model, vulnerability_models, scenario=scenario, year=year
        )

        hazard_bin_edges = results[assets[0]].event.intensity_bin_edges
        hazard_bin_probs = results[assets[0]].event.prob

        # check one:
        # the probability of inundation greater than 0.505m in a year is 1/10.0
        # the probability of inundation greater than 0.333m in a year is 1/5.0
        # therefore the probability of an inundation between 0.333 and 0.505 in a year is 1/5.0 - 1/10.0
        np.testing.assert_almost_equal(hazard_bin_edges[1:3], np.array([0.333, 0.505]))
        np.testing.assert_almost_equal(hazard_bin_probs[1], 0.1)

        # check that intensity bin edges for vilnerability matrix are same as for hazard
        vulnerability_intensity_bin_edges = results[assets[0]].vulnerability.intensity_bins
        np.testing.assert_almost_equal(vulnerability_intensity_bin_edges, hazard_bin_edges)

        # check the impact distribution the matrix is size [len(intensity_bins) - 1, len(impact_bins) - 1]
        cond_probs = results[assets[0]].vulnerability.prob_matrix[1, :]
        # check conditional prob for inundation intensity 0.333..0.505
        mean, std = np.mean(cond_probs), np.std(cond_probs)
        np.testing.assert_almost_equal(cond_probs.sum(), 1)
        np.testing.assert_allclose([mean, std], [0.09090909, 0.08184968], rtol=1e-6)

        # probability that impact occurs between impact bin edge 1 and impact bin edge 2
        prob_impact = np.dot(hazard_bin_probs, results[assets[0]].vulnerability.prob_matrix[:, 1])
        np.testing.assert_almost_equal(prob_impact, 0.19350789547968042)

        # no check with pre-calculated values for others:
        np.testing.assert_allclose(
            results[assets[0]].impact.prob,
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
        hazard_model = ZarrHazardModel(source_paths=calculation.get_default_zarr_source_paths(), store=store)

        # location="Europe", type="Buildings/Residential"
        assets = [
            RealEstateAsset(lat, lon, location="Asia", type="Buildings/Industrial")
            for lon, lat in zip(TestData.coastal_longitudes[0:1], TestData.coastal_latitudes[0:1])
        ]

        scenario = "rcp8p5"
        year = 2080

        vulnerability_models = {RealEstateAsset: [RealEstateCoastalInundationModel()]}

        results = calculation.calculate_impacts(
            assets, hazard_model, vulnerability_models, scenario=scenario, year=year
        )

        np.testing.assert_allclose(
            results[assets[0]].impact.prob,
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
