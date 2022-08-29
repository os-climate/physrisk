import unittest
from test.data.hazard_model_store import (
    mock_hazard_model_store_inundation,
)

import numpy as np
import numpy.testing
import scipy.interpolate
import zarr
from dotenv import load_dotenv

from physrisk.data.hazard_data_provider import get_source_path_wri_riverine_inundation
from physrisk.data.inventory import Inventory
from physrisk.data.zarr_reader import ZarrReader


class TestEventRetrieval(unittest.TestCase):

    def test_hazard_data_availability_summary(self):
        summary = Inventory().get_models_summary()
        self.assertEqual(summary["RiverineInundation"].years, [1980, 2030, 2050, 2080])

    def test_zarr_bilinear(self):
        # create suitable asymmetric data set and compare with scipy

        xt, yt = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
        data0 = np.exp(-(xt**2 / 25.0 + yt**2 / 16.0))
        data1 = np.exp(-(xt**2 / 36.0 + yt**2 / 25.0))

        data = np.stack([data0, data1], axis=0)

        # note that zarr array has index [z, y, x], e.g. 9, 21600, 43200 or [index, lat, lon]
        y = np.array([1.4, 2.8, 3.4])  # row indices
        x = np.array([3.2, 6.7, 7.9])  # column indices
        image_coords = np.stack([x, y])
        data_zarr = zarr.array(data)
        candidate_lin = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0, 1]), interpolation="linear"
        )
        candidate_max = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0, 1]), interpolation="max"
        )
        candidate_min = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0, 1]), interpolation="min"
        )

        image_coords_surr = np.stack(
            [
                np.concatenate([np.floor(x), np.floor(x) + 1, np.floor(x), np.floor(x) + 1]),
                np.concatenate([np.floor(y), np.floor(y), np.floor(y) + 1, np.floor(y) + 1]),
            ]
        )
        values_surr = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords_surr, np.array([0, 1]), interpolation="linear"
        ).reshape((4, 3, 2))
        interp_scipy0 = scipy.interpolate.interp2d(np.linspace(0, 9, 10), np.linspace(0, 9, 10), data0, kind="linear")
        interp_scipy1 = scipy.interpolate.interp2d(np.linspace(0, 9, 10), np.linspace(0, 9, 10), data1, kind="linear")
        expected0_lin = interp_scipy0(x, y).diagonal().reshape(len(y))
        expected1_lin = interp_scipy1(x, y).diagonal().reshape(len(y))
        expected0_max = np.max(values_surr[:, :, 0], axis=0)
        expected1_max = np.max(values_surr[:, :, 1], axis=0)
        expected0_min = np.min(values_surr[:, :, 0], axis=0)
        expected1_min = np.min(values_surr[:, :, 1], axis=0)

        numpy.testing.assert_allclose(candidate_lin[:, 0], expected0_lin, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_lin[:, 1], expected1_lin, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_max[:, 0], expected0_max, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_max[:, 1], expected1_max, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_min[:, 0], expected0_min, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_min[:, 1], expected1_min, rtol=1e-6)
        # array([0.43854423, 0.62290176, 0.50660137])
        # array([0.58346331, 0.72702827, 0.62629474])
        # array([0.60981554, 0.75222318, 0.72041193])
        # array([0.72817392, 0.82838968, 0.79717553])
        # array([0.34773063, 0.45343876, 0.45907147])
        # array([0.50555484, 0.59275348, 0.58789487])

    def test_zarr_geomax(self):
        lons_ = np.array([3.92783])
        lats_ = np.array([50.882394])
        curve = np.array(
            [0.00, 0.06997928, 0.2679602, 0.51508933, 0.69842442, 0.88040525, 1.11911115, 1.29562478, 1.47200677]
        )
        set_id = r"inundation/wri/v2\\inunriver_rcp8p5_MIROC-ESM-CHEM_2080"
        interpolation = "linear"
        delta_km = 0.100
        n_grid = 10
        store_ = mock_hazard_model_store_inundation(lons_, lats_, curve)
        zarrreader_ = ZarrReader(store_)
        curves_max_candidate, _ = zarrreader_.get_max_curves(
            set_id, lons_, lats_, interpolation=interpolation, delta_km=delta_km, n_grid=n_grid
        )
        curves_max_expected = np.array(
            [[0.00, 0.04917953, 0.1883151, 0.3619907, 0.49083358, 0.61872474, 0.78648075, 0.91052965, 1.03448614]]
        )
        numpy.testing.assert_allclose(curves_max_candidate, curves_max_expected, rtol=1e-6)

