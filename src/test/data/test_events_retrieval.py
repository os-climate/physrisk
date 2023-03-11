import unittest
from test.base_test import TestWithCredentials
from test.data.hazard_model_store import mock_hazard_model_store_inundation

import numpy as np
import numpy.testing
import scipy.interpolate
import zarr
from fsspec.implementations.memory import MemoryFileSystem

from physrisk.api.v1.hazard_data import HazardAvailabilityRequest, HazardResource, Scenario
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.zarr_reader import ZarrReader
from physrisk.requests import _get_hazard_data_availability


class TestEventRetrieval(TestWithCredentials):
    def test_hazard_data_availability_summary(self):
        # check validation passes calling in service-like way
        response = _get_hazard_data_availability(HazardAvailabilityRequest(sources=["embedded"]))  # , "hazard_test"])
        assert len(response.models) > 0  # rely on Pydantic validation for test

    def test_set_get_inventory(self):
        fs = MemoryFileSystem()
        reader = InventoryReader(fs=fs)
        reader.append("hazard_test", [self._test_hazard_model()])
        assert reader.read("hazard_test")[0].id == "test_model_id"

    @unittest.skip("S3 access needed")
    def test_set_get_inventory_s3(self):
        reader = InventoryReader()
        reader.append("hazard_test", [self._test_hazard_model()])
        assert reader.read("hazard_test")[0].id == "test_model_id"

    def _test_hazard_model(self):
        return HazardResource(
            type="TestHazardType",
            path="test_sub_type",
            id="test_model_id",
            array_name="test_array_name",
            display_name="Test model",
            description="Description of test model",
            scenarios=[Scenario(id="historical", years=[2010])],
            units="K",
        )

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

    def test_zarr_bilinear_with_bad_data(self):
        # create suitable asymmetric data set and compare with scipy

        xt, yt = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
        data = np.array([[[1.0, -9999.0], [2.0, 0.0]]])

        # note that zarr array has index [z, y, x], e.g. 9, 21600, 43200 or [index, lat, lon]
        y = np.array([0.4, 0.5, 0.8])  # row indices
        x = np.array([0.1, 0.6, 0.7])  # column indices
        image_coords = np.stack([x, y])
        data_zarr = zarr.array(data)

        candidate_lin = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0]), interpolation="linear"
        ).flatten()
        candidate_max = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0]), interpolation="max"
        ).flatten()
        candidate_min = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords, np.array([0]), interpolation="min"
        ).flatten()

        expected_lin = np.array([1.34042553, 0.85714286, 0.62790698])
        expected_max = np.array([2.0, 2.0, 2.0])
        expected_min = np.array([0.0, 0.0, 0.0])

        numpy.testing.assert_allclose(candidate_lin, expected_lin, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_max, expected_max, rtol=1e-6)
        numpy.testing.assert_allclose(candidate_min, expected_min, rtol=1e-6)

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
