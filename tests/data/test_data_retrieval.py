from collections import defaultdict
import logging
import os
import time
from typing import List, Optional, Sequence, Type
import unittest

# import fsspec.implementations.local as local  # type: ignore
import numpy as np
import numpy.testing
import scipy.interpolate
import zarr
from fsspec.implementations.memory import MemoryFileSystem
from shapely import Polygon

from physrisk.api.v1.hazard_data import (
    HazardAvailabilityRequest,
    HazardResource,
    Scenario,
)
from physrisk.data.hazard_data_provider import (
    HazardDataHint,
    HazardDataProvider,
    ResourcePaths,
    ScenarioPaths,
    ScenarioYear,
    SourcePaths,
)
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazard_model import HazardDataFailedResponse, HazardDataRequest
from physrisk.kernel.hazards import Hazard, RiverineInundation
from physrisk.requests import _get_hazard_data_availability

# from pathlib import PurePosixPath
from ..test_base import TestWithCredentials
from .test_hazard_model_store import (
    ZarrStoreMocker,
    mock_hazard_model_store_inundation,
)

logger = logging.getLogger(__name__)


class TestEventRetrieval(TestWithCredentials):
    @unittest.skip("S3 access needed")
    def test_inventory_change(self):
        # check validation passes calling in service-like way
        embedded = EmbeddedInventory()
        resources1 = embedded.to_resources()
        inventory = Inventory(resources1).json_ordered()
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "inventory.json"),
            "w",
        ) as f:
            f.write(inventory)

    def test_hazard_data_availability_summary(self):
        # check validation passes calling in service-like way
        inventory = EmbeddedInventory()
        response = _get_hazard_data_availability(
            HazardAvailabilityRequest(sources=["embedded"]),
            inventory,
            inventory.colormaps(),
        )  # , "hazard_test"])
        assert len(response.models) > 0  # rely on Pydantic validation for test

    def test_set_get_inventory(self):
        fs = MemoryFileSystem()
        reader = InventoryReader(fs=fs)
        reader.append("hazard_test", [self._test_hazard_model()])
        assert reader.read("hazard_test")[0].indicator_id == "test_indicator_id"

    @unittest.skip("S3 access needed")
    def test_set_get_inventory_s3(self):
        reader = InventoryReader()
        reader.append("hazard_test", [self._test_hazard_model()])
        assert reader.read("hazard_test")[0].id == "test_indicator_id"

    def _test_hazard_model(self):
        return HazardResource(
            hazard_type="TestHazardType",
            indicator_id="test_indicator_id",
            indicator_model_gcm="test_gcm",
            path="test_array_path",
            display_name="Test hazard indicator",
            description="Description of test hazard indicator",
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
                np.concatenate(
                    [np.floor(x), np.floor(x) + 1, np.floor(x), np.floor(x) + 1]
                ),
                np.concatenate(
                    [np.floor(y), np.floor(y), np.floor(y) + 1, np.floor(y) + 1]
                ),
            ]
        )
        values_surr = ZarrReader._linear_interp_frac_coordinates(
            data_zarr, image_coords_surr, np.array([0, 1]), interpolation="linear"
        ).reshape((4, 3, 2))
        interp_scipy0 = scipy.interpolate.RectBivariateSpline(
            np.linspace(0, 9, 10), np.linspace(0, 9, 10), data0.T, kx=1, ky=1
        )
        interp_scipy1 = scipy.interpolate.RectBivariateSpline(
            np.linspace(0, 9, 10), np.linspace(0, 9, 10), data1.T, kx=1, ky=1
        )
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

    def test_zarr_geomax_on_grid(self):
        lons_ = np.array([3.92783])
        lats_ = np.array([50.882394])
        curve = np.array(
            [
                0.00,
                0.06997928,
                0.2679602,
                0.51508933,
                0.69842442,
                0.88040525,
                1.11911115,
                1.29562478,
                1.47200677,
            ]
        )
        set_id = r"inundation/wri/v2\\inunriver_rcp8p5_MIROC-ESM-CHEM_2080"
        interpolation = "linear"
        delta_km = 0.100
        n_grid = 11
        store_ = mock_hazard_model_store_inundation(lons_, lats_, curve)
        zarrreader_ = ZarrReader(store_)

        lons_ = np.array([3.92916667, 3.925] + list(lons_))
        lats_ = np.array([50.87916667, 50.88333333] + list(lats_))
        curves_max_candidate, _ = zarrreader_.get_max_curves_on_grid(
            set_id,
            lons_,
            lats_,
            interpolation=interpolation,
            delta_km=delta_km,
            n_grid=n_grid,
        )

        curves_max_expected = np.array(
            [
                curve,
                [
                    0.0,
                    0.02272942,
                    0.08703404,
                    0.16730212,
                    0.22684974,
                    0.28595751,
                    0.3634897,
                    0.42082168,
                    0.47811095,
                ],
                [
                    0.0,
                    0.0432026,
                    0.16542863,
                    0.31799695,
                    0.43118118,
                    0.54352937,
                    0.69089751,
                    0.7998704,
                    0.90876211,
                ],
            ]
        )
        numpy.testing.assert_allclose(
            curves_max_candidate, curves_max_expected, rtol=1e-6
        )

    def test_zarr_geomax(self):
        longitudes = np.array([3.926])
        latitudes = np.array([50.878])
        curve = np.array(
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        )
        set_id = r"inundation/wri/v2\\inunriver_rcp8p5_MIROC-ESM-CHEM_2080"
        delta_deg = 0.1
        shapes = [
            Polygon(
                (
                    (x - 0.5 * delta_deg, y - 0.5 * delta_deg),
                    (x - 0.5 * delta_deg, y + 0.5 * delta_deg),
                    (x + 0.5 * delta_deg, y + 0.5 * delta_deg),
                    (x + 0.5 * delta_deg, y - 0.5 * delta_deg),
                )
            )
            for x, y in zip(longitudes, latitudes)
        ]
        store = mock_hazard_model_store_inundation(longitudes, latitudes, curve)
        zarr_reader = ZarrReader(store)
        curves_max_expected = np.array([[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]])

        curves_max_candidate, _, _ = zarr_reader.get_max_curves(
            set_id, shapes, interpolation="floor"
        )
        numpy.testing.assert_allclose(
            curves_max_candidate, curves_max_expected, rtol=1e-6
        )

        curves_max_candidate, _, _ = zarr_reader.get_max_curves(
            set_id, shapes, interpolation="linear"
        )
        numpy.testing.assert_allclose(
            curves_max_candidate, curves_max_expected / 4, rtol=1e-6
        )

    def test_reproject(self):
        """Test adding data in a non-ESPG-4326 coordinate reference system. Check that the round tip yields
        the correct results."""
        mocker = ZarrStoreMocker()
        lons = [1.1, -0.31]
        lats = [47.0, 52.0]
        mocker._add_curves(
            "test",
            lons,
            lats,
            "epsg:3035",
            [3, 39420, 38371],
            [100.0, 0.0, 2648100.0, 0.0, -100.0, 5404500],
            [10.0, 100.0, 1000.0],
            [1.0, 2.0, 3.0],
        )

    def test_years_interpolation(self):
        weights = HazardDataProvider._weights(
            "ssp585", [2050, 2060, 2080], [2040, 2050, 2065, 2090], 2025
        )
        assert (
            weights[ScenarioYear("ssp585", 2040)].weights[0][0].scenario == "historical"
        )
        assert weights[ScenarioYear("ssp585", 2040)].weights[0][1] == (
            2050.0 - 2040.0
        ) / (2050.0 - 2025.0)
        #
        assert weights[ScenarioYear("ssp585", 2050)].weights[0][0].year == 2050
        assert weights[ScenarioYear("ssp585", 2050)].weights[0][1] == 1.0
        #
        assert weights[ScenarioYear("ssp585", 2090)].weights[0][0].year == 2060
        assert weights[ScenarioYear("ssp585", 2090)].weights[0][1] == -(
            2090.0 - 2080.0
        ) / (2080.0 - 2060.0)
        # v_e = v_2 + (y_e - y_2) * (v_2 - v_1) / (y_2 - y_1)
        # w1 = - (y_e - y_2) / (y_2 - y_1)
        # w2 = 1 + (y_e - y_2) / (y_2 - y_1)
        weights = HazardDataProvider._weights("ssp585", [2050], [2040, 2090], 2025)
        assert weights[ScenarioYear("ssp585", 2090)].weights[0][0].year == -1

    def test_cascading_sources_and_interpolation(self):
        """This is a performance test to verify that it is sufficiently efficient to have
        unsorted HazardDataRequest objects as an input, which can then be sorted and processed.
        A few seconds per 5,000,000 requests is deemed acceptable, single-threaded.
        """
        n_samples = 5000000
        logger.info(f"Sampling asset locations for {n_samples} assets")
        latitudes = np.random.uniform(-80, 80, n_samples)
        longitudes = np.random.uniform(-180, 180, n_samples)
        logger.info("Creating request objects")
        requests = [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=float(lon),
                latitude=float(lat),
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2050,
            )
            for lat, lon in zip(latitudes, longitudes)
        ]
        logger.info(f"{len(requests)} request objects created.")
        logger.info("Grouping request objects (just for comparison)")
        groups = defaultdict(list)
        for req in requests:
            groups[req.group_key()].append(req)
        logger.info("Finding unique years")
        logger.info("Timing sorting step:")
        start = time.time()
        logger.info(
            "First latitudes and longitudes are identified, to process all scenarios and years together"
        )
        lookup = {}
        for req in requests:
            lookup.setdefault((req.latitude, req.longitude, req.buffer), len(lookup))

        times = {2050: []}
        logger.info(
            "These are then processed for all times and scenarios in index order"
        )
        logger.info("Finally, the index is looked up for each request again")
        for req in requests:
            index = lookup[(req.latitude, req.longitude, req.buffer)]
            value = times[req.year]

        elapsed = time.time() - start
        logger.info(f"(Additional) time for dealing with unsorted requests: {elapsed}s")
        assert index is not None
        assert value is not None


class SourcePathsTest(SourcePaths):
    def __init__(self, cascade: bool = True):
        self.cascade = cascade

    def hazard_types(self):
        return [RiverineInundation]

    def resource_paths(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        scenarios: Sequence[str],
        hint: Optional[HazardDataHint] = None,
    ) -> List[ResourcePaths]:
        # try Europe-specific first and then the whole-world
        result = [
            ResourcePaths(
                resource_path="",
                scenarios={
                    "ssp585": ScenarioPaths(
                        years=[2030, 2050, 2080],
                        path=lambda f: "test_set_europe_only",
                    ),
                    "historical": ScenarioPaths(
                        years=[-1], path=lambda f: "test_set_europe_only"
                    ),
                },
            )
        ]
        if self.cascade:
            result.append(
                ResourcePaths(
                    resource_path="",
                    scenarios={
                        "ssp585": ScenarioPaths(
                            years=[2030, 2050, 2080], path=lambda f: "test_set_world"
                        ),
                        "historical": ScenarioPaths(
                            years=[-1], path=lambda f: "test_set_world"
                        ),
                    },
                )
            )
        return result


def test_cascade():
    mocker = ZarrStoreMocker()
    # Europe, Europe, not Europe, not Europe, Europe
    lons = [1.1, -0.31, 32.5, -84.0, 1.15]
    lats = [47.0, 52.0, 16.0, 38.0, 47.1]
    mocker._add_curves(
        "test_set_europe_only",
        lons[0:2] + [lons[4]],
        lats[0:2] + [lats[4]],
        "epsg:3035",
        [3, 39420, 38371],
        [100.0, 0.0, 2648100.0, 0.0, -100.0, 5404500],
        [10.0, 100.0, 1000.0],
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    )
    mocker.add_curves_global(
        "test_set_world",
        lons,
        lats,
        [10.0, 20.0, 100.0, 1000.0],
        [4.0, 4.5, 5.0, 6.0],
    )
    requests = (
        [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=float(lon),
                latitude=float(lat),
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2050,
            )
            for lat, lon in zip(lats, lons)
        ]
        + [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=float(lon),
                latitude=float(lat),
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2080,
            )
            for lat, lon in zip(lats, lons)
        ]
        + [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=float(lon),
                latitude=float(lat),
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2070,
            )
            for lat, lon in zip(lats, lons)
        ]
        + [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=float(lon),
                latitude=float(lat),
                indicator_id="flood_depth",
                scenario="historical",
                year=-1,
            )
            for lat, lon in zip(lats, lons)
        ]
    )

    source_paths = SourcePathsTest(cascade=True)
    hazard_model = ZarrHazardModel(source_paths=source_paths, store=mocker.store)
    response = hazard_model.get_hazard_data(requests)
    np.testing.assert_almost_equal(
        response[requests[0]].intensities,
        [1.0, 2.0, 3.0],  # Europe
    )
    np.testing.assert_almost_equal(
        response[requests[2]].intensities,
        [4.0, 4.5, 5.0, 6.0],  # not Europe
    )
    np.testing.assert_almost_equal(
        response[requests[9]].intensities,
        [1.0, 2.0, 3.0],  #
    )
    assert isinstance(response[requests[10]], HazardDataFailedResponse)
    np.testing.assert_almost_equal(response[requests[15]].intensities, [1.0, 2.0, 3.0])


def test_error_cases():
    mocker = ZarrStoreMocker()
    # Europe, Europe, not Europe, not Europe, Europe
    lons = [1.1, -0.31, 32.5, -84.0, 1.15]
    lats = [47.0, 52.0, 16.0, 38.0, 47.1]
    mocker._add_curves(
        "test_set_europe_only",
        lons[0:2] + [lons[4]],
        lats[0:2] + [lats[4]],
        "epsg:3035",
        [3, 39420, 38371],
        [100.0, 0.0, 2648100.0, 0.0, -100.0, 5404500],
        [10.0, 100.0, 1000.0],
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [float("nan"), float("nan"), float("nan")],
            ]
        ),
    )
    mocker.add_curves_global(
        "test_set_world",
        lons,
        lats,
        [10.0, 20.0, 100.0, 1000.0],
        [4.0, 4.5, 5.0, 6.0],
    )
    requests = [
        HazardDataRequest(
            hazard_type=RiverineInundation,
            longitude=float(lon),
            latitude=float(lat),
            indicator_id="flood_depth",
            scenario="ssp585",
            year=2050,
        )
        for lat, lon in zip(lats, lons)
    ] + [
        HazardDataRequest(
            hazard_type=RiverineInundation,
            longitude=float(lon),
            latitude=float(lat),
            indicator_id="flood_depth",
            scenario="ssp245",
            year=2050,
        )
        for lat, lon in zip(lats[0:1], lons[0:1])
    ]

    source_paths = SourcePathsTest(cascade=False)
    hazard_model = ZarrHazardModel(source_paths=source_paths, store=mocker.store)
    response = hazard_model.get_hazard_data(requests)
    np.testing.assert_almost_equal(
        response[requests[0]].intensities,
        [1.0, 2.0, 3.0],  # Europe
    )
    # the out-of bounds data should come through as failed responses as no cascading
    assert isinstance(response[requests[2]], HazardDataFailedResponse)
    # for acute events, the not-a-number should be interpreted as zero intensity, as long as in-bounds
    # (in line with the convention for many acute sets).
    np.testing.assert_almost_equal(
        response[requests[4]].intensities,
        [0.0],  # Europe
    )
    # non-matching scenario
    assert isinstance(response[requests[5]], HazardDataFailedResponse)


class SourcePathsYearsInterpolationTest(SourcePaths):
    def hazard_types(self):
        return [RiverineInundation]

    def resource_paths(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        scenarios: Sequence[str],
        hint: Optional[HazardDataHint] = None,
    ) -> List[ResourcePaths]:
        result = [
            ResourcePaths(
                resource_path="",
                scenarios={
                    "ssp585": ScenarioPaths(
                        years=[2030, 2050, 2080],
                        path=lambda f: f"test_set_europe_only_{f}",
                    ),
                    "historical": ScenarioPaths(
                        years=[-1], path=lambda f: "test_set_europe_only_historical"
                    ),
                },
            )
        ]
        return result


def test_end_to_end_interpolation_years():
    mocker = ZarrStoreMocker()
    # Europe, Europe, not Europe, not Europe, Europe
    lons = [1.1, -0.31, 32.5, -84.0, 1.15]
    lats = [47.0, 52.0, 16.0, 38.0, 47.1]
    years_data = [
        [1.0, 1.5, 2.0],  # historical
        [1.1, 1.6, 2.1],  # 2030
        [1.2, 1.7, 2.2],  # 2040
        [1.3, 1.8, 2.3],
    ]  # 2050

    filenames = ["test_set_europe_only_historical"] + [
        f"test_set_europe_only_{year}" for year in [2030, 2050, 2080]
    ]
    for i, filename in enumerate(filenames):
        mocker._add_curves(
            filename,
            lons[0:2],
            lats[0:2],
            "epsg:3035",
            [3, 39420, 38371],
            [100.0, 0.0, 2648100.0, 0.0, -100.0, 5404500],
            [10.0, 100.0, 1000.0],
            np.array(
                [
                    years_data[i],
                    years_data[i],
                ]
            ),
        )
    requests = (
        [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=float(lon),
                latitude=float(lat),
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2027,
            )
            for lat, lon in zip(lats[0:1], lons[0:1])
        ]
        + [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=float(lon),
                latitude=float(lat),
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2040,
            )
            for lat, lon in zip(lats[0:1], lons[0:1])
        ]
        + [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=float(lon),
                latitude=float(lat),
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2090,
            )
            for lat, lon in zip(lats[0:1], lons[:1])
        ]
    )

    source_paths = SourcePathsYearsInterpolationTest()
    hazard_model = ZarrHazardModel(
        source_paths=source_paths, store=mocker.store, interpolate_years=True
    )
    response = hazard_model.get_hazard_data(requests)
    expected_2027 = (3.0 / 5.0) * np.array([1.0, 1.5, 2.0]) + (2.0 / 5.0) * np.array(
        [1.1, 1.6, 2.1]
    )
    expected_2040 = 0.5 * np.array([1.1, 1.6, 2.1]) + 0.5 * np.array([1.2, 1.7, 2.2])
    expected_2090 = np.array([1.3, 1.8, 2.3]) + 10.0 / 30.0 * (
        np.array([1.3, 1.8, 2.3]) - np.array([1.2, 1.7, 2.2])
    )
    np.testing.assert_almost_equal(response[requests[0]].intensities, expected_2027)
    np.testing.assert_almost_equal(response[requests[1]].intensities, expected_2040)
    np.testing.assert_almost_equal(response[requests[2]].intensities, expected_2090)


def test_buffer_integration():
    mocker = ZarrStoreMocker()
    # Europe, Europe, not Europe, not Europe, Europe
    lons = [1.1, -0.31, 32.5, -84.0, 1.15]
    lats = [47.0, 52.0, 16.0, 38.0, 47.1]
    mocker._add_curves(
        "test_set_europe_only",
        lons[0:2] + [lons[4]],
        lats[0:2] + [lats[4]],
        "epsg:3035",
        [3, 39420, 38371],
        [100.0, 0.0, 2648100.0, 0.0, -100.0, 5404500],
        [10.0, 100.0, 1000.0],
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [float("nan"), float("nan"), float("nan")],
            ]
        ),
    )
    mocker.add_curves_global(
        "test_set_world",
        lons,
        lats,
        [10.0, 20.0, 100.0, 1000.0],
        [4.0, 4.5, 5.0, 6.0],
    )
    requests = [
        HazardDataRequest(
            hazard_type=RiverineInundation,
            longitude=float(lon),
            latitude=float(lat),
            indicator_id="flood_depth",
            scenario="ssp585",
            year=2050,
            buffer=10,
        )
        for lat, lon in zip(lats, lons)
    ]

    source_paths = SourcePathsTest(cascade=False)
    hazard_model = ZarrHazardModel(source_paths=source_paths, store=mocker.store)
    response = hazard_model.get_hazard_data(requests)
    np.testing.assert_almost_equal(
        response[requests[0]].intensities,
        [1.0, 2.0, 3.0],  # Europe
    )
