import asyncio
import os

import numpy as np

from physrisk.data.zarr_reader import ZarrReader
from physrisk.utils.concurrency import available_cpu_count
from tests.data.test_hazard_model_store import (
    TestData,
    create_test_array,
    get_hazard_path_wri_riverine_inundation,
    mock_hazard_model_store_inundation,
    zarr_memory_store,
)


def test_ls_lists_group_members():
    # as used to find the zoom levels of a map pyramid
    store, root = zarr_memory_store()
    for zoom in [1, 2, 3]:
        create_test_array(root, f"flood/map/{zoom}", shape=(1, 2, 2))
    reader = ZarrReader(store)
    assert sorted(reader.ls("flood/map/")) == [
        "flood/map/1",
        "flood/map/2",
        "flood/map/3",
    ]


def test_async_and_sync_reads_agree():
    curve = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    store = mock_hazard_model_store_inundation(
        TestData.longitudes, TestData.latitudes, curve
    )
    reader = ZarrReader(store)
    path = get_hazard_path_wri_riverine_inundation(
        model="MIROC-ESM-CHEM", scenario="rcp8p5", year=2080
    )
    lons, lats = TestData.longitudes[0:3], TestData.latitudes[0:3]
    sync_values, _, _, _ = reader.get_curves(path, lons, lats)
    async_values, _, _, _ = asyncio.run(reader.get_curves_async(path, lons, lats))
    np.testing.assert_array_equal(sync_values, async_values)
    np.testing.assert_allclose(sync_values[0], curve, rtol=1e-6)


def test_available_cpu_count():
    assert 1 <= available_cpu_count() <= (os.cpu_count() or 1)
