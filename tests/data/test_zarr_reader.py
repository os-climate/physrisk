import asyncio
import collections
import concurrent.futures
import os
import threading

import numpy as np
import pytest
import zarr
import zarr.storage

from physrisk.data.zarr_reader import ZarrReader, configure_zarr_concurrency
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


class ReadStats:
    def __init__(self):
        self.delay_s = 0.0
        self.in_flight = 0
        self.max_in_flight = 0
        self.cancelled = threading.Event()


class SlowMemoryStore(zarr.storage.MemoryStore):
    """MemoryStore whose chunk reads take a while, recording reads in flight.
    Statistics are shared with the read-only copies zarr makes when opening."""

    def __init__(self, store_dict=None, *, read_only=False, stats=None):
        super().__init__(store_dict, read_only=read_only)
        self.stats = stats if stats is not None else ReadStats()

    def with_read_only(self, read_only=False):
        return type(self)(self._store_dict, read_only=read_only, stats=self.stats)

    async def get(self, key, prototype=None, byte_range=None):
        if key.rsplit("/", 1)[-1].startswith("."):  # metadata is not delayed
            return await super().get(key, prototype, byte_range)
        stats = self.stats
        stats.in_flight += 1
        stats.max_in_flight = max(stats.max_in_flight, stats.in_flight)
        try:
            await asyncio.sleep(stats.delay_s)
        except asyncio.CancelledError:
            stats.cancelled.set()
            raise
        finally:
            stats.in_flight -= 1
        return await super().get(key, prototype, byte_range)


def slow_store_with_arrays(n_arrays: int) -> SlowMemoryStore:
    store = SlowMemoryStore()
    root = zarr.open_group(store=store, mode="w", zarr_format=2)
    for i in range(n_arrays):
        z = create_test_array(root, f"array_{i}", shape=(1, 10, 10), chunks=(1, 10, 10))
        z.attrs["transform_mat3x3"] = [1.0, 0.0, 0.0, 0.0, -1.0, 10.0, 0.0, 0.0, 1.0]
        z[0, :, :] = float(i)
    return store


def test_cancelling_an_async_read_cancels_it_on_the_zarr_loop():
    store = slow_store_with_arrays(1)
    store.stats.delay_s = 10.0
    reader = ZarrReader(store)

    async def cancel_read():
        task = asyncio.ensure_future(reader.get_curves_async("array_0", [5.0], [5.0]))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_read())
    assert store.stats.cancelled.wait(timeout=5), (
        "read on the zarr loop was not cancelled"
    )


def test_reads_from_concurrent_callers_run_together_on_shared_loop():
    # e.g. parallel API requests, each running get_hazard_data in its own thread
    store = slow_store_with_arrays(4)
    store.stats.delay_s = 0.2
    reader = ZarrReader(store)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda i: asyncio.run(
                    reader.get_curves_async(f"array_{i}", [5.0], [5.0])
                ),
                range(4),
            )
        )
    for i, (values, _, _, _) in enumerate(results):
        assert values[0, 0] == float(i)
    # reads of single-chunk arrays are not held back by one another
    assert store.stats.max_in_flight == 4


def test_sync_read_from_inside_an_event_loop():
    store = slow_store_with_arrays(1)
    reader = ZarrReader(store)

    async def call_sync():
        return reader.get_curves("array_0", [5.0], [5.0])

    values, _, _, _ = asyncio.run(call_sync())
    assert values[0, 0] == 0.0


def test_read_and_array_info():
    store = slow_store_with_arrays(1)
    reader = ZarrReader(store)
    info = reader.array_info("array_0")
    assert info.shape == (1, 10, 10)
    assert "transform_mat3x3" in info.attrs
    np.testing.assert_array_equal(
        reader.read("array_0", (0, slice(0, 2), slice(0, 2))), np.zeros((2, 2))
    )


class CountingMemoryStore(zarr.storage.MemoryStore):
    """MemoryStore counting reads per key, shared with zarr's read-only copies."""

    def __init__(self, store_dict=None, *, read_only=False, gets=None):
        super().__init__(store_dict, read_only=read_only)
        self.gets = gets if gets is not None else collections.Counter()

    def with_read_only(self, read_only=False):
        return type(self)(self._store_dict, read_only=read_only, gets=self.gets)

    async def get(self, key, prototype=None, byte_range=None):
        self.gets[key] += 1
        return await super().get(key, prototype, byte_range)


@pytest.mark.parametrize(
    "cache_arrays, expected_metadata_reads", [(True, 1), (False, 3)]
)
def test_array_metadata_read_once_when_cached(cache_arrays, expected_metadata_reads):
    store = CountingMemoryStore()
    root = zarr.open_group(store=store, mode="w", zarr_format=2)
    z = create_test_array(root, "array_0", shape=(1, 10, 10), chunks=(1, 10, 10))
    z.attrs["transform_mat3x3"] = [1.0, 0.0, 0.0, 0.0, -1.0, 10.0, 0.0, 0.0, 1.0]
    z[0, :, :] = 1.0
    reader = ZarrReader(store, cache_arrays=cache_arrays)
    store.gets.clear()
    for _ in range(3):
        values, _, _, _ = reader.get_curves("array_0", [5.0], [5.0])
        assert values[0, 0] == 1.0
    assert store.gets["array_0/.zarray"] == expected_metadata_reads
    # chunks are always read from the store: only metadata is cached
    assert store.gets["array_0/0.0.0"] == 3


def test_missing_array_is_not_cached():
    store, root = zarr_memory_store()
    reader = ZarrReader(store)
    with pytest.raises(KeyError):
        reader.array_info("not_there")
    create_test_array(root, "not_there", shape=(1, 2, 2))
    assert reader.array_info("not_there").shape == (1, 2, 2)


def test_default_concurrency_sized_from_available_cpus():
    configure_zarr_concurrency()
    assert zarr.config.get("async.concurrency") == min(8, 2 * available_cpu_count())
    configure_zarr_concurrency(async_concurrency=3)
    assert zarr.config.get("async.concurrency") == 3
    configure_zarr_concurrency()
