import logging, os

import dask
import fsspec.implementations.local as local
from datasets.hazard.map_builder import MapBuilder
import hazard.utilities.xarray_utilities as xarray_utilities
import hazard.utilities.zarr_utilities as zarr_utilities
import hazard.utilities.map_utilities as map_utilities
from hazard.sources.osc_zarr import OscZarr
from hazard.task_runner import TaskRunner
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.transforms.degree_days import DegreeDays
import s3fs
import xarray as xr
import zarr
import rasterio


working_dir = "/Users/joemoorhouse/Code/data/heat"


def test_simple():
    path = r"/Users/joemoorhouse/Code/data/map_ht2kn3_inunriver_rcp4p5_MIROC-ESM-CHEM_2030_rp01000.tif"
    with rasterio.open(path) as f:
        data = f.data
    assert True



def test_degree_days_s3():
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030
    source = NexGddpCmip6()
    # cut down the transform
    transform = DegreeDays(source=source, window_years=1, gcms=[gcm], scenarios=[scenario], central_years=[year])
    writer = OscZarr(prefix="hazard_test") # test prefix is "hazard_test"; main one "hazard"
    runner = TaskRunner(transform, writer)
    runner.run_batch()
    assert True


def test_degree_days(caplog):
    caplog.set_level(logging.INFO)

    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030
    
    # read from local file system
    fs = local.LocalFileSystem()
    source = NexGddpCmip6(root=os.path.join(working_dir, NexGddpCmip6.bucket), fs=fs)
    
    # cut down the transform
    transform = DegreeDays(source=source, window_years=1, gcms=[gcm], scenarios=[scenario], central_years=[year])
    
    # write to local file system
    store = zarr.DirectoryStore(os.path.join(working_dir, 'hazard', 'hazard.zarr'))
    zarr_store = OscZarr(store=store)

    runner = TaskRunner(transform, zarr_store, map_builder=MapBuilder(zarr_store, working_directory=working_dir))
    runner.run_batch()

    map_utilities 

    assert True


def test_download():
    store = NexGddpCmip6()
    path, _ = store.path("NorESM2-MM", "ssp585", "tasmax", 2030)
    s3 = s3fs.S3FileSystem(anon=True)
    s3.download(path, os.path.join(working_dir, path))
    assert True


def test_load_dataset():    
    fs = local.LocalFileSystem()
    store = NexGddpCmip6(root=os.path.join(working_dir, "nex-gddp-cmip6"), fs=fs)
    with store.open_dataset("NorESM2-MM", "ssp585", "tasmax", 2030) as ds:
        print(ds)
