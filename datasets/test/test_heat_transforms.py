import logging, os, sys
import logging.handlers

import dask
import fsspec.implementations.local as local
from hazard.map_builder import MapBuilder
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
    #self.test_dir = tempfile.mkdtemp()
    #dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.getcwd())
    #dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
    #if os.path.exists(dotenv_path):
    #    load_dotenv(dotenv_path=dotenv_path, override=True)

    zarr_utilities.set_credential_env_variables()
    
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


def test_degree_days(): #caplog):
    #caplog.set_level(logging.INFO)

    handler = logging.handlers.WatchedFileHandler(filename=os.path.join(working_dir, "log.txt"),
        mode='a')

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger("hazard")
    root.setLevel(logging.DEBUG) 
    root.addHandler(handler)

    root.debug("Test")

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

    map_builder=MapBuilder(zarr_store, working_directory=working_dir)
    runner = TaskRunner(transform, zarr_store, map_builder=None)
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
