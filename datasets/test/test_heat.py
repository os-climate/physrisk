import os

import dask
import fsspec.implementations.local as local
import hazard.utilities.xarray_utilities as xarray_utilities
import hazard.utilities.zarr_utilities as zarr_utilities
from hazard.sources.osc_zarr import OscZarr
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.transforms.degree_days import DegreeDays
import s3fs
import xarray as xr
import zarr


working_dir = "/Users/joemoorhouse/Code/data/heat"


def test_simple():
    assert True

def test_degree_days_s3():
    store = NexGddpCmip6()
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030

    # load from local files
    open_dataset = store.open_dataset
    path, filename =store.path(gcm, scenario, "tas", year)

    res = DegreeDays.average_degree_days(open_dataset, gcm, scenario, year, 1)


def test_degree_days():
    store = NexGddpCmip6()
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030

    # load from local files
    def open_dataset_local(gcm, scenario, quantity, year):
        path, filename = store.path(gcm, scenario, quantity, year)
        return xr.open_dataset(os.path.join(working_dir, filename))

    def calc():
        return DegreeDays.average_degree_days(open_dataset_local, gcm, scenario, year, 1)

    delayed = dask.delayed(calc)()
    result = dask.compute(delayed)[0]

    data, transform, crs = xarray_utilities.get_array_components(result)

    zarr_utilities.load_dotenv()

    store = zarr.DirectoryStore(os.path.join(working_dir, 'hazard', 'hazard.zarr'))
    zarr_writer = OscZarr(OscZarr.default_staging_bucket, store=store) 
    z = zarr_writer.zarr_create('test', data.shape, transform, overwrite=True)
    z[0, :, :] = data[:,:]


    assert True


def test_download():
    store = NexGddpCmip6()
    path, filename = store.s3_path()
    s3 = s3fs.S3FileSystem(anon=True)
    s3.download(path, os.path.join(working_dir, filename))
    assert True


def test_load_dataset():    
    import zarr.storage

    fs = local.LocalFileSystem()
    store = zarr.storage.FSStore("s3://bucket/root")
    xr.load_dataset("/Users/joemoorhouse/Code/data/heat/tasmax_day_HadGEM3-GC31-MM_ssp585_r1i1p1f3_gn_2030.nc")
