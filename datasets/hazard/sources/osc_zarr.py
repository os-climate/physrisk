import os
from typing import MutableMapping, Tuple
from affine import Affine
import numpy as np
import s3fs
import hazard.utilities.xarray_utilities as xarray_utilities
import hazard.utilities.zarr_utilities as zarr_utilities
import xarray as xr
import zarr

class OscZarr:
    default_staging_bucket = "redhat-osc-physical-landing-647521352890"
    
    def __init__(self, bucket: str=default_staging_bucket, prefix: str="hazard", store: MutableMapping=None, s3: s3fs.S3File=None):
        """For reading and writing to OSC Climate Zarr storage. If store is provided this is used, otherwise if S3File is provided, this is used.
        Otherwise, store is created using credentials in environment variables.
        
        Args:
            bucket: Name of S3 bucket.
            root: Path to Zarr Group, i.e. objects are located in S3://{bucket}/{prefix}/hazard.zarr/{rest of key}.
            store: If provided, Zarr will use this store.
            s3: S3File to use if present and if store not provided. 
        """
        if store is None:
            if s3 is None:
                #zarr_utilities.load_dotenv() # to load environment variables
                s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])
            
            group_path = os.path.join(bucket, prefix, "hazard.zarr")
            store = s3fs.S3Map(root=group_path, s3=s3, check=False)
        
        self.root = zarr.group(store=store) 


    def read_numpy(self, path, index=0) -> Tuple[np.ndarray, Affine]:
        """Read index as two dimensional numpy array and affine transform.
        This is intended for small datasets, otherwise recommended to 
        use xarray.open_zarr."""
        z = self.root[path]
        t = z.attrs["transform_mat3x3"]  # type: ignore
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        return z[index, :, :], transform


    def if_exists_remove(self, path):
        if path in self.root:
            self.root.pop(path)


    def write(self, path: str, da: xr.DataArray):
        """Write DataArray to provided relative path."""
        data, transform, crs = xarray_utilities.get_array_components(da)
        z = self._zarr_create(path, da.shape, transform)
        z[0, :, :] = data[:,:]


    def _zarr_create(self, array_path, shape, transform, overwrite=False, return_periods=None):
        """
        Create Zarr array with given shape and affine transform.
        """

        z = self.root.create_dataset(
            array_path,
            shape=(1 if return_periods is None else len(return_periods), shape[0], shape[1]),
            chunks=(1 if return_periods is None else len(return_periods), 1000, 1000),
            dtype="f4",
            overwrite=overwrite,
        )  # array_path interpreted as path within group
        trans_members = [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ]
        mat3x3 = [x * 1.0 for x in trans_members] + [0.0, 0.0, 1.0]
        z.attrs["transform_mat3x3"] = mat3x3
        if return_periods is not None:
            z.attrs["index_values"] = return_periods
            z.attrs["index_name"] = "return period (years)"
        return z

