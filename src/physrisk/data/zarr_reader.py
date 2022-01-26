from affine import Affine
import numpy as np
import os
import s3fs
from typing import Any, Callable, Optional, MutableMapping, Union
import zarr

def get_env(key: str, default: str = None) -> str:
    value = os.environ.get(key)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f'environment variable {key} not present')
    else:
        return value

class ZarrReader():

    # environment variable names:
    __access_key = 'S3_ACCESS_KEY'
    __secret_key = 'S3_SECRET_KEY'
    __S3_bucket = 'S3_BUCKET' # e.g. redhat-osc-physical-landing-647521352890 on staging
    __zarr_path = 'S3_HAZARD_PATH' # e.g. hazard/hazard.zarr on staging
    
    def __init__(self,
                store : MutableMapping = None, 
                set_look_up : Callable[..., str] = None, 
                get_env : Callable[[str, Optional[str]], str] = get_env
                ):
        if store is None:
            # if no store is provided, attempt to connect to an S3 bucket
            if get_env is None:
                raise TypeError("if no store specified, get_env is required to provide credentials")   
            
            s3 = s3fs.S3FileSystem(anon=False,
                key = get_env(ZarrReader.__access_key, None),
                secret = get_env(ZarrReader.__secret_key, None))

            store = s3fs.S3Map(root = os.path.join(get_env(ZarrReader.__S3_bucket, 'redhat-osc-physical-landing-647521352890'),
                get_env(ZarrReader.__zarr_path, 'hazard/hazard.zarr')), s3 = s3, check = False)

        self._root = zarr.open(store, mode = 'r')
        self._set_look_up = set_look_up
        pass

    def get_curves(self, set_id, longitudes, latitudes):
        # assume that calls to this are large, not chatty
        if len(longitudes) != len(latitudes):
            raise ValueError('length of longitudes and latitudes not equal')
        path = self._set_look_up(set_id) if self._set_look_up is not None else set_id
        z = self._root[path] # e.g. inundation/wri/v2/<filename> 
        t = z.attrs['transform_mat3x3'] # type: ignore
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        return_periods = z.attrs['index_values'] # type: ignore
        image_coords = ZarrReader._get_coordinates(longitudes, latitudes, transform)
        iz = np.tile(np.arange(z.shape[0]), image_coords.shape[1]) # type: ignore
        ix = np.repeat(image_coords[1,:], len(return_periods))
        iy = np.repeat(image_coords[0,:], len(return_periods))

        data = z.get_coordinate_selection((iz, ix, iy)) # type: ignore

        return data.reshape([len(longitudes), len(return_periods)]), return_periods

    @staticmethod
    def _get_coordinates(longitudes, latitudes, transform : Affine):
        coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes)))) # type: ignore
        inv_trans = ~transform
        mat = np.array(inv_trans).reshape(3,3)
        frac_image_coords = mat @ coords
        image_coords = np.floor(frac_image_coords).astype(int)
        return image_coords



