from typing import Tuple

from affine import Affine
import numpy as np
import rioxarray
import rasterio
import xarray as xr


def get_array_components(array: xr.DataArray) -> Tuple[np.ndarray, Affine, any]:
    renamed = array.rename({ "lat": "latitude", "lon": "longitude" })
    data = renamed.data
    transform = renamed.rio.transform(recalc = True)
    crs = renamed.rio.crs
    
    if crs is None:
        # assumed default
        crs = rasterio.CRS.from_epsg(4326)

    return (data, transform, crs)


def write_array(array: xr.DataArray, path: str):
    array.rio.to_raster(raster_path=path, driver="COG")