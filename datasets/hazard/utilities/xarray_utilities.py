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


def enforce_conventions(da: xr.DataArray) -> xr.DataArray:
    """By convention, underlying data should have decreasing latitude
    and should be centred on longitude 0."""
    if da.lat[-1] > da.lat[0]:
        da = da.reindex(lat=da.lat[::-1])

    if np.any(da.lon > 180):
        lon = da.lon
        lon = np.where(lon > 180, lon - 360, lon) 
        da["lon"] = lon
        da = da.roll(lon=-len(da.lon) // 2, roll_coords=True)
    return da


def write_array(array: xr.DataArray, path: str):
    array.rio.to_raster(raster_path=path, driver="COG")