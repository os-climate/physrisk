from dataclasses import dataclass
import os
from typing import Dict, List

import s3fs, fsspec
import xarray as xr

from hazard.sources.interfaces import IOpenDatasetForYear

@dataclass
class Cmip6Item:
    gcm: str
    variant_id: str


class NexGddpCmip6(IOpenDatasetForYear):
    """Source class for loading in data from
    NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6)
    https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6
    """

    def __init__(self, fs: fsspec.spec.AbstractFileSystem=None):
        """
        Args:
            fs : fsspec.spec.AbstractFileSystem, optional existing filesystem to use for accessing data.
        """
        # subset of General Circulation Models (GCMs) and Variant IDs for analysis
        self.subset: Dict[str, Dict[str, str]] = {
            "ACCESS-CM2":       {"variantId": "r1i1p1f1"},
            "CMCC-ESM2":        {"variantId": "r1i1p1f1"},
            "CNRM-CM6-1":       {"variantId": "r1i1p1f2"},
            "MPI-ESM1-2-LR":    {"variantId": "r1i1p1f1"},
            "MIROC6":           {"variantId": "r1i1p1f1"},
            "NorESM2-MM":       {"variantId": "r1i1p1f1"},
        }
        self.bucket: str = "nex-gddp-cmip6"  # S3 bucket identifier
        self.fs = s3fs.S3FileSystem(anon=True) if fs is None else fs
        self.quantities = { "tas": {"name": "Daily average temperature"} }


    def path(self, gcm="NorESM2-MM", scenario="ssp585", quantity="tas", year=2030):
        component = self.subset[gcm]
        variantId = component["variantId"]
        filename = f"{quantity}_day_{gcm}_{scenario}_{variantId}_gn_{year}.nc"
        return (os.path.join(self.bucket, f"NEX-GDDP-CMIP6/{gcm}/{scenario}/{variantId}/{quantity}/") + filename, filename)


    def gcms(self) -> List[str]:
        return self.subset.keys()


    def open_dataset(self, gcm: str, scenario: str, quantity: str, year: int) -> xr.Dataset:
        # use "s3://bucket/root" ?
        path, _ = self.s3_path(gcm, scenario, quantity, year)
        with self.fs.open(path, 'rb') as f:
            return xr.open_dataset(f)