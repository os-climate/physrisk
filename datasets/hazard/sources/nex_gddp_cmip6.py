from contextlib import contextmanager
from dataclasses import dataclass
import logging, os
from typing import Dict, List

import s3fs, fsspec
import xarray as xr

from hazard.protocols import POpenDatasetForYear

logger = logging.getLogger(__name__)

@dataclass
class Cmip6Item:
    gcm: str
    variant_id: str


class NexGddpCmip6(POpenDatasetForYear):
    """Source class for loading in data from
    NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6)
    https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6
    """
    bucket: str = "nex-gddp-cmip6"

    def __init__(self, fs: fsspec.spec.AbstractFileSystem=None, root: str=None):
        """
        Args:
            fs: Optional existing filesystem to use for accessing data.
            root: Optional root path (bucket name in case of S3).
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
        self.fs = s3fs.S3FileSystem(anon=True) if fs is None else fs
        self.root = NexGddpCmip6.bucket if root is None else root
        self.quantities = { "tas": {"name": "Daily average temperature"} }


    def path(self, gcm="NorESM2-MM", scenario="ssp585", quantity="tas", year=2030):
        component = self.subset[gcm]
        variantId = component["variantId"]
        filename = f"{quantity}_day_{gcm}_{scenario}_{variantId}_gn_{year}.nc"
        return (os.path.join(self.root, f"NEX-GDDP-CMIP6/{gcm}/{scenario}/{variantId}/{quantity}/") + filename, filename)


    def gcms(self) -> List[str]:
        return self.subset.keys()


    @contextmanager
    def open_dataset(self, gcm: str, scenario: str, quantity: str, year: int, chunks=None) -> xr.Dataset:
        # use "s3://bucket/root" ?
        path, _ = self.path(gcm, scenario, quantity, year)
        logger.info(f"Opening DataSet, relative path={path}, chunks={chunks}")
        ds, f = None, None
        try:
            f = self.fs.open(path, 'rb')
            ds = xr.open_dataset(f, chunks=chunks)
            yield ds
        finally:
            if ds is not None:
                ds.close()
            if f is not None:
                f.close()
