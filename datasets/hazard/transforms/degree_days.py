from dataclasses import dataclass
import os, logging
from typing import Iterable
from hazard.utilities.xarray_utilities import enforce_conventions
from hazard.protocols import PTransform
from hazard.protocols import POpenDatasetForYear
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6

import dask
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

@dataclass
class BatchItem():
    gcm: str
    scenario: str
    year: int


class DegreeDays(PTransform):  
    """Calculates degree days from temperature data sets."""

    def __init__(self, source: POpenDatasetForYear=None,
            theshold: float=32,
            window_years: int=20,
            gcms: Iterable[str]=None,
            scenarios: Iterable[str]=None,
            central_years: Iterable[int]=None):

        self.source: POpenDatasetForYear = NexGddpCmip6() if source is None else source
        self.threshold: float = 273.15 + theshold # in Kelvin; degree days above 32C
        self.window_years = window_years
        self.gcms = self.source.gcms if gcms is None else gcms
        self.scenarios = ["historical", "ssp126", "ssp245", "ssp585"] if scenarios is None else scenarios
        # 1995, 2014 (2010)
        # 2021, 2040 (2030)
        # 2031, 2050 (2040)
        # 2041, 2060 (2050)
        
        self.central_years = [2010, 2030, 2040, 2050] if central_years is None else central_years


    @staticmethod
    def degree_days(ds: xr.Dataset, threshold: float) -> xr.DataArray:
        """Caculate degree days for Dataset provided."""
        # check DataArray 
        if any(coord not in ds.coords.keys() for coord in ['lat', 'lon', 'time']):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        # will raise error if taxmax not present    
        return xr.where(ds.tasmax > threshold, ds.tasmax - threshold, 0).sum(dim=["time"])


    def average_degree_days(self, gcm: str, scenario: str, central_year: int, window_years: int = 9) -> xr.DataArray:
        """Caclulate average annual degree days for given window."""
        years = range(central_year - window_years // 2, central_year + window_years // 2 + (window_years % 2))
        logging.info(f"Calculating average degree days, gcm={gcm}, scenario={scenario}, years={list(years)}")
        deg_days = []
        for year in years:
            with self.source.open_dataset(gcm, scenario, "tasmax", year) as ds:
                deg_days.append(DegreeDays.degree_days(ds, self.threshold))

        average = sum(deg_days) / float(len(years))
        return enforce_conventions(average)


    def batch_items(self) -> Iterable[BatchItem]:
        """Items to process."""
        for gcm in self.gcms:
            for scenario in self.scenarios:
                for central_year in self.central_years:
                    yield BatchItem(gcm=gcm, scenario=scenario, year=central_year)    

    
    def process_item(self, item: BatchItem) -> xr.DataArray:
        return self.average_degree_days(item.gcm, item.scenario, item.year, self.window_years)


    def item_path(self, item: BatchItem) -> str:
        path = "chronic_heat/osc/v2" # v2 uses downscaled data
        typ = "mean_degree_days_v2" # need v2 in filename to make unique
        levels = ['above', '32c']
        return os.path.join(path, f"{typ}_{levels[0]}_{levels[1]}_{item.gcm}_{item.scenario}_{item.year}")



