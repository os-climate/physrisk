from hazard.sources.interfaces import IOpenDatasetForYear
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6

import dask
import xarray as xr

class DegreeDays:
    """Calculates degree days from temperature data sets"""
    def __init__(self, open_dataset: IOpenDatasetForYear):
        self.source = NexGddpCmip6()
        self.open_dataset: IOpenDatasetForYear = open_dataset if open_dataset is not None else self.source.open_dataset


    @staticmethod
    def degree_days(ds: xr.Dataset) -> xr.DataArray:
        """Caculate degree days for Dataset provided"""
        threshold = 273.15 + 32 # degree days above 32C 
        return xr.where(ds.tas > threshold, ds.tas, 0).sum(dim=["time"])


    @staticmethod
    def average_degree_days(open_dataset: IOpenDatasetForYear, gcm: str, scenario: str, central_year: int, window_years: int = 9):
        """Caclulate average annual degree days for given window"""
        years = range(central_year - window_years // 2, central_year + window_years // 2 + 1)
        deg_days = [DegreeDays.degree_days(open_dataset(gcm, scenario, "tas", year)) for year in years]
        return sum(deg_days) / float(len(years))


    def calculate_all(self):
        """Caclulate all average degree days sets"""
        gcms = self.source.gcms
        central_years = [2030, 2050, 2080]
        scenarios = ["historical", "ssp126", "ssp245", "ssp585"]
        # we assume historical is centred on 2000

        results = []
        for scenario in scenarios:
            for central_year in central_years:
                for gcm in gcms:
                    result = dask.delayed(DegreeDays.average_degree_days)(self.source.open_dataset, gcm, scenario, central_year)
                    results.append(result)

        futures = dask.persist(*result)  
        results = dask.compute(*futures)

