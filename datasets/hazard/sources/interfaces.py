from typing import List, Protocol
import xarray as xr

class IOpenDatasetForYear(Protocol):
    """Open XArray Dataset for Global Circulation Model (GCM), scenario and quantity for whole year specified"""
    def __call__(self, gcm: str, scenario: str, quantity: str, year: int) -> xr.Dataset: ...

        