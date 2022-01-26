import logging, os.path, requests
import numpy as np
import physrisk.data.geotiff_reader as rr
from typing import Any, cast, Any, Callable, List, TypeVar, Union
from typing_extensions import Protocol
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.events import Event, Inundation, RiverineInundation

class SourcePath(Protocol):
    def __call__(self, year: int = None, scenario: str = None, model: str = None) -> str: ...

def _wri_inundation_prefix():
    return "inundation/wri/v2"

def get_source_path_wri_riverine_inundation(year: int = None, scenario: str = None, model: str = None):
    type = 'river'
    return os.path.join(_wri_inundation_prefix(), f"inun{type}_{scenario}_{model}_{year}")

class EventProvider():
    def __init__(self, get_source_path: SourcePath):
        self._get_source_path = get_source_path
        self._reader = ZarrReader()

    def get_intensity_curves(self,
        longitudes: List[float], 
        latitudes: List[float], 
        year: int = None, 
        scenario: str = None,
        model: str = None):
        
        path = self._get_source_path(year, scenario, model)
        curves, return_periods = self._reader.get_curves(path, longitudes, latitudes)
        return curves
            