from typing import Iterable, List, Protocol
import typing
import xarray as xr

class POpenDatasetForYear(Protocol):
    """Open XArray Dataset for Global Circulation Model (GCM), scenario and quantity for whole year specified"""
    def gcms() -> Iterable[str]:
        ...
    def open_dataset(self, gcm: str, scenario: str, quantity: str, year: int, chunks=None) -> xr.Dataset:
        ...
    #def __call__(self, gcm: str, scenario: str, quantity: str, year: int) -> xr.Dataset: ...


T = typing.TypeVar('T')

class PTransform(Protocol):
    
    def batch_items(self) -> Iterable[T]:
        ...
    def process_item(self, item: T) -> xr.DataArray:
        ...
    def item_path(self, item: T) -> str:
        ...
