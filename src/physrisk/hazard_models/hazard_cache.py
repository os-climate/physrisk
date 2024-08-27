import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Protocol, Sequence

import h3
from lmdbm import Lmdb


@dataclass
class Indicator:
    hazard_type: str
    indicator_id: str


class ItemType(str, Enum):
    request = "request"
    response = "response"


class Store(Protocol):
    def __init__(self, file: str): ...

    def setitems(self, items: Dict[str, Any]): ...

    def getitems(self, keys: Sequence[str]): ...

    def getall(self, prefix: str = "") -> Dict[str, bytes]: ...


def to_json(store: Store, prefix: str = ""):
    return json.dumps(
        {k: json.loads(v) for k, v in store.getall(prefix).items()}, indent=4
    )


class LMDBStore(Store):
    def __init__(self, file: str):
        self._file = file
        from pathlib import Path

        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with Lmdb.open(self._file, "c") as _:
            ...

    def setitems(self, items: Dict[str, Any]):
        with Lmdb.open(self._file, "c") as db:
            db.update(items)

    def getitems(self, keys: Sequence[str]):
        with Lmdb.open(self._file, "r") as db:
            return [db.get(k, None) for k in keys]

    def getall(self, prefix: str = ""):
        with Lmdb.open(self._file, "r") as db:
            return {
                k.decode(): v for k, v in db.items() if k.decode().startswith(prefix)
            }


class MemoryStore(Store):
    def __init__(
        self, file: Optional[str] = None, values: Optional[Dict[str, str]] = None
    ):
        self._dict = {}
        if file is not None and os.path.exists(file):
            with open(file, "r") as f:
                self._dict.update(json.loads(f.read()))
        if values is not None:
            self._dict.update(values)

    def setitems(self, items: Dict[str, Any]):
        self._dict.update(items)

    def getitems(self, keys: Sequence[str]):
        return [self._dict.get(k, None) for k in keys]

    def getall(self, prefix: str = ""):
        return {k: v for k, v in self._dict.items() if k.startswith(prefix)}


class H3BasedCache:
    def __init__(self, store: Store):
        self.resolution = 12  # resolution 9 ~ 200m; 12 ~ 10m
        self.store = store

    def spatial_key(self, latitude: float, longitude: float):
        return h3.geo_to_h3(latitude, longitude, self.resolution)

    def location_from_spatial_key(self, spatial_key: str):
        lat, lon = h3.h3_to_geo(spatial_key)
        # h3.h3_to_geo_boundary
        res = h3.h3_get_resolution(spatial_key)
        return lat, lon, res

    def key(self, provider_id: str, spatial_index: str):
        return str(PurePosixPath(provider_id, spatial_index))

    def getall(self, prefix: str = ""):
        return self.store.getall(prefix)

    def getitems(self, keys: List[str]):
        return self.store.getitems(keys)

    def setitems(self, items: Dict[str, Any]):
        self.store.setitems(items)
