import json
from pathlib import PurePosixPath
from typing import Callable, Iterable, List, Optional

import s3fs
from fsspec import AbstractFileSystem
from pydantic import BaseModel, parse_obj_as

from physrisk.api.v1.hazard_data import HazardModel

from .zarr_reader import get_env


class HazardModels(BaseModel):
    hazard_models: List[HazardModel]


class InventoryReader:
    # environment variable names:
    __access_key = "OSC_S3_ACCESS_KEY"
    __secret_key = "OSC_S3_SECRET_KEY"
    __S3_bucket = "OSC_S3_BUCKET"  # e.g. redhat-osc-physical-landing-647521352890

    def __init__(
        self,
        get_env: Callable[[str, Optional[str]], str] = get_env,
        fs: Optional[AbstractFileSystem] = None,
    ):
        """Class to read and update inventory stored in S3 or supplied AbstractFileSystem.

        Args:
            get_env (Callable[[str, Optional[str]], str], optional): Get environment variable. Defaults to get_env.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem. Defaults to None in which case S3FileSystem will be created. # noqa: E501
        """
        if fs is None:
            access_key = get_env(self.__access_key, None)
            secret_key = get_env(self.__secret_key, None)
            fs = s3fs.S3FileSystem(anon=False, key=access_key, secret=secret_key)

        self._bucket = get_env(self.__S3_bucket, "redhat-osc-physical-landing-647521352890")
        self._fs = fs

    def read(self, path: str) -> List[HazardModel]:
        """Read"""
        if not self._fs.exists(self._full_path(path)):
            return []
        json_str = self.read_json(path)
        models = parse_obj_as(HazardModels, json.loads(json_str)).hazard_models
        return models

    def read_json(self, path: str) -> str:
        with self._fs.open(self._full_path(path), "r") as f:
            json_str = f.read()
        return json_str

    def append(self, path: str, hazard_models: Iterable[HazardModel]):
        combined = dict((i.key(), i) for i in self.read(path))
        for model in hazard_models:
            combined[model.key()] = model
        models = HazardModels(hazard_models=list(combined.values()))
        json_str = json.dumps(models.dict())
        with self._fs.open(self._full_path(path), "w") as f:
            f.write(json_str)

    def _full_path(self, path: str):
        if path not in ["hazard", "hazard_test"]:
            raise ValueError("not supported path.")
        return str(PurePosixPath(self._bucket, path, "inventory.json"))
