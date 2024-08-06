import json
from pathlib import PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional

import s3fs
from fsspec import AbstractFileSystem
from pydantic import BaseModel, TypeAdapter

from physrisk.api.v1.hazard_data import HazardResource

from .zarr_reader import get_env


class HazardModels(BaseModel):
    resources: List[HazardResource]


class InventoryReader:
    # environment variable names:
    __access_key = "OSC_S3_ACCESS_KEY"
    __secret_key = "OSC_S3_SECRET_KEY"
    __S3_bucket = "OSC_S3_BUCKET"

    def __init__(
        self,
        *,
        get_env: Callable[[str, Optional[str]], str] = get_env,
        fs: Optional[AbstractFileSystem] = None,
        base_path: Optional[AbstractFileSystem] = None,
    ):
        """Class to read and update inventory stored in S3 or supplied AbstractFileSystem.

        Args:
            get_env (Callable[[str, Optional[str]], str], optional): Get environment variable. Defaults to get_env.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem. Defaults to None in which case S3FileSystem will be created. # noqa: E501
        """
        if fs is None:
            access_key = get_env(self.__access_key, "")
            secret_key = get_env(self.__secret_key, "")
            fs = s3fs.S3FileSystem(anon=False, key=access_key, secret=secret_key)

        bucket = get_env(self.__S3_bucket, "physrisk-hazard-indicators")
        self._base_path = bucket if base_path is None else base_path
        self._fs = fs

    def read(self, path: str) -> List[HazardResource]:
        """Read inventory at path provided and return HazardModels."""
        if not self._fs.exists(self._full_path(path)):
            return []
        json_str = self.read_json(path)
        models = (
            TypeAdapter(HazardModels).validate_python(json.loads(json_str)).resources
        )
        return models

    def read_description_markdown(self, paths: List[str]) -> Dict[str, str]:
        """Read description markdown at path provided."""
        md: Dict[str, str] = {}
        for path in paths:
            try:
                with self._fs.open(self._full_path(path), "r") as f:
                    md[path] = f.read()
            finally:
                continue
        return md

    def read_json(self, path: str) -> str:
        """Read inventory at path provided and return json."""
        with self._fs.open(self._full_path(path), "r") as f:
            json_str = f.read()
        return json_str

    def append(self, path: str, hazard_models: Iterable[HazardResource]):
        combined = dict((i.key(), i) for i in self.read(path))
        for model in hazard_models:
            combined[model.key()] = model
        models = HazardModels(resources=list(combined.values()))
        json_str = json.dumps(models.model_dump())
        with self._fs.open(self._full_path(path), "w") as f:
            f.write(json_str)

    def _full_path(self, path: str):
        if path not in ["hazard", "hazard_test"]:
            raise ValueError("not supported path.")
        return str(PurePosixPath(self._base_path, path, "inventory.json"))
