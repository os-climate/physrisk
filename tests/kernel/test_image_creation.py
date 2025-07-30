import io
import os
from typing import Dict
import unittest

from dependency_injector import providers
import numpy as np
import PIL.Image as Image
import zarr
import zarr.storage

from physrisk.api.v1.hazard_data import HazardResource, Scenario, MapInfo
from physrisk.container import Container
from physrisk.data import colormap_provider
from physrisk.data.hazard_data_provider import ScenarioPaths, SourcePaths
from physrisk.data.image_creator import ImageCreator
from physrisk.data.inventory import Inventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import (
    InventorySourcePaths,
    get_default_source_paths,
)
from physrisk.kernel.hazard_model import HazardModelFactory, Tile

from ..test_base import TestWithCredentials


class SourcePathsTest(SourcePaths):
    def hazard_types(self):
        return []

    def resource_paths(self, hazard_type, indicator_id, scenarios, hint=None):
        pass

    def scenario_paths_for_id(self, resource_id, scenarios, map):
        return {
            s: ScenarioPaths(
                years=[2030, 2050], path=lambda y, s=s: f"test_array_{s}_{y}"
            )
            for s in scenarios
        }


class TestImageCreation(TestWithCredentials):
    def test_image_creation(self):
        path = "test_array"
        store = zarr.storage.MemoryStore(root="hazard.zarr")
        root = zarr.open(store=store, mode="w")

        im = np.array([[1.2, 0.8], [0.5, 0.4]])
        z = root.create_dataset(  # type: ignore
            path,
            shape=(1, im.shape[0], im.shape[1]),
            chunks=(1, im.shape[0], im.shape[1]),
            dtype="f4",
        )
        z[0, :, :] = im
        converter = ImageCreator(
            reader=ZarrReader(store), source_paths=get_default_source_paths()
        )
        colormap = colormap_provider.colormap("test")

        def get_colors(index: int):
            return colormap[str(index)]

        result = converter._to_rgba(im, get_colors)
        # Max should be 255, min should be 1. Other values span the 253 elements from 2 to 254.
        expected = np.array(
            [
                [255, 2 + (0.8 - 0.4) * 253 / (1.2 - 0.4)],
                [2 + (0.5 - 0.4) * 253 / (1.2 - 0.4), 1],
            ]
        )
        np.testing.assert_equal(result, expected.astype(np.uint8))

    def _mock_inventory(self):
        return Inventory(
            hazard_resources=[
                HazardResource(
                    path="test_array_{scenario}_{year}",
                    hazard_type="",
                    indicator_id="",
                    indicator_model_gcm="",
                    display_name="",
                    description="",
                    scenarios=[Scenario(id="ssp585", years=[2030, 2050])],
                    units="",
                    map=MapInfo(
                        path="test_array_{scenario}_{year}",
                        bounds=[],
                        colormap=None,
                        source="",
                    ),
                )
            ]
        )

    def _store(self):
        store = zarr.storage.MemoryStore(root="hazard.zarr")
        root = zarr.open(store=store, mode="w")

        im = np.array([[1.2, 0.8], [0.5, 0.4]])
        im2 = np.array([[1.2 + 1.0, 0.8 + 1.0], [0.5 + 1.0, 0.4 + 1.0]])
        z = root.create_dataset(  # type: ignore
            "test_array_ssp585_2030/1",
            shape=(1, im.shape[0], im.shape[1]),
            chunks=(1, im.shape[0], im.shape[1]),
            dtype="f4",
        )
        z[0, :, :] = im
        z2 = root.create_dataset(  # type: ignore
            "test_array_ssp585_2050/1",
            shape=(1, im2.shape[0], im2.shape[1]),
            chunks=(1, im2.shape[0], im2.shape[1]),
            dtype="f4",
        )
        z2[0, :, :] = im2
        return store

    def test_interpolation(self):
        store = self._store()
        converter = ImageCreator(
            source_paths=SourcePathsTest(), reader=ZarrReader(store)
        )
        result = converter.create_image(
            "test_array_{scenario}_{year}",
            "ssp585",
            2040,
            min_value=0,
            max_value=2,
            tile=Tile(0, 0, 0),
        )
        rgba = np.array(
            [[3359392974, 3361568744], [3363412211, 3364135926]], dtype=np.uint32
        )
        image_bytes = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(image_bytes, format="PNG")
        expected = image_bytes.getvalue()
        np.testing.assert_equal(result, expected)

    def test_request(self):
        store = self._store()
        request_dict = {
            "resource": "test_array_{scenario}_{year}",
            "scenario_id": "ssp585",
            "year": 2040,
            "min_value": 0.0,
            "max_value": 2.0,
            "tile": Tile(0, 0, 0),
            "index": 2,
        }

        container = Container()

        inventory = self._mock_inventory()
        source_paths = InventorySourcePaths(inventory)

        class TestHazardModelFactory(HazardModelFactory):
            def hazard_model(
                self,
                interpolation: str = "floor",
                provider_max_requests: Dict[str, int] = {},
                interpolate_years: bool = False,
            ):
                return ZarrHazardModel(source_paths=source_paths, store=store)

            def image_creator(self):
                return ImageCreator(source_paths, ZarrReader(store=store))

        container.override_providers(
            hazard_model_factory=providers.Factory(TestHazardModelFactory)
        )
        container.override_providers(source_paths=providers.Factory(SourcePathsTest))
        container.override_providers(inventory=providers.Singleton(lambda: inventory))
        requester = container.requester()
        res = requester.get_image(request_dict)
        rgba = np.array(
            [[3359392974, 3361568744], [3363412211, 3364135926]], dtype=np.uint32
        )
        image_bytes = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(image_bytes, format="PNG")
        expected = image_bytes.getvalue()
        np.testing.assert_equal(res, expected)

    @unittest.skip("just example")
    def test_write_file(self):
        # show how to create image from zarr array
        # useful for testing image generation
        test_output_dir = "{set me}"
        test_path = "wildfire/jupiter/v1/wildfire_probability_ssp585_2050_map"
        store = zarr.DirectoryStore(
            os.path.join(test_output_dir, "hazard_test", "hazard.zarr")
        )
        inventory = self.mock_inventory()
        source_paths = InventorySourcePaths(inventory)
        creator = ImageCreator(ZarrReader(store), source_paths=source_paths)
        creator.to_file(os.path.join(test_output_dir, "test.png"), test_path)
