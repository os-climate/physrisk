import os
import unittest

import numpy as np
import zarr
import zarr.storage

from physrisk.data import colormap_provider
from physrisk.data.image_creator import ImageCreator
from physrisk.data.zarr_reader import ZarrReader

from ..test_base import TestWithCredentials


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
        converter = ImageCreator(reader=ZarrReader(store))
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
        converter.convert(
            path, colormap="test"
        )  # check no error running through mocked example.
        np.testing.assert_equal(result, expected.astype(np.uint8))

    @unittest.skip("just example")
    def test_write_file(self):
        # show how to create image from zarr array
        # useful for testing image generation
        test_output_dir = "{set me}"
        test_path = "wildfire/jupiter/v1/wildfire_probability_ssp585_2050_map"
        store = zarr.DirectoryStore(
            os.path.join(test_output_dir, "hazard_test", "hazard.zarr")
        )
        creator = ImageCreator(ZarrReader(store))
        creator.to_file(os.path.join(test_output_dir, "test.png"), test_path)
