from physrisk.data import colormap_provider
from test.base_test import TestWithCredentials

import numpy as np
import zarr
import zarr.storage

from physrisk.data.image_creator import ImageCreator


class TestImageCreation(TestWithCredentials):
    def test_image_creation(self):
        path = "test_array"
        store = zarr.storage.MemoryStore(root="hazard.zarr")
        root = zarr.open(store=store, mode="w")
        #x, y = np.meshgrid((np.arange(10) - 5.0) / 20.0, (np.arange(10) - 5.0) / 20.0)
        #im = np.exp(-(x**2 + y**2))
        im = np.array([[1.2, 0.8], [0.5, 0.4]])
        z = root.create_dataset(  # type: ignore
            path, shape=(1, im.shape[0], im.shape[1]), chunks=(1, im.shape[0], im.shape[1]), dtype="f4"
        )
        z[0, :, :] = im
        converter = ImageCreator(store=store)
        colormap = colormap_provider.colormap("test")
        
        def get_colors(index: int):
            return colormap[str(index)]

        result = converter.to_rgba(im, get_colors)
        # Max should be 255, min should be 1. Other values span the 253 elements from 2 to 254. 
        expected = np.array([[255, 1 + (0.8 - 0.4) * 253 / (1.2 - 0.4)], [1 + (0.5 - 0.4) * 253 / (1.2 - 0.4), 1]])
        converter.convert(path, colormap="test") # check no error running through mocked example.
        np.testing.assert_equal(result, expected.astype(np.uint8))
        

    #def test_image_creation_req(self):
    #    import physrisk.requests
    #    # http://127.0.0.1:5000/api/images/chronic_heat/osc/v1/mean_work_loss_high_ssp245_2050_map.png
    #    physrisk.requests.get_image("chronic_heat/osc/v1/mean_work_loss_high_ssp585_2050_map", max_value=1.0)
