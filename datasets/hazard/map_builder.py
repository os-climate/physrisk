

import logging
import os
from pathlib import Path
from typing import Iterable
import hazard.utilities.map_utilities as map_utilities
from hazard.sources.osc_zarr import OscZarr

logger = logging.getLogger(__name__)

class MapBuilder:
    
    def __init__(self, store: OscZarr=None, working_directory: str=None):
        self.store = OscZarr() if store is None else store
        self.working_directory = working_directory


    def create_maps(self, paths: Iterable[str]):
        if self.working_directory is None:
            raise 
        
        max_value = float("-inf")
        logger.info("Calculating max value")
        for path in paths:
            data, _ = self.store.read_numpy(path)
            max_value = max(max_value, data.max())     

        logger.info("Writing and uploading maps")
        access_token = os.environ["OSC_MAPBOX_UPLOAD_TOKEN"]
        for path in paths:
            data, transform = self.store.read_numpy(path)
            profile = map_utilities.geotiff_profile()
            filename = Path(path).stem + ".tif"
            path_out, colormap_path_out = map_utilities.write_map_geotiff_data(data,
                profile,
                data.shape[1],
                data.shape[0],
                transform, 
                filename, 
                self.working_directory, 
                nodata_threshold=0,
                zero_transparent=True,
                max_intensity=max_value,
                palette="heating")
           
            filename = os.path.basename(path_out)
            id = filename[4:10]
            map_utilities.upload_geotiff(path_out, id, access_token)