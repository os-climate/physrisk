import logging
import os
from pathlib import Path
from typing import Iterable
import dask
from hazard.map_builder import MapBuilder
from hazard.protocols import PTransform
import hazard.utilities.zarr_utilities as zarr_utilities
from hazard.sources.osc_zarr import OscZarr

logger = logging.getLogger(__name__)

class TaskRunner:
    """Runs transform tasks using dask."""
    def __init__(self, transform: PTransform,
            store: OscZarr=None,
            working_directory: str=None,
            map_builder: MapBuilder=None
        ):
        zarr_utilities.set_credential_env_variables()
        self.map_builder = map_builder
        self.transform = transform
        self.store = OscZarr() if store is None else store
        self.working_directoy = working_directory


    def run_batch(self):
        logger.info("Hazard batch")
        items = list(self.transform.batch_items())
        logger.info(f"Running {len(items)} items using dask")

        results = []    
        for item in items:
            result = dask.delayed(self._process_and_store)(item)
            results.append(result)

        futures = dask.persist(*results)  
        result_paths = dask.compute(*futures)

        if self.map_builder is not None:
            self.map_builder.create_maps(result_paths)
    

    def _process_and_store(self, item):
        """Process work item and store the result."""
        path = self.transform.item_path(item)
        logger.info(f"Processing item {path} with transform: {type(self.transform)}")
        da = self.transform.process_item(item)
        logger.info(f"Writing item {path} with writer {type(self.transform)}")
        
        try:
            self.store.if_exists_remove(path)
        except:
            logger.warn(f"Could not remove {path}") 
        self.store.write(path, da)
        return path

