import logging
import os
import pathlib
from importlib.resources import path

from dotenv import load_dotenv
from onboard_osc_heat import onboard_chronic_heat
from onboard_wri import create_map_geotiffs_riverine, onboard_wri_coastal_inundation, onboard_wri_riverine_inundation
from zarr_utilities import add_logging_output_to_stdout, set_credential_env_variables

LOG = logging.getLogger("Hazard onboarding")
add_logging_output_to_stdout(LOG)

set_credential_env_variables()

# writes
# 1) Zarr hazard data to zarr_store
# 2) Map-ready geotiffs using file system fs and to directory dir

dest_bucket = "redhat-osc-physical-landing-647521352890"


onboard_chronic_heat(dest_bucket)
# onboard_wri_riverine_inundation(dest_bucket)
# onboard_wri_coastal_inundation(dest_bucket)
# create_map_geotiffs_riverine()
