import logging
import os

import s3fs
from onboard_osc_heat import create_map_geotiffs_chronic_heat, onboard_chronic_heat
from onboard_wri import create_map_geotiffs_riverine, onboard_wri_coastal_inundation, onboard_wri_riverine_inundation
from zarr_utilities import add_logging_output_to_stdout, set_credential_env_variables

LOG = logging.getLogger("Hazard onboarding")
add_logging_output_to_stdout(LOG)

set_credential_env_variables()  # set credentials from credentials.env into environment variables

# settings for zarr storage
dest_bucket = "redhat-osc-physical-landing-647521352890"  # S3 bucket used for zarr storage (credentials must align)
dest_prefix = "hazard_test"  # _test" # prefix for zarr storage within bucket; use 'hazard_test' for testing, 'hazard' for final version
src_dir = "/opt/app-root/src/file_staging"  # where scripts do one-off conversion (i.e. resource not otherwise accessible e.g. via API) from files, this is path

# settings for map creation
map_working_dir = "/opt/app-root/src/map_working"

src_dir = "/Users/joemoorhouse/Code/data/heat/"
map_working_dir = "/Users/joemoorhouse/Code/working"

s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])

# on board chronic heat
# onboard_chronic_heat(
#    src_dir, dest_bucket=dest_bucket, dest_prefix=dest_prefix, s3_dest=s3
# )  # create zarr from file as one-off
# create map geotiffs from zarr
# create_map_geotiffs_chronic_heat(
#    dest_bucket=dest_bucket, dest_prefix=dest_prefix, map_working_dir=map_working_dir, dest_s3=s3, account="osc"
# )  # account=None|"osc"


# onboard_wri_riverine_inundation(dest_bucket)
# onboard_wri_coastal_inundation(dest_bucket)
create_map_geotiffs_riverine(map_working_dir)
