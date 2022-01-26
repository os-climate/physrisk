""" Test asset impact calculations."""
import json, os, pathlib, shutil, tempfile, unittest
from dotenv import dotenv_values, load_dotenv
import numpy as np
from physrisk.data.event_provider import EventProvider, get_source_path_wri_riverine_inundation
from physrisk.data.hazard.event_provider_wri import EventProviderWri
from physrisk.kernel.events import RiverineInundation
from physrisk.risk_requests import get_hazard_data, HazardEventDataRequest

class TestEventRetrieval(unittest.TestCase):
    """Tests asset impact calculations."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        dotenv_dir = os.environ.get('CREDENTIAL_DOTENV_DIR', os.getcwd())
        dotenv_path = pathlib.Path(dotenv_dir) / 'credentials.env'
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path,override=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    @unittest.skip("includes download of large files")
    def test_wri_from_web(self):
        cache_folder = self.test_dir 
        provider = EventProviderWri('web', cache_folder = cache_folder)
        lon = 19.885738
        lat = 45.268405
        events = provider.get_inundation_depth([lon], [lat])
        print(events)

    @unittest.skip("requires credentials")
    def test_zarr_reading(self):
        
        with open(os.path.join(os.getcwd(), "src/test/data", "coords.json"), 'r') as f:
            coords = json.load(f)
        
        request_dict = {
            'request_id': 'get_hazard_data',
            'event_type': 'RiverineInundation',
            'longitudes': coords['longitudes'][0:1000],
            'latitudes': coords['longitudes'][0:1000],
            'year': 2080,
            'scenario': "rcp8p5",
            'model': 'MIROC-ESM-CHEM'
        }
        request = HazardEventDataRequest(**request_dict) # validate request
        
        data_sources = { 
            RiverineInundation: EventProvider(get_source_path_wri_riverine_inundation).get_intensity_curves
        }
        
        get_hazard_data(request, data_sources)








        