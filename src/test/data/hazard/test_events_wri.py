""" Test asset impact calculations."""
import unittest
import shutil, tempfile
import numpy as np
from physrisk.data.hazard.event_provider_wri import EventProviderWri

class TestEventsWri(unittest.TestCase):
    """Tests asset impact calculations."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

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







        