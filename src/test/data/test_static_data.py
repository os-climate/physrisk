
import unittest
from physrisk.data.static.world import World, get_countries_json, get_countries_and_continents
from test.data.hazard_model_store import TestData

class TestStaticDate(unittest.TestCase):

    @unittest.skip("example that requires geopandas (consider adding for running tests only)")
    def test_get_countries_and_continents(self):       
        countries, continents = get_countries_and_continents(TestData.longitudes, TestData.latitudes)
        self.assertEqual(countries[0:3], ['Afghanistan', 'Afghanistan', 'Albania'])

    @unittest.skip("not really a test; just showing how world.json was generated")
    def test_get_countries_json(self):       
        with open('world.json', 'w') as f:
            world_json = get_countries_json()
            f.write(world_json)

    def test_get_load_world(self):       
        self.assertEqual(World.countries["United Kingdom"].continent, "Europe")