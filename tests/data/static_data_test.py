import os

from physrisk.data.static.world import (
    World,
    get_countries_and_continents,
    get_countries_json,
)
from ..data.hazard_model_store_test import TestData


def test_get_countries_and_continents():
    countries, continents = get_countries_and_continents(
        TestData.longitudes, TestData.latitudes
    )
    assert countries[0:3] == ["Afghanistan", "Afghanistan", "Albania"]


def test_get_countries_json(test_dir):
    with open(os.path.join(test_dir, "world.json"), "w") as f:
        world_json = get_countries_json()
        f.write(world_json)


def test_get_load_world():
    assert World.countries["United Kingdom"].continent == "Europe"
