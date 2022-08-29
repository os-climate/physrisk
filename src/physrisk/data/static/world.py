import importlib.resources
import json
from typing import Dict, List, Union

import numpy as np

import physrisk.data.static
from physrisk.api.v1.common import Countries, Country


def get_countries_from_resource():
    with importlib.resources.open_text(physrisk.data.static, "world.json") as f:
        countries = Countries(**json.load(f))
        return dict((c.country, c) for c in countries.items)


def get_countries_json():
    """Get countries and continents, populating json."""

    import geopandas as gpd

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))  # type: ignore
    countries = world[["continent", "name", "iso_a3"]]

    countries = [
        Country(continent=continent, country=country, country_iso_a3=code)
        for (continent, country, code) in zip(world["continent"], world["name"], world["iso_a3"])
    ]

    return json.dumps(Countries(items=countries).dict(), sort_keys=True, indent=4)


def get_countries_and_continents(longitudes: Union[List[float], np.ndarray], latitudes: Union[List[float], np.ndarray]):
    """Only for use when on-boarding; look up country and continent (e.g. for use in vulnerability models) by
    latitude and longitude."""

    # Geopandas draws in a number of libraries, including GDAL, so we probably(?)
    # want to confine its use to pre-processing / on-boarding of data
    # In particular, country/continent look-up is probably something to do pre-onboarding

    import geopandas as gpd

    # consider using map here https://gadm.org/download_world.html
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))  # type: ignore
    gdf = gpd.GeoDataFrame(crs=world.crs, geometry=gpd.points_from_xy(longitudes, latitudes))
    result = gpd.sjoin(gdf, world, how="left")

    return list(result["name"]), list(result["continent"])


class World:

    countries: Dict[str, Country] = get_countries_from_resource()
