import json
import os
import sys

sys.path.append("src")
a = os.getcwd()
import physrisk.data.static.world as wd


class Asset:
    def __init__(self, latitude: float, longitude: float, **kwargs):
        self.latitude = latitude
        self.longitude = longitude
        self.__dict__.update(kwargs)


class RealEstateAsset(Asset):
    def __init__(self, latitude: float, longitude: float, *, location: str, type: str):
        super().__init__(latitude, longitude)
        self.location = location
        self.type = type


cache_folder = r"/users/joemoorhouse/code/data"

with open(os.path.join(cache_folder, "assets_test_power_gen.json")) as f:
    assets = json.loads(f.read())

assets_new = []
latitudes = [a["latitude"] for a in assets["items"]]
longitudes = [a["longitude"] for a in assets["items"]]
countries, continents = wd.get_countries_and_continents(latitudes=latitudes, longitudes=longitudes)

items = []
for i in range(len(assets["items"])):
    if isinstance(continents[i], str):
        items.append(
            {
                "latitude": latitudes[i],
                "longitude": longitudes[i],
                "asset_class": "RealEstateAsset",
                "type": "Buildings/Industrial",
                "location": "Europe",
            }
        )  # continents[i]})

# for a in assets["items"]:
#    lat = a["latitude"]
#    lon = a["longitude"]
#    countries, continents = wd.get_countries_and_continents(latitudes = [lat], longitudes = [lon])
#    assets_new.append(RealEstateAsset(latitude = lat, longitude = lon, type = "Buildings/Industrial", location=continents[0]))

with open(os.path.join(cache_folder, "assets_test_real_estate.json"), "w") as f:
    f.write(json.dumps({"items": items}))
