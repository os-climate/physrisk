from importlib.resources import files
from typing import Dict, Optional, Sequence, Union

import geopandas as gpd
import pandas as pd

import physrisk.data.ne_10m_admin_0_map_subunits
from physrisk.kernel.assets import Asset


class Geocoder:
    def __init__(self):
        """Geocoder uses Natural Earth data to map lat/lon to country codes.
        High-resolution 'sub-unit' data is used, mainly to satisfy the needs
        of third-party APIs which require accurate country codes."""
        path = files(physrisk.data.ne_10m_admin_0_map_subunits).joinpath(
            "ne_10m_admin_0_map_subunits.shp"
        )
        world_orig = gpd.read_file(path)
        self.crs = world_orig.crs
        self.world = world_orig.to_crs(crs=3857)
        self.continents_from_a2 = (
            Geocoder.get_continent_and_country_from_code_iso_3166()[
                "Continent"
            ].to_dict()
        )
        self.subunit_mapping: Dict[str, Dict[str, str]] = {
            "ES": {"Canary Islands": "IC"},
            "-99": {"Crimea": "RU", "Cyprus No Mans Area": "CY", "Somaliland": "SO"},
            "PT": {"Madeira": "PT_M"},
        }

    def geocode_in_place(self, assets: Sequence[Asset]):
        no_country_assets = [
            a for a in assets if not getattr(a, "country_iso_a2", None)
        ]
        countries = self.get_countries(
            [a.latitude for a in assets if a in no_country_assets],
            [a.longitude for a in assets if a in no_country_assets],
        )
        for a, country_iso_a2 in zip(no_country_assets, countries):
            a.country_iso_a2 = country_iso_a2  # type: ignore[attr-defined]

    def get_countries(self, latitudes: Sequence[float], longitudes: Sequence[float]):
        gdf = gpd.GeoDataFrame(
            crs=self.crs, geometry=gpd.points_from_xy(longitudes, latitudes)
        ).to_crs(crs=3857)

        gdf["id"] = gdf.index
        result = gpd.sjoin_nearest(gdf, self.world, how="left")
        result = result.groupby(by="id").first()
        iso_a2s = result[
            "ISO_A2_EH"
        ].values  # ISO_A2 contains -99 https://github.com/nvkelso/natural-earth-vector/issues/268
        subunits = result["SUBUNIT"].values
        final_iso_a2 = []
        for iso_a2, subunit in zip(iso_a2s, subunits):
            if iso_a2 in self.subunit_mapping:
                iso_a2 = self.subunit_mapping[iso_a2].get(subunit, iso_a2)
            final_iso_a2.append(iso_a2)
        return final_iso_a2

    def get_continent(self, country_code_a2: str):
        return self.continents_from_a2.get(country_code_a2, "Generic")

    @staticmethod
    def get_continent_and_country_from_code_iso_3166(
        country_codes: Optional[Sequence[Union[str, int]]] = None,
    ) -> pd.DataFrame:
        path = files(physrisk.data.ne_10m_admin_0_map_subunits).joinpath(
            "country_codes.tsv"
        )
        df = pd.read_csv(path, sep="\t", keep_default_na=False)
        if country_codes is None:
            country_codes = list(df["Alpha-2 code"])
        numeric_codes = [
            int(country_code)
            for country_code in country_codes
            if str(country_code).isdecimal()
        ]
        alpha2_codes = [
            country_code
            for country_code in country_codes
            if not (str(country_code).isdecimal()) and len(str(country_code)) == 2
        ]
        alpha3_codes = [
            country_code
            for country_code in country_codes
            if not (str(country_code).isdecimal()) and len(str(country_code)) == 3
        ]
        if (len(numeric_codes) + len(alpha2_codes) + len(alpha3_codes)) != len(
            country_codes
        ):
            raise ValueError("invalid country codes")
        return pd.concat(
            [
                df[df["Numeric"].isin(numeric_codes)][
                    ["Numeric", "Country", "Continent"]
                ].rename(columns={"Numeric": "Code"}),
                df[df["Alpha-2 code"].isin(alpha2_codes)][
                    ["Alpha-2 code", "Country", "Continent"]
                ].rename(columns={"Alpha-2 code": "Code"}),
                df[df["Alpha-3 code"].isin(alpha3_codes)][
                    ["Alpha-3 code", "Country", "Continent"]
                ].rename(columns={"Alpha-3 code": "Code"}),
            ]
        ).set_index(keys="Code", drop=True, inplace=False)
