import importlib.resources
from typing import Sequence, Union

import geopandas as gpd
import pandas as pd

import physrisk.data.ne_110m_admin_0_countries


class Geocoder:
    def __init__(self):
        with importlib.resources.path(
            physrisk.data.ne_110m_admin_0_countries, "ne_110m_admin_0_countries.shp"
        ) as p:
            world_orig = gpd.read_file(p)
            self.crs = world_orig.crs
            self.world = world_orig.to_crs(crs=3857)

    def get_countries(self, latitudes: Sequence[float], longitudes: Sequence[float]):
        gdf = gpd.GeoDataFrame(
            crs=self.crs, geometry=gpd.points_from_xy(longitudes, latitudes)
        ).to_crs(crs=3857)
        result = gpd.sjoin_nearest(gdf, self.world, how="left")
        return result[
            "ISO_A2_EH"
        ].values  # ISO_A2 contains -99 https://github.com/nvkelso/natural-earth-vector/issues/268

    @staticmethod
    def get_continent_and_country_from_code_iso_3166(
        country_codes: Sequence[Union[str, int]],
    ) -> pd.DataFrame:
        with importlib.resources.path(
            physrisk.data.ne_110m_admin_0_countries, "country_codes.tsv"
        ) as path:
            df = pd.read_csv(path, sep="\t")
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
