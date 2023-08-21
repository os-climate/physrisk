""" Test asset impact calculations."""
import os
import unittest
from test.base_test import TestWithCredentials
from typing import List

import numpy as np

import physrisk.api.v1.common
import physrisk.data.static.world as wd
from physrisk.kernel import Asset, PowerGeneratingAsset
from physrisk.kernel.assets import IndustrialActivity, RealEstateAsset
from physrisk.kernel.hazard_model import HazardEventDataResponse
from physrisk.kernel.impact import calculate_impacts
from physrisk.utils.lazy import lazy_import
from physrisk.vulnerability_models.power_generating_asset_models import InundationModel

pd = lazy_import("pandas")


class TestPowerGeneratingAssetModels(TestWithCredentials):
    """Tests World Resource Institute (WRI) models for power generating assets."""

    def test_innundation(self):
        # exceedance curve
        return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
        base_depth = np.array(
            [0.0, 0.22372675, 0.3654859, 0.5393629, 0.6642473, 0.78564394, 0.9406518, 1.0539534, 1.1634114]
        )
        future_depth = np.array(
            [0.059601218, 0.33267087, 0.50511575, 0.71471703, 0.8641244, 1.0032823, 1.1491022, 1.1634114, 1.1634114]
        )

        # we mock the response of the data request
        responses_mock = [
            HazardEventDataResponse(return_periods, base_depth),
            HazardEventDataResponse(return_periods, future_depth),
        ]

        latitude, longitude = 45.268405, 19.885738
        assets = [Asset(latitude, longitude)]
        model = InundationModel(assets)

        impact, vul, event = model.get_impact_details(assets[0], responses_mock)
        mean = impact.mean_impact()

        self.assertAlmostEqual(mean, 4.8453897 / 365.0)

    @unittest.skip("example, not test")
    def test_create_synthetic_portfolios_and_test(self):
        # cache_folder = r"<cache folder>"

        cache_folder = r"/users/joemoorhouse/code/data"

        asset_list = pd.read_csv(os.path.join(cache_folder, "wri-all.csv"))
        # types = asset_list["primary_fuel"].unique()
        # interesting = [3, 8, 13, 14, 22, 25, 27, 28, 33, 40, 51, 64, 65, 66, 71, 72, 80, 88, 92, 109]

        filtered = asset_list[0:1000]

        longitudes = np.array(filtered["longitude"])
        latitudes = np.array(filtered["latitude"])
        primary_fuel = np.array(filtered["primary_fuel"])
        generation = np.array(filtered["estimated_generation_gwh"])

        _, continents = wd.get_countries_and_continents(latitudes=latitudes, longitudes=longitudes)

        # Power generating assets that are of interest
        assets = [
            PowerGeneratingAsset(lat, lon, generation=gen, primary_fuel=prim_fuel, location=continent, type=prim_fuel)
            for lon, lat, gen, prim_fuel, continent in zip(longitudes, latitudes, generation, primary_fuel, continents)
        ]
        detailed_results = calculate_impacts(assets, scenario="ssp585", year=2030)
        keys = list(detailed_results.keys())
        # detailed_results[keys[0]].impact.to_exceedance_curve()
        means = np.array([detailed_results[key].impact.mean_impact() for key in keys])
        interesting = [k for (k, m) in zip(keys, means) if m > 0]
        assets_out = self.api_assets(item[0] for item in interesting[0:10])
        with open(os.path.join(cache_folder, "assets_example_power_generating_small.json"), "w") as f:
            f.write(assets_out.json(indent=4))

        # Synthetic portfolio; industrial activity at different locations
        assets = [
            IndustrialActivity(lat, lon, type="Construction", location=continent)
            for lon, lat, continent in zip(longitudes, latitudes, continents)
        ]
        assets = [assets[i] for i in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]]
        detailed_results = calculate_impacts(assets, scenario="ssp585", year=2030)
        keys = list(detailed_results.keys())
        means = np.array([detailed_results[key].impact.mean_impact() for key in detailed_results.keys()])
        interesting = [k for (k, m) in zip(keys, means) if m > 0]
        assets_out = self.api_assets(item[0] for item in interesting[0:10])
        with open(os.path.join(cache_folder, "assets_example_industrial_activity_small.json"), "w") as f:
            f.write(assets_out.json(indent=4))

        # Synthetic portfolio; real estate assets at different locations
        assets = [
            RealEstateAsset(lat, lon, location=continent, type="Buildings/Industrial")
            for lon, lat, continent in zip(longitudes, latitudes, continents)
            if isinstance(continent, str) and continent != "Oceania"
        ]
        detailed_results = calculate_impacts(assets, scenario="ssp585", year=2030)
        keys = list(detailed_results.keys())
        means = np.array([detailed_results[key].impact.mean_impact() for key in detailed_results.keys()])
        interesting = [k for (k, m) in zip(keys, means) if m > 0]
        assets_out = self.api_assets(item[0] for item in interesting[0:10])
        with open(os.path.join(cache_folder, "assets_example_real_estate_small.json"), "w") as f:
            f.write(assets_out.json(indent=4))
        self.assertAlmostEqual(1, 1)

    def api_assets(self, assets: List[Asset]):
        items = [
            physrisk.api.v1.common.Asset(
                asset_class=type(a).__name__,
                type=getattr(a, "type") if hasattr(a, "type") else None,
                location=getattr(a, "location") if hasattr(a, "location") else None,
                latitude=a.latitude,
                longitude=a.longitude,
            )
            for a in assets
        ]
        return physrisk.api.v1.common.Assets(items=items)
