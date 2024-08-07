"""Test asset impact calculations."""

import os
from typing import List

import numpy as np
import pandas as pd
import pytest

import physrisk.api.v1.common
import physrisk.data.static.world as wd
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import (
    CoreFloodModels,
    CoreInventorySourcePaths,
)
from physrisk.kernel import calculation
from physrisk.kernel.assets import (
    Asset,
    IndustrialActivity,
    PowerGeneratingAsset,
    RealEstateAsset,
    ThermalPowerGeneratingAsset,
)
from physrisk.kernel.hazard_model import HazardEventDataResponse
from physrisk.kernel.impact import calculate_impacts
from physrisk.kernel.impact_distrib import EmptyImpactDistrib
from physrisk.kernel.vulnerability_model import DictBasedVulnerabilityModels
from physrisk.vulnerability_models.power_generating_asset_models import InundationModel
from physrisk.vulnerability_models.thermal_power_generation_models import (
    ThermalPowerGenerationAirTemperatureModel,
    ThermalPowerGenerationAqueductWaterRiskModel,
    ThermalPowerGenerationCoastalInundationModel,
    ThermalPowerGenerationDroughtModel,
    ThermalPowerGenerationHighFireModel,
    ThermalPowerGenerationLandslideModel,
    ThermalPowerGenerationRiverineInundationModel,
    ThermalPowerGenerationSevereConvectiveWindstormModel,
    ThermalPowerGenerationSubsidenceModel,
    ThermalPowerGenerationWaterStressModel,
    ThermalPowerGenerationWaterTemperatureModel,
)


@pytest.fixture
def setup_data(wri_power_plant_assets):
    """Fixture to set up the test data."""
    asset_list = wri_power_plant_assets

    filtered = asset_list.loc[
        asset_list["primary_fuel"].isin(["Coal", "Gas", "Nuclear", "Oil"])
    ]
    filtered = filtered[filtered["latitude"] > -60]

    longitudes = np.array(filtered["longitude"])
    latitudes = np.array(filtered["latitude"])
    primary_fuels = np.array(
        [
            primary_fuel.replace(" and ", "And").replace(" ", "")
            for primary_fuel in filtered["primary_fuel"]
        ]
    )
    capacities = np.array(filtered["capacity_mw"])
    asset_names = np.array(filtered["name"])

    countries, continents = wd.get_countries_and_continents(
        latitudes=latitudes, longitudes=longitudes
    )

    assets = [
        ThermalPowerGeneratingAsset(
            latitude,
            longitude,
            type=primary_fuel,
            location=continent,
            capacity=capacity,
        )
        for latitude, longitude, capacity, primary_fuel, continent in zip(
            latitudes,
            longitudes,
            capacities,
            primary_fuels,
            continents,
        )
    ]

    for i, asset_name in enumerate(asset_names):
        assets[i].__dict__.update({"asset_name": asset_name})

    return assets


@pytest.fixture
def setup_assets_extra(wri_power_plant_assets):
    asset_list = wri_power_plant_assets
    filtered = asset_list.loc[
        asset_list["primary_fuel"].isin(["Coal", "Gas", "Nuclear", "Oil"])
    ]
    filtered = filtered[-60 < filtered["latitude"]]

    longitudes = np.array(filtered["longitude"])
    latitudes = np.array(filtered["latitude"])
    primary_fuels = np.array(
        [
            primary_fuel.replace(" and ", "And").replace(" ", "")
            for primary_fuel in filtered["primary_fuel"]
        ]
    )
    capacities = np.array(filtered["capacity_mw"])

    countries, continents = wd.get_countries_and_continents(
        latitudes=latitudes, longitudes=longitudes
    )

    assets = [
        ThermalPowerGeneratingAsset(
            latitude,
            longitude,
            type=primary_fuel,
            location=country,
            capacity=capacity,
        )
        for latitude, longitude, capacity, primary_fuel, country in zip(
            latitudes,
            longitudes,
            capacities,
            primary_fuels,
            countries,
        )
        if country in ["Spain"]
    ]

    return assets


@pytest.fixture
def hazard_indicator_dict():
    """Fixture for hazard indicator dictionary."""
    return {
        "AirTemperature": "days_tas_above",
        "CoastalInundation": "flood_depth",
        "RiverineInundation": "flood_depth",
        "Drought": "months_spei3m_below_minus2",
        "WaterStress": "water_stress_and_water_supply",
        "WaterTemperature": "weeks_water_temp_above",
    }


@pytest.fixture
def vulnerability_models_dict():
    """Fixture for vulnerability models dictionary."""
    return {
        "historical_1980": [
            ThermalPowerGenerationRiverineInundationModel(),
            ThermalPowerGenerationCoastalInundationModel(),
        ],
        "historical_2005": [ThermalPowerGenerationAirTemperatureModel()],
        "ssp126_2030": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
        ],
        "ssp126_2040": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
        ],
        "ssp126_2050": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
        ],
        "ssp126_2060": [ThermalPowerGenerationWaterTemperatureModel()],
        "ssp126_2070": [ThermalPowerGenerationWaterTemperatureModel()],
        "ssp126_2080": [
            ThermalPowerGenerationWaterTemperatureModel(),
        ],
        "ssp126_2090": [ThermalPowerGenerationWaterTemperatureModel()],
        "ssp245_2030": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "ssp245_2040": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
        ],
        "ssp245_2050": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
            ThermalPowerGenerationRiverineInundationModel(),
            ThermalPowerGenerationCoastalInundationModel(),
        ],
        "ssp245_2060": [ThermalPowerGenerationWaterTemperatureModel()],
        "ssp245_2070": [ThermalPowerGenerationWaterTemperatureModel()],
        "ssp245_2080": [
            ThermalPowerGenerationWaterTemperatureModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "ssp245_2090": [ThermalPowerGenerationWaterTemperatureModel()],
        "ssp585_2005": [ThermalPowerGenerationDroughtModel()],
        "ssp585_2030": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationDroughtModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "ssp585_2040": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationDroughtModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
        ],
        "ssp585_2050": [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationDroughtModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "ssp585_2060": [ThermalPowerGenerationWaterTemperatureModel()],
        "ssp585_2070": [ThermalPowerGenerationWaterTemperatureModel()],
        "ssp585_2080": [
            ThermalPowerGenerationWaterTemperatureModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
            ThermalPowerGenerationDroughtModel(),
        ],
        "ssp585_2090": [ThermalPowerGenerationWaterTemperatureModel()],
    }


@pytest.fixture
def vulnerability_models_dict_water_stress():
    """Fixture for vulnerability models dictionary."""
    return {
        "historical_1999": [ThermalPowerGenerationWaterStressModel()],
        "ssp126_2030": [
            ThermalPowerGenerationWaterStressModel(),
        ],
        "ssp126_2050": [
            ThermalPowerGenerationWaterStressModel(),
        ],
        "ssp126_2080": [
            ThermalPowerGenerationWaterStressModel(),
        ],
        "ssp370_2030": [ThermalPowerGenerationWaterStressModel()],
        "ssp370_2050": [ThermalPowerGenerationWaterStressModel()],
        "ssp370_2080": [ThermalPowerGenerationWaterStressModel()],
        "ssp585_2030": [
            ThermalPowerGenerationWaterStressModel(),
        ],
        "ssp585_2050": [
            ThermalPowerGenerationWaterStressModel(),
        ],
        "ssp585_2080": [
            ThermalPowerGenerationWaterStressModel(),
        ],
    }


@pytest.fixture
def vul_models_dict_extra():
    return {
        "historical_1971": [
            ThermalPowerGenerationHighFireModel(),
            ThermalPowerGenerationSevereConvectiveWindstormModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "historical_1980": [
            ThermalPowerGenerationLandslideModel(),
            ThermalPowerGenerationSubsidenceModel(),
        ],
        "ssp126_2030": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "ssp126_2050": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "ssp126_2080": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "ssp370_2030": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "ssp370_2050": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "ssp370_2080": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "rcp45_2050": [
            ThermalPowerGenerationHighFireModel(),
            ThermalPowerGenerationSevereConvectiveWindstormModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "rcp45_2070": [
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "rcp45_2100": [
            ThermalPowerGenerationHighFireModel(),
            ThermalPowerGenerationSevereConvectiveWindstormModel(),
        ],
        "ssp585_2030": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "ssp585_2050": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "ssp585_2080": [ThermalPowerGenerationAqueductWaterRiskModel()],
        "rcp85_2050": [
            ThermalPowerGenerationHighFireModel(),
            ThermalPowerGenerationSevereConvectiveWindstormModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "rcp85_2070": [
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationRiverineInundationModel(),
        ],
        "rcp85_2100": [
            ThermalPowerGenerationHighFireModel(),
            ThermalPowerGenerationSevereConvectiveWindstormModel(),
        ],
    }


def api_assets(assets: List[Asset]):
    items = [
        physrisk.api.v1.common.Asset(
            asset_class=type(a).__name__,
            type=getattr(a, "type", None),
            location=getattr(a, "location", None),
            latitude=a.latitude,
            longitude=a.longitude,
        )
        for a in assets
    ]
    return physrisk.api.v1.common.Assets(items=items)


def test_inundation():
    # exceedance curve
    return_periods = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
    base_depth = np.array(
        [
            0.0,
            0.22372675,
            0.3654859,
            0.5393629,
            0.6642473,
            0.78564394,
            0.9406518,
            1.0539534,
            1.1634114,
        ]
    )
    future_depth = np.array(
        [
            0.059601218,
            0.33267087,
            0.50511575,
            0.71471703,
            0.8641244,
            1.0032823,
            1.1491022,
            1.1634114,
            1.1634114,
        ]
    )

    # Mock the response of the data request
    responses_mock = [
        HazardEventDataResponse(return_periods, base_depth),
        HazardEventDataResponse(return_periods, future_depth),
    ]

    latitude, longitude = 45.268405, 19.885738
    assets = [Asset(latitude, longitude)]
    model = InundationModel(assets)

    impact, _, _ = model.get_impact_details(assets[0], responses_mock)
    mean = impact.mean_impact()

    assert np.isclose(mean, 4.8453897 / 365.0)


@pytest.mark.skip(reason="Requires credentials.")
def test_create_synthetic_portfolios_and_test(
    wri_power_plant_assets, test_dir, load_credentials
):
    asset_list = wri_power_plant_assets
    filtered = asset_list[0:1000]

    longitudes = np.array(filtered["longitude"])
    latitudes = np.array(filtered["latitude"])
    primary_fuel = np.array(filtered["primary_fuel"])
    generation = np.array(filtered["estimated_generation_gwh_2017"])

    _, continents = wd.get_countries_and_continents(
        latitudes=latitudes, longitudes=longitudes
    )

    hazard_model = calculation.get_default_hazard_model()
    vulnerability_models = DictBasedVulnerabilityModels(
        calculation.get_default_vulnerability_models()
    )

    # Power generating assets that are of interest
    assets = [
        PowerGeneratingAsset(
            lat, lon, generation=gen, location=continent, type=prim_fuel
        )
        for lon, lat, gen, prim_fuel, continent in zip(
            longitudes, latitudes, generation, primary_fuel, continents
        )
    ]
    detailed_results = calculate_impacts(
        assets,
        hazard_model=hazard_model,
        vulnerability_models=vulnerability_models,
        scenario="ssp585",
        year=2030,
    )
    keys = list(detailed_results.keys())
    means = np.array([detailed_results[key][0].impact.mean_impact() for key in keys])
    interesting = [k for (k, m) in zip(keys, means) if m > 0]
    assets_out = api_assets(item[0] for item in interesting[0:10])
    with open(
        os.path.join(test_dir, "assets_example_power_generating_small.json"), "w"
    ) as f:
        f.write(assets_out.model_dump_json(indent=4))

    # Synthetic portfolio; industrial activity at different locations
    assets = [
        IndustrialActivity(lat, lon, type="Construction", location=continent)
        for lon, lat, continent in zip(longitudes, latitudes, continents)
    ]
    assets = [assets[i] for i in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]]
    detailed_results = calculate_impacts(
        assets,
        hazard_model=hazard_model,
        vulnerability_models=vulnerability_models,
        scenario="ssp585",
        year=2030,
    )
    keys = list(detailed_results.keys())
    means = np.array(
        [
            detailed_results[key][0].impact.mean_impact()
            for key in detailed_results.keys()
        ]
    )
    interesting = [k for (k, m) in zip(keys, means) if m > 0]
    assets_out = api_assets(item[0] for item in interesting[0:10])
    with open(
        os.path.join(test_dir, "assets_example_industrial_activity_small.json"), "w"
    ) as f:
        f.write(assets_out.model_dump_json(indent=4))

    # Synthetic portfolio; real estate assets at different locations
    assets = [
        RealEstateAsset(lat, lon, location=continent, type="Buildings/Industrial")
        for lon, lat, continent in zip(longitudes, latitudes, continents)
        if isinstance(continent, str) and continent != "Oceania"
    ]
    detailed_results = calculate_impacts(
        assets,
        hazard_model=hazard_model,
        vulnerability_models=vulnerability_models,
        scenario="ssp585",
        year=2030,
    )
    keys = list(detailed_results.keys())
    means = np.array(
        [
            detailed_results[key][0].impact.mean_impact()
            if detailed_results[key]
            else None
            for key in detailed_results.keys()
        ]
    )
    interesting = [k for (k, m) in zip(keys, means) if (m and m > 0)]
    assets_out = api_assets(item[0] for item in interesting[0:10])
    with open(
        os.path.join(test_dir, "assets_example_real_estate_small.json"), "w"
    ) as f:
        f.write(assets_out.model_dump_json(indent=4))
    assert len(os.listdir(test_dir)) == 3


@pytest.mark.skip(reason="Requires credentials.")
def test_thermal_power_generation_portfolio(
    wri_power_plant_assets, test_dir, load_credentials
):
    asset_list = wri_power_plant_assets

    filtered = asset_list.loc[
        asset_list["primary_fuel"].isin(["Coal", "Gas", "Nuclear", "Oil"])
    ]
    filtered = filtered[-60 < filtered["latitude"]]

    longitudes = np.array(filtered["longitude"])
    latitudes = np.array(filtered["latitude"])

    primary_fuels = np.array(
        [
            primary_fuel.replace(" and ", "And").replace(" ", "")
            for primary_fuel in filtered["primary_fuel"]
        ]
    )

    # Capacity describes a maximum electric power rate.
    # Generation describes the actual electricity output of the plant over a period of time.
    capacities = np.array(filtered["capacity_mw"])

    _, continents = wd.get_countries_and_continents(
        latitudes=latitudes, longitudes=longitudes
    )

    # Power generating assets that are of interest
    assets = [
        ThermalPowerGeneratingAsset(
            latitude,
            longitude,
            type=primary_fuel,
            location=continent,
            capacity=capacity,
        )
        for latitude, longitude, capacity, primary_fuel, continent in zip(
            latitudes,
            longitudes,
            capacities,
            primary_fuels,
            continents,
        )
    ]

    scenario = "ssp585"
    year = 2030

    hazard_model = calculation.get_default_hazard_model()
    vulnerability_models = DictBasedVulnerabilityModels(
        calculation.get_default_vulnerability_models()
    )

    results = calculate_impacts(
        assets, hazard_model, vulnerability_models, scenario=scenario, year=year
    )
    out = [
        {
            "asset": type(result.asset).__name__,
            "type": getattr(result.asset, "type", None),
            "capacity": getattr(result.asset, "capacity", None),
            "location": getattr(result.asset, "location", None),
            "latitude": result.asset.latitude,
            "longitude": result.asset.longitude,
            "impact_mean": (
                None
                if isinstance(results[key][0].impact, EmptyImpactDistrib)
                else results[key][0].impact.mean_impact()
            ),
            "hazard_type": key.hazard_type.__name__,
        }
        for result, key in zip(results, results.keys())
    ]
    pd.DataFrame.from_dict(out).to_csv(
        os.path.join(
            test_dir, f"thermal_power_generation_example_{scenario}_{year}.csv"
        )
    )
    assert len(out) == 53052
    assert len(os.listdir(test_dir)) == 1


@pytest.mark.skip(reason="Requires credentials.")
def test_thermal_power_generation_impacts(
    setup_data, hazard_indicator_dict, vulnerability_models_dict, load_credentials
):
    assets = setup_data
    hazard_model = calculation.get_default_hazard_model()
    out = []
    empty_impact_count = 0
    asset_subtype_none_count = 0
    empty_impact_scenarios = []
    asset_subtype_none_assets = []
    exception_scenarios = []
    for scenario_year, vulnerability_models in vulnerability_models_dict.items():
        scenario, year = scenario_year.split("_")

        vulnerability_models = DictBasedVulnerabilityModels(
            {ThermalPowerGeneratingAsset: vulnerability_models}
        )

        try:
            results = calculate_impacts(
                assets,
                hazard_model,
                vulnerability_models,
                scenario=scenario,
                year=int(year),
            )
        except Exception as e:
            exception_scenarios.append((scenario, year, str(e)))
            continue

        for result, key in zip(results, results.keys()):
            impact = results[key][0].impact
            if isinstance(impact, EmptyImpactDistrib):
                impact_mean = None
                hazard_type = None
                impact_distr_bin_edges = "0;0"
                impact_distr_p = "0;0"
                empty_impact_count += 1
                empty_impact_scenarios.append((scenario, year, result.asset.asset_name))
            else:
                impact_mean = impact.mean_impact()
                impact_distr_bin_edges = ";".join(impact.impact_bins.astype(str))
                impact_distr_p = ";".join(impact.prob.astype(str))
                hazard_type = (
                    impact.hazard_type.__name__
                    if impact.hazard_type.__name__ != "type"
                    else "Wind"
                )

            indicator_id = (
                hazard_indicator_dict.get(hazard_type) if hazard_type else None
            )
            asset_subtype = result.asset.type if hasattr(result.asset, "type") else None
            if asset_subtype is None:
                asset_subtype_none_count += 1
                asset_subtype_none_assets.append(result.asset.asset_name)

            out.append(
                {
                    "asset_name": result.asset.asset_name,
                    "latitude": result.asset.latitude,
                    "longitude": result.asset.longitude,
                    "hazard_type": hazard_type,
                    "indicator_id": indicator_id,
                    "scenario": scenario,
                    "year": int(year),
                    "impact_mean": impact_mean,
                    "impact_distr_bin_edges": impact_distr_bin_edges,
                    "impact_distr_p": impact_distr_p,
                }
            )

    # Out can be used when dealing with expected values.

    # Assert the counts and details for empty impacts, None asset_subtype, and exceptions

    assert (
        empty_impact_count == 0
    ), f"Found {empty_impact_count} EmptyImpactDistrib instances in scenarios: {empty_impact_scenarios}"
    assert (
        asset_subtype_none_count == 0
    ), f"Found {asset_subtype_none_count} assets with None asset_subtype: {asset_subtype_none_assets}"
    assert (
        not exception_scenarios
    ), f"Exceptions occurred in scenarios: {exception_scenarios}"


@pytest.mark.skip(reason="Requires credentials.")
def test_thermal_power_generation_impacts_water_stress(
    setup_data,
    hazard_indicator_dict,
    vulnerability_models_dict_water_stress,
    load_credentials,
):
    assets = setup_data
    hazard_model = calculation.get_default_hazard_model()
    out = []
    empty_impact_count = 0
    asset_subtype_none_count = 0
    empty_impact_scenarios = []
    asset_subtype_none_assets = []
    exception_scenarios = []
    for (
        scenario_year,
        vulnerability_models,
    ) in vulnerability_models_dict_water_stress.items():
        scenario, year = scenario_year.split("_")

        vulnerability_models = DictBasedVulnerabilityModels(
            {ThermalPowerGeneratingAsset: vulnerability_models}
        )

        try:
            results = calculate_impacts(
                assets,
                hazard_model,
                vulnerability_models,
                scenario=scenario,
                year=int(year),
            )
        except Exception as e:
            exception_scenarios.append((scenario, year, str(e)))
            continue

        for result, key in zip(results, results.keys()):
            impact = results[key][0].impact
            if isinstance(impact, EmptyImpactDistrib):
                impact_mean = None
                hazard_type = None
                impact_distr_bin_edges = "0;0"
                impact_distr_p = "0;0"
                empty_impact_count += 1
                empty_impact_scenarios.append((scenario, year, result.asset.asset_name))
            else:
                impact_mean = impact.mean_impact()
                impact_distr_bin_edges = ";".join(impact.impact_bins.astype(str))
                impact_distr_p = ";".join(impact.prob.astype(str))
                hazard_type = (
                    impact.hazard_type.__name__
                    if impact.hazard_type.__name__ != "type"
                    else "Wind"
                )

            indicator_id = (
                hazard_indicator_dict.get(hazard_type) if hazard_type else None
            )
            asset_subtype = result.asset.type if hasattr(result.asset, "type") else None
            if asset_subtype is None:
                asset_subtype_none_count += 1
                asset_subtype_none_assets.append(result.asset.asset_name)

            out.append(
                {
                    "asset_name": result.asset.asset_name,
                    "latitude": result.asset.latitude,
                    "longitude": result.asset.longitude,
                    "hazard_type": hazard_type,
                    "indicator_id": indicator_id,
                    "scenario": scenario,
                    "year": int(year),
                    "impact_mean": impact_mean,
                    "impact_distr_bin_edges": impact_distr_bin_edges,
                    "impact_distr_p": impact_distr_p,
                }
            )

    # Out can be used when dealing with expected values.

    # Assert the counts and details for empty impacts, None asset_subtype, and exceptions

    # There are 57 assets that are EmptyImpactDistrib objects, all for ThermalPowerGenerationWaterStressModel, which is used 10 times
    assert (
        empty_impact_count == 570
    ), f"Found {empty_impact_count} EmptyImpactDistrib instances in scenarios: {empty_impact_scenarios}"
    assert (
        asset_subtype_none_count == 0
    ), f"Found {asset_subtype_none_count} assets with None asset_subtype: {asset_subtype_none_assets}"
    assert (
        not exception_scenarios
    ), f"Exceptions occurred in scenarios: {exception_scenarios}"


@pytest.mark.skip(reason="Requires credentials.")
def test_thermal_power_generation_impacts_extra(
    load_credentials, setup_assets_extra, vul_models_dict_extra
):
    """Calculate impacts for the vulnerability models from use case id STRESSTEST."""
    assets = setup_assets_extra

    out = []
    empty_impact_count = 0
    asset_subtype_none_count = 0
    empty_impact_scenarios = []
    asset_subtype_none_assets = []
    exception_scenarios = []

    for scenario_year, vulnerability_models in vul_models_dict_extra.items():
        scenario, year = scenario_year.split("_")

        print(scenario_year)

        if vulnerability_models is ThermalPowerGenerationAqueductWaterRiskModel():
            reader = ZarrReader()
        else:
            devaccess = {
                "OSC_S3_ACCESS_KEY": os.environ.get("OSC_S3_ACCESS_KEY_DEV", None),
                "OSC_S3_SECRET_KEY": os.environ.get("OSC_S3_SECRET_KEY_DEV", None),
                "OSC_S3_BUCKET": os.environ.get("OSC_S3_BUCKET_DEV", None),
            }
            get_env = devaccess.get
            reader = ZarrReader(get_env=get_env)

        vulnerability_models = DictBasedVulnerabilityModels(
            {ThermalPowerGeneratingAsset: vulnerability_models}
        )

        # Use TUDelft flood models.
        hazard_model = ZarrHazardModel(
            source_paths=CoreInventorySourcePaths(
                EmbeddedInventory(), flood_model=CoreFloodModels.TUDelft
            ).source_paths(),
            reader=reader,
        )

        try:
            results = calculate_impacts(
                assets,
                hazard_model,
                vulnerability_models,
                scenario=scenario,
                year=int(year),
            )
        except Exception as e:
            exception_scenarios.append((scenario, year, str(e)))
            continue

        for result, key in zip(results, results.keys()):
            impact = results[key][0].impact
            if isinstance(impact, EmptyImpactDistrib):
                impact_mean = None
                hazard_type = None
                empty_impact_count += 1
                empty_impact_scenarios.append((scenario, year, result.asset.location))
            else:
                impact_mean = impact.mean_impact()
                hazard_type = (
                    impact.hazard_type.__name__
                    if impact.hazard_type.__name__ != "type"
                    else "Wind"
                )

            asset_subtype = result.asset.type if hasattr(result.asset, "type") else None
            if asset_subtype is None:
                asset_subtype_none_count += 1
                asset_subtype_none_assets.append(result.asset.location)

            out.append(
                {
                    "asset": type(result.asset).__name__,
                    "type": getattr(result.asset, "type", None),
                    "location": getattr(result.asset, "location", None),
                    "latitude": result.asset.latitude,
                    "longitude": result.asset.longitude,
                    "impact_mean": impact_mean,
                    "hazard_type": hazard_type if hazard_type else "Wind",
                    "scenario": scenario,
                    "year": int(year),
                }
            )

    # out can be used when dealing with expected values.

    # Assert the counts and details for empty impacts, None asset_subtype, and exceptions
    assert (
        empty_impact_count == 0
    ), f"Found {empty_impact_count} EmptyImpactDistrib instances in scenarios: {empty_impact_scenarios}"
    assert (
        asset_subtype_none_count == 0
    ), f"Found {asset_subtype_none_count} assets with None asset_subtype: {asset_subtype_none_assets}"
    assert (
        not exception_scenarios
    ), f"Exceptions occurred in scenarios: {exception_scenarios}"
