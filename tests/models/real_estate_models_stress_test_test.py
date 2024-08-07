"""Test asset impact calculations using pytest."""

import bz2
import json

import numpy as np
import pandas as pd
import pytest

import physrisk.data.static.world as wd
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel import calculation  # noqa: F401 ## Avoid circular imports
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.impact import calculate_impacts
from physrisk.kernel.impact_distrib import EmptyImpactDistrib
from physrisk.kernel.vulnerability_model import DictBasedVulnerabilityModels
from physrisk.vulnerability_models.real_estate_models import (
    CoolingModel,
    GenericTropicalCycloneModel,
    RealEstateCoastalInundationModel,
    RealEstateRiverineInundationModel,
)


@pytest.fixture
def load_assets():
    with bz2.open("./tests/api/housing_kaggle_spain.json.bz2") as f:
        houses = json.load(f)

    asset_df = pd.DataFrame(houses["items"])
    longitudes = asset_df.longitude
    latitudes = asset_df.latitude
    types = asset_df.type
    asset_names = asset_df.address
    asset_prices = asset_df.price

    countries, continents = wd.get_countries_and_continents(
        latitudes=latitudes, longitudes=longitudes
    )

    assets = [
        RealEstateAsset(latitude, longitude, type=type_, location="Europe")
        for latitude, longitude, type_ in zip(latitudes, longitudes, types)
    ]

    for i, asset_name in enumerate(asset_names):
        assets[i].__dict__.update({"asset_name": asset_name})

    for i, asset_price in enumerate(asset_prices):
        assets[i].__dict__.update({"asset_price": asset_price})

    entrada = np.random.choice(
        a=[10, 20, 30], size=len(asset_prices), p=[0.1, 0.2, 0.7]
    )
    loan_amounts = (1 - entrada / 100) * asset_prices.to_numpy().astype(float)

    for i, loan_amount in enumerate(loan_amounts):
        assets[i].__dict__.update({"asset_loan_amount": loan_amount})

    return assets


@pytest.fixture
def hazard_indicator_dict():
    return {
        "Wind": "max_speed",
        "CoastalInundation": "flood_depth",
        "RiverineInundation": "flood_depth",
        "ChronicHeat": '"mean_degree_days_above',
    }


@pytest.fixture
def vulnerability_models_dict():
    return {
        "ssp119_2050": [GenericTropicalCycloneModel()],
        "ssp126_2030": [CoolingModel()],
        "ssp126_2040": [CoolingModel()],
        "ssp126_2050": [CoolingModel()],
        "ssp245_2030": [CoolingModel()],
        "ssp245_2040": [CoolingModel()],
        "ssp245_2050": [CoolingModel(), GenericTropicalCycloneModel()],
        "ssp585_2030": [CoolingModel()],
        "ssp585_2040": [CoolingModel()],
        "ssp585_2050": [CoolingModel(), GenericTropicalCycloneModel()],
    }


@pytest.fixture
def rcp_vulnerability_models_dict():
    return {
        "rcp4p5_2030": [
            RealEstateCoastalInundationModel(),
            RealEstateRiverineInundationModel(),
        ],
        "rcp4p5_2050": [
            RealEstateCoastalInundationModel(),
            RealEstateRiverineInundationModel(),
        ],
        "rcp4p5_2080": [
            RealEstateCoastalInundationModel(),
            RealEstateRiverineInundationModel(),
        ],
        "rcp8p5_2030": [
            RealEstateCoastalInundationModel(),
            RealEstateRiverineInundationModel(),
        ],
        "rcp8p5_2050": [
            RealEstateCoastalInundationModel(),
            RealEstateRiverineInundationModel(),
        ],
        "rcp8p5_2080": [
            RealEstateCoastalInundationModel(),
            RealEstateRiverineInundationModel(),
        ],
    }


@pytest.mark.skip(reason="Requires credentials.")
def test_calculate_impacts(
    load_assets,
    hazard_indicator_dict,
    vulnerability_models_dict,
    rcp_vulnerability_models_dict,
    load_credentials,
):
    assets = load_assets

    reader = ZarrReader()

    out = []
    empty_impact_count = 0
    asset_subtype_none_count = 0
    empty_impact_scenarios = []
    asset_subtype_none_assets = []
    exception_scenarios = []

    for scenario_year, vulnerability_models in vulnerability_models_dict.items():
        scenario, year = scenario_year.split("_")

        vulnerability_models = DictBasedVulnerabilityModels(
            {RealEstateAsset: vulnerability_models}
        )

        hazard_model = ZarrHazardModel(
            source_paths=get_default_source_paths(), reader=reader
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

            if hazard_type is None:
                indicator_id = None
            else:
                indicator_id = hazard_indicator_dict.get(hazard_type)

            asset_subtype = result.asset.type if hasattr(result.asset, "type") else None
            if asset_subtype is None:
                asset_subtype_none_count += 1
                asset_subtype_none_assets.append(result.asset.asset_name)

            out.append(
                {
                    "asset_name": result.asset.asset_name,
                    "asset_price": result.asset.asset_price,
                    "asset_loan_amount": result.asset.asset_loan_amount,
                    "asset_subtype": asset_subtype,
                    "latitude": result.asset.latitude,
                    "longitude": result.asset.longitude,
                    "hazard_type": hazard_type,
                    "indicator_id": indicator_id,
                    "display_name": "display_name_vacio",
                    "model": "model_vacio",
                    "scenario": scenario,
                    "year": int(year),
                    "return_periods": {"0": "0"},
                    "parameter": 0,
                    "impact_mean": impact_mean,
                    "impact_distr_bin_edges": impact_distr_bin_edges,
                    "impact_distr_p": impact_distr_p,
                    "impact_exc_exceed_p": "0;0",
                    "impact_exc_values": "0;0",
                    "vuln_model_designer": "OS-C-RealEstate-LTV",
                }
            )

    for scenario_year, vulnerability_models in rcp_vulnerability_models_dict.items():
        scenario, year = scenario_year.split("_")

        vulnerability_models = DictBasedVulnerabilityModels(
            {RealEstateAsset: vulnerability_models}
        )

        hazard_model = ZarrHazardModel(
            source_paths=get_default_source_paths(), reader=reader
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

            if hazard_type is None:
                indicator_id = None
            else:
                indicator_id = hazard_indicator_dict.get(hazard_type)

            asset_subtype = result.asset.type if hasattr(result.asset, "type") else None
            if asset_subtype is None:
                asset_subtype_none_count += 1
                asset_subtype_none_assets.append(result.asset.asset_name)

            out.append(
                {
                    "asset_name": result.asset.asset_name,
                    "asset_price": result.asset.asset_price,
                    "asset_loan_amount": result.asset.asset_loan_amount,
                    "asset_subtype": asset_subtype,
                    "latitude": result.asset.latitude,
                    "longitude": result.asset.longitude,
                    "hazard_type": hazard_type,
                    "indicator_id": indicator_id,
                    "display_name": "display_name_vacio",
                    "model": "model_vacio",
                    "scenario": scenario,
                    "year": int(year),
                    "return_periods": {"0": "0"},
                    "parameter": 0,
                    "impact_mean": impact_mean,
                    "impact_distr_bin_edges": impact_distr_bin_edges,
                    "impact_distr_p": impact_distr_p,
                    "impact_exc_exceed_p": "0;0",
                    "impact_exc_values": "0;0",
                    "vuln_model_designer": "OS-C-RealEstate-LTV",
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
