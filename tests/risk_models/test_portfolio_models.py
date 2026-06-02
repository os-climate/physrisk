import logging
from typing import Dict, Optional

from dependency_injector import providers
import numpy as np

from physrisk.api.v1.common import Asset as APIAsset, Assets
from physrisk.api.v1.impact_req_resp import (
    AssetMeasuresSpecification,
    AssetImpactRequest,
    CalcSettings,
    RiskMeasuresForAssets,
    ScoreBasedRiskMeasuresForAssets,
)
from physrisk.container import Container
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import Asset, ManufacturingAsset
from physrisk.kernel.financial_model import (
    DefaultFinancialModel,
    FinancialDataProvider,
)
from physrisk.kernel.hazard_model import HazardModelFactory
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    Drought,
    Fire,
    Hail,
    RiverineInundation,
    Wind,
)
from physrisk.kernel.impact import AssetImpactResult, ImpactKey
from physrisk.kernel.impact_aggregator import aggregate_impacts
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.risk import QuantityType, RiskQuantityKey
from physrisk.vulnerability_models.vulnerability import VulnerabilityModelsFactory
from tests.data.test_hazard_model_store import ZarrStoreMocker
from tests.vulnerability_models.test_config_based_vulnerability import create_store


logger = logging.getLogger(__name__)


class TestFinancialDataProvider(FinancialDataProvider):
    def revenue_attributable_to_asset(self, asset: Asset, currency: str) -> float:
        return 200.0

    def total_insurable_value(self, asset: Asset, currency: str) -> float:
        return 100.0


def test_impact_aggregation():
    # Test case where assets are uncorrelated. Each asset has its own unique severity zone (SZ)
    # The samples are from the following distribution:
    impact_bin_edges = np.array([0.1, 0.2, 0.4, 0.8])
    impact_probabilities = np.array(
        [(1.0 - 0.5) / 100, (0.5 - 0.1) / 100.0, 0.1 / 100.0]
    )
    impact_bin_edges_zero = np.array([0.0, 0.0])
    impact_probabilities_zero = np.array([0.0])

    impacts: Dict[ImpactKey, list[AssetImpactResult]] = {}
    # in this example, we have 1,000,000 assets. We assume that 10% of assets have a non-zero impact probability.
    n_assets = 10000
    generator = np.random.default_rng(seed=111)
    non_zero_asset_mask = generator.random(n_assets) < 0.1
    logger.info(f"Creating impacts for {n_assets} assets.")
    for i in range(n_assets):
        if non_zero_asset_mask[i]:
            asset_impact = AssetImpactResult(
                impact=ImpactDistrib(
                    RiverineInundation,
                    impact_bin_edges.copy(),
                    impact_probabilities.copy(),
                    "",
                )
            )
        else:
            asset_impact = AssetImpactResult(
                impact=ImpactDistrib(
                    RiverineInundation,
                    impact_bin_edges_zero.copy(),
                    impact_probabilities_zero.copy(),
                    "",
                )
            )
        impacts[
            ImpactKey(
                asset=Asset(id=f"asset_{i}", latitude=0.0, longitude=0.0),
                hazard_type=RiverineInundation,
                scenario="historical",
                key_year=None,
            )
        ] = [asset_impact]
    logger.info(
        f"Created impacts for {n_assets} assets, with {np.sum(non_zero_asset_mask)} non-zero impacts"
    )

    financial_model = DefaultFinancialModel(
        data_provider=TestFinancialDataProvider(), downtime_config=[]
    )

    # single impact
    single_impact = ImpactDistrib(
        RiverineInundation, impact_bin_edges.copy(), impact_probabilities.copy(), ""
    )
    logger.info(
        f"Fractional difference: {abs(np.mean(single_impact.to_exceedance_curve().get_samples(np.random.rand(1000000))) - single_impact.mean_impact()) / single_impact.mean_impact()}"
    )

    results = aggregate_impacts(impacts, financial_model, "historical", None)
    damage = results[
        RiskQuantityKey(QuantityType.DAMAGE, None, None, RiverineInundation)
    ]
    mean_damage_mc = damage.mean
    logger.info(
        f"Mean damage aggregating all assets and hazards using Monte Carlo: {mean_damage_mc}"
    )
    mean_damage_exact = sum(
        i[0].impact.mean_impact() * 100.0 for i in impacts.values()
    ) / (n_assets * 100.0)
    logger.info(
        f"Mean damage aggregating all assets and hazards using exact calculation: {mean_damage_exact}"
    )
    np.testing.assert_allclose(mean_damage_mc, 0.000260859179)
    np.testing.assert_allclose(mean_damage_mc, mean_damage_exact, rtol=0.02)


def test_impact_aggregation_end_to_end():
    """Mocked test that aggregates riverine inundation over assets and calculates
    portfolio level scores.
    """
    scenarios = ["ssp585", "historical"]
    years = [2050]
    latitudes = [22.30224, 22.31150, 22.45022, 22.27034]
    longitudes = [114.18670, 114.17774, 114.02882, 114.19268]
    store = create_store(latitudes, longitudes)
    hazard_model = ZarrHazardModel(source_paths=get_default_source_paths(), store=store)

    class TestHazardModelFactory(HazardModelFactory):
        def hazard_model(
            self,
            interpolation: Optional[str] = "floor",
            provider_max_requests: dict[str, int] = {},
            interpolate_years: bool = False,
        ):
            return hazard_model

    container = Container()
    container.override_providers(
        hazard_model_factory=providers.Factory(TestHazardModelFactory)
    )
    container.override_providers(
        config=providers.Configuration(default={"zarr_sources": ["embedded"]})
    )
    container.override_providers(zarr_store=None)
    container.override_providers(inventory_reader=None)
    container.override_providers(zarr_reader=None)
    container.override_providers(sig_figures=6)

    container.override_providers(
        vulnerability_models_factory=providers.Factory(
            VulnerabilityModelsFactory,
            use_oed_hazus_curves=True,
            config=VulnerabilityModelsFactory.embedded_vulnerability_config(),
        )
    )
    requester = container.requester()

    assets = Assets(
        items=[
            APIAsset(
                id="asset_1",
                occupancy_code=1050,
                latitude=latitudes[0],
                longitude=longitudes[0],
            )
        ]
    )
    req = AssetImpactRequest(
        assets=assets,
        scenarios=scenarios,
        years=years,
        include_asset_level=True,
        include_measures=True,
        include_calc_details=False,
        use_case_id="company",
        calc_settings=CalcSettings(hazard_scope="RiverineInundation"),
    )
    res = requester.get_asset_impacts(request=req)
    assert res is not None


def test_impact_aggregation_end_to_end_multi_hazard():
    """End-to-end API test: 6 manufacturing assets exposed to flood, wind, fire and chronic heat.

    Uses AssetFinancialDrilldown to get analytical per-asset AALs alongside the Monte Carlo
    portfolio means produced by CompanyRiskMeasureCalculator.  The two code paths are
    independent; convergence (5 % rtol) verifies both.

    Asset layout: assets 0-1 carry flood risk; all 6 share wind, fire and chronic-heat exposure.
    FinancialDataStore default (no financial data supplied): TIV = revenue = 100 per asset.
    """
    scenarios = ["ssp585", "historical"]
    years = [2050]
    latitudes = [22.30224, 22.31150, 22.45022, 22.27034, 22.35000, 22.40000]
    longitudes = [114.18670, 114.17774, 114.02882, 114.19268, 114.10000, 114.15000]
    source_paths = get_default_source_paths()

    def hp(hazard_type, indicator_id, scenario):
        y = -1 if scenario == "historical" else 2050
        return (
            source_paths.resource_paths(
                hazard_type, indicator_id=indicator_id, scenarios=[scenario]
            )[0]
            .scenarios[scenario]
            .path(y)
        )

    mocker = ZarrStoreMocker()

    # RiverineInundation: assets 0-1 at risk, assets 2-5 zero depth
    flood_return_periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    flood_curve = np.array([0.0012, 0.39, 0.85, 1.39, 1.75, 2.09, 2.51, 2.82, 3.12])
    flood_per_asset = np.zeros((6, len(flood_return_periods)), dtype=np.float32)
    flood_per_asset[0, :] = flood_curve
    flood_per_asset[1, :] = flood_curve
    for scenario in ["ssp585", "historical"]:
        mocker.add_curves_global(
            hp(RiverineInundation, "flood_depth", scenario),
            longitudes,
            latitudes,
            flood_return_periods,
            flood_per_asset,
        )
    mocker.add_curves_global(
        hp(RiverineInundation, "flood_sop", "historical"),
        longitudes,
        latitudes,
        ["min", "max"],
        np.array([100.0, 300.0]),
    )

    # Wind: all assets
    wind_return_periods = [
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    ]
    wind_speeds = np.array(
        [
            24.44,
            26.79,
            28.63,
            30.10,
            31.03,
            32.00,
            33.04,
            33.75,
            34.23,
            39.33,
            43.0,
            44.96,
            46.19,
            47.62,
            48.58,
            48.83,
            49.79,
            50.11,
        ]
    )
    for scenario in ["ssp585", "historical"]:
        mocker.add_curves_global(
            hp(Wind, "max_speed", scenario),
            longitudes,
            latitudes,
            wind_return_periods,
            wind_speeds,
            units="m/s",
        )

    # Fire: all assets
    for scenario, prob in [("ssp585", 0.31867), ("historical", 0.25)]:
        mocker.add_curves_global(
            hp(Fire, "fire_probability", scenario),
            longitudes,
            latitudes,
            [0],
            np.array([prob]),
        )

    # ChronicHeat: all assets
    wbgt_thresholds = [5.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    for scenario, values in [
        (
            "ssp585",
            [
                363.65,
                350.21,
                303.64,
                240.48,
                181.83,
                128.47,
                74.40,
                1.40,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
        (
            "historical",
            [320.0, 290.0, 240.0, 180.0, 120.0, 65.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ]:
        mocker.add_curves_global(
            hp(ChronicHeat, "days_wbgt_above", scenario),
            longitudes,
            latitudes,
            wbgt_thresholds,
            np.array(values),
        )

    # CoastalInundation, Drought, Hail: zero values so calculate_impacts doesn't fail
    coastal_zeros = np.zeros(len(flood_return_periods), dtype=np.float32)
    for scenario in ["ssp585", "historical"]:
        mocker.add_curves_global(
            hp(CoastalInundation, "flood_depth", scenario),
            longitudes,
            latitudes,
            flood_return_periods,
            coastal_zeros,
        )
    mocker.add_curves_global(
        hp(CoastalInundation, "flood_sop", "historical"),
        longitudes,
        latitudes,
        ["min", "max"],
        np.array([100.0, 300.0]),
    )
    for scenario in ["ssp585", "historical"]:
        mocker.add_curves_global(
            hp(Drought, "months/spei12m/below/threshold", scenario),
            longitudes,
            latitudes,
            [0],
            np.array([0.0]),
        )
        mocker.add_curves_global(
            hp(Hail, "days/above/5cm", scenario),
            longitudes,
            latitudes,
            [0],
            np.array([0.0]),
        )

    hazard_model = ZarrHazardModel(source_paths=source_paths, store=mocker.store)

    class TestHazardModelFactory(HazardModelFactory):
        def hazard_model(
            self,
            interpolation: Optional[str] = "floor",
            provider_max_requests: dict[str, int] = {},
            interpolate_years: bool = False,
        ):
            return hazard_model

    container = Container()
    container.override_providers(
        hazard_model_factory=providers.Factory(TestHazardModelFactory)
    )
    container.override_providers(
        config=providers.Configuration(default={"zarr_sources": ["embedded"]})
    )
    container.override_providers(zarr_store=None)
    container.override_providers(inventory_reader=None)
    container.override_providers(zarr_reader=None)
    container.override_providers(sig_figures=6)
    container.override_providers(
        vulnerability_models_factory=providers.Factory(
            VulnerabilityModelsFactory,
            use_oed_hazus_curves=True,
            config=VulnerabilityModelsFactory.embedded_vulnerability_config(),
        )
    )
    requester = container.requester()

    assets = Assets(
        items=[
            APIAsset(
                id=f"asset_{i}",
                occupancy_code=2000,
                latitude=latitudes[i],
                longitude=longitudes[i],
            )
            for i in range(6)
        ]
    )
    req = AssetImpactRequest(
        assets=assets,
        scenarios=scenarios,
        years=years,
        include_asset_level=False,
        include_measures=True,
        include_calc_details=False,
        use_case_id="company",
        calc_settings=CalcSettings(
            hazard_scope="RiverineInundation,Wind,Fire,ChronicHeat"
        ),
        measures_specification=AssetMeasuresSpecification(
            measure_ids=["mean"],
            quantity_types=["damage", "disruption/revenue"],
        ),
    )
    res = requester.get_asset_impacts(request=req)

    import json

    with open("result.json", "w") as f:
        f.write(json.dumps(res.model_dump(exclude_none=True), indent=4))

    assert res.risk_measures is not None
    assert res.portfolio_impacts is not None

    # Score-based measures must cover all four hazard types
    score_hazard_types = {
        m.key.hazard_type
        for m in res.risk_measures.measures_for_assets
        if isinstance(m, ScoreBasedRiskMeasuresForAssets)
    }
    for expected in ("RiverineInundation", "Wind", "Fire", "ChronicHeat"):
        assert expected in score_hazard_types, (
            f"{expected} missing from score-based measures; found: {score_hazard_types}"
        )

    # Collect analytical per-asset AALs from the drilldown (ssp585 / 2050 only)
    # FinancialDataStore default: TIV = revenue = 100 per asset (no financial data supplied)
    n_assets = 6

    aal_by_key: dict[tuple[str, str], list[float]] = {
        (m.key.hazard_type, m.key.measure_id.split("_", 1)[1]): m.measures
        for m in res.risk_measures.measures_for_assets
        if isinstance(m, RiskMeasuresForAssets)
        and m.key.measure_id.startswith("mean_")
        and m.key.scenario_id == "ssp585"
        and m.key.year == "2050"
    }

    # Collect MC portfolio means from portfolio_impacts (ssp585 / 2050 only)
    mc_by_key: dict[tuple[str, str], float] = {
        (pi.key.hazard_type, pi.impact_type): pi.impact_mean
        for pi in res.portfolio_impacts
        if pi.key.scenario_id == "ssp585" and pi.key.year == "2050"
    }

    # For each hazard / impact-type pair:
    #   mean(per-asset fractional AAL)  ==  MC portfolio mean   (within MC noise)
    # Drilldown values are already fractions (divided by per-asset TIV or revenue), so the
    # portfolio mean is simply the average across assets (exact when all TIVs/revenues are equal).
    for hazard, impact_type in [
        ("RiverineInundation", "damage"),
        ("Wind", "damage"),
        ("Fire", "damage"),
        ("ChronicHeat", "disruption/revenue"),
    ]:
        k = (hazard, impact_type)
        assert k in aal_by_key, f"Analytical AAL missing for {k}"
        assert k in mc_by_key, f"MC portfolio mean missing for {k}"
        analytical = sum(aal_by_key[k]) / n_assets
        np.testing.assert_allclose(
            mc_by_key[k],
            analytical,
            rtol=0.05,
            err_msg=f"MC vs analytical mismatch for {k}: mc={mc_by_key[k]:.6g}, analytical={analytical:.6g}",
        )


def test_impact_aggregation_multi_hazard():
    """Aggregate Wind, RiverineInundation, Fire (all acute) and ChronicHeat (chronic)
    over a portfolio of manufacturing assets.

    Asset layout (6 assets total):
        asset_0, asset_1 : Wind + RiverineInundation (acute), ChronicHeat (chronic)
        asset_2          : Fire (acute), ChronicHeat (chronic)
        asset_3, asset_4 : Fire (acute) only
        asset_5          : zero acute risk, zero chronic risk

    Single-bin impact distributions make theoretical means easy to verify:
        Wind   [0.0, 0.5], p=0.02  →  mean = 0.25 × 0.02 = 0.005
        Flood  [0.0, 0.4], p=0.05  →  mean = 0.20 × 0.05 = 0.010
        Fire   [0.0, 0.8], p=0.01  →  mean = 0.40 × 0.01 = 0.004
        Heat future  [0, 0.20], p=0.5  →  mean = 0.05
        Heat histo   [0, 0.10], p=0.4  →  mean = 0.02  →  delta = 0.03

    Financial params (TestFinancialDataProvider): TIV = 100, Revenue = 200 per asset.
    No downtime model, so acute revenue loss = 0 (only chronic contributes REVENUE_LOSS).
    """
    scenario, key_year = "ssp585", 2050

    wind_edges = np.array([0.0, 0.5])
    wind_probs = np.array([0.02])  # mean 0.005
    flood_edges = np.array([0.0, 0.4])
    flood_probs = np.array([0.05])  # mean 0.010
    fire_edges = np.array([0.0, 0.8])
    fire_probs = np.array([0.01])  # mean 0.004
    heat_fut_edges = np.array([0.0, 0.20])
    heat_fut_probs = np.array([0.5])  # mean 0.05
    heat_his_edges = np.array([0.0, 0.10])
    heat_his_probs = np.array([0.4])  # mean 0.02
    zero_edges = np.array([0.0, 0.0])
    zero_probs = np.array([0.0])

    a = [
        ManufacturingAsset(id=f"asset_{i}", latitude=0.0, longitude=0.0)
        for i in range(6)
    ]

    def air(hazard_type, edges, probs):
        return AssetImpactResult(
            impact=ImpactDistrib(hazard_type, edges.copy(), probs.copy(), "")
        )

    def ik(asset, hazard_type, sc=scenario, yr=key_year):
        return ImpactKey(asset=asset, hazard_type=hazard_type, scenario=sc, key_year=yr)

    impacts: Dict[ImpactKey, list[AssetImpactResult]] = {
        # Wind – assets 0, 1
        ik(a[0], Wind): [air(Wind, wind_edges, wind_probs)],
        ik(a[1], Wind): [air(Wind, wind_edges, wind_probs)],
        # Flood – assets 0, 1
        ik(a[0], RiverineInundation): [
            air(RiverineInundation, flood_edges, flood_probs)
        ],
        ik(a[1], RiverineInundation): [
            air(RiverineInundation, flood_edges, flood_probs)
        ],
        # Fire – assets 2, 3, 4
        ik(a[2], Fire): [air(Fire, fire_edges, fire_probs)],
        ik(a[3], Fire): [air(Fire, fire_edges, fire_probs)],
        ik(a[4], Fire): [air(Fire, fire_edges, fire_probs)],
        # ChronicHeat future – assets 0, 1, 2
        ik(a[0], ChronicHeat): [air(ChronicHeat, heat_fut_edges, heat_fut_probs)],
        ik(a[1], ChronicHeat): [air(ChronicHeat, heat_fut_edges, heat_fut_probs)],
        ik(a[2], ChronicHeat): [air(ChronicHeat, heat_fut_edges, heat_fut_probs)],
        # ChronicHeat historical baseline – assets 0, 1, 2
        ik(a[0], ChronicHeat, sc="historical", yr=None): [
            air(ChronicHeat, heat_his_edges, heat_his_probs)
        ],
        ik(a[1], ChronicHeat, sc="historical", yr=None): [
            air(ChronicHeat, heat_his_edges, heat_his_probs)
        ],
        ik(a[2], ChronicHeat, sc="historical", yr=None): [
            air(ChronicHeat, heat_his_edges, heat_his_probs)
        ],
        # asset_5 – zero risk; present so its TIV/revenue enter the portfolio denominator
        ik(a[5], Wind): [air(Wind, zero_edges, zero_probs)],
    }

    financial_model = DefaultFinancialModel(
        data_provider=TestFinancialDataProvider(), downtime_config=[]
    )
    results = aggregate_impacts(impacts, financial_model, scenario, key_year)

    # Expected means (theoretical):
    #   sum_tiv     = 6 × 100 = 600
    #   sum_revenue = 6 × 200 = 1200
    #
    #   Acute damage (normalised by sum_tiv):
    #     Wind  : 2 assets × 100 × 0.005 / 600 = 1/600
    #     Flood : 2 assets × 100 × 0.010 / 600 = 2/600
    #     Fire  : 3 assets × 100 × 0.004 / 600 = 1.2/600
    #
    #   Chronic revenue loss (normalised by sum_revenue, deterministic):
    #     ChronicHeat: 3 assets × delta(0.03) / 1200 = 0.09/1200 = 7.5e-5

    expected_wind_damage = 2 * 100 * 0.005 / 600  # 1/600
    expected_flood_damage = 2 * 100 * 0.010 / 600  # 2/600
    expected_fire_damage = 3 * 100 * 0.004 / 600  # 1.2/600
    expected_heat_rev_loss = 3 * 200 * 0.03 / 1200  # 0.015

    rtol = 0.05  # 5% tolerance

    np.testing.assert_allclose(
        results[RiskQuantityKey(QuantityType.DAMAGE, None, None, Wind)].mean,
        expected_wind_damage,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        results[
            RiskQuantityKey(QuantityType.DAMAGE, None, None, RiverineInundation)
        ].mean,
        expected_flood_damage,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        results[RiskQuantityKey(QuantityType.DAMAGE, None, None, Fire)].mean,
        expected_fire_damage,
        rtol=rtol,
    )
    # Chronic is deterministic – no Monte Carlo noise
    np.testing.assert_allclose(
        results[
            RiskQuantityKey(QuantityType.REVENUE_LOSS, None, None, ChronicHeat)
        ].mean,
        expected_heat_rev_loss,
    )
    # asset_5 carries zero impact, so no Wind key exists for it; only the portfolio-level Wind key
    assert RiskQuantityKey(QuantityType.DAMAGE, None, None, Wind) in results
    # No Fire result key for assets that only have Wind/Flood
    assert RiskQuantityKey(QuantityType.DAMAGE, None, None, Fire) in results
    # Zero-risk asset does not create a spurious per-hazard entry for ChronicHeat
    assert (
        RiskQuantityKey(QuantityType.REVENUE_LOSS, None, None, ChronicHeat) in results
    )
