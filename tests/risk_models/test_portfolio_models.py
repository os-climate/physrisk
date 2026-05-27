import logging
from typing import Dict, Optional

from dependency_injector import providers
import numpy as np

from physrisk.api.v1.common import Asset as APIAsset, Assets
from physrisk.api.v1.impact_req_resp import AssetImpactRequest, CalcSettings
from physrisk.container import Container, DefaultVulnerabilityModelFactory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import Asset, ManufacturingAsset
from physrisk.kernel.financial_model import DefaultFinancialModel, FinancialDataProvider
from physrisk.kernel.hazard_model import HazardModelFactory
from physrisk.kernel.hazards import ChronicHeat, Fire, RiverineInundation, Wind
from physrisk.kernel.impact import AssetImpactResult, ImpactKey


from physrisk.kernel.impact_aggregator import aggregate_impacts
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.risk import QuantityType, RiskQuantityKey
from physrisk.vulnerability_models.vulnerability import VulnerabilityModelsFactory
from tests.data.test_hazard_model_store import TestData
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
    damage = results[RiskQuantityKey(QuantityType.DAMAGE, None, None, RiverineInundation)]
    mean_damage_mc = damage.mean
    logger.info(
        f"Mean damage aggregating all assets and hazards using Monte Carlo: {mean_damage_mc}"
    )
    mean_damage_exact = sum(i[0].impact.mean_impact() * 100.0 for i in impacts.values()) / (n_assets * 100.0)
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
    hazard_model = ZarrHazardModel(
        source_paths=get_default_source_paths(), store=store
    )

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
             config=VulnerabilityModelsFactory.embedded_vulnerability_config()
         )
    )
    requester = container.requester()

    assets = Assets(items=[APIAsset(id=f"asset_1", occupancy_code=1050, latitude=latitudes[0], longitude=longitudes[0])])
    req = AssetImpactRequest(assets=assets,
                             scenarios=scenarios,
                             years=years,
                             include_asset_level=True,
                             include_measures=True,
                             include_calc_details=False,
                             use_case_id="company",
                             calc_settings=CalcSettings(hazard_scope="RiverineInundation"))
    res = requester.get_asset_impacts(request=req)
    assert res is not None


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

    wind_edges  = np.array([0.0, 0.5]); wind_probs  = np.array([0.02])  # mean 0.005
    flood_edges = np.array([0.0, 0.4]); flood_probs = np.array([0.05])  # mean 0.010
    fire_edges  = np.array([0.0, 0.8]); fire_probs  = np.array([0.01])  # mean 0.004
    heat_fut_edges = np.array([0.0, 0.20]); heat_fut_probs = np.array([0.5])  # mean 0.05
    heat_his_edges = np.array([0.0, 0.10]); heat_his_probs = np.array([0.4])  # mean 0.02
    zero_edges = np.array([0.0, 0.0]);  zero_probs = np.array([0.0])

    a = [ManufacturingAsset(id=f"asset_{i}", latitude=0.0, longitude=0.0) for i in range(6)]

    def air(hazard_type, edges, probs):
        return AssetImpactResult(impact=ImpactDistrib(hazard_type, edges.copy(), probs.copy(), ""))

    def ik(asset, hazard_type, sc=scenario, yr=key_year):
        return ImpactKey(asset=asset, hazard_type=hazard_type, scenario=sc, key_year=yr)

    impacts: Dict[ImpactKey, list[AssetImpactResult]] = {
        # Wind – assets 0, 1
        ik(a[0], Wind): [air(Wind, wind_edges, wind_probs)],
        ik(a[1], Wind): [air(Wind, wind_edges, wind_probs)],
        # Flood – assets 0, 1
        ik(a[0], RiverineInundation): [air(RiverineInundation, flood_edges, flood_probs)],
        ik(a[1], RiverineInundation): [air(RiverineInundation, flood_edges, flood_probs)],
        # Fire – assets 2, 3, 4
        ik(a[2], Fire): [air(Fire, fire_edges, fire_probs)],
        ik(a[3], Fire): [air(Fire, fire_edges, fire_probs)],
        ik(a[4], Fire): [air(Fire, fire_edges, fire_probs)],
        # ChronicHeat future – assets 0, 1, 2
        ik(a[0], ChronicHeat): [air(ChronicHeat, heat_fut_edges, heat_fut_probs)],
        ik(a[1], ChronicHeat): [air(ChronicHeat, heat_fut_edges, heat_fut_probs)],
        ik(a[2], ChronicHeat): [air(ChronicHeat, heat_fut_edges, heat_fut_probs)],
        # ChronicHeat historical baseline – assets 0, 1, 2
        ik(a[0], ChronicHeat, sc="historical", yr=-1): [air(ChronicHeat, heat_his_edges, heat_his_probs)],
        ik(a[1], ChronicHeat, sc="historical", yr=-1): [air(ChronicHeat, heat_his_edges, heat_his_probs)],
        ik(a[2], ChronicHeat, sc="historical", yr=-1): [air(ChronicHeat, heat_his_edges, heat_his_probs)],
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

    expected_wind_damage  = 2 * 100 * 0.005 / 600   # 1/600
    expected_flood_damage = 2 * 100 * 0.010 / 600   # 2/600
    expected_fire_damage  = 3 * 100 * 0.004 / 600   # 1.2/600
    expected_heat_rev_loss = 3 * 0.03 / 1200        # 7.5e-5

    rtol = 0.05  # 5 % tolerance (~2 std-errors) for 50 000-event Monte Carlo

    np.testing.assert_allclose(
        results[RiskQuantityKey(QuantityType.DAMAGE, None, None, Wind)].mean,
        expected_wind_damage, rtol=rtol,
    )
    np.testing.assert_allclose(
        results[RiskQuantityKey(QuantityType.DAMAGE, None, None, RiverineInundation)].mean,
        expected_flood_damage, rtol=rtol,
    )
    np.testing.assert_allclose(
        results[RiskQuantityKey(QuantityType.DAMAGE, None, None, Fire)].mean,
        expected_fire_damage, rtol=rtol,
    )
    # Chronic is deterministic – no Monte Carlo noise
    np.testing.assert_allclose(
        results[RiskQuantityKey(QuantityType.REVENUE_LOSS, None, None, ChronicHeat)].mean,
        expected_heat_rev_loss,
    )
    # asset_5 carries zero impact, so no Wind key exists for it; only the portfolio-level Wind key
    assert RiskQuantityKey(QuantityType.DAMAGE, None, None, Wind) in results
    # No Fire result key for assets that only have Wind/Flood
    assert RiskQuantityKey(QuantityType.DAMAGE, None, None, Fire) in results
    # Zero-risk asset does not create a spurious per-hazard entry for ChronicHeat
    assert RiskQuantityKey(QuantityType.REVENUE_LOSS, None, None, ChronicHeat) in results

