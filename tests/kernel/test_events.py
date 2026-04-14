import logging
from typing import Dict

import numpy as np

from physrisk.kernel.assets import Asset
from physrisk.kernel.financial_model import DefaultFinancialModel, FinancialDataProvider
from physrisk.kernel.hazards import RiverineInundation
from physrisk.kernel.impact import AssetImpactResult, ImpactKey


from physrisk.kernel.impact_aggregator import aggregate_impacts
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.risk import QuantityType, RiskQuantityKey


logger = logging.getLogger(__name__)


class TestFinancialDataProvider(FinancialDataProvider):
    def revenue_attributable_to_asset(self, asset: Asset, currency: str) -> float:
        return 200.0

    def total_insurable_value(self, asset: Asset, currency: str) -> float:
        return 100.0


def test_event_samples():
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
                )
            )
        else:
            asset_impact = AssetImpactResult(
                impact=ImpactDistrib(
                    RiverineInundation,
                    impact_bin_edges_zero.copy(),
                    impact_probabilities_zero.copy(),
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
        RiverineInundation,
        impact_bin_edges.copy(),
        impact_probabilities.copy(),
    )
    logger.info(
        f"Fractional difference: {abs(np.mean(single_impact.to_exceedance_curve().get_samples(np.random.rand(1000000))) - single_impact.mean_impact()) / single_impact.mean_impact()}"
    )

    results = aggregate_impacts(impacts, financial_model)
    damage = results[RiskQuantityKey(QuantityType.DAMAGE, None, RiverineInundation)]
    mean_damage_mc = np.mean(damage)
    logger.info(
        f"Mean damage aggregating all assets and hazards using Monte Carlo: {mean_damage_mc}"
    )
    mean_damage_exact = sum(i[0].impact.mean_impact() * 100.0 for i in impacts.values())
    logger.info(
        f"Mean damage aggregating all assets and hazards using exact calculation: {mean_damage_exact}"
    )
    np.testing.assert_allclose(mean_damage_mc, 261.702737469)
    np.testing.assert_allclose(mean_damage_mc, mean_damage_exact, rtol=0.02)
