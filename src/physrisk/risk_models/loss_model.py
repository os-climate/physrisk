from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np

from physrisk.kernel.impact_distrib import ImpactDistrib, ImpactType

from ..kernel.assets import Asset
from ..kernel.calculation import (
    get_default_hazard_model,
    get_default_vulnerability_models,
)
from ..kernel.financial_model import FinancialModelBase
from ..kernel.hazard_model import HazardModel
from ..kernel.impact import calculate_impacts
from ..kernel.vulnerability_model import (
    DictBasedVulnerabilityModels,
    VulnerabilityModels,
)


class Aggregator(ABC):
    @abstractmethod
    def get_aggregation_keys(self, asset: Asset, impact: ImpactDistrib) -> List: ...


class DefaultAggregator(Aggregator):
    def get_aggregation_keys(self, asset: Asset, impact: ImpactDistrib) -> List:
        return [(impact.hazard_type.__name__), ("root")]


class LossModel:
    def __init__(
        self,
        hazard_model: Optional[HazardModel] = None,
        vulnerability_models: Optional[VulnerabilityModels] = None,
    ):
        self.hazard_model = (
            get_default_hazard_model() if hazard_model is None else hazard_model
        )
        self.vulnerability_models = (
            DictBasedVulnerabilityModels(get_default_vulnerability_models())
            if vulnerability_models is None
            else vulnerability_models
        )

    """Calculates the financial impact on a list of assets."""

    def get_financial_impacts(
        self,
        assets: Sequence[Asset],
        *,
        financial_model: FinancialModelBase,
        scenario: str,
        year: int,
        aggregator: Optional[Aggregator] = None,
        currency: str = "EUR",
        sims: int = 100000,
    ):
        if aggregator is None:
            aggregator = DefaultAggregator()

        aggregation_pools: Dict[str, np.ndarray] = {}

        results = calculate_impacts(
            assets,
            self.hazard_model,
            self.vulnerability_models,
            scenario=scenario,
            year=year,
        )
        # the impacts in the results are either fractional damage or a fractional disruption

        rg = np.random.Generator(np.random.MT19937(seed=111))

        for impact_key, impact_values in results.items():
            for result in impact_values:
                # look up keys for results
                impact = result.impact
                keys = aggregator.get_aggregation_keys(impact_key.asset, impact)
                # transform units of impact into currency for aggregation

                # Monte-Carlo approach: note that if correlations of distributions are simple and
                # model is otherwise linear then calculation by closed-form expression is preferred
                impact_samples = self.uncorrelated_samples(impact, sims, rg)

                if impact.impact_type == ImpactType.damage:
                    loss = financial_model.damage_to_loss(
                        impact_key.asset, impact_samples, currency
                    )
                else:  # impact.impact_type == ImpactType.disruption:
                    loss = financial_model.disruption_to_loss(
                        impact_key.asset, impact_samples, year, currency
                    )

                for key in keys:
                    if key not in aggregation_pools:
                        aggregation_pools[key] = np.zeros(sims)
                    aggregation_pools[key] += loss  # type: ignore

        measures = {}
        percentiles = [0, 10, 20, 40, 60, 80, 90, 95, 97.5, 99, 99.5, 99.9]
        for key, loss in aggregation_pools.items():
            measures[key] = {
                "percentiles": percentiles,
                "percentile_values": np.percentile(loss, percentiles),
                "mean": np.mean(loss),
            }

        return measures

    def uncorrelated_samples(
        self, impact: ImpactDistrib, samples: int, generator: np.random.Generator
    ) -> np.ndarray:
        return impact.to_exceedance_curve().get_samples(generator.uniform(size=samples))
