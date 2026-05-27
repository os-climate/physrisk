from collections import defaultdict
import logging
from typing import Generator, NamedTuple, Optional
from typing_extensions import Protocol

import numpy as np

from physrisk.kernel.hazards import Hazard
from physrisk.kernel.impact_distrib import ImpactDistrib
from physrisk.kernel.risk import Quantity, QuantityType, RiskQuantityKey
from physrisk.kernel.assets import Asset
from physrisk.kernel.curve import ExceedanceCurve
from physrisk.kernel.financial_model import FinancialModel
from physrisk.kernel.hazards import HazardKind
from physrisk.kernel.impact import AssetImpactResult, ImpactKey


logger = logging.getLogger(__name__)


class AggregationKeys(Protocol):
    def get_aggregation_keys(
        self, asset: Asset, hazard_type: type[Hazard], quantity: QuantityType
    ) -> list[RiskQuantityKey]: ...

    """Returns a list of keys for aggregation.
    Returns:
        List[RiskQuantityKey]: List of keys for aggregation.
    """


class ContributionKey(NamedTuple):
    # key describing the contribution to the aggregated result
    asset_id: Optional[str] = None
    quantity: Optional[QuantityType] = None
    hazard_type: Optional[type[Hazard]] = None


class Aggregator:
    def __init__(
        self, key_provider: AggregationKeys, size: Optional[tuple[int, ...]] = None
    ):
        self.key_provider = key_provider
        self.aggregation_pools: dict[RiskQuantityKey, np.ndarray] = {}
        # also keep a record of the contributors:
        self.aggregation_pool_contribs: dict[RiskQuantityKey, set[ContributionKey]] = (
            defaultdict(set)
        )
        self.size = size

    def zero(self, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
        if shape is not None:
            for k in self.aggregation_pools.keys():
                self.aggregation_pools[k] = np.zeros(shape)
        else:
            for k in self.aggregation_pools.keys():
                self.aggregation_pools[k] = np.zeros_like(self.aggregation_pools[k])

    def aggregate(
        self,
        asset: Asset,
        hazard_type: type[Hazard],
        quantity: QuantityType,
        values: np.ndarray,
        slice: Optional[tuple[slice, ...]] = None,
    ):
        for key in self.key_provider.get_aggregation_keys(asset, hazard_type, quantity):
            if key not in self.aggregation_pools:
                size = self.size if self.size is not None else values.shape
                self.aggregation_pools[key] = np.zeros(size)
            if slice is not None:
                self.aggregation_pools[key][slice] += values  # type: ignore
            else:
                self.aggregation_pools[key] += values  # type: ignore
            self.aggregation_pool_contribs[key].add(
                ContributionKey(
                    asset_id=asset.id, quantity=quantity, hazard_type=hazard_type
                )
            )

    def reset(self):
        self.aggregation_pools = {}
        self.aggregation_pool_contribs = defaultdict(set)


class ByAssetAggregationKeys(AggregationKeys):
    """Aggregator that provides aggregated results by asset and quantity type."""

    def get_aggregation_keys(
        self, asset: Asset, hazard_type: type[Hazard], quantity: QuantityType
    ) -> list[RiskQuantityKey]:
        return [
            RiskQuantityKey(asset=asset, quantity=quantity),
        ]


class HazardQuantityAggregationKeys(AggregationKeys):
    def get_aggregation_keys(
        self, asset: Asset, hazard_type: type[Hazard], quantity: QuantityType
    ) -> list[RiskQuantityKey]:
        return [
            RiskQuantityKey(quantity=quantity, hazard_type=hazard_type),
        ]


class DefaultAggregationKeys(AggregationKeys):
    def get_aggregation_keys(
        self, asset: Asset, hazard_type: type[Hazard], quantity: QuantityType
    ) -> list[RiskQuantityKey]:
        agg_id = getattr(asset, "agg_id", None)
        return [
            RiskQuantityKey(quantity=quantity, agg_id=agg_id),
            RiskQuantityKey(quantity=quantity, agg_id=agg_id, hazard_type=hazard_type),
        ]


def aggregate_impacts(
    impacts: dict[ImpactKey, list[AssetImpactResult]],
    financial_model: FinancialModel,
    scenario: str,
    key_year: Optional[int],
) -> dict[RiskQuantityKey, Quantity]:
    """Aggregate impacts over assets and hazards for a given scenario and year.
    For acute hazards, i.e. hazards associated with an event, a Monte Carlo approach is used whereby a large number of
    events is sampled, representing future possible years.
    In order to support correlation between assets and hazards, the concept of severity zones (SZ) is introduced. A severity
    zone is a geographical region where the assets within the region experience the same severity for a given event.
    A severity is a return period in years; the reciprocal of the severity is the event exceedance probability.
    It is emphasized that assets in the SZ do not experience the same hazard indicator value, only the same severity:
    For example for riverine inundation the severity zone could be a catchment. A severity of 250 years means that
    each asset in the catchment experiences a hazard indicator value with a 1/250 chance of being exceeded in a year,
    at the location of the asset. But for some assets a 250 year flood depth might be 1 m, for others 1 cm or 0 cm.

    The general approach of using severity zones is flexible enough to support several cases:
    1) Severity zones and severities supplied as an input (e.g. catastrophe model event sets, commercial or otherwise).
    2) Simple aggregation considering that all assets and hazards are uncorrelated, i.e. each asset has its own severity
    zone and the severities are independent between assets and hazards.
    3) Heuristic approaches whereby assets that are geographically close experience similar severities for the same event.

    In cases where severities refer to hazard indicator values, applying these to impacts is only correct if the impact
    is a non-decreasing function of the hazard indicator value (a common case). Otherwise, either a slightly more complex
    approach is needed or the approximation is accepted.

    Use of severity zones is perhaps the most natural fit for flood modelling, although can be treated as generic.
    The framework can be extended to model tropical cyclones for example, although arguably a less natural fit
    given approaches often concentrate on tropical cyclone tracks. Incorporating tracks may entail converting
    tracks to severities at each asset location: the severity zone is the asset location in such an approach.

    Args:
        impacts (Dict[ImpactKey, AssetImpactResult]): Impact results for each asset and hazard type.
        financial_model (FinancialModel): Financial model to convert impacts to financial losses.
    """

    # for acute hazards, for simplicity we add the constraint that only one ImpactDistrib is allowed
    # per asset and hazard type. At time of writing there are no known exceptions to this.
    acute_impacted_assets: dict[type[Hazard], set[Asset]] = defaultdict(set)
    all_assets: set[Asset] = set()
    impacts_exceed_curves: dict[ImpactKey, tuple[ImpactDistrib, ExceedanceCurve]] = {}
    acute_hazards_in_scope: set[type[Hazard]] = set()
    chronic_hazards_in_scope: set[type[Hazard]] = set()
    for ik, airs in impacts.items():  # AssetImpactResults
        if ik.scenario != scenario or ik.key_year != key_year:
            continue
        all_assets.add(ik.asset)
        if ik.hazard_type.kind != HazardKind.ACUTE:
            chronic_hazards_in_scope.add(ik.hazard_type)
            continue
        if len(airs) > 1:
            raise NotImplementedError(
                f"Multiple impacts for asset {ik.asset} and hazard type {ik.hazard_type}: not permitted for acute hazards."
            )
        air = airs[0]
        exceed_curve = air.impact.to_exceedance_curve()
        if exceed_curve.get_value(1.0 / 1500.0) > 1e-6:  # add a tolerance to 'zero'
            impacts_exceed_curves[ik] = (air.impact, exceed_curve)
            acute_impacted_assets[ik.hazard_type].add(ik.asset)
            acute_hazards_in_scope.add(ik.hazard_type)

    all_acute_impacted_assets_set: set[Asset] = set()
    for hazard_type, assets in acute_impacted_assets.items():
        logger.info(f"Hazard type {hazard_type} has {len(assets)} non-zero impacts.")
        all_acute_impacted_assets_set.update(assets)
    all_assets_list = sorted(all_assets, key=lambda a: a.id if a.id is not None else "")
    all_acute_impacted_assets = sorted(
        all_acute_impacted_assets_set, key=lambda a: a.id if a.id is not None else ""
    )
    logger.info(
        f"There are {len(all_acute_impacted_assets)} assets with non-zero impacts for at least one hazard type."
    )

    idx_lookup = {val: i for i, val in enumerate(all_acute_impacted_assets)}
    impacts_exceed_curves_sorted: dict[
        type[Hazard], list[tuple[ImpactDistrib, ExceedanceCurve]]
    ] = defaultdict(list)
    chronic_impacts_sorted: dict[type[Hazard], np.ndarray] = defaultdict(list)
    acute_impacted_asset_indices: dict[type[Hazard], list[int]] = defaultdict(list)

    for hazard_type, assets in acute_impacted_assets.items():
        impacts_exceed_curves_for_hazard = {
            ik.asset: v
            for ik, v in impacts_exceed_curves.items()
            if ik.hazard_type == hazard_type
        }
        # these are the assets with non-zero impact for the hazard
        sorted_assets_for_hazard = sorted(assets, key=lambda a: a.id if a.id is not None else "")
        indices = [idx_lookup[asset] for asset in sorted_assets_for_hazard]
        impacts_exceed_curves_sorted[hazard_type] = [
            impacts_exceed_curves_for_hazard[asset] for asset in sorted_assets_for_hazard
        ]
        acute_impacted_asset_indices[hazard_type] = indices

    for hazard_type in chronic_hazards_in_scope:
        chronic_impacts_future = np.array(
            [
                sum(
                    i.impact.mean_impact()
                    for i in impacts.get(
                        ImpactKey(
                            asset=asset,
                            hazard_type=hazard_type,
                            scenario=scenario,
                            key_year=key_year,
                        ),
                        [],
                    )
                )
                for asset in all_assets_list
            ]
        )
        chronic_impacts_histo = np.array(
            [
                sum(
                    i.impact.mean_impact()
                    for i in impacts.get(
                        ImpactKey(
                            asset=asset,
                            hazard_type=hazard_type,
                            scenario="historical",
                            key_year=-1,
                        ),
                        [],
                    )
                )
                for asset in all_assets_list
            ]
        )
        chronic_impacts_sorted[hazard_type] = (
            chronic_impacts_future - chronic_impacts_histo
        )

    logger.info("Created non-zero asset indices for each hazard type.")

    n_events = 50000
    event_batch_sz = 1000

    # the uncorrelated provider has one severity zone per asset and independence is assumed
    severity_provider = UncorrelatedEventSeverityProvider(
        {h: len(v) for h, v in acute_impacted_asset_indices.items()}
    )

    generator = np.random.default_rng(seed=111)
    # create a chunk of memory to store the impacts aggregated over hazards and assets for a batch of events.
    quantity_types = [
        QuantityType.DAMAGE,
        QuantityType.REVENUE_LOSS,
        QuantityType.COSTS_INCREASE,
    ]

    # aggregator used for the total impact, but just by batch of events
    by_asset_batch_agg = Aggregator(key_provider=ByAssetAggregationKeys())
    # aggregator used to keep track of the contribution to the total impact by hazard type for all events
    by_hazard_agg = Aggregator(
        key_provider=HazardQuantityAggregationKeys(), size=(n_events,)
    )

    # aggregated over hazards and assets for all events
    all_impacts: dict[QuantityType, np.ndarray] = {
        qt: np.zeros(shape=(n_events)) for qt in quantity_types
    }

    logger.info(
        f"Starting to aggregate impacts for {n_events} events, in batches of {event_batch_sz}, for {len(all_acute_impacted_assets)} assets."
    )
    asset_tiv = {
        asset: financial_model.financial_data_provider.total_insurable_value(
            asset, "EUR"
        )
        for asset in all_assets
    }
    asset_revenue = {
        asset: financial_model.financial_data_provider.revenue_attributable_to_asset(
            asset, "EUR"
        )
        for asset in all_assets
    }
    for event_start in range(0, n_events, event_batch_sz):
        by_asset_batch_agg.zero()
        event_end = min(event_start + event_batch_sz, n_events)
        # first handle acute risks, those that are based on events
        for (
            hazard_type,
            inv_severities,
        ) in severity_provider.next_inv_severities_in_batch(
            event_end - event_start, generator
        ):
            impacts_exceed_curves_sorted_for_hazard = impacts_exceed_curves_sorted[
                hazard_type
            ]
            non_zero_asset_indices_for_hazard = acute_impacted_asset_indices[
                hazard_type
            ]
            severity_zone_to_asset_indices = (
                severity_provider.severity_zone_to_asset_indices(hazard_type)
            )
            for sz_idx in range(inv_severities.shape[0]):
                # loop over assets in the severity zone
                for asset_idx in severity_zone_to_asset_indices[sz_idx]:
                    # asset_idx is the index for just those assets impacted by the hazard
                    _, exceed_curve = impacts_exceed_curves_sorted_for_hazard[asset_idx]
                    # in this case asset_idx is equal to sz_idx
                    impact_samples = exceed_curve.get_samples(1.0 - inv_severities[sz_idx, :])
                    all_non_zero_assets_idx = non_zero_asset_indices_for_hazard[asset_idx]
                    asset = all_acute_impacted_assets[all_non_zero_assets_idx]
                    damage, revenue_loss = (
                        financial_model.frac_damage_to_restoration_cost_and_revenue_loss(
                            asset, impact_samples, "EUR"
                        )
                    )
                    for val, qt in [
                        (damage, QuantityType.DAMAGE),
                        (revenue_loss, QuantityType.REVENUE_LOSS),
                    ]:
                        by_hazard_agg.aggregate(
                            asset,
                            hazard_type,
                            qt,
                            val,
                            slice=(slice(event_start, event_end),),
                        )
                        by_asset_batch_agg.aggregate(asset, hazard_type, qt, val)
        # now handle chronic risks: not based on events
        for hazard_type in chronic_hazards_in_scope:
            chronic_impacts = chronic_impacts_sorted[hazard_type]
            for idx, asset in enumerate(all_assets_list):
                by_hazard_agg.aggregate(
                    asset,
                    hazard_type,
                    QuantityType.REVENUE_LOSS,
                    chronic_impacts[idx],
                    slice=(slice(event_start, event_end),),
                )
                by_asset_batch_agg.aggregate(
                    asset, hazard_type, QuantityType.REVENUE_LOSS, chronic_impacts[idx]
                )

        # cap values per asset, aggregated over hazards
        for qt, cap in [
            (QuantityType.DAMAGE, asset_tiv),
            (QuantityType.REVENUE_LOSS, asset_revenue),
            (QuantityType.COSTS_INCREASE, asset_revenue),
        ]:
            for asset in all_assets:
                all_impacts[qt][event_start:event_end] += np.minimum(
                    by_asset_batch_agg.aggregation_pools.get(
                        RiskQuantityKey(quantity=qt, asset=asset), np.array(0.0)
                    ),
                    cap[asset],
                )

        if (event_end // event_batch_sz) % 20 == 0:
            logger.info(f"Processed {event_end} events out of {n_events}.")

    # the results are:
    # 1) the set of impacts aggregated over assets and hazards for required quantities (e.g. damage and revenue loss)
    # 2) the set of impacts aggregated over assets for each hazard and required quantities
    # a simple approach to compound damage is applied: damage/revenue-loss is additive but capped at 100% for a given year
    all_results = by_hazard_agg.aggregation_pools
    for qt in quantity_types:
        all_results[RiskQuantityKey(quantity=qt)] = all_impacts[qt]

    # convert back to relative
    sum_asset_tiv = sum(asset_tiv.values())
    sum_asset_revenue = sum(asset_revenue.values())
    for k, v in all_results.items():
        if k.quantity == QuantityType.DAMAGE:
            all_results[k] = v / sum_asset_tiv
        elif k.quantity == QuantityType.REVENUE_LOSS:
            all_results[k] = v / sum_asset_revenue

    return_periods = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    quantiles = 1.0 - 1.0 / return_periods
    summary_stats: dict[RiskQuantityKey, Quantity] = {}
    for k, v in all_results.items():
        exceed = ExceedanceCurve(1.0 / return_periods, np.quantile(v, quantiles))
        mean = np.mean(v)
        semi_std = np.sqrt(np.mean(np.square(v[v > mean] - mean)))
        summary_stats[k] = Quantity(
            values=v if k.hazard_type is None else None,
            exceedance_curve=exceed,
            mean=mean,
            semi_standard_deviation=semi_std,
        )
    return summary_stats


class EventSeverityProvider(Protocol):
    def next_inv_severities_in_batch(
        self, n_events: int, generator: np.random.Generator
    ) -> Generator[tuple[type[Hazard], np.ndarray], None, None]:
        """Returns a generator that gives the inverse severities for each hazard type.

        Args:
            n_events (int): Number of events in the batch.
            generator (np.random.Generator): Random number generator.

        Yields:
            Generator[tuple[type[Hazard], np.ndarray], None, None]: The severities for each hazard type.
        """
        ...

    def severity_zone_to_asset_indices(
        self, hazard_type: type[Hazard]
    ) -> list[list[int]]:
        """Returns a mapping from severity zone index to asset indices for a given hazard type."""
        ...


class UncorrelatedEventSeverityProvider(EventSeverityProvider):
    def __init__(self, n_non_zero_assets_by_hazard: dict[type[Hazard], int]):
        self.n_severity_zones_by_hazard = n_non_zero_assets_by_hazard
        self.severity_zone_to_asset_indices_by_hazard = {
            hazard_type: [[i] for i in range(n_non_zero_assets)]
            for hazard_type, n_non_zero_assets in n_non_zero_assets_by_hazard.items()
        }

    def next_inv_severities_in_batch(
        self, n_events: int, generator: np.random.Generator
    ) -> Generator[tuple[type[Hazard], np.ndarray], None, None]:
        for hazard_type in self.n_severity_zones_by_hazard.keys():
            randoms = generator.random(
                size=(self.n_severity_zones_by_hazard[hazard_type], n_events),
                dtype=np.float32,
            )
            yield hazard_type, randoms

    def severity_zone_to_asset_indices(
        self, hazard_type: type[Hazard]
    ) -> list[list[int]]:
        """Here there is one asset per severity zone for every hazard type."""
        return self.severity_zone_to_asset_indices_by_hazard[hazard_type]


class Events(object):
    def __init__(
        self,
        hazard_type: type[Hazard],
        event_id: np.ndarray,
        event_start: np.ndarray,
        event_end: np.ndarray,
        severity_zone: np.ndarray,
        severity: np.ndarray,
    ):
        """Representation of events based on severity zones. This is a geographical area such that all assets within the area
        have the same severity for a given event.

        Args:
            event_id (np.ndarray): Array of integer event identifiers.
            event_start (np.ndarray): Starts of events as DateTime64 array.
            event_end (np.ndarray): Ends of events as DateTime64 array.
            severities (List[np.ndarray]): List of arrays giving severities for each severity zone by hazard.
        """
        self.event_id = event_id
        self.event_start = event_start
        self.event_end = event_end
        self.severity_zones = severity_zone
        self.severity = severity
