from collections import defaultdict
from dataclasses import dataclass
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


@dataclass
class _SimulationInputs:
    all_assets: set[Asset]
    all_assets_list: list[Asset]
    all_acute_impacted_assets: list[Asset]
    impacts_exceed_curves_sorted: dict[type[Hazard], list[tuple[ImpactDistrib, ExceedanceCurve]]]
    acute_impacted_asset_indices: dict[type[Hazard], list[int]]
    chronic_impacts_sorted: dict[type[Hazard], np.ndarray]
    chronic_hazards_in_scope: set[type[Hazard]]


def _classify_impacts(
    impacts: dict[ImpactKey, list[AssetImpactResult]],
    scenario: str,
    key_year: Optional[int],
) -> tuple[
    set[Asset],
    dict[type[Hazard], set[Asset]],
    dict[ImpactKey, tuple[ImpactDistrib, ExceedanceCurve]],
    set[type[Hazard]],
]:
    """Filter impacts by scenario/year; split into acute (with exceedance curves) and chronic."""
    all_assets: set[Asset] = set()
    acute_impacted_assets: dict[type[Hazard], set[Asset]] = defaultdict(set)
    impacts_exceed_curves: dict[ImpactKey, tuple[ImpactDistrib, ExceedanceCurve]] = {}
    chronic_hazards_in_scope: set[type[Hazard]] = set()

    for ik, airs in impacts.items():
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
        if exceed_curve.get_value(1.0 / 1500.0) > 1e-6:
            impacts_exceed_curves[ik] = (air.impact, exceed_curve)
            acute_impacted_assets[ik.hazard_type].add(ik.asset)

    return all_assets, acute_impacted_assets, impacts_exceed_curves, chronic_hazards_in_scope


def _build_acute_structures(
    acute_impacted_assets: dict[type[Hazard], set[Asset]],
    impacts_exceed_curves: dict[ImpactKey, tuple[ImpactDistrib, ExceedanceCurve]],
) -> tuple[
    list[Asset],
    dict[type[Hazard], list[tuple[ImpactDistrib, ExceedanceCurve]]],
    dict[type[Hazard], list[int]],
]:
    """Build the sorted asset list, exceedance-curve lookups, and index maps for acute hazards."""
    all_acute_impacted_assets_set: set[Asset] = set()
    for hazard_type, assets in acute_impacted_assets.items():
        logger.info(f"Hazard type {hazard_type} has {len(assets)} non-zero impacts.")
        all_acute_impacted_assets_set.update(assets)

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
    acute_impacted_asset_indices: dict[type[Hazard], list[int]] = defaultdict(list)

    for hazard_type, assets in acute_impacted_assets.items():
        impacts_exceed_curves_for_hazard = {
            ik.asset: v
            for ik, v in impacts_exceed_curves.items()
            if ik.hazard_type == hazard_type
        }
        sorted_assets_for_hazard = sorted(
            assets, key=lambda a: a.id if a.id is not None else ""
        )
        indices = [idx_lookup[asset] for asset in sorted_assets_for_hazard]
        impacts_exceed_curves_sorted[hazard_type] = [
            impacts_exceed_curves_for_hazard[asset] for asset in sorted_assets_for_hazard
        ]
        acute_impacted_asset_indices[hazard_type] = indices

    return all_acute_impacted_assets, impacts_exceed_curves_sorted, acute_impacted_asset_indices


def _build_chronic_arrays(
    chronic_hazards_in_scope: set[type[Hazard]],
    all_assets_list: list[Asset],
    impacts: dict[ImpactKey, list[AssetImpactResult]],
    scenario: str,
    key_year: Optional[int],
) -> dict[type[Hazard], np.ndarray]:
    """Pre-compute per-asset chronic impact deltas (future minus historical) for each chronic hazard."""
    chronic_impacts_sorted: dict[type[Hazard], np.ndarray] = {}
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
        chronic_impacts_sorted[hazard_type] = chronic_impacts_future - chronic_impacts_histo
    return chronic_impacts_sorted


def _run_simulation(
    inputs: _SimulationInputs,
    financial_model: FinancialModel,
    asset_tiv: dict[Asset, float],
    asset_revenue: dict[Asset, float],
) -> dict[RiskQuantityKey, np.ndarray]:
    """Run Monte Carlo simulation; return per-event impact arrays keyed by RiskQuantityKey."""
    n_events = 50000
    event_batch_sz = 1000
    quantity_types = [
        QuantityType.DAMAGE,
        QuantityType.REVENUE_LOSS,
        QuantityType.COSTS_INCREASE,
    ]

    severity_provider = UncorrelatedEventSeverityProvider(
        {h: len(v) for h, v in inputs.acute_impacted_asset_indices.items()}
    )
    generator = np.random.default_rng(seed=111)

    by_asset_batch_agg = Aggregator(key_provider=ByAssetAggregationKeys())
    by_hazard_agg = Aggregator(
        key_provider=HazardQuantityAggregationKeys(), size=(n_events,)
    )
    all_impacts: dict[QuantityType, np.ndarray] = {
        qt: np.zeros(shape=(n_events)) for qt in quantity_types
    }

    logger.info(
        f"Starting to aggregate impacts for {n_events} events, in batches of {event_batch_sz}, "
        f"for {len(inputs.all_acute_impacted_assets)} assets."
    )

    for event_start in range(0, n_events, event_batch_sz):
        by_asset_batch_agg.zero()
        event_end = min(event_start + event_batch_sz, n_events)

        for hazard_type, inv_severities in severity_provider.next_inv_severities_in_batch(
            event_end - event_start, generator
        ):
            impacts_ec = inputs.impacts_exceed_curves_sorted[hazard_type]
            non_zero_indices = inputs.acute_impacted_asset_indices[hazard_type]
            sz_to_assets = severity_provider.severity_zone_to_asset_indices(hazard_type)
            for sz_idx in range(inv_severities.shape[0]):
                for asset_idx in sz_to_assets[sz_idx]:
                    _, exceed_curve = impacts_ec[asset_idx]
                    impact_samples = exceed_curve.get_samples(1.0 - inv_severities[sz_idx, :])
                    asset = inputs.all_acute_impacted_assets[non_zero_indices[asset_idx]]
                    damage, revenue_loss = (
                        financial_model.frac_damage_to_restoration_cost_and_revenue_loss(
                            asset, impact_samples, "EUR"
                        )
                    )
                    for val, qt in [
                        (damage, QuantityType.DAMAGE),
                        (revenue_loss, QuantityType.REVENUE_LOSS),
                    ]:
                        # aggregated impacts for all events for (hazard, quantity) combinations - not asset, important to reduce memory use 
                        by_hazard_agg.aggregate(
                            asset, hazard_type, qt, val,
                            slice=(slice(event_start, event_end),),
                        )
                        # aggregated impacts for batch of events for (asset, quantity) combinations - more combinations, but only for the batch
                        by_asset_batch_agg.aggregate(asset, hazard_type, qt, val)

        # chronic impacts applies to revenue loss:
        for hazard_type in inputs.chronic_hazards_in_scope:
            chronic_impacts = inputs.chronic_impacts_sorted[hazard_type]
            for idx, asset in enumerate(inputs.all_assets_list):
                by_hazard_agg.aggregate(
                    asset, hazard_type, QuantityType.REVENUE_LOSS, chronic_impacts[idx],
                    slice=(slice(event_start, event_end),),
                )
                by_asset_batch_agg.aggregate(
                    asset, hazard_type, QuantityType.REVENUE_LOSS, chronic_impacts[idx]
                )

        # cap per-asset totals (aggregated over hazards) at TIV / revenue
        for qt, cap in [
            (QuantityType.DAMAGE, asset_tiv),
            (QuantityType.REVENUE_LOSS, asset_revenue),
            (QuantityType.COSTS_INCREASE, asset_revenue),
        ]:
            for asset in inputs.all_assets:
                all_impacts[qt][event_start:event_end] += np.minimum(
                    by_asset_batch_agg.aggregation_pools.get(
                        RiskQuantityKey(quantity=qt, asset=asset), np.array(0.0)
                    ),
                    cap[asset],
                )

        if (event_end // event_batch_sz) % 20 == 0:
            logger.info(f"Processed {event_end} events out of {n_events}.")

    # return both by hazard and 
    all_results = by_hazard_agg.aggregation_pools
    for qt in quantity_types:
        all_results[RiskQuantityKey(quantity=qt)] = all_impacts[qt]
    return all_results


def _summarise_results(
    all_results: dict[RiskQuantityKey, np.ndarray],
    asset_tiv: dict[Asset, float],
    asset_revenue: dict[Asset, float],
) -> dict[RiskQuantityKey, Quantity]:
    """Normalise per-event arrays by portfolio totals and build exceedance-curve summaries."""
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
    # acute_impacted_assets: just those assets with non-zero acute impact for a given hazard type 
    all_assets, acute_impacted_assets, impacts_exceed_curves, chronic_hazards_in_scope = (
        _classify_impacts(impacts, scenario, key_year)
    )
    all_assets_list = sorted(all_assets, key=lambda a: a.id if a.id is not None else "")
    # all_acute_impacted_assets: sorted list of all assets with any acute impact (i.e. from any hazard type)
    # acute_impacted_asset_indices: index of asset in all_acute_impacted_assets, for assets impacted by the given hazard type
    # impacts_exceed_curves_sorted: the corresponding ImpactDistribs and ExceedanceCurves
    all_acute_impacted_assets, impacts_exceed_curves_sorted, acute_impacted_asset_indices = (
        _build_acute_structures(acute_impacted_assets, impacts_exceed_curves)
    )
    # chronic impacts per hazard for all assets
    chronic_impacts_sorted = _build_chronic_arrays(
        chronic_hazards_in_scope, all_assets_list, impacts, scenario, key_year
    )
    logger.info("Created non-zero asset indices for each hazard type.")

    sim_inputs = _SimulationInputs(
        all_assets=all_assets, # set of all assets
        all_assets_list=all_assets_list, # sorted list of all assets
        all_acute_impacted_assets=all_acute_impacted_assets, 
        impacts_exceed_curves_sorted=impacts_exceed_curves_sorted, # impacts and exceedance curves for assets with acute impact
        # for hazard
        acute_impacted_asset_indices=acute_impacted_asset_indices, # index 
        chronic_impacts_sorted=chronic_impacts_sorted,
        chronic_hazards_in_scope=chronic_hazards_in_scope,
    )
    asset_tiv = {
        asset: financial_model.financial_data_provider.total_insurable_value(asset, "EUR")
        for asset in all_assets
    }
    asset_revenue = {
        asset: financial_model.financial_data_provider.revenue_attributable_to_asset(asset, "EUR")
        for asset in all_assets
    }
    all_results = _run_simulation(sim_inputs, financial_model, asset_tiv, asset_revenue)
    return _summarise_results(all_results, asset_tiv, asset_revenue)


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
