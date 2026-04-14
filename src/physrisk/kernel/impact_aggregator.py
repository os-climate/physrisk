from asyncio import Protocol
from collections import defaultdict
import logging
from typing import Generator, NamedTuple, Optional
import logging

import numpy as np

from physrisk.kernel.hazards import Hazard
from physrisk.kernel.impact_distrib import ImpactDistrib, ImpactType
from physrisk.kernel.risk import Quantity, QuantityType, RiskQuantityKey
from physrisk.kernel.assets import Asset
from physrisk.kernel.curve import ExceedanceCurve
from physrisk.kernel.events import event_samples
from physrisk.kernel.financial_model import FinancialModel
from physrisk.kernel.hazards import Hazard, HazardKind
from physrisk.kernel.impact import AssetImpactResult, ImpactKey

from numba.typed.typedlist import List as NumbaList


logger = logging.getLogger(__name__)


class AggregationKeys(Protocol):
    def get_aggregation_keys(
        self, asset: Asset, hazard_type: type[Hazard], quantity: QuantityType
    ) -> list[RiskQuantityKey]:
        """Returns a list of keys for aggregation.
        Returns:
            List[RiskQuantityKey]: List of keys for aggregation.
        """
        ...


class ContributionKey(NamedTuple):
    # key describing the contribution to the aggregated result
    asset_id: Optional[str] = None
    quantity: Optional[Quantity] = None
    hazard_type: Optional[type[Hazard]] = None


class Aggregator:
    def __init__(self, key_provider: AggregationKeys, size: Optional[tuple[int, ...]] = None):
        self.key_provider = key_provider
        self.aggregation_pools: dict[RiskQuantityKey, np.ndarray] = {}
        # also keep a record of the contributors:
        self.aggregation_pool_contribs: dict[
            RiskQuantityKey, set[ContributionKey]
        ] = defaultdict(set)
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
        quantity: Quantity,
        values: np.ndarray,
        slice: Optional[tuple[slice, ...]] = None
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
        self.aggregation_pool_contribs = defaultdict(list)


class HazardQuantityAggregationKeys(AggregationKeys):
    def get_aggregation_keys(
        self, asset: Asset, hazard_type: str, quantity: Quantity
    ) -> list[RiskQuantityKey]:
        return [
            RiskQuantityKey(quantity=quantity, hazard_type=hazard_type),
        ]


class DefaultAggregationKeys(AggregationKeys):
    def get_aggregation_keys(
        self, asset: Asset, hazard_type: str, quantity: Quantity
    ) -> list[RiskQuantityKey]:
        agg_id = getattr(asset, "agg_id", None)
        return [
            RiskQuantityKey(quantity=quantity, agg_id=agg_id),
            RiskQuantityKey(
                quantity=quantity, agg_id=agg_id, hazard_type=hazard_type
            ),
        ]


def aggregate_impacts(impacts: dict[ImpactKey, list[AssetImpactResult]], financial_model: FinancialModel):
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
    non_zero_assets: dict[type[Hazard], set[Asset]] = defaultdict(set)
    impacts_exceed_curves: dict[ImpactKey, tuple[ImpactDistrib, ExceedanceCurve]] = {}
    acute_hazards_in_scope: set[type[Hazard]] = set()
    for k, airs in impacts.items(): # AssetImpactResults
        if k.hazard_type.kind != HazardKind.ACUTE:
            continue
        if len(airs) > 1:
            raise NotImplementedError(f"Multiple impacts for asset {k.asset} and hazard type {k.hazard_type}: not permitted for acute hazards.")
        air = airs[0]
        exceed_curve = air.impact.to_exceedance_curve()
        if exceed_curve.get_value(1./1500.) > 1e-6: # add a tolerance to 'zero'
            impacts_exceed_curves[k] = (air.impact, exceed_curve)
            non_zero_assets[k.hazard_type].add(k.asset)
            acute_hazards_in_scope.add(k.hazard_type)

    all_non_zero_assets: set[Asset] = set()
    for hazard_type, assets in non_zero_assets.items():
        logger.info(f"Hazard type {hazard_type} has {len(assets)} non-zero impacts.")
        all_non_zero_assets.update(assets)
    all_non_zero_assets = sorted(all_non_zero_assets, key=lambda a: a.id)
    logger.info(f"There are {len(all_non_zero_assets)} assets with non-zero impacts for at least one hazard type.")    

    idx_lookup = {val: i for i, val in enumerate(all_non_zero_assets)}
    impacts_exceed_curves_sorted: dict[type[Hazard], list[tuple[ImpactDistrib, ExceedanceCurve]]] = defaultdict(list)
    non_zero_asset_indices: dict[type[Hazard], list[int]] = defaultdict(list)
    
    for hazard_type, assets in non_zero_assets.items():
        impacts_exceed_curves_for_hazard = {k.asset:v for k, v in impacts_exceed_curves.items() if k.hazard_type == hazard_type}
        sorted_assets = sorted(assets, key=lambda a: a.id)
        indices = [idx_lookup[asset] for asset in sorted_assets]
        impacts_exceed_curves_sorted[hazard_type] = [impacts_exceed_curves_for_hazard[asset] for asset in sorted_assets]
        non_zero_asset_indices[hazard_type] = indices

    logger.info(f"Created non-zero asset indices for each hazard type.")

    n_all_non_zero_assets = len(all_non_zero_assets)
    n_events = 100000
    event_batch_sz = 1000

    # the uncorrelated provider has one severity zone per asset and independence is assumed
    severity_provider = UncorrelatedEventSeverityProvider({h:len(v) for h, v in non_zero_asset_indices.items()})
    
    generator = np.random.default_rng(seed=111)
    # create a chunk of memory to store the impacts aggregated over hazards and assets for a batch of events.
    quantity_types = [QuantityType.DAMAGE, QuantityType.REVENUE_LOSS]

    hazard_quantity_agg = Aggregator(key_provider=HazardQuantityAggregationKeys(), size=(n_events))

    # aggregated over hazards and assets for all events
    all_impacts: dict[QuantityType, np.ndarray] = {qt: np.zeros(shape=(n_events)) for qt in quantity_types}
    # aggregated over hazards but not assets for a batch of events
    batch_impacts: dict[QuantityType, np.ndarray] = {qt: np.zeros(shape=(n_all_non_zero_assets, event_batch_sz)) for qt in quantity_types}
    
    logger.info(f"Starting to aggregate impacts for {n_events} events, in batches of {event_batch_sz}, for {len(all_non_zero_assets)} assets.")
    asset_tiv = np.array([financial_model.financial_data_provider.total_insurable_value(asset, "EUR") for asset in all_non_zero_assets])
    asset_revenue = np.array([financial_model.financial_data_provider.revenue_attributable_to_asset(asset, "EUR") for asset in all_non_zero_assets])
    for event_start in range(0, n_events, event_batch_sz):
        event_end = min(event_start + event_batch_sz, n_events)
        logger.info(f"Processing events {event_start} to {event_end}...")
        batch_impacts[QuantityType.DAMAGE].fill(0.)
        batch_impacts[QuantityType.REVENUE_LOSS].fill(0.)
        for hazard_type, inv_severities in severity_provider.next_inv_severities_in_batch(event_end - event_start, generator):
            impacts_exceed_curves_sorted_for_hazard = impacts_exceed_curves_sorted[hazard_type]
            non_zero_asset_indices_for_hazard = non_zero_asset_indices[hazard_type]
            severity_zone_to_asset_indices = severity_provider.severity_zone_to_asset_indices(hazard_type)
            for sz_idx in range(inv_severities.shape[0]):
                for asset_idx in severity_zone_to_asset_indices[sz_idx]:
                    _, exceed_curve = impacts_exceed_curves_sorted_for_hazard[asset_idx]
                    # in this case asset_idx is equal to sz_idx
                    impact_samples = exceed_curve.get_samples(inv_severities[sz_idx, :])
                    all_non_zero_assets_idx = non_zero_asset_indices_for_hazard[sz_idx]
                    asset = all_non_zero_assets[all_non_zero_assets_idx]
                    damage, revenue_loss = financial_model.frac_damage_to_restoration_cost_and_revenue_loss(asset, impact_samples, "EUR")
                    batch_impacts[QuantityType.DAMAGE][all_non_zero_assets_idx, 0:len(damage)] += damage
                    batch_impacts[QuantityType.REVENUE_LOSS][all_non_zero_assets_idx, 0:len(revenue_loss)] += revenue_loss
                    hazard_quantity_agg.aggregate(asset, hazard_type, QuantityType.DAMAGE, damage, slice=slice(event_start, event_end))
                    hazard_quantity_agg.aggregate(asset, hazard_type, QuantityType.REVENUE_LOSS, revenue_loss, slice=slice(event_start, event_end))
        # cap values per asset, aggregated over hazards
        for qt, cap in [(QuantityType.DAMAGE, asset_tiv), (QuantityType.REVENUE_LOSS, asset_revenue)]:
            batch_impacts[qt] = np.minimum(batch_impacts[qt], cap[:, None])
            all_impacts[qt][event_start:event_end] = np.sum(batch_impacts[qt], axis=0)

        # the results are: 
        # 1) the set of impacts aggregated over assets and hazards for required quantities (e.g. damage and revenue loss)
        # 2) the set of impacts aggregated over assets for each hazard and required quantities
        # a simple approach to compound damage is applied: damage/revenue-loss is additive but capped at 100% for a given year
        all_results = hazard_quantity_agg.aggregation_pools
        for qt in quantity_types:
            all_results[RiskQuantityKey(quantity=qt)] = all_impacts[qt]
        return all_results


class EventSeverityProvider(Protocol):
    def next_inv_severities_in_batch(self,
                                    n_events: int,
                                    generator: np.random.Generator) -> Generator[tuple[type[Hazard], np.ndarray], None, None]:
        """Returns a generator that gives the inverse severities for each hazard type.

        Args:
            n_events (int): Number of events in the batch.
            generator (np.random.Generator): Random number generator.

        Yields:
            Generator[tuple[type[Hazard], np.ndarray], None, None]: The severities for each hazard type.
        """
        ...

    def severity_zone_to_asset_indices(self, hazard_type: type[Hazard]) -> list[list[int]]:
        """Returns a mapping from severity zone index to asset indices for a given hazard type."""
        ...


class UncorrelatedEventSeverityProvider(EventSeverityProvider):
    def __init__(self, n_non_zero_assets_by_hazard: dict[type[Hazard],int]):
        self.n_severity_zones_by_hazard = n_non_zero_assets_by_hazard
        self.severity_zone_to_asset_indices_by_hazard = {hazard_type: [[i] for i in range(n_non_zero_assets)] for hazard_type, n_non_zero_assets in n_non_zero_assets_by_hazard.items()}

    def next_inv_severities_in_batch(self,
                                     n_events: int,
                                     generator: np.random.Generator) -> Generator[tuple[type[Hazard], np.ndarray], None, None]:
        for hazard_type in self.n_severity_zones_by_hazard.keys():
            randoms = generator.uniform(size=(self.n_severity_zones_by_hazard[hazard_type], n_events))        
            yield hazard_type, randoms

    def severity_zone_to_asset_indices(self, hazard_type: type[Hazard]) -> list[list[int]]:
        """Here there is one asset per severity zone for every hazard type."""
        return self.severity_zone_to_asset_indices_by_hazard[hazard_type]


class Events(object):
    def __init__(self,
                 hazard_type: type[Hazard], 
                 event_id: np.ndarray,
                 event_start: np.ndarray,
                 event_end: np.ndarray,
                 severity_zone: np.ndarray,
                 severity: np.ndarray):
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