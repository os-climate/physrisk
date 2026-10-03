import logging
from enum import Enum
from typing import NamedTuple, Protocol, Sequence

from physrisk.api.v1.hazard_data import HazardResource
from physrisk.data.hazard_data_provider import (
    HazardDataHint,
    HazardResourceSelector,
)
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    Drought,
    Hazard,
    Landslide,
    PluvialInundation,
    RiverineInundation,
    Wind,
    hazard_class,
)

logger = logging.getLogger(__name__)


class ResourceSelectionRule(Protocol):
    """Filter or order candidate resources for a hazard/indicator combination."""

    def __call__(
        self, candidates: Sequence[HazardResource]
    ) -> list[HazardResource]: ...


class ResourceSelectionKey(NamedTuple):
    hazard_type: type[Hazard]
    indicator_id: str


class InventoryHazardResourceSelector(HazardResourceSelector):
    """Select hazard resources from the inventory in cascade order."""

    def __init__(self, inventory: Inventory):
        self._inventory = inventory
        self._selection_rules: dict[ResourceSelectionKey, ResourceSelectionRule] = {}

    def add_selection_rule(
        self, hazard_type: type[Hazard], indicator_id: str, rule: ResourceSelectionRule
    ):
        self._selection_rules[ResourceSelectionKey(hazard_type, indicator_id)] = rule

    def hazard_indicators(self) -> dict[type[Hazard], list[str]]:
        result: dict[type[Hazard], list[str]] = {}
        for (
            hazard,
            indicator_id,
        ), resources in self._inventory.resources_by_type_id.items():
            if not resources:
                continue
            try:
                hazard_type = hazard_class(hazard)
            except AttributeError:
                logger.warning(
                    f"unable to find hazard class for hazard {hazard}, skipping"
                )
                continue
            result.setdefault(hazard_type, []).append(indicator_id)
        return result

    def get_resources(
        self,
        hazard_type: type[Hazard],
        indicator_id: str,
        hint: HazardDataHint | None = None,
    ) -> list[HazardResource]:
        candidate_resources = list(
            self._inventory.resources_by_type_id[(hazard_type.__name__, indicator_id)]
        )
        if not candidate_resources:
            raise RuntimeError(
                f"unable to find any resources for hazard {hazard_type.__name__} "
                f"and indicator ID {indicator_id}"
            )
        try:
            if hint is not None:
                matching_resources = [
                    resource
                    for resource in candidate_resources
                    if resource.path == hint.path
                ]
                return [matching_resources[0]]
            rule = self._selection_rules.get(
                ResourceSelectionKey(hazard_type, indicator_id)
            )
            return (
                rule(candidate_resources)
                if rule is not None
                else [candidate_resources[0]]
            )
        except Exception as error:
            raise RuntimeError(
                f"unable to select resources for hazard {hazard_type.__name__} "
                f"and indicator ID {indicator_id}: {error}"
            ) from error


class CoreFloodModels(Enum):
    WRI = 1
    TUDelft = 2


class CoreInventoryHazardResourceSelector(InventoryHazardResourceSelector):
    def __init__(
        self, inventory: Inventory, flood_model: CoreFloodModels = CoreFloodModels.WRI
    ):
        super().__init__(inventory)
        for indicator_id in [
            "mean_work_loss/low",
            "mean_work_loss/medium",
            "mean_work_loss/high",
        ]:
            self.add_selection_rule(
                ChronicHeat, indicator_id, chronic_heat_selection_rule
            )
        self.add_selection_rule(
            ChronicHeat, "mean/degree/days/above/32c", chronic_heat_selection_rule
        )
        self.add_selection_rule(
            Drought, "months/spei12m/below/index", drought_selection_rule
        )  # legacy
        self.add_selection_rule(
            Drought, "months/spei12m/below/threshold", drought_selection_rule
        )
        self.add_selection_rule(
            PluvialInundation, "flood_depth", pluvial_inundation_selection_rule
        )
        self.add_selection_rule(
            RiverineInundation,
            "flood_depth",
            riverine_inundation_selection_rule
            if flood_model == CoreFloodModels.WRI
            else riverine_inundation_tudelft_selection_rule,
        )
        self.add_selection_rule(
            CoastalInundation, "flood_depth", coastal_inundation_selection_rule
        )
        self.add_selection_rule(Wind, "max_speed", wind_selection_rule)
        self.add_selection_rule(
            Landslide, "landslide_susceptability", landslide_selection_rule
        )


def chronic_heat_selection_rule(
    candidates: Sequence[HazardResource],
) -> list[HazardResource]:
    matches = [
        resource
        for resource in candidates
        if resource.indicator_model_gcm == "ACCESS-CM2"
    ]
    return [matches[0]]


def coastal_inundation_selection_rule(
    candidates: Sequence[HazardResource],
) -> list[HazardResource]:
    matches = [
        resource for resource in candidates if resource.indicator_model_id == "wtsub/95"
    ]
    return [matches[0]]


def drought_selection_rule(
    candidates: Sequence[HazardResource],
) -> list[HazardResource]:
    matches = [
        resource
        for resource in candidates
        if resource.indicator_model_gcm == "multi_model_0"
    ]
    return [matches[0]]


def pluvial_inundation_selection_rule(
    candidates: Sequence[HazardResource],
) -> list[HazardResource]:
    # JBA resources are API-only, so the Zarr cascade excludes them.
    return [resource for resource in candidates if not resource.path.startswith("jba_")]


def riverine_inundation_selection_rule(
    candidates: Sequence[HazardResource],
) -> list[HazardResource]:
    # Use this GCM for historical data too, to avoid discontinuities with the baseline dataset.
    matches = [
        resource
        for resource in candidates
        if resource.indicator_model_gcm == "MIROC-ESM-CHEM"
    ]
    return [matches[0]]


def riverine_inundation_tudelft_selection_rule(
    candidates: Sequence[HazardResource],
) -> list[HazardResource]:
    tudelft = [
        resource for resource in candidates if resource.indicator_model_id == "tudelft"
    ]
    wri = [
        resource
        for resource in candidates
        if resource.indicator_model_gcm == "MIROC-ESM-CHEM"
    ]
    return [tudelft[0], wri[0]]


def wind_selection_rule(candidates: Sequence[HazardResource]) -> list[HazardResource]:
    preferred = [resource for resource in candidates if resource.group_id == "iris_osc"]
    return [preferred[0] if preferred else candidates[0]]


def landslide_selection_rule(
    candidates: Sequence[HazardResource],
) -> list[HazardResource]:
    matches = [
        resource for resource in candidates if resource.group_id == "landslide_jrc"
    ]
    return [matches[0]]


def get_default_hazard_resource_selector(
    inventory: Inventory = EmbeddedInventory(),
) -> HazardResourceSelector:
    return CoreInventoryHazardResourceSelector(inventory)
