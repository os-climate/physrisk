from enum import Enum
from typing import Dict, Iterable, NamedTuple, Optional, Protocol

from physrisk.api.v1.hazard_data import HazardResource
from physrisk.data.hazard_data_provider import HazardDataHint, SourcePath
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.kernel import hazards
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    RiverineInundation,
    Wind,
)


class ResourceSubset:
    def __init__(self, resources: Iterable[HazardResource]):
        self.resources = list(resources)

    def any(self):
        return any(self.resources)

    def first(self):
        return next(r for r in self.resources)

    def match(self, hint: HazardDataHint):
        return next(r for r in self.resources if r.path == hint.path)

    def prefer_group_id(self, group_id: str):
        with_condition = self.with_group_id(group_id)
        return with_condition if with_condition.any() else self

    def with_group_id(self, group_id: str):
        return ResourceSubset(r for r in self.resources if r.group_id == group_id)

    def with_model_gcm(self, gcm: str):
        return ResourceSubset(r for r in self.resources if r.indicator_model_gcm == gcm)

    def with_model_id(self, model_id: str):
        return ResourceSubset(
            r for r in self.resources if r.indicator_model_id == model_id
        )


class ResourceSelector(Protocol):
    """For a particular hazard type and indicator_id (specifying the type of indicator),
    defines the rule for selecting a resource from
    all matches. The selection rule depends on scenario and year."""

    def __call__(
        self,
        *,
        candidates: ResourceSubset,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ) -> HazardResource: ...


class ResourceSelectorKey(NamedTuple):
    hazard_type: type
    indicator_id: str


class InventorySourcePaths:
    """Class used to generate SourcePaths by selecting the appropriate HazardResource from the
    Inventory of HazardResources.
    """

    def __init__(self, inventory: Inventory):
        self._inventory = inventory
        self._selectors: Dict[ResourceSelectorKey, ResourceSelector] = {}

    def source_paths(self) -> Dict[type, SourcePath]:
        all_hazard_types = list(
            set(
                htype
                for ((htype, _), _) in self._inventory.resources_by_type_id.items()
            )
        )
        source_paths: Dict[type, SourcePath] = {}
        for hazard_type in all_hazard_types:
            source_paths[hazards.hazard_class(hazard_type)] = (
                self._get_resource_source_path(
                    hazard_type,
                )
            )
        return source_paths

    def add_selector(
        self, hazard_type: type, indicator_id: str, selector: ResourceSelector
    ):
        self._selectors[ResourceSelectorKey(hazard_type, indicator_id)] = selector

    def _get_resource_source_path(self, hazard_type: str):
        def _get_source_path(
            *,
            indicator_id: str,
            scenario: str,
            year: int,
            hint: Optional[HazardDataHint] = None,
        ):
            # all matching resources in the inventory
            selector = self._selectors.get(
                ResourceSelectorKey(
                    hazard_type=hazards.hazard_class(hazard_type),
                    indicator_id=indicator_id,
                ),
                self._no_selector,
            )
            resources = self._inventory.resources_by_type_id[
                (hazard_type, indicator_id)
            ]
            if len(resources) == 0:
                raise RuntimeError(
                    f"unable to find any resources for hazard {hazard_type} "
                    f"and indicator ID {indicator_id}"
                )
            candidates = ResourceSubset(resources)
            try:
                if hint is not None:
                    resource = candidates.match(hint)
                else:
                    resource = selector(
                        candidates=candidates, scenario=scenario, year=year, hint=hint
                    )
            except Exception:
                raise RuntimeError(
                    f"unable to select unique resource for hazard {hazard_type} "
                    f"and indicator ID {indicator_id}"
                )
            proxy_scenario = (
                cmip6_scenario_to_rcp(scenario)
                if resource.scenarios[0].id.startswith("rcp")
                or resource.scenarios[-1].id.startswith("rcp")
                else scenario
            )
            if scenario == "historical":
                # there are some cases where there is no historical scenario or -
                # more commonly - we do not want to use. We have seen cases where there is
                # an apparent inconsistency.
                # in such cases we allow for a proxy whereby the earliest year of the scenario with
                # lowest net flux in the identifier is used.
                scenarios = next(
                    iter(s for s in resource.scenarios if s.id == "historical"), None
                )
                if scenarios is None:
                    scenarios = next(
                        s
                        for s in sorted(resource.scenarios, key=lambda s: min(s.years))
                    )
                proxy_scenario = scenarios.id
                year = min(scenarios.years)
            return resource.path.format(
                id=indicator_id, scenario=proxy_scenario, year=year
            )

        return _get_source_path

    @staticmethod
    def _no_selector(
        candidates: ResourceSubset,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.first()


class CoreFloodModels(Enum):
    WRI = 1
    TUDelft = 2


class CoreInventorySourcePaths(InventorySourcePaths):
    def __init__(
        self, inventory: Inventory, flood_model: CoreFloodModels = CoreFloodModels.WRI
    ):
        super().__init__(inventory)
        for indicator_id in [
            "mean_work_loss/low",
            "mean_work_loss/medium",
            "mean_work_loss/high",
        ]:
            self.add_selector(ChronicHeat, indicator_id, self._select_chronic_heat)
        self.add_selector(
            ChronicHeat, "mean/degree/days/above/32c", self._select_chronic_heat
        )
        self.add_selector(
            RiverineInundation,
            "flood_depth",
            self._select_riverine_inundation
            if flood_model == CoreFloodModels.WRI
            else self._select_riverine_inundation_tudelft,
        )
        self.add_selector(
            CoastalInundation, "flood_depth", self._select_coastal_inundation
        )
        self.add_selector(Wind, "max_speed", self._select_wind)

    def resources_with(self, *, hazard_type: type, indicator_id: str):
        return ResourceSubset(
            self._inventory.resources_by_type_id[(hazard_type.__name__, indicator_id)]
        )

    @staticmethod
    def _select_chronic_heat(
        candidates: ResourceSubset,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.with_model_gcm("ACCESS-CM2").first()

    @staticmethod
    def _select_coastal_inundation(
        candidates: ResourceSubset,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.with_model_id("wtsub/95").first()

    @staticmethod
    def _select_riverine_inundation(
        candidates: ResourceSubset,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        # we use this GCM, even for the historical scenario, where the earliest year is used.
        # because of noted discontinuities between baseline and GCM data sets.
        return candidates.with_model_gcm("MIROC-ESM-CHEM").first()

    @staticmethod
    def _select_riverine_inundation_tudelft(
        candidates: ResourceSubset,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.with_model_id("tudelft").first()

    @staticmethod
    def _select_wind(
        candidates: ResourceSubset,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.prefer_group_id("iris_osc").first()


def cmip6_scenario_to_rcp(scenario: str):
    """Convention is that CMIP6 scenarios are expressed by identifiers:
    SSP1-2.6: 'ssp126'
    SSP2-4.5: 'ssp245'
    SSP5-8.5: 'ssp585' etc.
    Here we translate to form
    RCP-4.5: 'rcp4p5'
    RCP-8.5: 'rcp8p5' etc.
    """
    if scenario == "ssp126":
        return "rcp2p6"
    elif scenario == "ssp245":
        return "rcp4p5"
    elif scenario == "ssp585":
        return "rcp8p5"
    else:
        if scenario not in ["rcp2p6", "rcp4p5", "rcp6p0", "rcp8p5", "historical"]:
            raise ValueError(f"unexpected scenario {scenario}")
        return scenario


def get_default_source_path_provider(inventory: Inventory = EmbeddedInventory()):
    return CoreInventorySourcePaths(inventory)


def get_default_source_paths(inventory: Inventory = EmbeddedInventory()):
    return CoreInventorySourcePaths(inventory).source_paths()
