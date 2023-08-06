from typing import Callable, Dict, List, Optional, Tuple

from physrisk.api.v1.hazard_data import HazardResource
from physrisk.data.hazard_data_provider import HazardDataHint, SourcePath
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.kernel import hazards
from physrisk.kernel.hazards import ChronicHeat, CoastalInundation, RiverineInundation

Selector = Callable[[List[HazardResource], str, int], HazardResource]


class ResourceSelector:
    def __init__(self):
        self._selectors: Dict[Tuple[type, str], Selector] = {}

    def select_resource(
        self, hazard_type: type, indicator_id: str, resources: List[HazardResource], scenario: str, year: int
    ):
        if (hazard_type, indicator_id) not in self._selectors:
            return resources[0]
        return self._selectors[(hazard_type, indicator_id)](resources, scenario, year)

    def _add_selector(self, hazard_type: type, indicator_id: str, selector: Selector):
        self._selectors[(hazard_type, indicator_id)] = selector

    def _use_gcm(self, gcm: str) -> Selector:
        selector: Selector = lambda resources, scenario, year: next(
            r for r in resources if r.indicator_model_gcm == gcm
        )
        return selector


class EmbeddedResourceSelector(ResourceSelector):
    def __init__(self):
        super().__init__()

        # whether for retrospective or prospective scenarios use GCM:
        self._add_selector(ChronicHeat, "mean_degree_days/above/32c", self._use_gcm("ACCESS-CM2"))

        for intensity in ["low", "medium", "high"]:
            self._add_selector(ChronicHeat, f"mean_work_loss/{intensity}", self._use_gcm("ACCESS-CM2"))

        # for retrospective scenarios use historical data (as opposed to bias-corrected retrospective run of GCM):
        selector: Selector = (
            lambda resources, scenario, year: next(r for r in resources if r.indicator_model_gcm == "historical")
            if scenario == "historical"
            else next(r for r in resources if r.indicator_model_gcm == "MIROC-ESM-CHEM")
        )
        self._add_selector(RiverineInundation, "flood_depth", selector)
        self._add_selector(CoastalInundation, "flood_depth", selector)


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
        if scenario not in ["rcp2p6", "rcp4p5", "rcp8p5", "historical"]:
            raise ValueError(f"unexpected scenario {scenario}")
        return scenario


def get_default_source_paths(inventory: Inventory = EmbeddedInventory()):
    return get_source_paths(inventory, EmbeddedResourceSelector())


def get_source_paths(inventory: Inventory, selector: ResourceSelector):
    all_hazard_types = list(set(htype for ((htype, _), _) in inventory.resources_by_type_id.items()))
    source_paths: Dict[type, SourcePath] = {}
    for hazard_type in all_hazard_types:
        source_paths[hazards.hazard_class(hazard_type)] = get_resource_source_path(hazard_type, inventory, selector)
    return source_paths


def get_resource_source_path(hazard_type: str, inventory: Inventory, selector: ResourceSelector):
    def get_source_path(*, indicator_id: str, scenario: str, year: int, hint: Optional[HazardDataHint] = None):
        # all matching resources in the inventory
        if hint is not None:
            return hint.path
        resources = inventory.resources_by_type_id[hazard_type, indicator_id]
        selected_resource = selector.select_resource(
            hazards.hazard_class(hazard_type), indicator_id, resources, scenario, year
        )
        proxy_scenario = (
            cmip6_scenario_to_rcp(scenario) if selected_resource.scenarios[0].id.startswith("rcp") else scenario
        )
        if scenario == "historical":
            year = next(y for y in next(s for s in selected_resource.scenarios if s.id == "historical").years)
        return selected_resource.path.format(id=indicator_id, scenario=proxy_scenario, year=year)

    return get_source_path
