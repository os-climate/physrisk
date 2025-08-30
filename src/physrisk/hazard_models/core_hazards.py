from enum import Enum
from pathlib import PosixPath, PurePosixPath
from typing import Dict, Iterable, List, NamedTuple, Optional, Protocol, Sequence, Type

from physrisk.api.v1.hazard_data import HazardResource
from physrisk.data.hazard_data_provider import (
    HazardDataHint,
    ResourcePaths,
    ScenarioPaths,
    SourcePaths,
)
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.kernel import hazards
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    Drought,
    Hazard,
    RiverineInundation,
    Wind,
)


class ResourceSubset:
    def __init__(self, resources: Iterable[HazardResource]):
        self.resources = list(resources)

    def any(self):
        return any(self.resources)

    def first(self):
        return [next(r for r in self.resources)]

    def match(self, hint: HazardDataHint):
        return [next(r for r in self.resources if r.path == hint.path)]

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
    all matches."""

    def __call__(
        self,
        *,
        candidates: ResourceSubset,
        hint: Optional[HazardDataHint] = None,
    ) -> List[HazardResource]: ...


class ResourceSelectorKey(NamedTuple):
    hazard_type: type
    indicator_id: str


class InventorySourcePaths(SourcePaths):
    """Class used to generate SourcePaths by selecting the appropriate HazardResource from the
    Inventory of HazardResources.
    """

    def __init__(self, inventory: Inventory):
        self._inventory = inventory
        self._selectors: Dict[ResourceSelectorKey, ResourceSelector] = {}

    def add_selector(
        self, hazard_type: type, indicator_id: str, selector: ResourceSelector
    ):
        self._selectors[ResourceSelectorKey(hazard_type, indicator_id)] = selector

    def all_hazards(self):
        return set(
            htype for ((htype, _), _) in self._inventory.resources_by_type_id.items()
        )

    def hazard_types(self):
        return [hazards.hazard_class(ht) for ht in self.all_hazards()]

    def resource_paths(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        scenarios: Sequence[str],
        hint: Optional[HazardDataHint] = None,
    ) -> List[ResourcePaths]:
        resources = self.get_resources(hazard_type, indicator_id, hint=hint)
        result = []
        for r in resources:
            result.append(
                ResourcePaths(
                    resource_path=r.path,
                    scenarios={
                        s: InventorySourcePaths.scenario_paths_for_resource(r, s)
                        for s in scenarios
                    },
                )
            )
        return result

    def scenario_paths_for_id(
        self,
        resource_id: str,
        scenarios: Sequence[str],
        map: bool = False,
        map_zoom: Optional[int] = None,
    ):
        r = self._inventory.resources[resource_id]
        return {
            s: self.scenario_paths_for_resource(r, s, map, map_zoom) for s in scenarios
        }

    @staticmethod
    def scenario_paths_for_resource(
        resource: HazardResource,
        scenario_id: str,
        map: bool = False,
        map_zoom: Optional[int] = None,
    ):
        if map:
            assert resource.map is not None
            # is this a pyramid of tiles?
            is_pyramid = resource.map.source != "map_array"
            path = (
                str(PosixPath(resource.path).with_name(resource.map.path))
                if len(PosixPath(resource.map.path).parts) == 1
                else resource.map.path
            )
            if is_pyramid:
                path = str(PurePosixPath(path, str(map_zoom)))
        else:
            path = resource.path

        if scenario_id == "historical":
            # there are some cases where there is no historical scenario or -
            # more commonly - we do not want to use. We have seen cases where there is
            # an apparent inconsistency.
            # in such cases we allow for a proxy whereby the earliest year of the scenario with
            # lowest net flux in the identifier is used.
            scenario = next(
                iter(s for s in resource.scenarios if s.id == "historical"), None
            )
            if scenario is None:
                scenario = next(
                    s for s in sorted(resource.scenarios, key=lambda s: min(s.years))
                )
            assert scenario is not None
            year = min(scenario.years)
            return ScenarioPaths(
                [-1],
                lambda y: path.format(
                    id=resource.indicator_id,
                    scenario=scenario.id,  # type:ignore
                    year=year,
                ),
            )
        proxy_scenario_id = (
            cmip6_scenario_to_rcp(scenario_id)
            if resource.scenarios[0].id.startswith("rcp")
            or resource.scenarios[-1].id.startswith("rcp")
            else scenario_id
        )
        scenario = next(
            iter(s for s in resource.scenarios if s.id == proxy_scenario_id), None
        )
        if scenario is None:
            return ScenarioPaths([], lambda y: "")
        else:
            return ScenarioPaths(
                scenario.years,
                lambda y: path.format(
                    id=resource.indicator_id, scenario=proxy_scenario_id, year=y
                )
                + ("/indicator" if (resource.store_netcdf_coords and not map) else ""),
            )

    def get_resources(
        self,
        hazard_type: Type[Hazard],
        indicator_id: str,
        hint: Optional[HazardDataHint] = None,
    ) -> List[HazardResource]:
        # all matching resources in the inventory
        selector = self._selectors.get(
            ResourceSelectorKey(
                hazard_type=hazard_type,
                indicator_id=indicator_id,
            ),
            self._no_selector,
        )
        resources = self._inventory.resources_by_type_id[
            (hazard_type.__name__, indicator_id)
        ]
        if len(resources) == 0:
            raise RuntimeError(
                f"unable to find any resources for hazard {hazard_type.__name__} "
                f"and indicator ID {indicator_id}"
            )
        candidates = ResourceSubset(resources)
        try:
            if hint is not None:
                resources = candidates.match(hint)
            else:
                resources = selector(candidates=candidates)
        except Exception as e:
            raise RuntimeError(
                f"unable to select resources for hazard {hazard_type.__name__} "
                f"and indicator ID {indicator_id}: {str(e)}"
            )
        return resources

    @staticmethod
    def _no_selector(
        candidates: ResourceSubset,
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
        self.add_selector(Drought, "months/spei12m/below/index", self._select_drought)
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
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.with_model_gcm("ACCESS-CM2").first()

    @staticmethod
    def _select_coastal_inundation(
        candidates: ResourceSubset,
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.with_model_id("wtsub/95").first()

    @staticmethod
    def _select_drought(
        candidates: ResourceSubset,
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.with_model_gcm("MIROC6").first()

    @staticmethod
    def _select_riverine_inundation(
        candidates: ResourceSubset,
        hint: Optional[HazardDataHint] = None,
    ):
        # we use this GCM, even for the historical scenario, where the earliest year is used.
        # because of noted discontinuities between baseline and GCM data sets.
        return candidates.with_model_gcm("MIROC-ESM-CHEM").first()

    @staticmethod
    def _select_riverine_inundation_tudelft(
        candidates: ResourceSubset,
        hint: Optional[HazardDataHint] = None,
    ):
        return (
            candidates.with_model_id("tudelft").first()
            + candidates.with_model_gcm("MIROC-ESM-CHEM").first()
        )

    @staticmethod
    def _select_wind(
        candidates: ResourceSubset,
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
    return CoreInventorySourcePaths(inventory)
