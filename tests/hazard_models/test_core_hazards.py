from physrisk.api.v1.hazard_data import HazardResource, Scenario
from physrisk.data.inventory import Inventory
from physrisk.hazard_models.core_hazards import (
    InventoryHazardResourceSelector,
)
from physrisk.kernel.hazards import ChronicHeat


def _hazard_resource(hazard_type: str, indicator_id: str) -> HazardResource:
    return HazardResource(
        path=f"test/{hazard_type}/{indicator_id}/{{scenario}}_{{year}}",
        hazard_type=hazard_type,
        indicator_id=indicator_id,
        indicator_model_gcm="test_gcm",
        display_name=f"{hazard_type} {indicator_id}",
        description="Test hazard resource",
        scenarios=[Scenario(id="ssp585", years=[2030])],
        units="test_units",
    )


def test_hazard_indicators_skips_hazards_without_class() -> None:
    known_resource = _hazard_resource("ChronicHeat", "test_indicator")
    unknown_resource = _hazard_resource("UnknownHazard", "test_indicator")
    inventory = Inventory(hazard_resources=[known_resource, unknown_resource])
    resource_selector = InventoryHazardResourceSelector(inventory)

    assert resource_selector.hazard_indicators() == {ChronicHeat: ["test_indicator"]}
