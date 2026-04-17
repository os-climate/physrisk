from typing import Dict, List
from pydantic import BaseModel, Field
from physrisk.api.v1.hazard_data import Scenario


class AvailabilitySourcesRequest(BaseModel):
    """Hazard data availability request"""

    include_all: bool = Field(
        False,
        description="If true, brings back all available information about years and scenarios per each hazard",
    )
    use_case_id: str = Field(
        "DEFAULT",
        description="Use case id determines which set of hazards will be taken into account depending on the selected vulnerability models",
    )
    selected_hazards_list: list[str] = Field(
        [], description="Specify which list of hazards the data is required for"
    )


class HazardTypeAvailability(BaseModel):
    scenarios: List[Scenario] = []
    available_for_assets_type: List[str] = []
    indicator_display_name: str = ""


class AvailabilitySourcesResponse(BaseModel):
    hazards: Dict[str, Dict[str, HazardTypeAvailability]] = Field(
        {},
        description="Information of available sources for the hazards in the inventory",
    )
    message: str = Field(
        "",
        description="Aditional information related to the obtained available sources",
    )
