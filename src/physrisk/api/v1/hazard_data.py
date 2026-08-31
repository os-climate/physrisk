from collections.abc import Iterable, Sequence
from enum import Flag, auto
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, computed_field, field_validator

from physrisk.api.v1.common import BaseHazardRequest, IntensityCurve
from physrisk.kernel.hazards import all_hazards


class Colormap(BaseModel):
    """Provides details of colormap."""

    min_index: int | None = Field(
        1,
        description="Value of colormap minimum. Constant min for a group of maps can facilitate comparison.",
    )
    min_value: float = Field(
        description="Value of colormap minimum. Constant min for a group of maps can facilitate comparison."
    )
    max_index: int | None = Field(
        255,
        description="Value of colormap maximum. Constant max for a group of maps can facilitate comparison.",
    )
    max_value: float = Field(
        description="Value of colormap maximum. Constant max for a group of maps can facilitate comparison."
    )
    name: str = Field(description="Name of colormap, e.g. 'flare', 'heating'.")
    nodata_index: int | None = Field(0, description="Index used for no data.")
    units: str = Field(description="Units, e.g. 'degree days', 'metres'.")
    scaling: Literal["linear", "log"] = Field(
        "linear",
        description="Value-to-colour scaling: 'linear' (default) or 'log'. "
        "'log' requires min_value > 0.",
    )


class MapInfo(BaseModel):
    """Provides information about map layer."""

    colormap: Colormap | None = Field(description="Details of colormap.")
    bounds: list[tuple[float, float]] = Field(
        [(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
        description="Bounds (top/left, top/right, bottom/right, bottom/left) as degrees. Note applied to map reprojected into Web Mercator CRS.",
    )
    bbox: list[float] | None = Field(default=[-180.0, -85.0, 180.0, 85.0])
    index_values: Sequence[float | int | str] | None = Field(
        default=None,
        description="Index values to include in maps. If None, the last index value only is included.",
    )
    path: str = Field(
        description="Name of array reprojected to Web Mercator for on-the-fly display or to hash to obtain tile ID. If not supplied, convention is to add '_map' to path."
    )
    # note that the bounds should be consistent with the array attributes
    source: str | None = Field(
        description="""Source of map image. These are
                            'map_array': single Mercator projection array at path above
                            'map_array_pyramid': pyramid of Mercator projection arrays
                            'mapbox'.
                            """
    )


class Period(BaseModel):
    """Provides information about a period, which currently corresponds to a year, belonging to a scenario."""

    year: int
    map_id: str = Field(
        description="If present, identifier to be used for looking up map tiles from server."
    )


class Scenario(BaseModel):
    """Scenario ID and the list of available years for that scenario e.g. RCP8.5 = 'rcp8.5'"""

    id: str
    years: list[int]
    # periods: Optional[List[Period]]


def expanded(item: str, key: str, param: str):
    return item and item.replace("{" + key + "}", param)


class HazardResource(BaseModel):
    """Provides information about a set of hazard indicators, including available scenarios and years."""

    hazard_type: str = Field(description="Type of hazard.")
    group_id: str | None = Field(
        "public",
        description="Identifier of the resource group (used for authentication).",
    )
    path: str = Field(description="Full path to the indicator array.")
    indicator_id: str = Field(
        description="Identifier of the hazard indicator (i.e. the modelled quantity), e.g. 'flood_depth'."
    )
    indicator_model_id: str | None = Field(
        None,
        description="Identifier specifying the type of model used in the derivation of the indicator "
        "(e.g. whether flood model includes impact of sea-level rise).",
    )
    indicator_model_gcm: str = Field(
        description="Identifier of general circulation model(s) used in the derivation of the indicator."
    )
    params: dict[str, list[str]] = Field(
        {}, description="Parameters used to expand wild-carded fields."
    )
    display_name: str = Field(description="Text used to display indicator.")
    display_groups: list[str] = Field(
        [], description="Text used to group the (expanded) indicators for display."
    )
    description: str = Field(
        description="Brief description in mark down of the indicator and model that generated the indicator."
    )
    map: MapInfo | None = Field(
        None,
        description="Optional information used for display of the indicator in a map.",
    )
    scenarios: list[Scenario] = Field(
        description="Climate change scenarios for which the indicator is available."
    )
    store_netcdf_coords: bool = Field(
        False,
        description="If True, NetCDF-style coordinates are also stored, which allows XArray to read the array \
            natively. In this case, path still points to the array; the coordinates are stored in an array group \
                in the parent folder. That is, path should be in the form path_to_array_group/array_group/array",
    )
    units: str = Field(description="Units of the hazard indicator.")

    def expand(self):
        keys = list(self.params.keys())
        return expand_resource(self, keys, self.params)

    def key(self):
        """Unique key for the resource. array_path should be unique, although indicator_id is typically not.
        Vulnerability models request a hazard indicator by indicator_id from the Hazard Model. The Hazard Model
        selects based on its own logic (e.g. selects a particular General Circulation Model)."""
        return self.path


def expand(item: str, key: str, param: str):
    return item and item.replace("{" + key + "}", param)


def expand_resource(
    resource: HazardResource, keys: list[str], params: dict[str, list[str]]
) -> Iterable[HazardResource]:
    if len(keys) == 0:
        yield resource.model_copy(deep=True, update={"params": {}})
    else:
        keys = keys.copy()
        key = keys.pop()
        for item in expand_resource(resource, keys, params):
            for param in params[key]:
                yield item.model_copy(
                    deep=True,
                    update={
                        "indicator_id": expand(item.indicator_id, key, param),
                        "indicator_model_gcm": expand(
                            item.indicator_model_gcm, key, param
                        ),
                        "display_name": expand(item.display_name, key, param),
                        "path": expand(item.path, key, param),
                        "map": (
                            None
                            if item.map is None
                            else (
                                item.map.model_copy(
                                    deep=True,
                                    update={
                                        "path": expand(
                                            item.map.path
                                            if item.map.path is not None
                                            else "",
                                            key,
                                            param,
                                        )
                                    },
                                )
                            )
                        ),
                    },
                )


class InventorySource(Flag):
    """Source of inventory. Where multiple are selected, order is as shown:
    results from HAZARD_TEST override those of HAZARD and EMBEDDED (to facilitate testing).
    """

    EMBEDDED = auto()  # inventory embedded in physrisk
    HAZARD = auto()  # inventory stored in the S3 hazard location
    HAZARD_TEST = auto()  # inventory stored in the S3 hazard_test location


class HazardAvailabilityRequest(BaseModel):
    types: list[str] | None = []  # e.g. ["RiverineInundation"]
    sources: list[str] | None = Field(
        None,
        description="Sources of inventory, can be 'embedded', 'hazard' or 'hazard_test'.",
    )


class HazardAvailabilityResponse(BaseModel):
    models: list[HazardResource]
    colormaps: dict


class HazardDescriptionRequest(BaseModel):
    paths: list[str] = Field(description="List of paths to markdown objects.")


class HazardDescriptionResponse(BaseModel):
    descriptions: dict[str, str] = Field(
        description="For each path (key), the description markdown (value)."
    )


class StaticInformationResponse(BaseModel):
    scenario_descriptions: dict[str, str] = Field(
        description="For each scenario identifier (key, e.g. 'ssp585', 'rcp8p5'), the description markdown (value)."
    )
    oed_occupancy_codes: dict[int, str] = Field(
        description="OED occupancy codes: mapping of OED Code (integer) to OED Label."
    )


def _supported_hazard_type_names() -> set[str]:
    return {hazard.__name__ for hazard in all_hazards()}


class HazardDataRequestItem(BaseModel):
    longitudes: list[float]
    latitudes: list[float]
    request_item_id: str
    hazard_type: str | None = Field(
        default=None,
        description="Type of hazard, e.g. RiverineInundation.",
        json_schema_extra={"enum": sorted(_supported_hazard_type_names())},
        validation_alias=AliasChoices("hazard_type", "event_type"),
    )
    indicator_id: str
    indicator_model_gcm: str | None = ""
    path: str | None = None
    scenario: str  # e.g. rcp8p5
    year: int

    @field_validator("hazard_type")
    @classmethod
    def validate_hazard_type(cls, hazard_type: str | None) -> str | None:
        if hazard_type is None:
            return None
        supported_hazard_types = _supported_hazard_type_names()
        if hazard_type not in supported_hazard_types:
            raise ValueError(
                f"hazard_type must be one of the hazard classes defined in physrisk.kernel.hazards. "
                f"Got {hazard_type!r}; expected one of {sorted(supported_hazard_types)}."
            )
        return hazard_type

    @computed_field(description="Deprecated: same as hazard_type")  # type: ignore[misc]
    @property
    def event_type(self) -> str | None:
        """Deprecated: use hazard_type instead."""
        return self.hazard_type


class HazardDataRequest(BaseHazardRequest):
    """Request hazard indicator data for a set of locations, specified by latitude and longitude
    (in coordinate reference system EPSG:4326).
    A hazard indicator is defined by the type of the hazard ('hazard_type') and the indicator ('indicator_id').
    A hazard indicator is a measure that is used to quantify the hazard. For example, for hazard type
    'RiverineInundation', flood depth (unprotected) is given by the indicator 'flood_depth'.
    In addition, 'scenario' and 'year' are required. For 'historical' scenario, year is ignored and can
    be set to e.g. '-1'. Scenarios can be RCPs and SSPs; convention is e.g. 'rcp8p5' (RCP 8.5) and
    'ssp585'.

    If only 'hazard_type' and 'indicator_id' are provided, then the default physrisk logic will be used to
    identify the data sets. If a specific data set is required, 'path' can be supplied. The path is the
    unique identifier of a 'hazard resource'. A hazard resource is a collection of data arrays for different
    scenarios and years, typically made available by the same model provider.

    Current behaviour is that empty arrays are returned if there is no match for the scenario and year in the
    selected / specified resource (an alternative would be to allow interpolation of years and proxying/interpolation
    of scenarios).
    """

    interpolation: str = "floor"
    interpolate_years: bool = True
    provider_max_requests: dict[str, int] = Field(
        {},
        description="The maximum permitted number of \
        requests to external providers. This setting is intended in particular for paid-for data. The key \
        is the provider ID and the value is the maximum permitted requests.",
    )
    items: list[HazardDataRequestItem]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {
                            "request_item_id": "my_id",
                            "hazard_type": "RiverineInundation",
                            "indicator_id": "flood_depth",
                            "latitudes": [45.4089, 43.4923],
                            "longitudes": [20.3162, 4.7877],
                            "path": "inundation/river_tudelft/v2/flood_depth_unprot_{scenario}_{year}",
                            "scenario": "rcp8p5",
                            "year": 2035,
                        }
                    ]
                },
                {
                    "items": [
                        {
                            "request_item_id": "my_id",
                            "hazard_type": "RiverineInundation",
                            "indicator_id": "flood_depth",
                            "latitudes": [45.4089, 43.4923],
                            "longitudes": [20.3162, 4.7877],
                            "path": "inundation/river_tudelft/v2/flood_depth_unprot_{scenario}_{year}",
                            "scenario": "historical",
                            "year": -1,
                        }
                    ]
                },
            ]
        }
    }


class HazardDataResponseItem(BaseModel):
    intensity_curve_set: list[IntensityCurve]
    request_item_id: str
    hazard_type: str | None
    indicator_id: str
    scenario: str
    year: int

    @computed_field(description="Deprecated: same as hazard_type")  # type: ignore[misc]
    @property
    def event_type(self) -> str | None:
        """Deprecated: use hazard_type instead."""
        return self.hazard_type

    @computed_field(description="Deprecated: same as indicator_id")  # type: ignore[misc]
    @property
    def model(self) -> str:
        """Deprecated: use indicator_id instead."""
        return self.indicator_id


class HazardDataResponse(BaseModel):
    items: list[HazardDataResponseItem]
