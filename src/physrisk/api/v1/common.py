from enum import Enum
from typing import Annotated, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer


def before_validator(list: list) -> npt.NDArray:
    """Deserialize a list into a numpy array."""
    return np.array(list)


def serialize_array(array: npt.NDArray):
    """Serialize a numpy array into a list."""
    return array.tolist()


NDArray = Annotated[
    npt.NDArray,
    BeforeValidator(before_validator),
    PlainSerializer(serialize_array, return_type=list),
]


class HazardType(str, Enum):
    coastal_inundation = "CoastalInundation"
    chronic_heat = "ChronicHeat"
    drought = "Drought"
    fire = "Fire"
    hail = "Hail"
    pluvial_inundation = "PluvialInundation"
    precipitation = "Precipitation"
    riverine_inundation = "RiverineInundation"
    subsidence = "Subsidence"
    water_risk = "WaterRisk"
    wind = "Wind"


class Asset(BaseModel):
    """Defines an asset.

    Assets can be specified in two ways
    1) 'asset_class' set explicitly. This must correspond to a physrisk Asset subclass, e.g. PowerGeneratingAsset',
    ManufacturingAsset, or the parent Asset. physrisk will create an instance of the asset and set its
    attributes from the other fields provided (which can therefore be specific to the class).
    2) Inferred from 'occupancy_code'.

    https://github.com/OasisLMF/ODS_OpenExposureData/blob/main/OpenExposureData/OEDInputFields.csv

    An asset's value may be impacted through damage or through disruption
    disruption being reduction of an asset's ability to generate cashflows
    (or equivalent value, e.g. by increased expense or reduced sales).
    """

    asset_class: Optional[str] = Field(
        default=None,
        description="name of asset class; corresponds to physrisk class names, e.g. PowerGeneratingAsset. If not provided, "
        "physrisk will infer the class from 'occupancy_code'.",
    )
    latitude: Optional[float] = Field(
        default=None, description="Latitude in degrees, specified in WGS84 (EPSG:4326)."
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Longitude in degrees, specified in WGS84 (EPSG:4326).",
    )
    wkt: Optional[str] = Field(
        default=None,
        description="Well Known Text representation of asset geometry, specified in WGS84 (EPSG:4326).",
    )
    buffer: float = Field(
        default=0.0,
        description="Buffer to be applied to the geometry, in metres. "
        "If 'wkt' is not provided, the buffer is applied.",
    )
    type: Optional[str] = Field(
        None,
        description="Type of the asset <level_1>/<level_2>/<level_3>. This is free-text that matches lines in the vulnerability config, or"
        " programmatic vulnerability models",
    )
    location: Optional[str] = Field(
        default=None,
        description="Location (e.g. Africa, Asia, Europe, North America, Oceania, South America); this is used to select location-specific"
        "vulnerability functions where available.",
    )
    occupancy_code: int = Field(
        default=1000,
        description="Occupancy code. This is Open Exposure Data occupancy code, unless otherwise specified (and currently no other scheme supported)."
        "Defaults to 1000, which is 'unknown', in which case physrisk asset class is inferred from 'asset_class'.",
    )
    number_of_storeys: Optional[int] = Field(
        default=-1,
        description="Number of storeys. Can also take special values: "
        "-1 = unknown number of storeys - low rise, -2 = unknown number of storeys - mid rise, -3 = Unknown number of storeys = high rise)",
    )
    basement: int = Field(
        default=0,
        description="0 = unknown / default, 1 = unfinished, 2 = completely finished, 6 = no basement",
    )
    construction_code: int = Field(
        default=5000,
        description="Construction code. This is Open Exposure Data construction code, unless otherwise specified.",
    )
    first_floor_height: float = Field(
        default=0.305,
        description="First floor height in metres above grade (local ground level). A basement, if present,"
        "is not considered to be the first floor in this definition. Default is 0.305 m (1 foot). In the US, a crawl-space foundation is commonly 4 foot for"
        "example.",
    )
    # see https://www.fema.gov/sites/default/files/documents/fema_hazus-inventory-technical-manual-6.1.pdf for information about first floor heights in US
    capacity: Optional[float] = Field(
        default=None,
        description="Power generation capacity in MW for power generating assets.",
        kw_only=True,
    )

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Commercial",
                    "latitude": 22.2972,
                    "longitude": 91.8062,
                    "location": "Asia",
                }
            ]
        },
    }


class Assets(BaseModel):
    """Defines a collection of assets."""

    items: List[Asset]


class BaseHazardRequest(BaseModel):
    group_ids: List[str] = Field(
        ["public"],
        description="""List of data groups which can be used to service the request,
            e.g. 'osc': available to OS-Climate members (e.g. pending license decision),
                 'public'.""",
    )


class Country(BaseModel):
    """Country information."""

    country: str
    continent: str
    country_iso_a3: str


class Countries(BaseModel):
    """List of Country."""

    items: List[Country]


class IntensityCurve(BaseModel):
    """Hazard indicator intensity curve. Acute hazards are parameterized by event intensities and
    return periods in years. Chronic hazards are parameterized by a set of index values.
    Index values are defined per indicator."""

    intensities: List[float] = Field([], description="Hazard indicator intensities.")
    return_periods: Optional[Sequence[float]] = Field(
        [],
        description="[Deprecated] Return period in years in the case of an acute hazard.",
    )
    index_values: Optional[Union[Sequence[float], Sequence[str]]] = Field(
        [],
        description="Set of index values. \
            This is return period in years in the case of an acute hazard or \
            a set of indicator value thresholds in the case of a multi-threshold chronic hazard.",
    )
    index_name: str = Field(
        "",
        description="Name of the index. In the case of an acute hazard this is 'return period'; \
            for a multi-threshold chronic hazard this is 'threshold'.",
    )


class ExceedanceCurve(BaseModel):
    """General exceedance curve (e.g. hazazrd, impact)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    values: NDArray = Field(default_factory=lambda: np.zeros(10), description="")
    exceed_probabilities: NDArray = Field(
        default_factory=lambda: np.zeros(10), description=""
    )


class Distribution(BaseModel):
    """General exceedance curve (e.g. hazazrd, impact)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    bin_edges: NDArray = Field(default_factory=lambda: np.zeros(11), description="")
    probabilities: NDArray = Field(default_factory=lambda: np.zeros(10), description="")


class HazardEventDistrib(BaseModel):
    """Intensity curve of an acute hazard."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    intensity_bin_edges: NDArray = Field(
        default_factory=lambda: np.zeros(10), description=""
    )
    probabilities: NDArray = Field(default_factory=lambda: np.zeros(10), description="")
    path: List[str] = Field([], description="Path to the hazard indicator data source.")


class VulnerabilityCurve(BaseModel):
    """Defines a damage or disruption curve."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    asset_type: str = Field(...)
    location: str = Field(...)
    event_type: str = Field(description="hazard event type, e.g. RiverineInundation")
    impact_type: str = Field(description="'Damage' or 'Disruption'")
    intensity: List[float] = Field(...)
    intensity_units: str = Field(description="units of the intensity")
    impact_mean: List[float] = Field(description="mean impact (damage or disruption)")
    impact_std: List[float] = Field(
        description="standard deviation of impact (damage or disruption)"
    )


class VulnerabilityCurves(BaseModel):
    """List of VulnerabilityCurve."""

    items: List[VulnerabilityCurve]


class VulnerabilityDistrib(BaseModel):
    """Defines a vulnerability matrix."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    intensity_bin_edges: NDArray = Field(
        default_factory=lambda: np.zeros(10), description=""
    )
    impact_bin_edges: NDArray = Field(
        default_factory=lambda: np.zeros(10), description=""
    )
    prob_matrix: NDArray = Field(default_factory=lambda: np.zeros(10), description="")
