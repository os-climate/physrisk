from typing import Annotated, Dict, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer


# def deserialize_list(list: list) -> npt.NDArray:
#     """Deserialize a list into a numpy array."""
#     return np.array(list)


def serialize_array(array: npt.NDArray) -> str:
    """Serialize a numpy array into a list."""
    return array.tolist()


NDArray = Annotated[
    npt.NDArray,
    # AfterValidator(deserialize_list),
    PlainSerializer(serialize_array, return_type=list),
]


class Asset(BaseModel):
    """Defines an asset.

    An asset is identified first by its asset_class and then by its type within the class.
    An asset's value may be impacted through damage or through disruption
    disruption being reduction of an asset's ability to generate cashflows
    (or equivalent value, e.g. by reducing expenses or increasing sales).
    """

    # model_config = ConfigDict(extra="allow")

    asset_class: str = Field(
        description="name of asset class; corresponds to physrisk class names, e.g. PowerGeneratingAsset"
    )
    latitude: float = Field(description="Latitude in degrees")
    longitude: float = Field(description="Longitude in degrees")
    type: Optional[str] = Field(
        None, description="Type of the asset <level_1>/<level_2>/<level_3>"
    )
    location: Optional[str] = Field(
        None,
        description="Location (e.g. Africa, Asia, Europe, North America, Oceania, South America); ",
    )
    capacity: Optional[float] = Field(None, description="Power generation capacity")
    attributes: Optional[Dict[str, str]] = Field(
        None,
        description="Bespoke attributes (e.g. number of storeys, structure type, occupancy type)",
    )

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "asset_class": "RealEstateAsset",
                    "type": "Buildings/Industrial",
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
    # intensity: Array = Field(...)
    # intensity: np.ndarray = np.zeros(1) #Field(default_factory=lambda: np.zeros(1))
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
