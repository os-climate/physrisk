from typing import List

import numpy as np
from pydantic import BaseModel, Field


class TypedArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return np.array(val, dtype=cls.inner_type)  # type: ignore


class ArrayMeta(type):
    def __getitem__(self, t):
        return type("Array", (TypedArray,), {"inner_type": t})


class Array(np.ndarray, metaclass=ArrayMeta):
    pass


class Asset(BaseModel):
    """Defines an asset. An asset is identified first by its asset_class and then by its type within the class.
    An asset's value may be impacted through damage or through disruption
    disruption being reduction of an asset's ability to generate cashflows
    (or equivalent value, e.g. by reducing expenses or increasing sales).
    """

    asset_class: str = Field(
        description="name of asset class; corresponds to physrisk class names, e.g. PowerGeneratingAsset"
    )
    type: str = Field(description="Type of the asset <level_1>/<level_2>/<level_3>")
    location: str
    latitude: float = Field(description="Latitude in degrees")
    longitude: float = Field(description="Longitude in degrees")


class Assets(BaseModel):
    """Defines a collection of assets."""

    items: List[Asset]


class Country(BaseModel):
    """Country information."""

    country: str
    continent: str
    country_iso_a3: str


class Countries(BaseModel):
    """List of Country."""

    items: List[Country]


class IntensityCurve(BaseModel):
    """Intensity curve of an acute hazard."""

    intensities: List[float]
    return_periods: List[float]


class HazardEventDistrib(BaseModel):
    """Intensity curve of an acute hazard."""

    intensity_bin_edges: np.ndarray = Field(default_factory=lambda: np.zeros(10), description="")
    probabilities: np.ndarray = Field(default_factory=lambda: np.zeros(10), description="")

    class Config:
        arbitrary_types_allowed = True


class VulnerabilityCurve(BaseModel):
    """Defines a damage or disruption curve."""

    asset_type: str = Field(...)
    location: str = Field(...)
    event_type: str = Field(description="hazard event type, e.g. RiverineInundation")
    impact_type: str = Field(description="'Damage' or 'Disruption'")
    # intensity: Array = Field(...)
    # intensity: np.ndarray = np.zeros(1) #Field(default_factory=lambda: np.zeros(1))
    intensity: List[float] = Field(...)
    intensity_units: str = Field(description="units of the intensity")
    impact_mean: List[float] = Field(description="mean impact (damage or disruption)")
    impact_std: List[float] = Field(description="standard deviation of impact (damage or disruption)")

    class Config:
        arbitrary_types_allowed = True


class VulnerabilityCurves(BaseModel):
    """List of VulnerabilityCurve."""

    items: List[VulnerabilityCurve]


class VulnerabilityDistrib(BaseModel):
    """Defines a vulnerability matrix."""

    intensity_bin_edges: np.ndarray = Field(default_factory=lambda: np.zeros(10), description="")
    impact_bin_edges: np.ndarray = Field(default_factory=lambda: np.zeros(10), description="")
    prob_matrix: np.ndarray = Field(default_factory=lambda: np.zeros(10), description="")

    class Config:
        arbitrary_types_allowed = True
