from typing import List

from pydantic import BaseModel, Field


class Asset(BaseModel):
    """Defines an asset. An asset is identified first by its class and then by its type within the class."""

    asset_class: str = Field(
        description="name of asset class; corresponds to physrisk class names, e.g. PowerGeneratingAsset"
    )
    type: str = Field(description="type of the asset <level_1>/<level_2>/<level_3>")
    latitude: float = Field(description="Latitude in degrees")
    longitude: float = Field(description="Longitude in degrees")


class Assets(BaseModel):
    """Defines a collection of assets."""

    items: List[Asset]
