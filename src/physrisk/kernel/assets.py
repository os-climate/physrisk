from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pyproj import Transformer
from shapely.ops import transform
from shapely import Point
from shapely.geometry.base import BaseGeometry
import shapely.wkt

project_4326_to_3857 = Transformer.from_crs(
    "EPSG:4326", "EPSG:3857", always_xy=True
).transform
project_3857_to_4326 = Transformer.from_crs(
    "EPSG:3857", "EPSG:4326", always_xy=True
).transform


# 'primary_fuel' entries in Global Power Plant Database v1.3.0 (World Resources Institute)
# https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3
class FuelKind(Enum):
    Biomass = 1
    Coal = 2
    Cogeneration = 3
    Gas = 4
    Geothermal = 5
    Hydro = 6
    Nuclear = 7
    Oil = 8
    Other = 9
    Petcoke = 10
    Solar = 11
    Storage = 12
    Waste = 13
    WaveAndTidal = 14
    Wind = 15


class CoolingKind(Enum):
    # Air Temperature, Inundation
    Dry = 1

    # Drought, Inundation, Water Temperature, Water Stress
    OnceThrough = 2

    # Drought, Inundation, Water Temperature, Water Stress (TO CLARIFY), Wet-Bulb Temperature
    Recirculating = 3


class TurbineKind(Enum):
    Gas = 1
    Steam = 2


class Asset:
    """Class holding information about an asset that is used by vulnerability models.

    Generic vulnerability models
    ----------------------------
    Generic vulnerability models cover a significant number of use-cases. For example, inundation vulnerability
    models for a wide variety of assets may apply vulnerability curves which are looked up from the different
    attributes of the asset (e.g. occupancy identifier and properties such as number of storeys).

    Asset class sub-types
    ---------------------
    There may be cases however where vulnerability models have specificities associated with the class of the
    asset. For example, thermal power generating assets might require a model for estimating efficiency loss as air
    temperature rises. Such a model and its inputs are quite specific to the asset class and it can be convenient
    to define a vulnerability model to encapsulate that logic. To this end, a number of asset classes, sub-types of
    Asset as defined to facilitate plugging in of different vulnerability models.

    Adherence to conventions
    ------------------------
    In general asset attributes should follow the Open Exposure Data (OED) standard.
    OED and NACE codes contribute to naming of asset sub-types although as noted above, sub-types are defined at
    the level of granularity to facilitate the plugging in of different models defined in code.

    Asset 'archetypes' for sectorial models
    ---------------------------------------
    Calculations might have detailed asset information from which precise vulnerability functions can be
    obtained, but sometimes - for example in the case of sectorial calculations - information might be limited to
    the broad category of the asset. This can be specified via `type` (aka asset type as distinct from asset class)
    and `location` attributes of assets.

    Current Asset sub-type classes are:

    AgricultureAsset
    ConstructionAsset
    ManufacturingAsset
    OilGasAsset
    PowerGeneratingAsset
    RealEstateAsset
    ThermalPowerGeneratingAsset
    TransportationAsset
    UtilityAsset

    It is emphasized that the sub-types exist to facilitate development of plug-in models which can each deal with
    a large number of types of asset.

    """

    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        wkt_geometry: Optional[str] = None,
        buffer: float = 0.0,
        id: Optional[str] = None,
        **kwargs,
    ):
        """
        Create Asset instance. Asset can be point-like, or have a geometry defined via WKT. Buffering can
        be specified as a convenience: if buffer is non-zero, the geometry is buffered by the specified
        distance in metres. This is also applied to point-like assets created from latitude/longitude.

        Args:
            latitude (Optional[float], optional): Latitude in degrees (EPSG:4326). Defaults to None.
            longitude (Optional[float], optional): Longitude in degrees (EPSG:4326). Defaults to None.
            wkt_geometry (Optional[str], optional): Well-Known Text representation of geometry. Defaults to None.
            buffer (float, optional): Buffer distance in metres. Defaults to 0.0.
            id (Optional[str], optional): Identifier for the asset. Defaults to None.
        Raises:
            ValueError: If neither lat/lon nor wkt_geometry is provided.
        """
        self.id = id
        if latitude is None or longitude is None:
            if wkt_geometry is None:
                raise ValueError("either latitude/longitude or wkt must be provided")
        if wkt_geometry is not None:
            self.geometry = shapely.wkt.loads(wkt_geometry).normalize()
            if buffer != 0.0:
                self.geometry = self.buffered_geometry(self.geometry, buffer)
            centroid = self.geometry.centroid
            self.latitude = centroid.y
            self.longitude = centroid.x
        else:
            if buffer != 0.0:
                self.geometry = Asset.buffered_geometry(
                    Point(longitude, latitude), buffer
                )
            else:
                self.geometry = None
            self.latitude = latitude
            self.longitude = longitude

        self.__dict__.update(kwargs)

    @staticmethod
    def buffered_geometry(geometry: BaseGeometry, buffer: float) -> BaseGeometry:
        """Relatively simple buffer, suitable for small buffers in metres around lat/lon points.

        Args:
            geometry (BaseGeometry): Geometry in EPSG:4326.
            buffer (float): Buffer in metres.

        Returns:
            BaseGeometry: Buffered geometry in EPSG:4326.
        """
        geom_proj: BaseGeometry = transform(project_4326_to_3857, geometry)
        buffered = geom_proj.buffer(buffer, quad_segs=4)
        return transform(project_3857_to_4326, buffered)


class OEDAsset(Asset):
    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        occupancy_code: int = 1000,
        wkt_geometry: Optional[str] = None,
        buffer: float = 0.0,
        number_of_storeys: int = 0,  # -1 = unknown No. storeys - low rise, -2 = unknown No. storeys - mid rise, -3 = Unknown no. storeys = high rise).
        basement: int = 0,  # 0 = unknown / default, 1 = unfinished, 2 = 100% finished
        construction_code: int = 5000,
        first_floor_height: float = 0.305,
        **kwargs,
    ):
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            wkt_geometry=wkt_geometry,
            buffer=buffer,
            **kwargs,
        )
        self.occupancy_code = occupancy_code
        self.number_of_storeys = number_of_storeys
        self.basement = basement
        self.construction_code = construction_code
        self.first_floor_height = first_floor_height


class SimpleTypeLocationAsset(Asset):
    def __init__(
        self, *, location: Optional[str] = None, type: Optional[str] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.location = location
        self.type = type


class AgricultureAsset(OEDAsset, SimpleTypeLocationAsset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ConstructionAsset(OEDAsset, SimpleTypeLocationAsset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class IndustrialActivity(OEDAsset, SimpleTypeLocationAsset):
    """To be deprecated. Preferred model is that loss for Assets is calculated both on the asset value
    (i.e. loss from damage) and the revenue-generation associated with the Asset. That is, revenue-generation
    is not separated out as it is here.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class ManufacturingAsset(OEDAsset, SimpleTypeLocationAsset):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class OilGasAsset(OEDAsset, SimpleTypeLocationAsset):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class PowerGeneratingAsset(OEDAsset, SimpleTypeLocationAsset):
    def __init__(
        self,
        capacity: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.capacity: Optional[float] = capacity
        self.primary_fuel: Optional[FuelKind] = None

        if self.type is not None:
            archetypes = self.type.split("/")
            if 0 < len(archetypes):
                self.primary_fuel = FuelKind[archetypes[0]]


class RealEstateAsset(OEDAsset, SimpleTypeLocationAsset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ThermalPowerGeneratingAsset(PowerGeneratingAsset):
    def __init__(
        self,
        *,
        type: Optional[str] = None,
        location: Optional[str] = None,
        capacity: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            type=type,
            location=location,
            capacity=capacity,
            **kwargs,
        )

        self.turbine: Optional[TurbineKind] = None
        self.cooling: Optional[CoolingKind] = None

        if self.type is not None:
            archetypes = self.type.split("/")
            if 1 < len(archetypes):
                self.turbine = TurbineKind[archetypes[1]]
                if 2 < len(archetypes):
                    assert self.turbine == TurbineKind.Steam
                    self.cooling = CoolingKind[archetypes[2]]

    # Designed to be protected against 250-year inundation events in the
    # baseline except for nuclear power plants which are designed to be
    # protected against 10,000-year inundation events in the baseline:
    def get_inundation_protection_return_period(self):
        if self.primary_fuel is not None:
            if self.primary_fuel == FuelKind.Nuclear:
                return 10000.0
        return 250.0


class TestAsset(Asset):
    pass


class TransportationAsset(OEDAsset, SimpleTypeLocationAsset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class UtilityAsset(OEDAsset, SimpleTypeLocationAsset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class WindTurbine(Asset):
    capacity: Optional[float] = None
    hub_height: Optional[float] = None
    cut_in_speed: Optional[float] = None
    cut_out_speed: Optional[float] = None
    fixed_base: Optional[bool] = True
    rotor_diameter: Optional[float] = None
