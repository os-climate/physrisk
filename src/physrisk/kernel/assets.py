from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
        self, latitude: float, longitude: float, id: Optional[str] = None, **kwargs
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.id = id
        self.__dict__.update(kwargs)


class AgricultureAsset(Asset):
    def __init__(
        self, latitude: float, longitude: float, *, location: str, type: str, **kwargs
    ):
        super().__init__(latitude, longitude, **kwargs)
        self.location = location
        self.type = type


class ConstructionAsset(Asset):
    def __init__(
        self, latitude: float, longitude: float, *, location: str, type: str, **kwargs
    ):
        super().__init__(latitude, longitude, **kwargs)
        self.location = location
        self.type = type


class IndustrialActivity(Asset):
    """To be deprecated. Preferred model is that loss for Assets is calculated both on the asset value
    (i.e. loss from damage) and the revenue-generation associated with the Asset. That is, revenue-generation
    is not separated out as it is here.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        location: Optional[str] = None,
        type: str,
        **kwargs,
    ):
        super().__init__(latitude, longitude, **kwargs)
        self.location = location
        self.type = type


class ManufacturingAsset(Asset):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        location: Optional[str] = None,
        type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(latitude, longitude, **kwargs)
        self.location = location
        self.type = type


class OilGasAsset(Asset):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        location: Optional[str] = None,
        type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(latitude, longitude, **kwargs)
        self.location = location
        self.type = type


class PowerGeneratingAsset(Asset):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        type: Optional[str] = None,
        location: Optional[str] = None,
        capacity: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(latitude, longitude, **kwargs)

        self.type: Optional[str] = type
        self.location: Optional[str] = location
        self.capacity: Optional[float] = capacity
        self.primary_fuel: Optional[FuelKind] = None

        if type is not None:
            archetypes = type.split("/")
            if 0 < len(archetypes):
                self.primary_fuel = FuelKind[archetypes[0]]


class RealEstateAsset(Asset):
    def __init__(
        self, latitude: float, longitude: float, *, location: str, type: str, **kwargs
    ):
        super().__init__(latitude, longitude, **kwargs)
        self.location = location
        self.type = type


class ThermalPowerGeneratingAsset(PowerGeneratingAsset):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        type: Optional[str] = None,
        location: Optional[str] = None,
        capacity: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            latitude,
            longitude,
            type=type,
            location=location,
            capacity=capacity,
            **kwargs,
        )

        self.turbine: Optional[TurbineKind] = None
        self.cooling: Optional[CoolingKind] = None

        if type is not None:
            archetypes = type.split("/")
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


class TransportationAsset(Asset):
    def __init__(
        self, latitude: float, longitude: float, *, location: str, type: str, **kwargs
    ):
        super().__init__(latitude, longitude, **kwargs)
        self.location = location
        self.type = type


class UtilityAsset(Asset):
    def __init__(
        self, latitude: float, longitude: float, *, location: str, type: str, **kwargs
    ):
        super().__init__(latitude, longitude, **kwargs)
        self.location = location
        self.type = type


@dataclass
class WindTurbine(Asset):
    capacity: Optional[float] = None
    hub_height: Optional[float] = None
    cut_in_speed: Optional[float] = None
    cut_out_speed: Optional[float] = None
    fixed_base: Optional[bool] = True
    rotor_diameter: Optional[float] = None
