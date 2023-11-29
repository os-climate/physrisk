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
    def __init__(self, latitude: float, longitude: float, **kwargs):
        self.latitude = latitude
        self.longitude = longitude
        self.__dict__.update(kwargs)


# WindFarm as separate
@dataclass
class WindTurbine(Asset):
    capacity: Optional[float] = None
    hub_height: Optional[float] = None
    cut_in_speed: Optional[float] = None
    cut_out_speed: Optional[float] = None
    fixed_base: Optional[bool] = True
    rotor_diameter: Optional[float] = None


class PowerGeneratingAsset(Asset):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        type: Optional[str] = None,
        location: Optional[str] = None,
        capacity: Optional[float] = None
    ):
        super().__init__(latitude, longitude)

        self.type: Optional[str] = type
        self.location: Optional[str] = location
        self.capacity: Optional[float] = capacity

        if type is not None:
            self.primary_fuel: Optional[FuelKind] = None
            archetypes = type.split("/")
            if 0 < len(archetypes):
                self.primary_fuel = FuelKind[archetypes[0]]


class ThermalPowerGeneratingAsset(PowerGeneratingAsset):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        type: Optional[str] = None,
        location: Optional[str] = None,
        capacity: Optional[float] = None
    ):
        super().__init__(latitude, longitude, type=type, location=location, capacity=capacity)

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


class RealEstateAsset(Asset):
    def __init__(self, latitude: float, longitude: float, *, location: str, type: str):
        super().__init__(latitude, longitude)
        self.location = location
        self.type = type


class IndustrialActivity(Asset):
    def __init__(self, latitude: float, longitude: float, *, location: Optional[str] = None, type: str):
        super().__init__(latitude, longitude)
        self.location = location
        self.type = type


class TestAsset(Asset):
    pass
