from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast


# 'primary_fuel' entries in Global Power Plant Database v1.3.0 (World Resources Institute)
# https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3
class FuelKind(Enum):
    biomass = 1
    coal = 2
    cogeneration = 3
    gas = 4
    geothermal = 5
    hydro = 6
    nuclear = 7
    oil = 8
    other = 9
    petcoke = 10
    solar = 11
    storage = 12
    waste = 13
    wave_and_tidal = 14
    wind = 15


class CoolingKind(Enum):
    dry = 1
    once_through = 2
    recirculating = 3


class Cooling:
    @staticmethod
    def kind(cooling_type):
        return cast(CoolingKind, cooling_type.kind)


# Air Temperature, Inundation
class DryCooling(Cooling):
    kind = CoolingKind.dry


# Drought, Inundation, Water Temperature, Water Stress
class OnceThroughCooling(Cooling):
    kind = CoolingKind.once_through


# Drought, Inundation, Water Temperature, Water Stress (TO CLARIFY), Wet-Bulb Temperature,
class RecirculatingCooling(Cooling):
    kind = CoolingKind.recirculating


class TurbineKind(Enum):
    gas = 1
    steam = 2


class Turbine:
    @staticmethod
    def kind(turbine_type):
        return cast(TurbineKind, turbine_type.kind)


# Inundation, Air Temperature
class GasTurbine(Turbine):
    kind: TurbineKind = TurbineKind.gas


class SteamTurbine(Turbine):
    kind: TurbineKind = TurbineKind.steam
    cooling: Optional[Cooling] = None


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
        capacity: Optional[float] = None,
        primary_fuel: Optional[FuelKind] = None,
    ):
        super().__init__(latitude, longitude)
        self.type: Optional[str] = type
        self.location: Optional[str] = location
        self.capacity: Optional[float] = capacity
        self.primary_fuel: Optional[FuelKind] = primary_fuel


# Designed to be protected against 250-year inundation events in the baseline
# except for "nuclear" which is designed to be protected against 10,000-year
# inundation events in the baseline
class ThermalPowerGeneratingAsset(PowerGeneratingAsset):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        type: Optional[str] = None,
        location: Optional[str] = None,
        capacity: Optional[float] = None,
        primary_fuel: Optional[FuelKind] = None,
        turbine: Optional[Turbine] = None,
    ):
        super().__init__(
            latitude, longitude, type=type, location=location, capacity=capacity, primary_fuel=primary_fuel
        )
        self.turbine: Optional[Turbine] = turbine

    def get_inundation_protection_return_period(self):
        if self.primary_fuel is not None and self.primary_fuel == FuelKind.nuclear:
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
