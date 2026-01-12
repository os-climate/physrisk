import inspect
import sys
from enum import Enum
from typing import Dict, Type


class IndicatorData(Enum):
    EVENT = 1
    PARAMETERS = 2


class HazardKind(Enum):
    ACUTE = 1
    CHRONIC = 2
    UNKNOWN = 3


class Hazard:
    kind = HazardKind.UNKNOWN
    indicator_data: Dict[str, IndicatorData] = {}


def hazard_kind(hazard_type: Type[Hazard]):
    return hazard_type.kind


def indicator_data(hazard_type: Type[Hazard], indicator_id: str):
    default = (
        IndicatorData.EVENT
        if hazard_type.kind == HazardKind.ACUTE
        else IndicatorData.PARAMETERS
    )
    return hazard_type.indicator_data.get(indicator_id, default)


class ChronicHeat(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {
        "mean_degree_days/above/32c": IndicatorData.PARAMETERS,
        "days/above/35c": IndicatorData.PARAMETERS,
        "mean_work_loss": IndicatorData.PARAMETERS,
        "days_tas/above/": IndicatorData.PARAMETERS,
        "mean_degree_days/above/index": IndicatorData.PARAMETERS,
        "mean_degree_days/below/index": IndicatorData.PARAMETERS,
        "weeks_water_temp_above": IndicatorData.PARAMETERS,
        "days_wbgt_above": IndicatorData.PARAMETERS,
    }
    pass


class Inundation(Hazard):
    kind = HazardKind.ACUTE
    indicator_data = {
        "flood_depth": IndicatorData.EVENT,
        "flood_sop": IndicatorData.PARAMETERS,
        "flood_depth_unprot": IndicatorData.EVENT,
    }
    pass


class AirTemperature(ChronicHeat):
    pass


class CoastalInundation(Inundation):
    pass


class ChronicWind(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {
        "severe_windstorm_probability": IndicatorData.PARAMETERS,
        "extreme_windstorm_probability": IndicatorData.PARAMETERS,
    }
    pass


class CombinedInundation(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {
        "flooded_fraction": IndicatorData.PARAMETERS,
    }
    pass


class Drought(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {
        "months/spei12m/below/index": IndicatorData.PARAMETERS,
        "months/spei3m/below/-2": IndicatorData.PARAMETERS,
        "cdd": IndicatorData.PARAMETERS,
        "spi6": IndicatorData.PARAMETERS,
    }
    pass


class Fire(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {
        "fire_probability": IndicatorData.PARAMETERS,
        "daily_probability_fwi20": IndicatorData.PARAMETERS,
    }
    pass


class Hail(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {"days/above/5cm": IndicatorData.PARAMETERS}
    pass


class PluvialInundation(Inundation):
    pass


class Precipitation(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {"max/daily/water_equivalent": IndicatorData.PARAMETERS}
    pass


class RiverineInundation(Inundation):
    pass


class WaterRisk(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {
        "water_demand": IndicatorData.PARAMETERS,
        "water_supply": IndicatorData.PARAMETERS,
        "water_stress": IndicatorData.PARAMETERS,
        "water_depletion": IndicatorData.PARAMETERS,
        "water_stress_category": IndicatorData.PARAMETERS,
        "water_depletion_category": IndicatorData.PARAMETERS,
    }
    pass


class WaterTemperature(ChronicHeat):
    pass


class Wind(Hazard):
    kind = HazardKind.ACUTE
    indicator_data = {"max_speed": IndicatorData.EVENT}
    pass


class Subsidence(Hazard):
    kind = HazardKind.CHRONIC
    indicator_data = {"subsidence_susceptability": IndicatorData.PARAMETERS}
    pass


class Landslide(Hazard):
    kind = HazardKind.ACUTE
    indicator_data = {"susceptability": IndicatorData.EVENT}
    pass


def all_hazards():
    return [
        obj
        for _, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and issubclass(obj, Hazard)
    ]


def hazard_class(name: str):
    return getattr(sys.modules[__name__], name)
