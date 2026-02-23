import inspect
import sys
from enum import Enum
from typing import Dict, Type


class IndicatorData(Enum):
    # The hazard data comprises return periods and corresponding hazard indicator values.
    # e.g. for return periods 200, 500, 1000 years, the corresponding flood depth.
    EVENT = 1
    # The hazard data comprises a set of parameter definitions and corresponding parameters.
    # A common case is that the parameters are hazard indicator values at specific thresholds, given by
    # the parameter definitions.
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
    pass


class Inundation(Hazard):
    kind = HazardKind.ACUTE
    indicator_data = {
        "flood_depth": IndicatorData.EVENT,
        "flood_sop": IndicatorData.PARAMETERS,
    }
    pass


class AirTemperature(ChronicHeat):
    pass


class CoastalInundation(Inundation):
    pass


class ChronicWind(Hazard):
    kind = HazardKind.CHRONIC
    pass


class CombinedInundation(Hazard):
    kind = HazardKind.CHRONIC
    pass


class Drought(Hazard):
    kind = HazardKind.CHRONIC
    pass


class Fire(Hazard):
    kind = HazardKind.ACUTE
    indicator_data = {
        "fire_probability": IndicatorData.PARAMETERS,
    }
    pass


class Hail(Hazard):
    kind = HazardKind.ACUTE
    indicator_data = {
        "days/above/5cm": IndicatorData.PARAMETERS,
    }
    pass


class PluvialInundation(Inundation):
    pass


class Precipitation(Hazard):
    kind = HazardKind.CHRONIC
    pass


class RiverineInundation(Inundation):
    pass


class WaterRisk(Hazard):
    kind = HazardKind.CHRONIC
    pass


class WaterTemperature(ChronicHeat):
    pass


class Wind(Hazard):
    kind = HazardKind.ACUTE
    pass


class Subsidence(Hazard):
    kind = HazardKind.CHRONIC
    pass


def all_hazards():
    return [
        obj
        for _, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and issubclass(obj, Hazard)
    ]


def hazard_class(name: str):
    return getattr(sys.modules[__name__], name)
