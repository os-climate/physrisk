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
    kind = HazardKind.ACUTE
    indicator_data = {
        "subsidence_probability": IndicatorData.PARAMETERS,
    }
    pass


def all_hazards() -> list[type[Hazard]]:
    return [
        obj
        for _, obj in sorted(globals().items())
        if isinstance(obj, type) and issubclass(obj, Hazard) and obj is not Hazard
    ]


def hazard_class(name: str) -> type[Hazard]:
    """Return the hazard class with the supplied name.

    Args:
        name: Name of a class defined in this module.

    Returns:
        The matching ``Hazard`` class or subclass.

    Raises:
        AttributeError: The name does not identify a ``Hazard`` class.
    """
    candidate = globals().get(name)
    if not isinstance(candidate, type) or not issubclass(candidate, Hazard):
        raise AttributeError(f"unknown hazard class {name!r}")
    return candidate


class Landslide(Hazard):
    kind = HazardKind.ACUTE
    indicator_data = {
        "landslide_probability": IndicatorData.PARAMETERS,
    }
    pass


class Earthquake(Hazard):
    kind = HazardKind.ACUTE
