import sys
from enum import Enum
from typing import cast


class HazardKind(Enum):
    acute = (1,)
    chronic = 2


class InundationType(Enum):
    riverine = (1,)
    coastal = 2


class Hazard:
    @staticmethod
    def kind(hazard_type):
        return cast(HazardKind, hazard_type.kind)


class Inundation(Hazard):
    kind = HazardKind.acute
    pass


class CoastalInundation(Inundation):
    kind = HazardKind.acute
    pass


class ChronicHeat(Hazard):
    kind = HazardKind.chronic
    pass


class CombinedInundation(Hazard):
    kind = HazardKind.chronic
    pass


class Drought(Hazard):
    kind = HazardKind.chronic
    pass


class Fire(Hazard):
    kind = HazardKind.chronic
    pass


class Hail(Hazard):
    kind = HazardKind.chronic
    pass


class Precipitation(Hazard):
    kind = HazardKind.chronic
    pass


class RiverineInundation(Inundation):
    kind = HazardKind.acute
    pass


class Wind(Hazard):
    kind = HazardKind.acute
    pass


def hazard_class(name: str):
    return getattr(sys.modules[__name__], name)
