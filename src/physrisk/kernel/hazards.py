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


class Drought(Hazard):
    pass


class ChronicHeat(Hazard):
    kind = HazardKind.chronic
    pass


class Inundation(Hazard):
    kind = HazardKind.acute
    pass


class RiverineInundation(Inundation):
    kind = HazardKind.acute
    pass


class CoastalInundation(Inundation):
    kind = HazardKind.acute
    pass
