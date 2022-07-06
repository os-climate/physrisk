from enum import Enum


class HazardKind(Enum):
    acute = (1,)
    chronic = 2


class InundationType(Enum):
    riverine = (1,)
    coastal = 2


class Event:
    pass


class Drought(Event):
    pass


class ChronicHeat(Event):
    kind = HazardKind.chronic
    pass


class Inundation(Event):
    kind = HazardKind.acute
    pass


class RiverineInundation(Inundation):
    kind = HazardKind.acute
    pass


class CoastalInundation(Inundation):
    kind = HazardKind.acute
    pass
