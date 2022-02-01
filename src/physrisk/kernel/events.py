from enum import Enum


class InundationType(Enum):
    riverine = (1,)
    coastal = 2


class Event:
    pass


class Drought(Event):
    pass


class HighTemperature(Event):
    pass


class Inundation(Event):
    pass


class RiverineInundation(Inundation):
    pass


class CoastalInundation(Inundation):
    pass
