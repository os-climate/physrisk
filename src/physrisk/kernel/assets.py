from typing import Optional


class Asset:
    def __init__(self, latitude: float, longitude: float, **kwargs):
        self.latitude = latitude
        self.longitude = longitude
        self.__dict__.update(kwargs)


class PowerGeneratingAsset(Asset):
    pass


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
