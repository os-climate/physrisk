class Asset:
    def __init__(self, latitude: float, longitude: float, **kwargs):
        self.latitude = latitude
        self.longitude = longitude
        self.__dict__.update(kwargs)


class PowerGeneratingAsset(Asset):
    pass


class RealEstateAsset(Asset):
    pass


class TestAsset(Asset):
    pass
