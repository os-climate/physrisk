class Asset:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

class PowerGeneratingAsset(Asset) : pass