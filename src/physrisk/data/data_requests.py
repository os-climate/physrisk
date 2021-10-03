class EventDataRequest:
    def __init__(self, event_type: type, latitude: float, longitude: float):
        self._event_type = event_type
        self._latitude = latitude
        self._longitude = longitude
