from collections import namedtuple
from typing import List

from physrisk.data_objects.hazard_event_requests import Model


class EventTypeInfo:
    def __init__(self, hazard_type):
        self.hazard_type = hazard_type
        self.model_ids = set()
        self.scenarios = set()
        self.years = set()


class Inventory:
    """Contains an inventory of available hazard data.
    model id is given by <model group identifier>/<version>/<model identifier>
    """

    def __init__(self):
        self.models = [
            {
                "event_type": "RiverineInundation",
                "id": "wri/v2/00000NorESM1-M",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "id": "wri/v2/0000GFDL-ESM2M",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "id": "wri/v2/00IPSL-CM5A-LR",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "id": "wri/v2/MIROC-ESM-CHEM",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]

    @staticmethod
    def get_models_summary(models: List[Model]):  # -> Dict[str, EventTypeInfo]:
        """Unpack models."""

        info = {}
        HazardInfo = namedtuple("HazardInfo", "event_type model_ids scenarios years")

        for model in models:
            if model.event_type not in info:
                hazard_info = HazardInfo(event_type=model.event_type, model_ids=set(), scenarios=set(), years=set())
                info[model.event_type] = hazard_info
            else:
                hazard_info = info[model.event_type]

            hazard_info.model_ids.add(model.id)

            for s in model.scenarios:
                hazard_info.scenarios.add(s.id)

            for s in model.scenarios:
                for y in s.years:
                    hazard_info.years.add(y)

        result = {}
        for key, value in info.items():
            result[key] = HazardInfo(
                event_type=key,
                model_ids=list(sorted(value.model_ids)),
                scenarios=list(sorted(value.scenarios)),
                years=list(sorted(value.years)),
            )

        return result
