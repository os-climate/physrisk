from collections import namedtuple
from typing import Dict, List, Sequence

from pydantic import parse_obj_as

from ..data_objects.hazard_event_requests import Model


class EventTypeInfo:
    def __init__(self, hazard_type, model_ids: Sequence[str], scenarios: Sequence[str], years: Sequence[int]):
        self.hazard_type = hazard_type
        self.model_ids = list(sorted(model_ids))
        self.scenarios = list(sorted(scenarios))
        self.years = list(sorted(years))


class Inventory:
    """Contains an inventory of available hazard data.
    model id is given by {event_type}/{model group identifier}/{version}/{model identifier}
    """

    def __init__(self):
        wri_riverine_inundation_models = [
            {
                "event_type": "RiverineInundation",
                "id": "riverine_inundation/wri/v2/000000000WATCH",
                "display_name": "Baseline condition",
                "filename": "inun{type}_{scenario}_{model}_{year}",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "id": "riverine_inundation/wri/v2/00000NorESM1-M",
                "display_name": "GCM model: Bjerknes Centre for Climate Research, Norwegian Meteorological Institute",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "id": "riverine_inundation/wri/v2/0000GFDL-ESM2M",
                "display_name": "GCM model: Geophysical Fluid Dynamics Laboratory (NOAA)",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "id": "riverine_inundation/wri/v2/0000HadGEM2-ES",
                "display_name": "GCM model: Met Office Hadley Centre",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "id": "riverine_inundation/wri/v2/00IPSL-CM5A-LR",
                "display_name": "GCM model: Institut Pierre Simon Laplace",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "id": "riverine_inundation/wri/v2/MIROC-ESM-CHEM",
                "display_name": """GCM model: Atmosphere and Ocean Research Institute
 (The University of Tokyo), National Institute for Environmental Studies, and Japan Agency
 for Marine-Earth Science and Technology""",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]

        wri_coastal_inundation_models = [
            {
                "event_type": "CoastalInundation",
                "id": "coastal_inundation/wri/v2/nosub",
                "display_name": "Baseline condition; no subsidence",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "id": "coastal_inundation/wri/v2/nosub_95",
                "display_name": "No subsidence; 95th percentile sea rise",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "id": "coastal_inundation/wri/v2/nosub_05",
                "display_name": "No subsidence; 5th percentile sea rise",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "id": "coastal_inundation/wri/v2/nosub_50",
                "display_name": "No subsidence; 50th percentile sea rise",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "id": "coastal_inundation/wri/v2/wtsub",
                "display_name": "Baseline condition; with subsidence",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "id": "coastal_inundation/wri/v2/wtsub_95",
                "display_name": "With subsidence; 95th percentile sea rise",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "id": "coastal_inundation/wri/v2/wtsub_05",
                "display_name": "With subsidence; 5th percentile sea rise",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "id": "coastal_inundation/wri/v2/wtsub_50",
                "display_name": "With subsidence; 50th percentile sea rise",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]

        self.models = wri_riverine_inundation_models + wri_coastal_inundation_models

    def get_models_summary(self) -> Dict[str, EventTypeInfo]:
        """Get the possible models, scenarios and years."""

        models = parse_obj_as(List[Model], self.models)

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
            result[key] = EventTypeInfo(value.event_type, value.model_ids, value.scenarios, value.years)

        return result
