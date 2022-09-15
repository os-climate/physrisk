from collections import namedtuple
from typing import Dict, List, Sequence

from pydantic import parse_obj_as

from ..api.v1.hazard_data import HazardModel


class EventTypeInfo:
    def __init__(
        self,
        hazard_type: str,
        paths: Sequence[str],
        ids: Sequence[str],
        names: Sequence[str],
        descriptions: Sequence[str],
        filenames: Sequence[str],
        scenarios: Sequence[str],
        years: Sequence[int],
    ):
        self.hazard_type = hazard_type
        self.paths = list(sorted(paths))
        self.ids = list(sorted(ids))
        self.names = list(sorted(names))
        self.descriptions = list(sorted(descriptions))
        self.filenames = list(sorted(filenames))
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
                "path": "riverine_inundation/wri/v2",
                "id": "000000000WATCH",
                "display_name": "WRI/Baseline",
                "description": "Baseline condition",
                "filename": "inunriver_{scenario}_{id}_{year}",
                "scenarios": [
                    {"id": "historical", "years": [1980]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "path": "riverine_inundation/wri/v2",
                "id": "00000NorESM1-M",
                "display_name": "WRI/NorESM1-M",
                "description": "GCM model: Bjerknes Centre for Climate Research, Norwegian Meteorological Institute",
                "filename": "inunriver_{scenario}_{id}_{year}",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "path": "riverine_inundation/wri/v2",
                "id": "0000GFDL-ESM2M",
                "display_name": "WRI/GFDL-ESM2M",
                "description": "GCM model: Geophysical Fluid Dynamics Laboratory (NOAA)",
                "filename": "inunriver_{scenario}_{id}_{year}",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "path": "riverine_inundation/wri/v2",
                "id": "0000HadGEM2-ES",
                "display_name": "WRI/HadGEM2-ES",
                "description": "GCM model: Met Office Hadley Centre",
                "filename": "inunriver_{scenario}_{id}_{year}",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "path": "riverine_inundation/wri/v2",
                "id": "00IPSL-CM5A-LR",
                "display_name": "WRI/IPSL-CM5A-LR",
                "description": "GCM model: Institut Pierre Simon Laplace",
                "filename": "inunriver_{scenario}_{id}_{year}",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "RiverineInundation",
                "path": "riverine_inundation/wri/v2",
                "id": "MIROC-ESM-CHEM",
                "display_name": "WRI/MIROC-ESM-CHEM",
                "description": """GCM model: Atmosphere and Ocean Research Institute
 (The University of Tokyo), National Institute for Environmental Studies, and Japan Agency
 for Marine-Earth Science and Technology""",
                "filename": "inunriver_{scenario}_{id}_{year}",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]

        wri_coastal_inundation_models = [
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "nosub",
                "display_name": "WRI/Baseline no subsidence",
                "description": "Baseline condition; no subsidence",
                "filename": "inuncoast_{scenario}_nosub_{year}_0",
                "scenarios": [{"id": "historical", "years": [1980]}],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "nosub/95",
                "display_name": "WRI/95% no subsidence",
                "description": "No subsidence; 95th percentile sea level rise",
                "filename": "inuncoast_{scenario}_nosub_{year}_0",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "nosub/5",
                "display_name": "WRI/5% no subsidence",
                "description": "No subsidence; 5th percentile sea level rise",
                "filename": "inuncoast_{scenario}_nosub_{year}_0_perc_05",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "nosub/50",
                "display_name": "WRI/50% no subsidence",
                "description": "No subsidence; 50th percentile sea level rise",
                "filename": "inuncoast_{scenario}_nosub_{year}_0_perc_50",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "wtsub",
                "display_name": "WRI/Baseline with subsidence",
                "description": "Baseline condition; with subsidence",
                "filename": "inuncoast_{scenario}_wtsub_{year}_0",
                "scenarios": [{"id": "historical", "years": [1980]}],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "wtsub/95",
                "display_name": "WRI/95% with subsidence",
                "description": "With subsidence; 95th percentile sea level rise",
                "filename": "inuncoast_{scenario}_wtsub_{year}_0",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "wtsub/5",
                "display_name": "WRI/5% with subsidence",
                "description": "With subsidence; 5th percentile sea level rise",
                "filename": "inuncoast_{scenario}_wtsub_{year}_0_perc_05",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "wtsub/50",
                "display_name": "WRI/50% with subsidence",
                "description": "With subsidence; 50th percentile sea level rise",
                "filename": "inuncoast_{scenario}_wtsub_{year}_0_perc_50",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]

        self.models = wri_riverine_inundation_models + wri_coastal_inundation_models

    def get_models_summary(self) -> Dict[str, EventTypeInfo]:
        """Get the possible models, scenarios and years."""

        models = parse_obj_as(List[HazardModel], self.models)

        info = {}
        HazardInfo = namedtuple("HazardInfo", "event_type paths ids names descriptions filenames scenarios years")

        for model in models:
            if model.event_type not in info:
                info[model.event_type] = HazardInfo(
                    event_type=model.event_type,
                    paths=set(),
                    ids=set(),
                    names=set(),
                    descriptions=set(),
                    filenames=set(),
                    scenarios=set(),
                    years=set(),
                )

            hazard_info = info[model.event_type]
            hazard_info.paths.add(model.path)
            hazard_info.ids.add(model.id)
            hazard_info.names.add(model.display_name)
            hazard_info.descriptions.add(model.description)
            hazard_info.filenames.add(model.filename)

            for s in model.scenarios:
                hazard_info.scenarios.add(s.id)
                for y in s.years:
                    hazard_info.years.add(y)

        result = {}
        for key, value in info.items():
            result[key] = EventTypeInfo(
                value.event_type,
                value.paths,
                value.ids,
                value.names,
                value.descriptions,
                value.filenames,
                value.scenarios,
                value.years,
            )

        return result
