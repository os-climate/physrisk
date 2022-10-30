# flake8: noqa: E501
import hashlib
from typing import List

from pydantic import parse_obj_as

import physrisk.data.colormap_provider as colormap_provider

from ..api.v1.hazard_data import HazardModel, Period

methodology_doc = """
For more details and relevant citations see the
[OS-Climate Physical Climate Risk Methodology document](https://github.com/os-climate/physrisk/blob/main/methodology/PhysicalRiskMethodology.pdf).
"""

aqueduct_description = (
    """
The World Resources Institute (WRI) [Aqueduct Floods model](https://www.wri.org/aqueduct) is an acute riverine and
coastal flood hazard model with a spatial resolution of 30 × 30 arc seconds (approx. 1 km at the equator). Flood intensity is provided
as a _return period_ map: each point comprises a curve of inundation depths for 9 different return periods (also known as
reoccurrence periods). If a flood event has depth $d_i$ with return period of $r_i$ this implies that the probability of a flood
event with depth greater than $d_i$ occurring in any one year is $1 / r_i$; this is the _exceedance probability_.

Aqueduct Floods is based on Global Flood Risk with IMAGE Scenarios (GLOFRIS); see [here](https://www.wri.org/aqueduct/publications)
for more details.

"""
    + methodology_doc
)


class Inventory:
    """Contains an inventory of available hazard data.
    model id is given by {event_type}/{model group identifier}/{version}/{model identifier}
    """

    def __init__(self):
        osc_chronic_heat_models = [
            {
                "event_type": "ChronicHeat",
                "path": "",
                "id": "mean_degree_days/above/32c",
                "display_name": "Mean degree days above 32°C",
                "description": """
Degree days indicators are calculated by integrating over time the absolute difference in temperature
of the medium over a reference temperature. The exact method of calculation may vary;
here the daily average temperature is used to calculate an annual indicator:
$$
I^\\text{dd} = \\sum_{i = 1}^{365} |  T^\\text{avg}_i - T^\\text{ref} | \\nonumber
$$
$I^\\text{dd}$ is the indicator, $T^\\text{avg}$ is the daily average surface temperature
and $T^\\text{ref}$ is the reference temperature of 32°C. The OS-Climate-generated indicators are inferred
from CMIP6 data, averaged over 6 models: ACCESS-CM2, CMCC-ESM2, CNRM-CM6-1, MPI-ESM1-2-LR, MIROC6 and NorESM2-MM.
The indicators are generated for periods: 'historical' (averaged over 1995-2014), 2030 (2021-2040), 2040 (2031-2050)
and 2050 (2041-2060).

Applications for indicators based on surface temperature degree days include models of:
- heating and cooling requirements
- labour loss caused by extreme heat

                """
                + methodology_doc,
                "filename": "mean_degree_days_above_32c_{scenario}_{year}",
                "map": {"colormap": "mean_degree_days/above/32c"},
                "units": "degree days",
                "scenarios": [
                    {"id": "ssp585", "years": [2030, 2040, 2050]},
                    {"id": "historical", "years": [1980]},
                ],
            },
            {
                "event_type": "ChronicHeat",
                "path": "",
                "id": "mean_work_loss/high",
                "display_name": "Mean work loss",
                "description": """
The mean work loss indicator is calculated from the 'Wet Bulb Globe Temperature' (WBGT) indicator:
$$
I^\\text{WBGT}_i = 0.567 \\times T^\\text{avg}_i + 0.393 \\times P^\\text{vapour}_i
$$
$I^\\text{WBGT}_i$ is the WBGT indicator, $T^\\text{avg}_i$ is the daily average surface temperature (in degress Celsius) on day index, $i$, and $P^\\text{vapour}$
is the water vapour partial pressure (in kPa). $P^\\text{vapour}$ is calculated from relative humidity $H_R$ via:
$$
P^\\text{vapour}_i = \\frac{H_R}{100} \\times 6.105 \\times \\exp \\left( \\frac{17.27 \\times T^\\text{avg}_i}{237.7 \\times T^\\text{avg}_i} \\right)
$$
The work ability indicator, $I^{\\text{WA}}$ is finally calculated via:
$$
I^{\\text{WA}}_i = 0.1 + 0.9 / \\left( 1 + (I^\\text{WBGT}_i / \\alpha_1)^{\\alpha_2} \\right)
$$
An annual average work ability indiator, $I^{\\text{WA}}$ is calculated via:
$$
I^{\\text{WA}} = \\frac{1}{365} \\sum_{i = 1}^{365} I^{\\text{WA}}_i
$$
The OS-Climate-generated indicators are inferred from CMIP6 data, averaged over 6 models: ACCESS-CM2, CMCC-ESM2, CNRM-CM6-1, MPI-ESM1-2-LR, MIROC6 and NorESM2-MM.
The indicators are generated for periods: 'historical' (averaged over 1995-2014), 2030 (2021-2040), 2040 (2031-2050) and 2050 (2041-2060).

                """
                + methodology_doc,
                "filename": "mean_work_loss_high_{scenario}_{year}",
                "map": {"colormap": "mean_work_loss/high"},
                "units": "fractional loss",
                "scenarios": [
                    {"id": "ssp585", "years": [2030, 2040, 2050]},
                    {"id": "ssp245", "years": [2030, 2040, 2050]},
                    {"id": "historical", "years": [1980]},
                ],
            },
        ]

        wri_riverine_inundation_models = [
            {
                "event_type": "RiverineInundation",
                "path": "riverine_inundation/wri/v2",
                "id": "000000000WATCH",
                "display_name": "WRI/Baseline",
                "description": """
World Resources Institute Aqueduct Floods baseline riverine model using historical data.

                """
                + aqueduct_description,
                "filename": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                },
                "units": "metres",
                "scenarios": [{"id": "historical", "years": [1980], "periods": [{"year": 1980, "map_id": "gw4vgq"}]}],
            },
            {
                "event_type": "RiverineInundation",
                "path": "riverine_inundation/wri/v2",
                "id": "00000NorESM1-M",
                "display_name": "WRI/NorESM1-M",
                "description": """
World Resources Institute Aqueduct Floods riverine model using GCM model from
Bjerknes Centre for Climate Research, Norwegian Meteorological Institute.

                """
                + aqueduct_description,
                "filename": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                },
                "units": "metres",
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
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model from
Geophysical Fluid Dynamics Laboratory (NOAA).

                """
                + aqueduct_description,
                "filename": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                },
                "units": "metres",
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
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model:
Met Office Hadley Centre.

                """
                + aqueduct_description,
                "filename": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                },
                "units": "metres",
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
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model from
Institut Pierre Simon Laplace

                """
                + aqueduct_description,
                "filename": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                },
                "units": "metres",
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
                "description": """World Resource Institute Aqueduct Floods riverine model using
 GCM model from Atmosphere and Ocean Research Institute
 (The University of Tokyo), National Institute for Environmental Studies, and Japan Agency
 for Marine-Earth Science and Technology.

                """
                + aqueduct_description,
                "filename": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                },
                "units": "metres",
                "scenarios": [
                    {
                        "id": "rcp4p5",
                        "years": [2030, 2050, 2080],
                        "periods": [
                            {"year": 2030, "map_id": "ht2kn3"},
                            {"year": 2050, "map_id": "1k4boi"},
                            {"year": 2080, "map_id": "3rok7b"},
                        ],
                    },
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
                "description": """
World Resources Institute Aqueduct Floods baseline coastal model using historical data. Model excludes subsidence.

                """
                + aqueduct_description,
                "filename": "inuncoast_historical_nosub_hist_0",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inuncoast_historical_nosub_hist_rp{return_period:04d}_0",
                },
                "units": "metres",
                "scenarios": [{"id": "historical", "years": [1980]}],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "nosub/95",
                "display_name": "WRI/95% no subsidence",
                "description": """
World Resource Institute Aqueduct Floods coastal model, exclusing subsidence; 95th percentile sea level rise.

                """
                + aqueduct_description,
                "filename": "inuncoast_{scenario}_nosub_{year}_0",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inuncoast_{scenario}_nosub_{year}_rp{return_period:04d}_0",
                },
                "units": "metres",
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
                "description": """
World Resource Institute Aqueduct Floods coastal model, excluding subsidence; 5th percentile sea level rise.

                """
                + aqueduct_description,
                "filename": "inuncoast_{scenario}_nosub_{year}_0_perc_05",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inuncoast_{scenario}_nosub_{year}_rp{return_period:04d}_0_perc_05",
                },
                "units": "metres",
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
                "description": """
World Resource Institute Aqueduct Floods model, excluding subsidence; 50th percentile sea level rise.

                """
                + aqueduct_description,
                "filename": "inuncoast_{scenario}_nosub_{year}_0_perc_50",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inuncoast_{scenario}_nosub_{year}_rp{return_period:04d}_0_perc_50",
                },
                "units": "metres",
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
                "description": """
World Resource Institute Aqueduct Floods model, excluding subsidence; baseline (based on historical data).

                """
                + aqueduct_description,
                "filename": "inuncoast_historical_wtsub_hist_0",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inuncoast_historical_wtsub_hist_rp{return_period:04d}_0",
                },
                "units": "metres",
                "scenarios": [{"id": "historical", "years": [1980]}],
            },
            {
                "event_type": "CoastalInundation",
                "path": "coastal_inundation/wri/v2",
                "id": "wtsub/95",
                "display_name": "WRI/95% with subsidence",
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 95th percentile sea level rise.

                """
                + aqueduct_description,
                "filename": "inuncoast_{scenario}_wtsub_{year}_0",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inuncoast_{scenario}_wtsub_{year}_rp{return_period:04d}_0",
                },
                "units": "metres",
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
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 5th percentile sea level rise.

                """
                + aqueduct_description,
                "filename": "inuncoast_{scenario}_wtsub_{year}_0_perc_05",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inuncoast_{scenario}_wtsub_{year}_rp{return_period:04d}_0_perc_05",
                },
                "units": "metres",
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
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 50th percentile sea level rise.

                """
                + aqueduct_description,
                "filename": "inuncoast_{scenario}_wtsub_{year}_0_perc_50",
                "map": {
                    "colormap": "flare_intensity_0_2",
                    "filename": "inuncoast_{scenario}_wtsub_{year}_rp{return_period:04d}_0_perc_50",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]

        self.models = osc_chronic_heat_models + wri_riverine_inundation_models + wri_coastal_inundation_models

    def to_hazard_models(self) -> List[HazardModel]:
        models = parse_obj_as(List[HazardModel], self.models)

        # we populate map_id hashes programmatically
        for model in models:

            for scenario in model.scenarios:
                test_periods = scenario.periods
                scenario.periods = []
                for year in scenario.years:
                    name_format = (
                        model.filename if model.map is None or model.map.filename is None else model.map.filename
                    )
                    filename = name_format.format(scenario=scenario.id, year=year, id=model.id, return_period=1000)
                    id = alphanumeric(filename)[0:6]
                    scenario.periods.append(Period(year=year, map_id=id))
                # if a period was specifed explicitly, we check that hash is the same: a build-in check
                if test_periods is not None:
                    for (period, test_period) in zip(scenario.periods, test_periods):
                        if period.map_id != test_period.map_id:
                            raise Exception(
                                f"validation error: hash {period.map_id} different to specified hash {test_period.map_id}"  # noqa: E501
                            )

        return models

    def colormaps(self):
        """Color maps. Key can be identical to a model identifier or more descriptive (if shared by many models)."""
        return colormap_provider.colormaps()


def alphanumeric(text):
    """Return alphanumeric hash from supplied string."""
    hash_int = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest(), "big")
    return base36encode(hash_int)


def base36encode(number, alphabet="0123456789abcdefghijklmnopqrstuvwxyz"):
    """Converts an integer to a base36 string."""
    if not isinstance(number, int):
        raise TypeError("number must be an integer")

    base36 = ""

    if number < 0:
        raise TypeError("number must be positive")

    if 0 <= number < len(alphabet):
        return alphabet[number]

    while number != 0:
        number, i = divmod(number, len(alphabet))
        base36 = alphabet[i] + base36

    return base36
