import pathlib
from typing import List, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import zarr
import zarr.storage
from affine import Affine
from physrisk import requests
from physrisk.api.v1.common import Assets
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.assets import (
    Asset,
    ManufacturingAsset,
    PowerGeneratingAsset,
    RealEstateAsset,
    ThermalPowerGeneratingAsset,
)

from physrisk.vulnerability_models.config_based_model import (
    ConfigBasedVulnerabilityModelAcute,
    ImpactCurveKey,
    PiecewiseLinearImpactCurve,
    config_items_from_csv,
    config_items_to_csv,
)
from physrisk.vulnerability_models.vulnerability import VulnerabilityConfigItem, VulnerabilityModelsFactory

from ..conftest import working_dir


def vulnerability_config():
    return [
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="Asset",
            asset_identifier="type=Generic,location=Europe",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units="",
            curve_type="indicator/piecewise_linear",
            points_x=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 130],
            points_y=[0, 0, 0, 0.0025, 0.01, 0.02, 0.04, 0.08, 0.12, 0.17, 0.23, 0.29, 0.35, 0.42, 0.48, 1.0],
            activation_of_points_x=40,
            cap_of_points_x=None,
            cap_of_points_y=1,
        ),
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="Asset",
            asset_identifier="type=Generic,location=Asia",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units="",
            curve_type="indicator/piecewise_linear",
            points_x=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 130],
            points_y=[0, 0, 0, 0.01, 0.04, 0.09, 0.17, 0.25, 0.34, 0.42, 0.49, 0.55, 0.61, 0.66, 0.71, 1.0],
            activation_of_points_x=40,
            cap_of_points_x=None,
            cap_of_points_y=1,
        ),
    ]


def config_based_vulnerability_models():
    return [
        VulnerabilityConfigItem(
            hazard_class="Fire",
            asset_class="RealEstateAsset",
            asset_identifier="type=Generic,location=Generic",
            indicator_id="fire_probability",
            indicator_units="",
            impact_id="damage",
            impact_units="",
            curve_type="threshold/piecewise_linear",
            points_x=0,
            points_y=0.74,
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=1,
        ),
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="RealEstateAsset",
            asset_identifier="type=Generic,location=Generic",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 130],
            points_y=[0, 0, 0, 0.0025, 0.01, 0.02, 0.04, 0.08, 0.12, 0.17, 0.23, 0.29, 0.35, 0.42, 0.48, 1.0],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="RealEstateAsset",
            asset_identifier="type=Generic,location=Europe",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 130],
            points_y=[0, 0, 0, 0.0025, 0.01, 0.02, 0.04, 0.08, 0.12, 0.17, 0.23, 0.29, 0.35, 0.42, 0.48, 1.0],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="RealEstateAsset",
            asset_identifier="type=Generic,location=North America",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 120],
            points_y=[0, 0, 0, 0.01, 0.07, 0.15, 0.26, 0.39, 0.5, 0.6, 0.68, 0.75, 0.8, 0.83, 0.87, 1.0],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="RealEstateAsset",
            asset_identifier="type=Generic,location=South America",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 130],
            points_y=[0, 0, 0, 0.0025, 0.01, 0.02, 0.04, 0.08, 0.12, 0.17, 0.23, 0.29, 0.35, 0.42, 0.48, 1.0],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="RealEstateAsset",
            asset_identifier="type=Generic,location=Asia",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 130],
            points_y=[0, 0, 0, 0.01, 0.04, 0.09, 0.17, 0.25, 0.34, 0.42, 0.49, 0.55, 0.61, 0.66, 0.71, 1.0],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="RealEstateAsset",
            asset_identifier="type=Generic,location=Oceania",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 110],
            points_y=[0, 0, 0, 0.03, 0.11, 0.23, 0.37, 0.52, 0.63, 0.71, 0.78, 0.83, 0.87, 0.89, 0.92, 1.0],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Hail",
            asset_class="PowerGeneratingAsset",
            asset_identifier="type=Wind,location=Generic",
            indicator_id="days/above/5cm",
            indicator_units="days/year",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0, 6],
            points_y=[0, 0.2],
            activation_of_points_x=None,
            cap_of_points_x=6,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Hail",
            asset_class="PowerGeneratingAsset",
            asset_identifier="type=Solar,location=Generic",
            indicator_id="days/above/5cm",
            indicator_units="days/year",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0, 6],
            points_y=[0, 1.0],
            activation_of_points_x=None,
            cap_of_points_x=6,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Hail",
            asset_class="PowerGeneratingAsset",
            asset_identifier="type=Generic,location=Generic",
            indicator_id="days/above/5cm",
            indicator_units="days/year",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0, 6],
            points_y=[0, 0.034],
            activation_of_points_x=None,
            cap_of_points_x=6,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Hail",
            asset_class="RealEstateAsset",
            asset_identifier="type=Generic,location=Generic",
            indicator_id="days/above/5cm",
            indicator_units="days/year",
            impact_id="damage",
            impact_units="",
            curve_type="indicator/piecewise_linear",
            points_x=[0, 6],
            points_y=[0, 0.139],
            activation_of_points_x=None,
            cap_of_points_x=6,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Hail",
            asset_class="ManufacturingAsset",
            asset_identifier="type=Generic,location=Generic",
            indicator_id="days/above/5cm",
            indicator_units="days/year",
            impact_id="damage",
            impact_units="",
            curve_type="indicator/piecewise_linear",
            points_x=[0, 6],
            points_y=[0, 0.074],
            activation_of_points_x=None,
            cap_of_points_x=6,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation,",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Residential,location=Europe",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95, 1],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Residential,location=North America",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0.201804543482798,
                0.443269856567027,
                0.582754693231323,
                0.682521911580079,
                0.783957148211158,
                0.854348922258608,
                0.923670100884942,
                0.958522772593124,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Residential,location=South America",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.490885950756559,
                0.711294066621163,
                0.842026010706856,
                0.949369095816464,
                0.98363697705803,
                1,
                1,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Residential,location=Asia",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.32655650202936,
                0.494050323540452,
                0.616572124453847,
                0.720711764425329,
                0.869528212561713,
                0.93148708390482,
                0.983604148013016,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Residential,location=Africa",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.219925400932904,
                0.37822684609281,
                0.530589081762227,
                0.635636732936187,
                0.816939779852045,
                0.90343468839945,
                0.957152173012357,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Residential,location=Oceania",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.475418119166991,
                0.640393124360772,
                0.714614661507739,
                0.787726347552177,
                0.928779884269488,
                0.967381852500423,
                0.982795443758899,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Commercial,location=Europe",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[0, 0.15, 0.3, 0.45, 0.55, 0.75, 0.9, 1, 1],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Commercial,location=North America",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0.0184049079754601,
                0.239263803680982,
                0.374233128834356,
                0.466257668711656,
                0.552147239263804,
                0.687116564417178,
                0.822085889570552,
                0.907975460122699,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Commercial,location=South America",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[0, 0.611477586608912, 0.839531094460437, 0.923588457451102, 0.99197247706422, 1, 1, 1, 1],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Commercial,location=Asia",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.376789623183568,
                0.537681618645996,
                0.659336683951557,
                0.762845232498637,
                0.883348655829433,
                0.941854894898346,
                0.980759380222282,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Commercial,location=Oceania",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[0, 0.238953574549086, 0.481199681707545, 0.673795091431068, 0.864583333333333, 1, 1, 1, 1],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="RealEstateAsset",
            asset_identifier="type=Buildings/Commercial,location=Generic",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.323296917604509,
                0.506529104729667,
                0.634595580309077,
                0.744309656431999,
                0.864093044049322,
                0.93278815689378,
                0.977746968068996,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="Asset",
            asset_identifier="type=Generic,location=Europe",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[0, 0.15, 0.27, 0.4, 0.52, 0.7, 0.85, 1, 1],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="Asset",
            asset_identifier="type=Generic,location=North America",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0.0257142857142857,
                0.322857142857143,
                0.511428571428571,
                0.637142857142857,
                0.74,
                0.86,
                0.937142857142857,
                0.98,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="Asset",
            asset_identifier="type=Generic,location=South America",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[0, 0.667019400352734, 0.888712522045855, 0.94673721340388, 1, 1, 1, 1, 1],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="Asset",
            asset_identifier="type=Generic,location=Asia",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.28318152390365,
                0.481615653142461,
                0.629218893846413,
                0.717240588020605,
                0.856675029760043,
                0.908577003967013,
                0.955327463022467,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="Asset",
            asset_identifier="type=Generic,location=Africa",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.0626820428336079,
                0.247196046128501,
                0.403329983525535,
                0.49448863261944,
                0.684652388797364,
                0.91858978583196,
                1,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="Asset",
            asset_identifier="type=Generic,location=Generic",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0.05, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
            points_y=[
                0,
                0.297148021989427,
                0.479790558549078,
                0.603285789583737,
                0.694345844128009,
                0.820265483711481,
                0.922861929388366,
                0.987065492604493,
                1,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Drought",
            asset_class="Asset",
            asset_identifier="type=Buildings/Commercial,location=Generic",
            indicator_id="months/spei3m/below/-2",
            indicator_units="months/year",
            impact_id="damage",
            impact_units="",
            curve_type="indicator/piecewise_linear",
            points_x=[0, 0.5, 3],
            points_y=[0, 0, 0.01],
            activation_of_points_x=0.5,
            cap_of_points_x=3,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Drought",
            asset_class="ThermalPowerGeneratingAsset",
            asset_identifier="type=Generic,location=Generic",
            indicator_id="months/spei3m/below/-2",
            indicator_units="months/year",
            impact_id="disruption",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0, 12],
            points_y=[0, 0.2],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Drought",
            asset_class="ManufacturingAsset",
            asset_identifier="type=Chemical,location=Generic",
            indicator_id="months/spei3m/below/-2",
            indicator_units="months/year",
            impact_id="disruption",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[0, 12],
            points_y=[0, 0.3],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="ChronicHeat",
            asset_class="PowerGeneratingAsset",
            asset_identifier="type=Solar,location=Generic",
            indicator_id="days_tas/above/{temp_c}c",
            indicator_units="days/year",
            impact_id="disruption",
            impact_units=None,
            curve_type="threshold/piecewise_linear",
            points_x=[0, 3, 103],
            points_y=[0, 0, 0.37],
            activation_of_points_x=3,
            cap_of_points_x=None,
            cap_of_points_y=1.0,
        ),
        VulnerabilityConfigItem(
            hazard_class="ChronicHeat",
            asset_class="ManufacturingAsset",
            asset_identifier="type=Chemical,location=Generic",
            indicator_id="days_tas/above/{temp_c}c",
            indicator_units="days/year",
            impact_id="disruption",
            impact_units=None,
            curve_type="threshold/piecewise_linear",
            points_x=[15, 25, 35, 45, 55, 65],
            points_y=[0, 0.1, 0.25, 0.5, 0.8, 1.0],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="ChronicHeat",
            asset_class="ManufacturingAsset",
            asset_identifier="type=Chemical,location=Generic",
            indicator_id="days_wbgt_above",
            indicator_units="days/year",
            impact_id="disruption",
            impact_units=None,
            curve_type="threshold/piecewise_linear",
            points_x=[0, 32, 35],
            points_y=[0, 0.5, 0.75],
            activation_of_points_x=32,
            cap_of_points_x=35,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="ChronicHeat",
            asset_class="ThermalPowerGeneratingAsset",
            asset_identifier="type=Generic,location=Generic",
            indicator_id="days_wbgt_above",
            indicator_units="days/year",
            impact_id="disruption",
            impact_units=None,
            curve_type="threshold/piecewise_linear",
            points_x=[0, 32, 35],
            points_y=[0, 0.5, 0.75],
            activation_of_points_x=32,
            cap_of_points_x=35,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="CoastalInundation,PluvialInundation,RiverineInundation",
            asset_class="Asset",
            asset_identifier="hazus_fl_building_dmg_id=1035",
            indicator_id="flood_depth",
            indicator_units="metres",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[
                -1.2192,
                -0.9144,
                -0.6096,
                -0.3048,
                0,
                0.3048,
                0.6096,
                0.9144,
                1.2192,
                1.524,
                1.8288,
                2.1336,
                2.4384,
                2.7432,
                3.048,
                3.3528,
                3.6576,
                3.9624,
                4.2672,
                4.572,
                4.8768,
                5.1816,
                5.4864,
                5.7912,
                6.096,
                6.4008,
                6.7056,
                7.0104,
                7.3152,
            ],
            points_y=[
                0,
                0,
                0,
                0,
                0.01,
                0.02,
                0.05,
                0.15,
                0.3,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
            ],
            activation_of_points_x=0,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
        VulnerabilityConfigItem(
            hazard_class="Wind",
            asset_class="Asset",
            asset_identifier="hazus_ws_building_dmg_id=1035",
            indicator_id="max_speed",
            indicator_units="m/s",
            impact_id="damage",
            impact_units=None,
            curve_type="indicator/piecewise_linear",
            points_x=[
                22.3527777777778,
                24.5861111111111,
                26.8222222222222,
                29.0583333333333,
                31.2916666666667,
                33.5277777777778,
                35.7638888888889,
                37.9972222222222,
                40.2333333333333,
                42.4694444444444,
                44.7027777777778,
                46.9388888888889,
                49.175,
                51.4083333333333,
                53.6444444444444,
                55.8805555555556,
                58.1138888888889,
                60.35,
                62.5861111111111,
                64.8194444444444,
                67.0555555555556,
                69.2916666666667,
                71.5277777777778,
                73.7611111111111,
                75.9972222222222,
                78.2333333333333,
                80.4666666666667,
                82.7027777777778,
                84.9388888888889,
                87.1722222222222,
                89.4083333333333,
                91.6444444444444,
                93.8777777777778,
                96.1138888888889,
                98.35,
                100.583333333333,
                102.819444444444,
                105.055555555556,
                107.288888888889,
                109.525,
                111.761111111111,
            ],
            points_y=[
                0,
                0,
                0.00034,
                0.00068,
                0.00131,
                0.0024,
                0.00422,
                0.00697,
                0.01091,
                0.01623,
                0.02302,
                0.03255,
                0.04449,
                0.06182,
                0.08373,
                0.11759,
                0.15836,
                0.20806,
                0.26597,
                0.3268,
                0.39282,
                0.46269,
                0.53287,
                0.5896,
                0.6463,
                0.70293,
                0.75299,
                0.78164,
                0.82841,
                0.86405,
                0.89275,
                0.90493,
                0.92122,
                0.93966,
                0.95623,
                0.96491,
                0.97141,
                0.977,
                0.98491,
                0.98868,
                0.98868,
            ],
            activation_of_points_x=None,
            cap_of_points_x=None,
            cap_of_points_y=None,
        ),
    ]


def shape_transform_21600_43200(
    width: int = 43200, height: int = 21600, return_periods: Union[List[float], npt.NDArray] = [0.0]
):
    t = [360.0 / width, 0.0, -180.0, 0.0, -180.0 / height, 90.0, 0.0, 0.0, 1.0]
    return (len(return_periods), height, width), t


def zarr_memory_store(path="hazard.zarr"):
    store = zarr.storage.MemoryStore(root=path)
    return store, zarr.open(store=store, mode="w")


def add_curves(
    root: zarr.Group,
    longitudes,
    latitudes,
    array_path: str,
    shape: Tuple[int, int, int],
    curve: np.ndarray,
    return_periods: List[float],
    trans: List[float],
):
    z = root.create_dataset(  # type: ignore
        array_path, shape=(shape[0], shape[1], shape[2]), chunks=(shape[0], 1000, 1000), dtype="f4"
    )
    z.attrs["transform_mat3x3"] = trans
    z.attrs["index_values"] = return_periods

    trans = z.attrs["transform_mat3x3"]
    transform = Affine(trans[0], trans[1], trans[2], trans[3], trans[4], trans[5])

    coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))
    inv_trans = ~transform
    mat = np.array(inv_trans).reshape(3, 3)
    frac_image_coords = mat @ coords
    image_coords = np.floor(frac_image_coords).astype(int)
    for j in range(len(longitudes)):
        z[:, image_coords[1, j], image_coords[0, j]] = curve


def test_create_vulnerability_model():
    factory = VulnerabilityModelsFactory(config=vulnerability_config())
    vulnerability_models = factory.vulnerability_models()
    generic_model = vulnerability_models.vuln_model_for_asset_of_type(Asset)[0]
    assert isinstance(generic_model, ConfigBasedVulnerabilityModelAcute)
    asset = Asset(location="Europe", latitude=0, longitude=0)
    damage_curve = generic_model.curves[ImpactCurveKey.get(asset, generic_model.asset_attributes)]
    assert isinstance(damage_curve, PiecewiseLinearImpactCurve)
    expected_item = next(v for v in vulnerability_config() if v.asset_identifier == "type=Generic,location=Europe")
    np.testing.assert_array_almost_equal(
        cast(PiecewiseLinearImpactCurve, damage_curve).points_y, expected_item.points_y
    )


def test_read_write_utilities(working_dir):
    config = vulnerability_config()
    path = str(pathlib.Path(working_dir, "test_config.csv"))
    config_items_to_csv(config, path)
    items = config_items_from_csv(path)


def test_config_based_vulnerability_models(working_dir):

    location = "North America"
    latitudes = np.array([32.6017])
    longitudes = np.array([-87.7811])

    assets = [
        RealEstateAsset(
            location=location,
            latitude=latitudes[0],
            longitude=longitudes[0],
            type="Buildings/Commercial",
        ),
        PowerGeneratingAsset(
            location=location,
            latitude=latitudes[0],
            longitude=longitudes[0],
            type="Solar",
        ),
        ThermalPowerGeneratingAsset(
            location=location,
            latitude=latitudes[0],
            longitude=longitudes[0],
            type=None,
        ),
        ManufacturingAsset(
            location=location,
            latitude=latitudes[0],
            longitude=longitudes[0],
            type="Chemical",
        ),
        Asset(
            location=location,
            latitude=latitudes[0],
            longitude=longitudes[0],
            type=None,
        ),
        Asset(
            location=location,
            latitude=latitudes[0],
            longitude=longitudes[0],
            type=None,
            hazus_fl_building_dmg_id=1035,
            hazus_ws_building_dmg_id=1035,
        ),
        RealEstateAsset(
            location=location,
            latitude=latitudes[0],
            longitude=longitudes[0],
            type=None,
        ),
    ]

    setattr(assets[-1], "hazus_fl_building_dmg_id", 1035)
    setattr(assets[-1], "hazus_ws_building_dmg_id", 1035)

    request = requests.AssetImpactRequest(
        assets=Assets(items=[]),
        include_asset_level=True,
        include_calc_details=True,
        years=[2050],
        scenarios=["ssp585"],
    )

    path = str(pathlib.Path(working_dir, "test_vulnerability_config.csv"))
    config_items = config_based_vulnerability_models()
    config_items_to_csv(config_items, path)
    config_items = config_items_from_csv(path)
    factory = VulnerabilityModelsFactory(config=config_items)
    vulnerability_models = factory.vulnerability_models(disable_api_calls=True)

    store, root = zarr_memory_store()

    return_periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    shape, t = shape_transform_21600_43200(return_periods=return_periods)
    add_curves(
        root,
        longitudes,
        latitudes,
        "inundation/wri/v2/inunriver_rcp8p5_MIROC-ESM-CHEM_2050",
        shape,
        np.array(
            [
                0.001158079132437706,
                0.3938717246055603,
                0.8549619913101196,
                1.3880255222320557,
                1.7519289255142212,
                2.0910017490386963,
                2.5129663944244385,
                2.8202412128448486,
                3.115604877471924,
            ]
        ),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "inundation/wri/v2/inuncoast_rcp8p5_wtsub_2050_0",
        shape,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        return_periods,
        t,
    )

    return_periods = [20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    shape, t = shape_transform_21600_43200(return_periods=return_periods)
    add_curves(
        root,
        longitudes,
        latitudes,
        "wind/iris/v1/max_speed_ssp585_2050",
        shape,
        np.array(
            [
                24.443750381469727,
                26.787500381469727,
                28.631250381469727,
                30.096874237060547,
                31.034374237060547,
                32.00312423706055,
                33.037498474121094,
                33.75,
                34.23125076293945,
                39.32500076293945,
                43.0,
                44.95624923706055,
                46.1875,
                47.618751525878906,
                48.57500076293945,
                48.82500076293945,
                49.79375076293945,
                50.10625076293945,
            ]
        ),
        return_periods,
        t,
    )

    return_periods = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    shape, t = shape_transform_21600_43200(return_periods=return_periods)
    add_curves(
        root,
        longitudes,
        latitudes,
        "chronic_heat/osc/v2/days_wbgt_above_ACCESS-CM2_ssp585_2050",
        shape,
        np.array(
            [
                363.65054,
                350.21094,
                303.6388,
                240.48442,
                181.82924,
                128.46844,
                74.400276,
                1.3997267,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        return_periods,
        t,
    )

    return_periods = [0.0]
    shape, t = shape_transform_21600_43200(return_periods=return_periods)
    add_curves(
        root,
        longitudes,
        latitudes,
        "fire/jupiter/v1/fire_probability_ssp585_2050",
        shape,
        np.array([0.31867]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "drought/jupiter/v1/months_spei3m_below_-2_ssp585_2050",
        shape,
        np.array([0.17918]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "hail/jupiter/v1/days_above_5cm_ssp585_2050",
        shape,
        np.array([0.67936]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "chronic_heat/osc/v2/days_tas_above_25c_ACCESS-CM2_ssp585_2050",
        shape,
        np.array([148.55369567871094]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "chronic_heat/osc/v2/days_tas_above_30c_ACCESS-CM2_ssp585_2050",
        shape,
        np.array([65.30751037597656]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "chronic_heat/osc/v2/days_tas_above_35c_ACCESS-CM2_ssp585_2050",
        shape,
        np.array([0.6000000238418579]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "chronic_heat/osc/v2/days_tas_above_40c_ACCESS-CM2_ssp585_2050",
        shape,
        np.array([0.0]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "chronic_heat/osc/v2/days_tas_above_45c_ACCESS-CM2_ssp585_2050",
        shape,
        np.array([0.0]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "chronic_heat/osc/v2/days_tas_above_50c_ACCESS-CM2_ssp585_2050",
        shape,
        np.array([0.0]),
        return_periods,
        t,
    )
    add_curves(
        root,
        longitudes,
        latitudes,
        "chronic_heat/osc/v2/days_tas_above_55c_ACCESS-CM2_ssp585_2050",
        shape,
        np.array([0.0]),
        return_periods,
        t,
    )

    source_paths = get_default_source_paths(EmbeddedInventory())

    response = requests._get_asset_impacts(
        request,
        ZarrHazardModel(source_paths=source_paths, reader=ZarrReader(store)),
        vulnerability_models=vulnerability_models,
        assets=assets,
    )

    impacts = {impact.hazard_type: impact.impact_mean for impact in response.asset_impacts[0].impacts}
    np.testing.assert_almost_equal(impacts["Fire"], 0.23581580340862274 / 500.0)
    np.testing.assert_almost_equal(impacts["Wind"], 0.0010401051562173025)
    np.testing.assert_almost_equal(impacts["Hail"], 0.015738506029049557)
    np.testing.assert_almost_equal(impacts["CoastalInundation"], 0.00920245398773005)
    np.testing.assert_almost_equal(impacts["RiverineInundation"], 0.10152082730070953)
    np.testing.assert_almost_equal(impacts["Drought"], 0.0)

    impacts = {impact.hazard_type: impact.impact_mean for impact in response.asset_impacts[1].impacts}
    np.testing.assert_almost_equal(impacts["Hail"], 0.1132266620794932)
    np.testing.assert_almost_equal(impacts["ChronicHeat"], 0.04023474373249919)
    np.testing.assert_almost_equal(impacts["CoastalInundation"], 0.01285714285714285)
    np.testing.assert_almost_equal(impacts["RiverineInundation"], 0.13741988317883205)

    impacts = {impact.hazard_type: impact.impact_mean for impact in response.asset_impacts[2].impacts}
    np.testing.assert_almost_equal(impacts["ChronicHeat"], 0.2084267637692392)
    np.testing.assert_almost_equal(impacts["Drought"], 0.0029863332708676654)
    np.testing.assert_almost_equal(impacts["Hail"], 0.003849706510702769)
    np.testing.assert_almost_equal(impacts["CoastalInundation"], 0.01285714285714285)
    np.testing.assert_almost_equal(impacts["RiverineInundation"], 0.13741988317883205)

    impacts = {impact.hazard_type: impact.impact_mean for impact in response.asset_impacts[3].impacts}
    np.testing.assert_almost_equal(impacts["Hail"], 0.008378772993882497)
    np.testing.assert_almost_equal(impacts["Drought"], 0.004479499906301498)
    np.testing.assert_almost_equal(impacts["ChronicHeat"], 0.06771319298422895)
    np.testing.assert_almost_equal(impacts["CoastalInundation"], 0.01285714285714285)
    np.testing.assert_almost_equal(impacts["RiverineInundation"], 0.13741988317883205)

    impacts = {impact.hazard_type: impact.impact_mean for impact in response.asset_impacts[4].impacts}
    np.testing.assert_almost_equal(impacts["CoastalInundation"], 0.01285714285714285)
    np.testing.assert_almost_equal(impacts["RiverineInundation"], 0.13741988317883205)

    impacts = {impact.hazard_type: impact.impact_mean for impact in response.asset_impacts[5].impacts}
    np.testing.assert_almost_equal(impacts["Wind"], 0.0002031498336482825)
    np.testing.assert_almost_equal(impacts["CoastalInundation"], 0.0)
    np.testing.assert_almost_equal(impacts["RiverineInundation"], 0.043914584926972436)

    impacts = {impact.hazard_type: impact.impact_mean for impact in response.asset_impacts[6].impacts}
    np.testing.assert_almost_equal(impacts["Fire"], 0.23581580340862274 / 500.0)
    np.testing.assert_almost_equal(impacts["Wind"], 0.0002031498336482825)
    np.testing.assert_almost_equal(impacts["Hail"], 0.015738506029049557)
    np.testing.assert_almost_equal(impacts["CoastalInundation"], 0.0)
    np.testing.assert_almost_equal(impacts["RiverineInundation"], 0.043914584926972436)


@pytest.mark.skip("Only to be run when importing HAZUS damage functions.")
def test_import_hazus_damage_function(working_dir):
    tab = "FL_BuildingDamageFns"
    # https://bnpparibas.sharepoint.com/:x:/r/sites/RISKESG-CreditRiskQuantification/Documents%20partages/Credit%20Risk%20Quantification%20Channel/damage%20functions/Tables%20supporting%20flood%20and%20wind%20loss%20modeling%20with%20HAZUS%20-%20EI%20v3.0.xlsx?d=wcb24e9bb305446d2b0a27918a228aef5&csf=1&web=1&e=4bSc1E
    input_path = pathlib.Path(working_dir, "Tables supporting flood and wind loss modeling with HAZUS - EI v3.0.xlsx")
    df = pd.read_excel(io=str(input_path), sheet_name=tab)
    if "damageFnId" not in df.columns:
        raise ValueError(f"{tab} does not contain any 'damageFnId' column")
    attribute = "hazus_" + tab.replace("DamageFns", "_dmg_id").lower()
    impact_id = "disruption" if "_downtime_" in attribute else "damage"
    if tab.split("_")[0].upper() == "FL":
        hazard_class = "CoastalInundation,PluvialInundation,RiverineInundation"
        indicator_id = "flood_depth"
        indicator_units = "metres"
        if np.all([column.endswith("m") for column in df.columns if column.startswith("dmg_")]):
            points_x = [float(column[4:-1]) for column in df.columns if column.startswith("dmg_")]
        else:
            raise ValueError(f"Unsupported indicator units for {tab}")
    elif tab.split("_")[0].upper() == "WS":
        hazard_class = "Wind"
        indicator_id = "max_speed"
        indicator_units = "m/s"
        if np.all([column.endswith("kmh") for column in df.columns if column.startswith("dmg_")]):
            points_x = [float(column[4:-3]) / 3.6 for column in df.columns if column.startswith("dmg_")]
        elif np.all([column.endswith("ms") for column in df.columns if column.startswith("dmg_")]):
            points_x = [float(column[4:-2]) for column in df.columns if column.startswith("dmg_")]
        else:
            raise ValueError(f"Unsupported indicator units for {tab}")
    else:
        raise ValueError(f"Unsupported hazard class for {tab}")
    config_items = list(
        df.apply(
            lambda row: VulnerabilityConfigItem(
                hazard_class=hazard_class,
                asset_class="Asset",
                asset_identifier=attribute + "=" + str(row["damageFnId"]),
                indicator_id=indicator_id,
                indicator_units=indicator_units,
                impact_id=impact_id,
                impact_units=None,
                curve_type="indicator/piecewise_linear",
                points_x=points_x,
                points_y=[float(row[key]) for key in row.keys() if key.startswith("dmg_")],
                cap_of_points_x=None,
                cap_of_points_y=None,
                activation_of_points_x=None,
            ),
            axis=1,
        ).values
    )
    output_path = pathlib.Path(working_dir, attribute + ".csv")
    config_items_to_csv(config_items, str(output_path))
