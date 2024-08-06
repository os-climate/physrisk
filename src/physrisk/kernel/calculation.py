from typing import Dict, Sequence, Type

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.hazards import ChronicHeat, Drought, Fire, Hail, Precipitation
from physrisk.kernel.impact_distrib import ImpactType
from physrisk.kernel.risk import RiskMeasureCalculator, RiskMeasuresFactory
from physrisk.risk_models.generic_risk_model import GenericScoreBasedRiskMeasures
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures
from physrisk.vulnerability_models import power_generating_asset_models as pgam
from physrisk.vulnerability_models.chronic_heat_models import ChronicHeatGZNModel
from physrisk.vulnerability_models.example_models import PlaceholderVulnerabilityModel
from physrisk.vulnerability_models.real_estate_models import (
    CoolingModel,
    GenericTropicalCycloneModel,
    RealEstateCoastalInundationModel,
    RealEstateRiverineInundationModel,
)
from physrisk.vulnerability_models.thermal_power_generation_models import (
    Asset,
    ThermalPowerGenerationAirTemperatureModel,
    ThermalPowerGenerationCoastalInundationModel,
    ThermalPowerGenerationDroughtModel,
    ThermalPowerGenerationRiverineInundationModel,
    ThermalPowerGenerationWaterStressModel,
    ThermalPowerGenerationWaterTemperatureModel,
)

from .assets import (
    IndustrialActivity,
    PowerGeneratingAsset,
    RealEstateAsset,
    TestAsset,
    ThermalPowerGeneratingAsset,
)
from .hazard_model import HazardModel
from .vulnerability_model import VulnerabilityModelBase


def get_default_hazard_model() -> HazardModel:
    # Model that gets hazard event data from Zarr storage
    return ZarrHazardModel(source_paths=get_default_source_paths())


def get_default_vulnerability_models() -> Dict[type, Sequence[VulnerabilityModelBase]]:
    """Get default exposure/vulnerability models for different asset types."""
    return {
        Asset: [
            # This is for a generic 'unknown' asset. To be replaced by a config-based model (when complete)
            # but for now, treat like a real estate asset.
            RealEstateCoastalInundationModel(),
            RealEstateRiverineInundationModel(),
            GenericTropicalCycloneModel(),
            PlaceholderVulnerabilityModel("fire_probability", Fire, ImpactType.damage),
            PlaceholderVulnerabilityModel(
                "days/above/35c", ChronicHeat, ImpactType.damage
            ),
            PlaceholderVulnerabilityModel("days/above/5cm", Hail, ImpactType.damage),
            PlaceholderVulnerabilityModel(
                "months/spei3m/below/-2", Drought, ImpactType.damage
            ),
            PlaceholderVulnerabilityModel(
                "max/daily/water_equivalent", Precipitation, ImpactType.damage
            ),
        ],
        PowerGeneratingAsset: [pgam.InundationModel()],
        RealEstateAsset: [
            RealEstateCoastalInundationModel(),
            RealEstateRiverineInundationModel(),
            GenericTropicalCycloneModel(),
            CoolingModel(),
        ],
        IndustrialActivity: [ChronicHeatGZNModel()],
        ThermalPowerGeneratingAsset: [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationDroughtModel(),
            ThermalPowerGenerationRiverineInundationModel(),
            ThermalPowerGenerationWaterStressModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
        ],
        TestAsset: [pgam.TemperatureModel()],
    }


def get_default_risk_measure_calculators() -> Dict[Type[Asset], RiskMeasureCalculator]:
    """For asset-level risk measure, define the measure calculators to use."""
    return {RealEstateAsset: RealEstateToyRiskMeasures()}


class DefaultMeasuresFactory(RiskMeasuresFactory):
    def calculators(self, use_case_id: str) -> Dict[Type[Asset], RiskMeasureCalculator]:
        if use_case_id == "generic":
            return {Asset: GenericScoreBasedRiskMeasures()}
        return get_default_risk_measure_calculators()
