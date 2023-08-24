from typing import Dict, Sequence

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.risk import RiskMeasureCalculator
from physrisk.risk_models.risk_models import RealEstateToyRiskMeasures
from physrisk.vulnerability_models import power_generating_asset_models as pgam
from physrisk.vulnerability_models.chronic_heat_models import ChronicHeatGZNModel
from physrisk.vulnerability_models.real_estate_models import (
    RealEstateCoastalInundationModel,
    RealEstateRiverineInundationModel,
)

from .assets import IndustrialActivity, PowerGeneratingAsset, RealEstateAsset, TestAsset
from .hazard_model import HazardModel
from .vulnerability_model import VulnerabilityModelBase


def get_default_hazard_model() -> HazardModel:
    # Model that gets hazard event data from Zarr storage
    return ZarrHazardModel(source_paths=get_default_source_paths())


def get_default_vulnerability_models() -> Dict[type, Sequence[VulnerabilityModelBase]]:
    """Get default exposure/vulnerability models for different asset types."""
    return {
        PowerGeneratingAsset: [pgam.InundationModel()],
        RealEstateAsset: [RealEstateCoastalInundationModel(), RealEstateRiverineInundationModel()],
        IndustrialActivity: [ChronicHeatGZNModel()],
        TestAsset: [pgam.TemperatureModel()],
    }


def get_default_risk_measure_calculators() -> Dict[type, RiskMeasureCalculator]:
    """For asset-level risk measure, define the measure calculators to use."""
    return {RealEstateAsset: RealEstateToyRiskMeasures()}
