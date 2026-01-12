import pathlib
from typing import Dict, Optional, Sequence, Type


from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.hazards import (
    ChronicHeat,
    CoastalInundation,
    Drought,
    Fire,
    Hail,
    Precipitation,
    RiverineInundation,
    Wind,
)
from physrisk.kernel.impact_distrib import ImpactType
from physrisk.kernel.risk import (
    PortfolioRiskMeasureCalculator,
    RiskMeasureCalculator,
    RiskMeasuresFactory,
)
from physrisk.risk_models.generic_risk_model import GenericScoreBasedRiskMeasures
from physrisk.kernel.risk import (
    NullAssetBasedPortfolioRiskMeasureCalculator,
)
from physrisk.risk_models.risk_models import (
    ECBScoreRiskMeasures,
)
from physrisk.vulnerability_models import power_generating_asset_models as pgam
from physrisk.vulnerability_models.chronic_heat_models import ChronicHeatGZNModel
from physrisk.vulnerability_models.config_based_impact_curves import (
    config_items_from_csv,
)
from physrisk.vulnerability_models.example_models import PlaceholderVulnerabilityModel
from physrisk.vulnerability_models.real_estate_models import (
    CoolingModel,
    AKFire,
    GenericTropicalCycloneModel,
    IPCCDroughtModelcdd,
    IPCCDroughtModelspi6,
    RealEstateCoastalInundationModel,
    RealEstatePluvialInundationModel,
    RealEstateRiverineInundationModel,
    SubsidenceModel,
    WaterstressModel,
)

from physrisk.vulnerability_models.thermal_power_generation_models import (
    Asset,
    ThermalPowerGenerationAirTemperatureModel,
    ThermalPowerGenerationCoastalInundationModel,
    ThermalPowerGenerationDroughtModel,
    ThermalPowerGenerationRiverineInundationModel,
    ThermalPowerGenerationWaterStressModel,
    ThermalPowerGenerationWaterTemperatureModel,
    ThermalPowerGenerationAqueductWaterRiskModel,
    ThermalPowerGenerationSubsidenceModel,
)

from physrisk.kernel.assets import (
    IndustrialActivity,
    PowerGeneratingAsset,
    RealEstateAsset,
    TestAsset,
    ThermalPowerGeneratingAsset,
)
from physrisk.kernel.hazard_model import HazardModel
from physrisk.kernel.vulnerability_model import (
    VulnerabilityModelsFactory as PVulnerabilityModelsFactory,
    DictBasedVulnerabilityModels,
    VulnerabilityModelBase,
)
from physrisk.vulnerability_models.vulnerability import VulnerabilityModelsFactory


def get_default_hazard_model() -> HazardModel:
    # Model that gets hazard event data from Zarr storage
    return ZarrHazardModel(source_paths=get_default_source_paths())


def get_placeholder_models() -> Sequence[VulnerabilityModelBase]:
    return [
        PlaceholderVulnerabilityModel("fire_probability", Fire, ImpactType.damage),
        PlaceholderVulnerabilityModel("days/above/35c", ChronicHeat, ImpactType.damage),
        PlaceholderVulnerabilityModel("days/above/5cm", Hail, ImpactType.damage),
        PlaceholderVulnerabilityModel(
            "months/spei3m/below/-2", Drought, ImpactType.damage
        ),
        PlaceholderVulnerabilityModel(
            "max/daily/water_equivalent", Precipitation, ImpactType.damage
        ),
    ]


def get_default_vulnerability_models() -> Dict[type, Sequence[VulnerabilityModelBase]]:
    """Base set of programmatic models; other models are added on top of these
    There is a specific treatment for power generating assets and real estate assets.
    """
    return {
        PowerGeneratingAsset: [pgam.InundationModel()],
        RealEstateAsset: [
            RealEstateCoastalInundationModel(),
            RealEstatePluvialInundationModel(),
            RealEstateRiverineInundationModel(),
        ],
        ThermalPowerGeneratingAsset: [
            ThermalPowerGenerationAirTemperatureModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            ThermalPowerGenerationDroughtModel(),
            ThermalPowerGenerationRiverineInundationModel(),
            ThermalPowerGenerationWaterStressModel(),
            ThermalPowerGenerationWaterTemperatureModel(),
        ],
    }


def alternate_default_vulnerability_models_scores() -> Dict[
    type, Sequence[VulnerabilityModelBase]
]:
    """A vulnerability models set that combines loss-based and exposure-based scores."""
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
            GenericTropicalCycloneModel(),
        ],
        TestAsset: [pgam.TemperatureModel()],
    }


def get_ecb_vulnerability_models() -> Dict[type, Sequence[VulnerabilityModelBase]]:
    """Get exposure/vulnerability models for different asset types.

    This set uses the data used in the stress test article from the ECB.
    """
    base_dir = (
        pathlib.Path(__file__).parent.parent / "data" / "static" / "vulnerability"
    )
    source_dir = base_dir / "vulnerability_config.csv"
    config_items = config_items_from_csv(str(source_dir))
    factory = VulnerabilityModelsFactory(config=config_items)
    vulnerability_models = factory.vulnerability_models(disable_api_calls=True)
    conf = list(vulnerability_models.models.values())
    vulnerability_models_list = list(conf[0])

    wind_model = [
        model for model in vulnerability_models_list if model.hazard_type is Wind
    ][0]

    riverine_model = [
        model
        for model in vulnerability_models_list
        if model.hazard_type is RiverineInundation
    ][0]

    coastal_model = [
        model
        for model in vulnerability_models_list
        if model.hazard_type is CoastalInundation
    ][0]

    return {
        RealEstateAsset: [
            riverine_model,
            coastal_model,
            IPCCDroughtModelcdd(),
            IPCCDroughtModelspi6(),
            SubsidenceModel(),
            WaterstressModel(),
            wind_model,
            GenericTropicalCycloneModel(),
            AKFire(),
        ],
        ThermalPowerGeneratingAsset: [
            ThermalPowerGenerationRiverineInundationModel(),
            ThermalPowerGenerationCoastalInundationModel(),
            IPCCDroughtModelcdd(),
            IPCCDroughtModelspi6(),
            ThermalPowerGenerationSubsidenceModel(),
            ThermalPowerGenerationAqueductWaterRiskModel(),
        ],
    }


def get_default_risk_measure_calculators() -> Dict[type[Asset], RiskMeasureCalculator]:
    """For asset-level risk measure, define the measure calculators to use."""
    return {Asset: GenericScoreBasedRiskMeasures()}
    # return {RealEstateAsset: RealEstateToyRiskMeasures()}


def get_ecb_risk_measure_calculators() -> Dict[type[Asset], RiskMeasureCalculator]:
    """For asset-level stress test risk measure, define the measure calculators to use."""
    return {Asset: ECBScoreRiskMeasures()}


def get_generic_risk_measure_calculators() -> Dict[type[Asset], RiskMeasureCalculator]:
    """For asset-level generic risk measure, define the measure calculators to use."""
    return {Asset: GenericScoreBasedRiskMeasures()}


class DefaultMeasuresFactory(RiskMeasuresFactory):
    """Factory class for selecting appropriate risk measure calculators based on the use case."""

    def asset_calculators(
        self, use_case_id: str
    ) -> Dict[Type[Asset], RiskMeasureCalculator]:
        """Get the appropriate risk measure calculators based on the use case identifier."""
        if use_case_id.upper() == "GENERIC":
            return get_generic_risk_measure_calculators()
        elif use_case_id.upper() == "ECB_SCORES":
            return get_ecb_risk_measure_calculators()
        return get_default_risk_measure_calculators()

    def portfolio_calculator(self, use_case_id: str) -> PortfolioRiskMeasureCalculator:
        return NullAssetBasedPortfolioRiskMeasureCalculator()


class DictBasedVulnerabilityModelsFactory(PVulnerabilityModelsFactory):
    """Factory class for selecting appropriate vulnerability models based on the use case."""

    def __init__(self, use_case_id: str = "DEFAULT"):
        """Initialize the factory with a specific use case identifier.

        Parameters
        ----------
        use_case_id : str, optional
            An identifier for the use case to determine the appropriate vulnerability models.
            Defaults to DEFAULT.

        """
        self.use_case_id = use_case_id

    def vulnerability_models(
        self, hazard_scope: Optional[set[type]] = None
    ) -> DictBasedVulnerabilityModels:
        """Get the appropriate vulnerability model based on the use case identifier."""
        if self.use_case_id.upper() == "DEFAULT":
            models = get_default_vulnerability_models()
        elif self.use_case_id.upper() == "ECB_SCORES":
            models = get_ecb_vulnerability_models()
        else:
            raise ValueError("Unsupported use_case_id")
        return DictBasedVulnerabilityModels(models)
