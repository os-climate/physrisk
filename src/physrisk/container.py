from typing import Dict, MutableMapping, Optional

from dependency_injector import containers, providers

from physrisk.data.hazard_data_provider import SourcePaths
from physrisk.data.image_creator import ImageCreator
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.credentials_provider import (
    CredentialsProvider,
    EnvCredentialsProvider,
)
from physrisk.hazard_models.hazard_cache import GeometryH3BasedCache, MemoryStore
from physrisk.hazard_models.hazard_model_factory import CompositeHazardModel
from physrisk.hazard_models.jba_image_creator import (
    CombinedImageCreator,
    JBAImageCreator,
)
from physrisk.kernel import calculation as calc
from physrisk.kernel.hazard_model import HazardModelFactory
from physrisk.kernel.hazards import Hazard
from physrisk.kernel.vulnerability_model import (
    DictBasedVulnerabilityModels,
    VulnerabilityModels as PVulnerabilityModels,
    VulnerabilityModelsFactory as PVulnerabilityModelsFactory,
)
from physrisk.requests import (
    PhysriskDefaultEncoder,
    Requester,
    _create_inventory,
    create_source_paths,
)
from physrisk.vulnerability_models.config_based_vuln_model_acute import (
    StandardOfProtection,
)
from physrisk.vulnerability_models.configuration.asset_factory import (
    DefaultAssetFactory,
)
from physrisk.vulnerability_models.vulnerability import VulnerabilityModelsFactory


class DefaultHazardModelFactory(HazardModelFactory):
    def __init__(
        self,
        cache_store: GeometryH3BasedCache,
        credentials: CredentialsProvider,
        inventory: Inventory,
        source_paths: SourcePaths,
        store: Optional[MutableMapping] = None,
        reader: Optional[ZarrReader] = None,
        default_interpolation: str = "floor",
        zarr_max_workers: int = 32,
    ):
        self.cache_store = cache_store
        self.inventory = inventory
        self.source_paths = source_paths
        self.store = store
        self.reader = reader
        self.default_interpolation = default_interpolation
        self.credentials = credentials
        self.zarr_max_workers = zarr_max_workers
        self.zarr_image_creator = (
            ImageCreator(inventory, source_paths, reader)
            if reader is not None
            else None
        )
        self.jba_image_creator = (
            JBAImageCreator(credentials)
            if credentials.jba_vision_password() != ""
            else None
        )

    def hazard_model(
        self,
        interpolation: Optional[str] = None,
        provider_max_requests: Dict[str, int] = {},
        interpolate_years: bool = True,
    ):
        # this is done to allow interpolation etc to be set dynamically,
        # e.g. different requests can have different parameters.
        return CompositeHazardModel(
            cache_store=self.cache_store,
            credentials=self.credentials,
            source_paths=self.source_paths,
            store=self.store,
            reader=self.reader,
            interpolation=interpolation
            if interpolation is not None
            else self.default_interpolation,
            provider_max_requests=provider_max_requests,
            restrict_coverage=False,
            interpolate_years=interpolate_years,
            use_jba_coastal=False,
            zarr_max_workers=self.zarr_max_workers,
        )

    def image_creator(self):
        return CombinedImageCreator(self.zarr_image_creator, self.jba_image_creator)


class DictBasedVulnerabilityModelsFactory(PVulnerabilityModelsFactory):
    def vulnerability_models(
        self,
        hazard_scope: dict[type[Hazard], set[str] | None] | None = None,
        map_unknown_occ: bool = True,
    ) -> PVulnerabilityModels:
        return DictBasedVulnerabilityModels(
            calc.alternate_default_vulnerability_models_scores()
        )


class DefaultVulnerabilityModelsFactory(VulnerabilityModelsFactory):
    """Default vulnerability approach. 'default_vulnerability_models'
    programmatic models are used, to which FEMA Hazus vulnerability-based
    models are added and finally configuration.
    FEMA Hazus vulnerability-based models excluded by default (until non-experimental).
    """

    def __init__(self, use_oed_hazus_curves: bool = True):
        super().__init__(
            config=VulnerabilityModelsFactory.embedded_vulnerability_config(),
            programmatic_models=calc.default_vulnerability_models(),
            use_oed_hazus_curves=use_oed_hazus_curves,
            standard_of_protection=StandardOfProtection.CONSTANT_DEPTH,
        )


class Container(containers.DeclarativeContainer):
    asset_factory = providers.Factory(DefaultAssetFactory)

    cache_store = providers.Singleton(GeometryH3BasedCache, store=MemoryStore())

    colormaps = providers.Singleton(lambda: EmbeddedInventory().colormaps())

    config = providers.Configuration(
        default={"zarr_sources": ["embedded"], "zarr_max_workers": 32}
    )

    credentials = providers.Singleton(EnvCredentialsProvider, disable_api_calls=False)

    inventory_reader = providers.Singleton(InventoryReader)

    inventory = providers.Singleton(
        _create_inventory, reader=inventory_reader, sources=config.zarr_sources
    )

    json_encoder_cls = providers.Object(PhysriskDefaultEncoder)

    sig_figures = providers.Object(4)  # -1 indicates no rounding

    source_paths = providers.Factory(create_source_paths, inventory=inventory)

    zarr_store = providers.Singleton(ZarrReader.create_s3_zarr_store)

    zarr_reader = providers.Singleton(ZarrReader, store=zarr_store)

    # why do we have factories for hazard models, vulnerability models and measures?
    # this is because the models may need to be created with different parameters for different requests

    hazard_model_factory = providers.Factory(
        DefaultHazardModelFactory,
        cache_store=cache_store,
        credentials=credentials,
        inventory=inventory,
        reader=zarr_reader,
        source_paths=source_paths,
        zarr_max_workers=config.zarr_max_workers,
    )

    measures_factory = providers.Factory(calc.DefaultMeasuresFactory)

    vulnerability_models_factory = providers.Factory(DefaultVulnerabilityModelsFactory)

    requester = providers.Singleton(
        Requester,
        asset_factory=asset_factory,
        hazard_model_factory=hazard_model_factory,
        vulnerability_models_factory=vulnerability_models_factory,
        inventory=inventory,
        source_paths=source_paths,
        inventory_reader=inventory_reader,
        reader=zarr_reader,
        colormaps=colormaps,
        measures_factory=measures_factory,
        json_encoder_cls=json_encoder_cls,
        sig_figures=sig_figures,
    )
