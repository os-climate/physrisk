from typing import Dict, MutableMapping, Optional, Set, Type

from dependency_injector import containers, providers

from physrisk.data.hazard_data_provider import SourcePaths
from physrisk.data.image_creator import ImageCreator
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel import calculation as calc
from physrisk.kernel.hazard_model import HazardModelFactory
from physrisk.kernel.hazards import Hazard
from physrisk.kernel.vulnerability_model import (
    DictBasedVulnerabilityModels,
    VulnerabilityModels,
    VulnerabilityModelsFactory,
)
from physrisk.requests import (
    PhysriskDefaultEncoder,
    Requester,
    _create_inventory,
    create_source_paths,
)


class ZarrHazardModelFactory(HazardModelFactory):
    def __init__(
        self,
        inventory: Inventory,
        source_paths: SourcePaths,
        store: Optional[MutableMapping] = None,
        reader: Optional[ZarrReader] = None,
        default_interpolation: str = "floor",
    ):
        self.inventory = inventory
        self.source_paths = source_paths
        self.store = store
        self.reader = reader
        self.default_interpolation = default_interpolation

    def hazard_model(
        self,
        interpolation: Optional[str] = None,
        provider_max_requests: Dict[str, int] = {},
        interpolate_years: bool = False,
    ):
        # this is done to allow interpolation to be set dynamically, e.g. different requests can have different
        # parameters.
        return ZarrHazardModel(
            source_paths=self.source_paths,
            store=self.store,
            reader=self.reader,
            interpolation=interpolation
            if interpolation is not None
            else self.default_interpolation,
            interpolate_years=interpolate_years,
        )

    def image_creator(self):
        return ImageCreator(self.inventory, self.source_paths, self.reader)


class DictBasedVulnerabilityModelsFactory(VulnerabilityModelsFactory):
    def vulnerability_models(
        self, hazard_scope: Optional[Set[Type[Hazard]]] = None
    ) -> VulnerabilityModels:
        return DictBasedVulnerabilityModels(calc.get_default_vulnerability_models())


class Container(containers.DeclarativeContainer):
    config = providers.Configuration(default={"zarr_sources": ["embedded", "hazard"]})

    colormaps = providers.Singleton(lambda: EmbeddedInventory().colormaps())

    inventory_reader = providers.Singleton(InventoryReader)

    inventory = providers.Singleton(
        _create_inventory, reader=inventory_reader, sources=config.zarr_sources
    )

    json_encoder_cls = providers.Object(PhysriskDefaultEncoder)

    sig_figures = providers.Object(-1)  # -1 indicates no rounding

    source_paths = providers.Factory(create_source_paths, inventory=inventory)

    zarr_store = providers.Singleton(ZarrReader.create_s3_zarr_store)

    zarr_reader = providers.Singleton(ZarrReader, store=zarr_store)

    hazard_model_factory = providers.Factory(
        ZarrHazardModelFactory,
        inventory=inventory,
        reader=zarr_reader,
        source_paths=source_paths,
    )

    measures_factory = providers.Factory(calc.DefaultMeasuresFactory)

    vulnerability_models_factory = providers.Factory(
        DictBasedVulnerabilityModelsFactory
    )

    requester = providers.Singleton(
        Requester,
        hazard_model_factory=hazard_model_factory,
        vulnerability_models_factory=vulnerability_models_factory,
        inventory=inventory,
        inventory_reader=inventory_reader,
        reader=zarr_reader,
        colormaps=colormaps,
        measures_factory=measures_factory,
        json_encoder_cls=json_encoder_cls,
        sig_figures=sig_figures,
    )
