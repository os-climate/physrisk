from dependency_injector import containers, providers

from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.requests import Requester, _create_inventory, create_source_paths


class Container(containers.DeclarativeContainer):
    config = providers.Configuration(default={"zarr_sources": ["embedded", "hazard"]})

    colormaps = providers.Singleton(lambda: EmbeddedInventory().colormaps())

    inventory_reader = providers.Singleton(InventoryReader)

    inventory = providers.Singleton(_create_inventory, reader=inventory_reader, sources=config.zarr_sources)

    source_paths = providers.Factory(create_source_paths, inventory=inventory)

    zarr_store = providers.Singleton(ZarrReader.create_s3_zarr_store)

    zarr_reader = providers.Singleton(ZarrReader, store=zarr_store)

    hazard_model = providers.Singleton(
        ZarrHazardModel, reader=zarr_reader, source_paths=source_paths, interpolation="floor"
    )

    requester = providers.Singleton(
        Requester,
        hazard_model=hazard_model,
        inventory=inventory,
        inventory_reader=inventory_reader,
        reader=zarr_reader,
        colormaps=colormaps,
    )
