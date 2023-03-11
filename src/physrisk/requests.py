import importlib
import json
import os
from importlib import import_module
from pathlib import PosixPath
from typing import Any, Dict, List, Optional, cast

import fsspec.implementations.local as local
import numpy as np
import zarr

import physrisk.data.static.example_portfolios
from physrisk.api.v1.common import Distribution, ExceedanceCurve, VulnerabilityDistrib
from physrisk.api.v1.hazard_image import HazardImageRequest
from physrisk.data.inventory_reader import InventoryReader

from .api.v1.hazard_data import (
    HazardAvailabilityRequest,
    HazardAvailabilityResponse,
    HazardDescriptionRequest,
    HazardDescriptionResponse,
    HazardEventDataRequest,
    HazardEventDataResponse,
    HazardEventDataResponseItem,
    HazardResource,
    IntensityCurve,
)
from .api.v1.impact_req_resp import (
    AcuteHazardCalculationDetails,
    AssetImpactRequest,
    AssetImpactResponse,
    AssetLevelImpact,
    Assets,
    AssetSingleHazardImpact,
)
from .data.image_creator import ImageCreator
from .data.inventory import EmbeddedInventory, Inventory
from .data.pregenerated_hazard_model import ZarrHazardModel
from .kernel import Asset, Hazard
from .kernel import calculation as calc
from .kernel.hazard_model import HazardDataRequest
from .kernel.hazard_model import HazardEventDataResponse as hmHazardEventDataResponse
from .kernel.hazard_model import HazardParameterDataResponse

# module level singletons, populated/updated on hazard data availability request
_inventory: Optional[Inventory] = None
_colormaps: Optional[Dict[str, Any]] = None

# hooks to facilitate testing
_hazard_test_local_path = ""


def _create_inventory_reader():
    global _hazard_test_local_path
    if _hazard_test_local_path == "":
        return InventoryReader()
    # otherwise, use local test version
    return InventoryReader(fs=local.LocalFileSystem(), base_path=_hazard_test_local_path)


def _create_zarr_store():
    global _hazard_test_local_path
    if _hazard_test_local_path == "":
        return None
    # otherwise, use local test version
    return zarr.DirectoryStore(os.path.join(_hazard_test_local_path, "hazard_test", "hazard.zarr"))


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dumps(dict):
    return json.dumps(dict, cls=NumpyArrayEncoder)


def hazard_resources() -> Inventory:
    global _inventory, _colormaps
    if _inventory is None:
        _inventory, _colormaps = _get_updated_hazard_resources()
    return _inventory


def colormaps() -> Dict[str, Any]:
    global _inventory, _colormaps
    if _colormaps is None:
        _inventory, _colormaps = _get_updated_hazard_resources(["embedded", "hazard_test"])
    return _colormaps


def get(*, request_id, request_dict, store=None):
    if store is None:
        store = _create_zarr_store()
    if request_id == "get_hazard_data":
        request = HazardEventDataRequest(**request_dict)
        return json.dumps(_get_hazard_data(request, store=store).dict())
    elif request_id == "get_hazard_data_availability":
        request = HazardAvailabilityRequest(**request_dict)
        return json.dumps(_get_hazard_data_availability(request).dict())
    elif request_id == "get_hazard_data_description":
        request = HazardDescriptionRequest(**request_dict)
        return json.dumps(_get_hazard_data_description(request).dict())
    elif request_id == "get_asset_impact":
        request = AssetImpactRequest(**request_dict)
        return dumps(_get_asset_impacts(request, store=store).dict())
    elif request_id == "get_example_portfolios":
        return dumps(_get_example_portfolios())
    else:
        raise ValueError(f"request type '{request_id}' not found")


def get_image(*, request_dict):
    global _create_zarr_store
    request = HazardImageRequest(**request_dict)
    if not _read_permitted(request.group_ids, hazard_resources().resources[request.resource]):
        raise PermissionError()
    model = hazard_resources().resources[request.resource]
    path = str(PosixPath(model.path, model.map.array_name)).format(scenario=request.scenarioId, year=request.year)
    creator = ImageCreator(_create_zarr_store())  # store=ImageCreator.test_store(path))
    return creator.convert(path, colormap=request.colormap, min_value=request.min_value, max_value=request.max_value)


def _read_permitted(group_ids: List[str], resource: HazardResource):
    """_summary_

    Args:
        group_ids (List[str]): Groups to which requester belongs.
        resourceId (str): Resource identifier.

    Returns:
        bool: True is requester is permitted access to models comprising resource.
    """
    return ("osc" in group_ids) or resource.group_id == "public"


def _get_hazard_data_availability(request: HazardAvailabilityRequest):
    global _inventory, _colormaps
    _inventory, _colormaps = _get_updated_hazard_resources(request.sources)
    assert _inventory is not None
    response = HazardAvailabilityResponse(
        models=list(_inventory.resources.values()), colormaps=_colormaps
    )  # type: ignore
    return response


def _get_hazard_data_description(request: HazardDescriptionRequest):
    reader = InventoryReader()
    descriptions = reader.read_description_markdown(request.paths)
    return HazardDescriptionResponse(descriptions=descriptions)


def _get_hazard_data(request: HazardEventDataRequest, source_paths=None, store=None):
    if any(
        not _read_permitted(request.group_ids, hazard_resources().resources_by_type_id[(i.event_type, i.model)][0])
        for i in request.items
    ):
        raise PermissionError()
    hazard_model = _create_hazard_model(
        interpolation=request.interpolation, inventory=hazard_resources(), source_paths=source_paths, store=store
    )

    # get hazard event types:
    event_types = Hazard.__subclasses__()
    event_dict = dict((et.__name__, et) for et in event_types)
    event_dict.update((est.__name__, est) for et in event_types for est in et.__subclasses__())

    # flatten list to let event processor decide how to group
    item_requests = []
    all_requests = []
    for item in request.items:
        event_type = event_dict[item.event_type]

        data_requests = [
            HazardDataRequest(event_type, lon, lat, model=item.model, scenario=item.scenario, year=item.year)
            for (lon, lat) in zip(item.longitudes, item.latitudes)
        ]

        all_requests.extend(data_requests)
        item_requests.append(data_requests)

    response_dict = hazard_model.get_hazard_events(all_requests)
    # responses comes back as a dictionary because requests may be executed in different order to list
    # to optimise performance.

    response = HazardEventDataResponse(items=[])

    for i, item in enumerate(request.items):
        requests = item_requests[i]
        resps = (response_dict[req] for req in requests)
        intensity_curves = [
            IntensityCurve(intensities=list(resp.intensities), return_periods=list(resp.return_periods))
            if isinstance(resp, hmHazardEventDataResponse)
            else IntensityCurve(intensities=[resp.parameter], return_periods=[])
            if isinstance(resp, HazardParameterDataResponse)
            else None
            for resp in resps
        ]
        response.items.append(
            HazardEventDataResponseItem(
                intensity_curve_set=intensity_curves,
                request_item_id=item.request_item_id,
                event_type=item.event_type,
                model=item.model,
                scenario=item.scenario,
                year=item.year,
            )
        )

    return response


def _get_updated_hazard_resources(sources: Optional[List[str]] = None):
    global _create_inventory_reader
    models: List[HazardResource] = []
    colormaps: Dict[str, Dict[str, Any]] = {}
    request_sources = ["embedded"] if sources is None else [s.lower() for s in sources]
    reader = None
    for source in request_sources:
        if source == "embedded":
            inventory = EmbeddedInventory()
            for model in inventory.to_resources():
                models.append(model)
            colormaps.update(inventory.colormaps())
        elif source == "hazard" or source == "hazard_test":
            if reader is None:
                reader = _create_inventory_reader()
            for model in reader.read(source):
                models.append(model)
    return Inventory(models), colormaps


def create_assets(assets: Assets):
    """Create list of Asset objects from the Assets API object:"""
    module = import_module("physrisk.kernel.assets")
    asset_objs = []
    for asset in assets.items:
        asset_obj = cast(
            Asset,
            getattr(module, asset.asset_class)(
                asset.latitude, asset.longitude, type=asset.type, location=asset.location
            ),
        )
        asset_objs.append(asset_obj)
    return asset_objs


def _create_hazard_model(interpolation="floor", inventory: Optional[Inventory] = None, source_paths=None, store=None):
    if source_paths is None:
        source_paths = calc.get_default_zarr_source_paths()

    if inventory is not None:
        source_paths = calc.get_source_paths_from_inventory(inventory, source_paths)

    hazard_model = ZarrHazardModel(source_paths, store=store, interpolation=interpolation)

    return hazard_model


def _get_asset_impacts(request: AssetImpactRequest, source_paths=None, store=None):
    hazard_model = _create_hazard_model(source_paths=source_paths, store=store)
    vulnerability_models = calc.get_default_vulnerability_models()

    # we keep API definition of asset separate from internal Asset class; convert by reflection
    # based on asset_class:
    assets = create_assets(request.assets)

    results = calc.calculate_impacts(
        assets, hazard_model, vulnerability_models, scenario=request.scenario, year=request.year
    )

    # note that this does rely on ordering of dictionary (post 3.6)
    impacts: Dict[Asset, List[AssetSingleHazardImpact]] = {}
    for (asset, hazard_type), v in results.items():
        # calculation details
        if v.event is not None and v.vulnerability is not None:
            hazard_exceedance = v.event.to_exceedance_curve()

            vulnerability_distribution = VulnerabilityDistrib(
                intensity_bin_edges=v.vulnerability.intensity_bins,
                impact_bin_edges=v.vulnerability.impact_bins,
                prob_matrix=v.vulnerability.prob_matrix,
            )

            calc_details = AcuteHazardCalculationDetails(
                hazard_exceedance=ExceedanceCurve(
                    values=hazard_exceedance.values, exceed_probabilities=hazard_exceedance.probs
                ),
                hazard_distribution=Distribution(bin_edges=v.event.intensity_bin_edges, probabilities=v.event.prob),
                vulnerability_distribution=vulnerability_distribution,
            )

        impact_exceedance = v.impact.to_exceedance_curve()
        hazard_impacts = AssetSingleHazardImpact(
            hazard_type=v.impact.hazard_type.__name__,
            impact_type=v.impact.impact_type.name,
            impact_exceedance=ExceedanceCurve(
                values=impact_exceedance.values, exceed_probabilities=impact_exceedance.probs
            ),
            impact_distribution=Distribution(bin_edges=v.impact.impact_bins, probabilities=v.impact.prob),
            impact_mean=v.impact.mean_impact(),
            impact_std_deviation=0,  # TODO!
            calc_details=None if v.event is None else calc_details,
        )

        impacts.setdefault(asset, []).append(hazard_impacts)

    asset_impacts = [AssetLevelImpact(asset_id="", impacts=a) for a in impacts.values()]

    return AssetImpactResponse(asset_impacts=asset_impacts)


def _get_example_portfolios() -> List[Assets]:
    portfolios = []
    for file in importlib.resources.contents(physrisk.data.static.example_portfolios):
        if not str(file).endswith(".json"):
            continue
        with importlib.resources.open_text(physrisk.data.static.example_portfolios, file) as f:
            portfolio = Assets(**json.load(f))
            portfolios.append(portfolio)
    return portfolios
