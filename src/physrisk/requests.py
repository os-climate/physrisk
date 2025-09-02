import importlib
import json
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union, cast

import numpy as np

import physrisk.data.image_creator
import physrisk.data.static.example_portfolios
import physrisk.kernel.hazard_model
from physrisk.api.v1.common import Distribution, ExceedanceCurve, VulnerabilityDistrib
from physrisk.api.v1.exposure_req_resp import (
    AssetExposure,
    AssetExposureRequest,
    AssetExposureResponse,
    Exposure,
)
from physrisk.api.v1.hazard_image import (
    HazardImageInfoRequest,
    HazardImageInfoResponse,
    HazardImageRequest,
)
from physrisk.data.hazard_data_provider import HazardDataHint
from physrisk.data.inventory import expand
from physrisk.data.inventory_reader import InventoryReader
from physrisk.data.static.scenarios import scenario_description
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.kernel.exposure import JupterExposureMeasure, calculate_exposures
from physrisk.kernel.hazards import Hazard
from physrisk.kernel.impact import AssetImpactResult, ImpactKey  # , ImpactKey
from physrisk.kernel.impact_distrib import EmptyImpactDistrib, PlaceholderImpactDistrib
from physrisk.kernel.risk import (
    AssetLevelRiskModel,
    Measure,
    MeasureKey,
    RiskMeasureCalculator,
    RiskMeasuresFactory,
)
from physrisk.kernel.vulnerability_model import (
    DictBasedVulnerabilityModels,
    VulnerabilityModels,
    VulnerabilityModelsFactory,
)
from physrisk.utils import encoder
from physrisk.utils.encoder import PhysriskDefaultEncoder

from .api.v1.hazard_data import (
    HazardAvailabilityRequest,
    HazardAvailabilityResponse,
    HazardDataRequest,
    HazardDataResponse,
    HazardDataResponseItem,
    HazardDescriptionRequest,
    HazardDescriptionResponse,
    HazardResource,
    IntensityCurve,
    Scenario,
    StaticInformationResponse,
)
from .api.v1.impact_req_resp import (
    CalculationDetails,
    AssetImpactRequest,
    AssetImpactResponse,
    AssetLevelImpact,
    Assets,
    AssetSingleImpact,
)
from .api.v1.impact_req_resp import ImpactKey as APIImpactKey
from .api.v1.impact_req_resp import (
    RiskMeasureKey,
    RiskMeasures,
    RiskMeasuresForAssets,
    ScoreBasedRiskMeasureDefinition,
    ScoreBasedRiskMeasureSetDefinition,
)
from .data.inventory import EmbeddedInventory, Inventory
from .kernel.assets import Asset
from .kernel import calculation as calc
from .kernel.hazard_model import (
    HazardDataRequest as hmHazardDataRequest,
    HazardImageCreator,
)
from .kernel.hazard_model import HazardEventDataResponse as hmHazardEventDataResponse
from .kernel.hazard_model import (
    HazardModel,
    HazardModelFactory,
    HazardParameterDataResponse,
)

Colormaps = Dict[str, Any]


class Requester:
    def __init__(
        self,
        hazard_model_factory: HazardModelFactory,
        vulnerability_models_factory: VulnerabilityModelsFactory,
        inventory: Inventory,
        inventory_reader: InventoryReader,
        reader: ZarrReader,
        colormaps: Colormaps,
        measures_factory: RiskMeasuresFactory,
        json_encoder_cls: Type[json.JSONEncoder] = PhysriskDefaultEncoder,
        sig_figures: int = -1,
    ):
        self.colormaps = colormaps
        self.json_encoder_cls = json_encoder_cls
        self.hazard_model_factory = hazard_model_factory
        self.measures_factory = measures_factory
        self.sig_figures = sig_figures
        self.vulnerability_models_factory = vulnerability_models_factory
        self.inventory = inventory
        self.inventory_reader = inventory_reader
        self.zarr_reader = reader

    def get(self, *, request_id, request_dict):
        if request_id == "get_hazard_data":
            request = HazardDataRequest(**request_dict)
            return self.dumps(
                self.get_hazard_data(request).model_dump()  # , allow_nan=False)
            )
        elif request_id == "get_hazard_data_availability":
            request = HazardAvailabilityRequest(**request_dict)
            return self.dumps(self.get_hazard_data_availability(request).model_dump())
        elif request_id == "get_hazard_data_description":
            request = HazardDescriptionRequest(**request_dict)
            return self.dumps(self.get_hazard_data_description(request).model_dump())
        elif request_id == "get_static_information":
            return self.dumps(self.get_static_information().model_dump())
        elif request_id == "get_asset_exposure":
            request = AssetExposureRequest(**request_dict)
            return self.dumps(
                self.get_asset_exposures(request).model_dump(exclude_none=True)
            )
        elif request_id == "get_asset_impact":
            request = AssetImpactRequest(**request_dict)
            return self.dumps(
                self.get_asset_impacts(request).model_dump(exclude_none=True)
            )
        elif request_id == "get_example_portfolios":
            return self.dumps(_get_example_portfolios())
        elif request_id == "get_image_info":
            return self.dumps(self.get_image_info().model_dump())
        else:
            raise ValueError(f"request type '{request_id}' not found")

    def get_hazard_data(self, request: HazardDataRequest):
        hazard_model = self.hazard_model_factory.hazard_model(
            interpolation=request.interpolation,
            provider_max_requests=request.provider_max_requests,
        )
        return _get_hazard_data(
            request, hazard_model=hazard_model, sig_figures=self.round_sig_figures
        )

    def get_hazard_data_availability(self, request: HazardAvailabilityRequest):
        return _get_hazard_data_availability(request, self.inventory, self.colormaps)

    def get_hazard_data_description(self, request: HazardDescriptionRequest):
        return _get_hazard_data_description(request, self.inventory_reader)

    def get_static_information(self):
        return StaticInformationResponse(
            scenario_descriptions=scenario_description.description()
        )

    def get_asset_exposures(self, request: AssetExposureRequest):
        hazard_model = self.hazard_model_factory.hazard_model(
            interpolation=request.calc_settings.hazard_interp,
            provider_max_requests=request.provider_max_requests,
        )
        return _get_asset_exposures(request, hazard_model)

    def get_asset_impacts(self, request: AssetImpactRequest) -> AssetImpactResponse:
        hazard_model = self.hazard_model_factory.hazard_model(
            interpolation=request.calc_settings.hazard_interp,
            provider_max_requests=request.provider_max_requests,
        )
        vulnerability_models = self.vulnerability_models_factory.vulnerability_models()
        measure_calculators = self.measures_factory.calculators(request.use_case_id)
        return _get_asset_impacts(
            request,
            hazard_model,
            vulnerability_models,
            measure_calculators,
            sig_figures=self.round_sig_figures,
        )

    def get_image(self, request_or_dict: Union[HazardImageRequest, Dict]):
        if isinstance(request_or_dict, Dict):
            request = HazardImageRequest(**request_or_dict)
        else:
            request = request_or_dict
        inventory = self.inventory
        if not _read_permitted(
            request.group_ids, inventory.resources[request.resource]
        ):
            raise PermissionError()
        model = inventory.resources[request.resource]
        assert model.map is not None
        colormap = (
            request.colormap
            if request.colormap is not None
            else (model.map.colormap.name if model.map.colormap is not None else "None")
        )
        creator: HazardImageCreator = self.hazard_model_factory.image_creator()
        return creator.create_image(
            request.resource,
            request.scenario_id,
            request.year,
            colormap=colormap,
            tile=None
            if request.tile is None
            else physrisk.kernel.hazard_model.Tile(
                request.tile.x, request.tile.y, request.tile.z
            ),
            min_value=request.min_value,
            max_value=request.max_value,
            index_value=request.index_value,
        )

    def get_image_info(self, request: HazardImageInfoRequest):
        creator: HazardImageCreator = self.hazard_model_factory.image_creator()
        all_index_values, available_index_values, index_display_name, index_units = (
            creator.get_info(request.resource, request.scenario_id, request.year)
        )
        return HazardImageInfoResponse(
            all_index_values=all_index_values,
            available_index_values=available_index_values,
            index_display_name=index_display_name,
            index_units=index_units,
        )

    def dumps(self, dict):
        return json.dumps(dict, cls=self.json_encoder_cls)

    def round_sig_figures(self, x: Union[np.ndarray, float]):
        if self.sig_figures == -1:
            return x
        return encoder.sig_figures(x, self.sig_figures)


def _create_inventory(
    reader: Optional[InventoryReader] = None, sources: Optional[List[str]] = None
):
    resources: List[HazardResource] = []
    colormaps: Dict[str, Dict[str, Any]] = {}
    request_sources = ["embedded"] if sources is None else [s.lower() for s in sources]
    for source in request_sources:
        if source == "embedded":
            inventory = EmbeddedInventory()
            for res in inventory.resources.values():
                resources.append(res)
            colormaps.update(inventory.colormaps())
        elif source == "hazard" or source == "hazard_test":
            if reader is not None:
                for resource in reader.read(source):
                    resources.extend(expand([resource]))
    return Inventory(resources)


def create_source_paths(inventory: Inventory):
    return get_default_source_paths(inventory)


def _read_permitted(group_ids: List[str], resource: HazardResource):
    """_summary_

    Args:
        group_ids (List[str]): Groups to which requester belongs.
        resourceId (str): Resource identifier.

    Returns:
        bool: True is requester is permitted access to models comprising resource.
    """
    return ("osc" in group_ids) or resource.group_id == "public"


def _get_hazard_data_availability(
    request: HazardAvailabilityRequest, inventory: Inventory, colormaps: dict
):
    response = HazardAvailabilityResponse(
        models=list(inventory.resources.values()), colormaps=colormaps
    )  # type: ignore
    return response


def _get_hazard_data_description(
    request: HazardDescriptionRequest, reader: InventoryReader
):
    descriptions = reader.read_description_markdown(request.paths)
    return HazardDescriptionResponse(descriptions=descriptions)


def _get_hazard_data(
    request: HazardDataRequest,
    hazard_model: HazardModel,
    sig_figures: Callable[[Union[np.ndarray, float]], np.ndarray] = lambda x: x,
):
    # if any(
    #     not _read_permitted(request.group_ids, inventory.resources_by_type_id[(i.event_type, i.model)][0])
    #     for i in request.items
    # ):
    #     raise PermissionError()

    # get hazard event types:
    event_types = Hazard.__subclasses__()
    event_dict = dict((et.__name__, et) for et in event_types)
    event_dict.update(
        (est.__name__, est) for et in event_types for est in et.__subclasses__()
    )

    # flatten list to let event processor decide how to group
    item_requests = []
    all_requests = []
    for item in request.items:
        hazard_type = (
            item.hazard_type
            if item.hazard_type is not None
            else item.event_type
            if item.event_type is not None
            else ""
        )
        event_type = event_dict[hazard_type]
        hint = None if item.path is None else HazardDataHint(path=item.path)

        data_requests = [
            hmHazardDataRequest(
                event_type,
                lon,
                lat,
                indicator_id=item.indicator_id,
                scenario=item.scenario,
                year=item.year,
                hint=hint,
            )
            for (lon, lat) in zip(item.longitudes, item.latitudes)
        ]

        all_requests.extend(data_requests)
        item_requests.append(data_requests)

    response_dict = hazard_model.get_hazard_data(all_requests)
    # responses comes back as a dictionary because requests may be executed in different order to list
    # to optimise performance.

    response = HazardDataResponse(items=[])

    for i, item in enumerate(request.items):
        requests = item_requests[i]
        resps = (response_dict[req] for req in requests)
        intensity_curves = [
            (
                IntensityCurve(
                    intensities=list(sig_figures(resp.intensities)),
                    index_values=list(sig_figures(resp.return_periods)),
                    index_name="return period",
                    return_periods=[],
                )
                if isinstance(resp, hmHazardEventDataResponse)
                else (
                    IntensityCurve(
                        intensities=list(sig_figures(resp.parameters)),
                        index_values=list(sig_figures(resp.param_defns)),
                        index_name="threshold",
                        return_periods=[],
                    )
                    if isinstance(resp, HazardParameterDataResponse)
                    else IntensityCurve(
                        intensities=[],
                        index_values=[],
                        index_name="",
                        return_periods=[],
                    )
                )
            )
            for resp in resps
        ]
        response.items.append(
            HazardDataResponseItem(
                intensity_curve_set=intensity_curves,
                request_item_id=item.request_item_id,
                event_type=item.event_type,
                model=item.indicator_id,
                scenario=item.scenario,
                year=item.year,
            )
        )

    return response


def create_assets(api_assets: Assets, assets: Optional[List[Asset]] = None):
    """Create list of Asset objects from the Assets API object:"""
    if assets is not None:
        if len(api_assets.items) != 0:
            raise ValueError(
                "Cannot provide asset items in the request while specifying an explicit asset list"
            )
        return assets
    else:
        module = import_module("physrisk.kernel.assets")
        asset_objs = []
        for item in api_assets.items:
            if hasattr(module, item.asset_class):
                init = getattr(module, item.asset_class)
                kwargs = {}
                kwargs.update(item.__dict__)
                if item.model_extra is not None:
                    kwargs.update(item.model_extra)
                del kwargs["asset_class"], kwargs["latitude"], kwargs["longitude"]
                asset_obj = cast(
                    Asset,
                    init(item.latitude, item.longitude, **kwargs),
                )
                asset_objs.append(asset_obj)
            else:
                raise ValueError(f"asset type '{item.asset_class}' not found")
        return asset_objs


def _get_asset_exposures(
    request: AssetExposureRequest,
    hazard_model: HazardModel,
    assets: Optional[List[Asset]] = None,
):
    _assets = create_assets(request.assets, assets)
    measure = JupterExposureMeasure()
    results = calculate_exposures(
        _assets, hazard_model, measure, scenario="ssp585", year=2030
    )
    return AssetExposureResponse(
        items=[
            AssetExposure(
                asset_id="",
                exposures=dict(
                    (t.__name__, Exposure(category=c.name, value=v, path=p))
                    for (t, (c, v, p)) in r.hazard_categories.items()
                ),
            )
            for (a, r) in results.items()
        ]
    )


def _get_asset_impacts(
    request: AssetImpactRequest,
    hazard_model: HazardModel,
    vulnerability_models: Optional[VulnerabilityModels] = None,
    measure_calculators: Optional[Dict[Type[Asset], RiskMeasureCalculator]] = None,
    assets: Optional[List[Asset]] = None,
    sig_figures: Callable[
        [Union[np.ndarray, float]], Union[np.ndarray, float]
    ] = lambda x: x,
):
    vulnerability_models = (
        DictBasedVulnerabilityModels(calc.get_default_vulnerability_models())
        if vulnerability_models is None
        else vulnerability_models
    )
    # we keep API definition of asset separate from internal Asset class; convert by reflection
    # based on asset_class:
    _assets = create_assets(request.assets, assets)
    measure_calculators = (
        calc.get_default_risk_measure_calculators()
        if measure_calculators is None
        else measure_calculators
    )
    risk_model = AssetLevelRiskModel(
        hazard_model, vulnerability_models, measure_calculators
    )

    scenarios = (
        [request.scenario]
        if request.scenarios is None or len(request.scenarios) == 0
        else request.scenarios
    )
    years = (
        [request.year]
        if request.years is None or len(request.years) == 0
        else request.years
    )
    risk_measures = None
    if request.include_measures:
        impacts, measures = risk_model.calculate_risk_measures(
            _assets, scenarios, years
        )
        measure_ids_for_asset, definitions = risk_model.populate_measure_definitions(
            _assets
        )
        # create object for API:
        risk_measures = _create_risk_measures(
            measures,
            measure_ids_for_asset,
            definitions,
            _assets,
            scenarios,
            years,
            sig_figures,
        )
    elif request.include_asset_level:
        impacts = risk_model.calculate_impacts(_assets, scenarios, years)

    if request.include_asset_level:
        asset_impacts = compile_asset_impacts(
            impacts, _assets, request.include_calc_details, sig_figures
        )
    else:
        asset_impacts = None
    return AssetImpactResponse(asset_impacts=asset_impacts, risk_measures=risk_measures)


def compile_asset_impacts(
    impacts: Dict[ImpactKey, List[AssetImpactResult]],
    assets: List[Asset],
    include_calc_details: bool,
    sig_figures: Callable[
        [Union[np.ndarray, float]], Union[np.ndarray, float]
    ] = lambda x: x,
):
    """Convert (internal) list of AssetImpactResult objects to a list of AssetLevelImpact
    objects ready for serialization.

    Args:
        impacts (Dict[ImpactKey, List[AssetImpactResult]]): Impact results.
        assets (List[Asset]): Assets: the list will be returned using this order.
        include_calc_details (bool): Include calculation details.
        sig_figures: Function to round results.

    Returns:
        List[AssetLevelImpact]: AssetImpactResult objects for serialization.
    """
    ordered_impacts: Dict[Asset, List[AssetSingleImpact]] = {}
    for asset in assets:
        ordered_impacts[asset] = []
    for k, value in impacts.items():
        for v in value:
            if isinstance(v.impact, EmptyImpactDistrib):
                continue
            calc_details = None
            if include_calc_details:
                if isinstance(v.impact, PlaceholderImpactDistrib):
                    # if the impact is a placeholder, measures are calculated based
                    # only on hazard indicators. Still populate the hazard_path field in this case.
                    calc_details = CalculationDetails(
                        hazard_exceedance=None,
                        hazard_distribution=None,
                        vulnerability_distribution=None,
                        hazard_path=[]
                        if v.hazard_data is None
                        else [h.path for h in v.hazard_data],
                    )

                elif v.event is not None and v.vulnerability is not None:
                    hazard_exceedance = v.event.to_exceedance_curve()
                    vulnerability_distribution = VulnerabilityDistrib(
                        intensity_bin_edges=sig_figures(v.vulnerability.intensity_bins),
                        impact_bin_edges=sig_figures(v.vulnerability.impact_bins),
                        prob_matrix=sig_figures(v.vulnerability.prob_matrix),
                    )
                    calc_details = CalculationDetails(
                        hazard_exceedance=ExceedanceCurve(
                            values=sig_figures(hazard_exceedance.values),
                            exceed_probabilities=sig_figures(hazard_exceedance.probs),
                        ),
                        hazard_distribution=Distribution(
                            bin_edges=sig_figures(v.event.intensity_bin_edges),
                            probabilities=sig_figures(v.event.prob),
                        ),
                        vulnerability_distribution=vulnerability_distribution,
                        hazard_path=v.impact.path,
                    )

            key = APIImpactKey(
                hazard_type=k.hazard_type.__name__,
                scenario_id=k.scenario,
                year=str(k.key_year),
            )
            if isinstance(v.impact, PlaceholderImpactDistrib):
                # only calc_details relevant here:
                hazard_impacts = AssetSingleImpact(
                    key=key,
                    impact_type="n/a",
                    impact_distribution=None,
                    impact_exceedance=None,
                    impact_mean=float("nan"),
                    impact_std_deviation=float("nan"),
                    calc_details=calc_details,
                )
            else:
                impact_exceedance = v.impact.to_exceedance_curve()
                hazard_impacts = AssetSingleImpact(
                    key=key,
                    impact_type=v.impact.impact_type.name,
                    impact_exceedance=ExceedanceCurve(
                        values=sig_figures(impact_exceedance.values),
                        exceed_probabilities=sig_figures(impact_exceedance.probs),
                    ),
                    impact_distribution=Distribution(
                        bin_edges=sig_figures(v.impact.impact_bins),
                        probabilities=sig_figures(v.impact.prob),
                    ),
                    impact_mean=sig_figures(v.impact.mean_impact()),
                    impact_std_deviation=sig_figures(v.impact.standard_deviation()),
                    impact_semi_std_deviation=sig_figures(
                        v.impact.semi_standard_deviation()
                    ),
                    calc_details=None if v.event is None else calc_details,
                )
            # note that this does rely on ordering of dictionary (post 3.6)
            ordered_impacts[k.asset].append(hazard_impacts)

    return [
        AssetLevelImpact(asset_id=k.id if k.id is not None else "", impacts=v)
        for k, v in ordered_impacts.items()
    ]


def _create_risk_measures(
    measures: Dict[MeasureKey, Measure],
    measure_ids_for_asset: Dict[Type[Hazard], List[str]],
    definitions: Dict[ScoreBasedRiskMeasureDefinition, str],
    assets: List[Asset],
    scenarios: Sequence[str],
    years: Sequence[int],
    sig_figures: Callable[
        [Union[np.ndarray, float]], Union[np.ndarray, float]
    ] = lambda x: x,
) -> RiskMeasures:
    """Prepare RiskMeasures object for (JSON) output from measure results.

    Args:
        measures (Dict[MeasureKey, Measure]): The score-based risk measures.
        measure_ids_for_asset (Dict[Type[Hazard], List[str]]): IDs of the score-based risk measures
            for each asset.
        definitions (Dict[ScoreBasedRiskMeasureDefinition, str]): Map of the score-based risk measures
            definitions to ID.
        assets (List[Asset]): Assets.
        scenarios (Sequence[str]): Scenario IDs.
        years (Sequence[int]): Years.

    Returns:
        RiskMeasures: Output for writing to JSON.
    """

    nan_value = -9999.0  # Nan not part of JSON spec
    hazard_types = set(k.hazard_type for k in measures.keys())
    # hazard_types = all_hazards()
    measure_set_id = "measure_set_0"
    measures_for_assets: List[RiskMeasuresForAssets] = []
    for hazard_type in hazard_types:
        for scenario_id in scenarios:
            for year in [None] if scenario_id == "historical" else years:
                # we calculate and tag results for each scenario, year and hazard
                score_key = RiskMeasureKey(
                    hazard_type=hazard_type.__name__,
                    scenario_id=scenario_id,
                    year=str(year),
                    measure_id=measure_set_id,
                )
                scores = [-1] * len(assets)
                # measures_0 = [float("nan")] * len(assets)
                measures_0 = [nan_value] * len(assets)
                for i, asset in enumerate(assets):
                    # look up result using the MeasureKey:
                    measure_key = MeasureKey(
                        asset=asset,
                        prosp_scen=scenario_id,
                        year=year,
                        hazard_type=hazard_type,
                    )
                    measure = measures.get(measure_key, None)
                    if measure is not None:
                        scores[i] = measure.score
                        measures_0[i] = measure.measure_0
                measures_for_assets.append(
                    RiskMeasuresForAssets(
                        key=score_key,
                        scores=scores,
                        measures_0=sig_figures(measures_0),
                        measures_1=None,
                    )
                )
    score_based_measure_set_defn = ScoreBasedRiskMeasureSetDefinition(
        measure_set_id=measure_set_id,
        asset_measure_ids_for_hazard={
            k.__name__: v for k, v in measure_ids_for_asset.items()
        },
        score_definitions={v: k for (k, v) in definitions.items()},
    )
    return RiskMeasures(
        measures_for_assets=measures_for_assets,
        score_based_measure_set_defn=score_based_measure_set_defn,
        measures_definitions=None,
        scenarios=[Scenario(id=scenario, years=list(years)) for scenario in scenarios],
        asset_ids=[
            f"asset_{i}" if a.id is None else a.id for i, a in enumerate(assets)
        ],
    )


def _get_example_portfolios() -> List[Assets]:
    portfolios = []
    for file in importlib.resources.contents(physrisk.data.static.example_portfolios):
        if not str(file).endswith(".json"):
            continue
        with importlib.resources.open_text(
            physrisk.data.static.example_portfolios, file
        ) as f:
            portfolio = Assets(**json.load(f))
            portfolios.append(portfolio)
    return portfolios
