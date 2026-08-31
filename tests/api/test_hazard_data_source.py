import numpy as np

from physrisk.api.v1.hazard_data import (
    HazardDataRequest as APIHazardDataRequest,
    HazardDataRequestItem,
)
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import (
    HazardDataSource,
    HazardEventDataResponse,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import Fire
from physrisk.kernel.impact import AssetImpactResult, ImpactKey
from physrisk.kernel.impact_distrib import PlaceholderImpactDistrib
from physrisk.requests import _compile_asset_impacts, _get_hazard_data


SOURCE = HazardDataSource(
    resource_id="resource_{scenario}_{year}",
    path="resource_historical_2005",
    scenario="historical",
    year=2005,
)


class SourceHazardModel:
    def get_hazard_data(self, requests):
        return {
            request: HazardEventDataResponse(
                np.array([10.0]),
                np.array([1.0]),
                units="m",
                path=SOURCE.resource_id,
                source=SOURCE,
            )
            for request in requests
        }


def test_hazard_data_api_serializes_actual_source_per_curve():
    request = APIHazardDataRequest(
        items=[
            HazardDataRequestItem(
                longitudes=[1.0],
                latitudes=[2.0],
                request_item_id="item",
                hazard_type="RiverineInundation",
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2080,
            )
        ]
    )

    response = _get_hazard_data(request, SourceHazardModel())  # type: ignore[arg-type]

    item = response.items[0]
    assert item.scenario == "ssp585"
    assert item.year == 2080
    assert item.intensity_curve_set[0].source is not None
    assert item.intensity_curve_set[0].source.model_dump() == {
        "resource_id": SOURCE.resource_id,
        "path": SOURCE.path,
        "scenario": SOURCE.scenario,
        "year": SOURCE.year,
    }


def test_asset_impact_calculation_details_serialize_hazard_sources():
    asset = Asset(latitude=2.0, longitude=1.0, id="asset")
    hazard_response = HazardParameterDataResponse(
        np.array([0.1]),
        units="index",
        path=SOURCE.resource_id,
        source=SOURCE,
    )
    impact_result = AssetImpactResult(
        PlaceholderImpactDistrib(),
        hazard_data=[hazard_response],
    )
    impacts = {
        ImpactKey(asset, Fire, "ssp585", 2080): [impact_result],
    }

    response = _compile_asset_impacts(impacts, [asset], include_calc_details=True)

    details = response[0].impacts[0].calc_details
    assert details is not None
    assert details.hazard_sources[0] is not None
    assert details.hazard_sources[0].path == SOURCE.path
    assert details.hazard_sources[0].scenario == "historical"
    assert details.hazard_sources[0].year == 2005


def test_response_source_defaults_to_none():
    response = HazardEventDataResponse(np.array([10.0]), np.array([1.0]))

    assert response.source is None
