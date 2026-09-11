import numpy as np
import pytest

from physrisk.api.v1.hazard_data import HazardResource, Scenario
from physrisk.data.hazard_data_provider import DataSourcingError
from physrisk.data.pregenerated_hazard_model import HierarchicalZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazard_model import HazardDataFailedResponse, HazardDataRequest
from physrisk.kernel.hazards import RiverineInundation

from tests.data.test_hazard_model_store import ZarrStoreMocker


class OrderedHazardResourceProvider:
    def __init__(self, resources):
        self.resources = resources

    def hazard_indicators(self):
        return {RiverineInundation: ["flood_depth"]}

    def get_resources(self, hazard_type, indicator_id, hint=None):
        if hint is not None and hint.path is not None:
            return [
                resource for resource in self.resources if resource.path == hint.path
            ]
        return self.resources


def hierarchical_resource(path: str, scenarios: list[Scenario]):
    return HazardResource(
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        indicator_model_gcm="test",
        path=path,
        display_name="Test flood depth",
        description="Test resource",
        scenarios=scenarios,
        units="m",
    )


def test_prefers_first_resource_and_traces_concrete_source(monkeypatch):
    store_mocker = ZarrStoreMocker()
    longitude, latitude = 1.1, 47.0
    preferred_path = "preferred_historical_2005"
    exact_path = "exact_ssp585_2080"
    for path, values in [(preferred_path, [1.0, 2.0]), (exact_path, [8.0, 9.0])]:
        store_mocker.add_curves_global(
            path,
            [longitude],
            [latitude],
            [10.0, 100.0],
            values,
            width=360,
            height=180,
            units="m",
        )
    resources = [
        hierarchical_resource(
            "preferred_{scenario}_{year}",
            [Scenario(id="historical", years=[2005])],
        ),
        hierarchical_resource(
            "exact_{scenario}_{year}",
            [Scenario(id="ssp585", years=[2080])],
        ),
    ]
    reader = ZarrReader(store=store_mocker.store)
    original_get_curves = reader.get_curves
    read_paths = []

    def tracking_get_curves(path, *args, **kwargs):
        read_paths.append(path)
        return original_get_curves(path, *args, **kwargs)

    monkeypatch.setattr(reader, "get_curves", tracking_get_curves)
    model = HierarchicalZarrHazardModel(
        resource_provider=OrderedHazardResourceProvider(resources), reader=reader
    )
    requests = [
        HazardDataRequest(
            RiverineInundation,
            longitude,
            latitude,
            indicator_id="flood_depth",
            scenario=scenario,
            year=year,
        )
        for scenario, year in [("ssp585", 2080), ("ssp245", 2050)]
    ]

    responses = model.get_hazard_data(requests)

    for request in requests:
        response = responses[request]
        np.testing.assert_allclose(response.intensities, [1.0, 2.0])
        assert response.source is not None
        assert response.source.resource_id == "preferred_{scenario}_{year}"
        assert response.source.path == preferred_path
        assert response.source.scenario == "historical"
        assert response.source.year == 2005
    assert read_paths == [preferred_path]


def test_cascades_geographically_and_traces_each_source():
    store_mocker = ZarrStoreMocker()
    index_values = [10.0, 100.0]
    store_mocker._add_curves(
        "first_historical_2005",
        [0.5],
        [0.5],
        "epsg:4326",
        (2, 1, 1),
        [1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0],
        index_values,
        [1.0, 2.0],
        units="m",
    )
    store_mocker._add_curves(
        "second_ssp585_2080",
        [1.5],
        [0.5],
        "epsg:4326",
        (2, 1, 1),
        [1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0],
        index_values,
        [8.0, 9.0],
        units="m",
    )
    resources = [
        hierarchical_resource(
            "first_{scenario}_{year}",
            [Scenario(id="historical", years=[2005])],
        ),
        hierarchical_resource(
            "second_{scenario}_{year}",
            [Scenario(id="ssp585", years=[2080])],
        ),
    ]
    model = HierarchicalZarrHazardModel(
        resource_provider=OrderedHazardResourceProvider(resources),
        store=store_mocker.store,
    )
    requests = [
        HazardDataRequest(
            RiverineInundation,
            longitude,
            0.5,
            indicator_id="flood_depth",
            scenario="ssp585",
            year=2080,
        )
        for longitude in [0.5, 1.5]
    ]

    responses = model.get_hazard_data(requests)

    np.testing.assert_allclose(responses[requests[0]].intensities, [1.0, 2.0])
    np.testing.assert_allclose(responses[requests[1]].intensities, [8.0, 9.0])
    assert responses[requests[0]].source.path == "first_historical_2005"
    assert responses[requests[1]].source.path == "second_ssp585_2080"


def test_fulfils_complete_proxy_scenario_year_matrix():
    store_mocker = ZarrStoreMocker()
    longitude, latitude = 1.1, 47.0
    source_path = "historical_2005"
    store_mocker.add_curves_global(
        source_path,
        [longitude],
        [latitude],
        [10.0],
        [1.0],
        width=360,
        height=180,
        units="m",
    )
    resources = [
        hierarchical_resource(
            "{scenario}_{year}",
            [Scenario(id="historical", years=[1980, 2005])],
        )
    ]
    model = HierarchicalZarrHazardModel(
        resource_provider=OrderedHazardResourceProvider(resources),
        store=store_mocker.store,
    )
    requests = [
        HazardDataRequest(
            RiverineInundation,
            longitude,
            latitude,
            indicator_id="flood_depth",
            scenario=scenario,
            year=year,
        )
        for scenario, years in [
            ("historical", [-1]),
            ("ssp126", [2030, 2050, 2080]),
            ("ssp245", [2030, 2050, 2080]),
            ("ssp585", [2030, 2050, 2080]),
        ]
        for year in years
    ]

    responses = model.get_hazard_data(requests)

    assert len(responses) == 10
    for request in requests:
        response = responses[request]
        assert not isinstance(response, HazardDataFailedResponse)
        np.testing.assert_allclose(response.intensities, [1.0])
        assert response.source is not None
        assert response.source.resource_id == "{scenario}_{year}"
        assert response.source.path == source_path
        assert response.source.scenario == "historical"
        assert response.source.year == 2005


def test_skips_resources_without_available_scenario_years():
    store_mocker = ZarrStoreMocker()
    longitude, latitude = 1.1, 47.0
    store_mocker.add_curves_global(
        "usable_rcp4p5_2050",
        [longitude],
        [latitude],
        [10.0],
        [4.0],
        width=360,
        height=180,
        units="m",
    )
    resources = [
        hierarchical_resource("empty_{scenario}_{year}", []),
        hierarchical_resource(
            "usable_{scenario}_{year}",
            [Scenario(id="rcp4p5", years=[2050])],
        ),
    ]
    model = HierarchicalZarrHazardModel(
        resource_provider=OrderedHazardResourceProvider(resources),
        store=store_mocker.store,
    )
    request = HazardDataRequest(
        RiverineInundation,
        longitude,
        latitude,
        indicator_id="flood_depth",
        scenario="ssp245",
        year=2050,
    )

    response = model.get_hazard_data([request])[request]

    assert response.source is not None
    assert response.source.path == "usable_rcp4p5_2050"


def test_rejects_inconsistent_read_coverage(monkeypatch):
    resources = [
        hierarchical_resource(
            "resource_{scenario}_{year}",
            [
                Scenario(id="historical", years=[2005]),
                Scenario(id="ssp585", years=[2080]),
            ],
        )
    ]
    reader = ZarrReader(store=ZarrStoreMocker().store)

    def in_bounds(path, longitudes, latitudes, interpolation=None):
        return np.ones(len(longitudes), dtype=bool)

    async def read_single_item(
        reader, interpolation, item, latitudes, longitudes, buffer, path
    ):
        return (
            item,
            np.ones((len(longitudes), 1)),
            np.full(len(longitudes), item.scenario == "historical"),
            np.array([1]),
            "m",
            path,
        )

    monkeypatch.setattr(reader, "in_bounds", in_bounds)
    monkeypatch.setattr(
        "physrisk.data.hierarchical_hazard_data_provider.read_single_item",
        read_single_item,
    )
    model = HierarchicalZarrHazardModel(
        resource_provider=OrderedHazardResourceProvider(resources), reader=reader
    )
    requests = [
        HazardDataRequest(
            RiverineInundation,
            1.1,
            47.0,
            indicator_id="flood_depth",
            scenario=scenario,
            year=year,
        )
        for scenario, year in [("historical", -1), ("ssp585", 2080)]
    ]

    with pytest.raises(
        ValueError,
        match="inconsistent geographic coverage across scenario/year sources",
    ):
        model.get_hazard_data(requests)


def test_rejects_inconsistent_units_across_geographic_sources(monkeypatch):
    resources = [
        hierarchical_resource(
            f"{name}_{{scenario}}_{{year}}",
            [Scenario(id="ssp585", years=[2080])],
        )
        for name in ["first", "second"]
    ]
    reader = ZarrReader(store=ZarrStoreMocker().store)

    def in_bounds(path, longitudes, latitudes, interpolation=None):
        if path.startswith("first"):
            return np.array([True, False])
        return np.ones(len(longitudes), dtype=bool)

    async def read_single_item(
        reader, interpolation, item, latitudes, longitudes, buffer, path
    ):
        return (
            item,
            np.ones((len(longitudes), 1)),
            np.ones(len(longitudes), dtype=bool),
            np.array([1]),
            "m" if path.startswith("first") else "cm",
            path,
        )

    monkeypatch.setattr(reader, "in_bounds", in_bounds)
    monkeypatch.setattr(
        "physrisk.data.hierarchical_hazard_data_provider.read_single_item",
        read_single_item,
    )
    model = HierarchicalZarrHazardModel(
        resource_provider=OrderedHazardResourceProvider(resources), reader=reader
    )
    requests = [
        HazardDataRequest(
            RiverineInundation,
            longitude,
            47.0,
            indicator_id="flood_depth",
            scenario="ssp585",
            year=2080,
        )
        for longitude in [1.1, 2.2]
    ]

    with pytest.raises(ValueError, match="inconsistent units"):
        model.get_hazard_data(requests)


def test_raises_data_sourcing_error_for_missing_selected_array():
    model = HierarchicalZarrHazardModel(
        resource_provider=OrderedHazardResourceProvider(
            [
                hierarchical_resource(
                    "missing_{scenario}_{year}",
                    [Scenario(id="ssp585", years=[2080])],
                )
            ]
        ),
        store=ZarrStoreMocker().store,
    )
    request = HazardDataRequest(
        RiverineInundation,
        1.1,
        47.0,
        indicator_id="flood_depth",
        scenario="ssp585",
        year=2080,
    )

    with pytest.raises(DataSourcingError, match="Dataset not found"):
        model.get_hazard_data([request])
