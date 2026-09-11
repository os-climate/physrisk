import pytest

from physrisk.api.v1.hazard_data import HazardResource, MapInfo, Scenario


def hazard_resource(*, map: MapInfo | None = None) -> HazardResource:
    return HazardResource(
        hazard_type="TestHazard",
        indicator_id="indicator",
        indicator_model_gcm="test",
        path="resource/{id}/{scenario}/{year}",
        display_name="Test resource",
        description="Test resource",
        scenarios=[Scenario(id="ssp245", years=[2050])],
        store_netcdf_coords=True,
        units="test",
        map=map,
    )


def test_hazard_resource_builds_concrete_data_path():
    resource = hazard_resource()

    assert (
        resource.path_for_scenario_year("ssp245", 2050)
        == "resource/indicator/ssp245/2050/indicator"
    )


@pytest.mark.parametrize(
    ("map_info", "zoom", "expected"),
    [
        (
            MapInfo(path="map", source="map_array", colormap=None),
            None,
            "resource/indicator/ssp245/map",
        ),
        (
            MapInfo(
                path="maps/{scenario}/{year}", source="map_array_pyramid", colormap=None
            ),
            4,
            "maps/ssp245/2050/4",
        ),
    ],
)
def test_hazard_resource_builds_concrete_map_path(map_info, zoom, expected):
    resource = hazard_resource(map=map_info)

    assert resource.map_path_for_scenario_year("ssp245", 2050, zoom) == expected


def test_hazard_resource_requires_map_configuration_for_map_path():
    with pytest.raises(ValueError, match="has no map configuration"):
        hazard_resource().map_path_for_scenario_year("ssp245", 2050)
