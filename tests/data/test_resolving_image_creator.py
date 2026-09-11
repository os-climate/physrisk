from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from physrisk.api.v1.hazard_data import HazardResource, MapInfo, Scenario
from physrisk.api.v1.hazard_image import TileNotAvailableError
from physrisk.data.hazard_data_provider import ScenarioYear
from physrisk.data.image_creator import ResolvingImageCreator
from physrisk.data.inventory import Inventory
from physrisk.data.zarr_reader import ZarrReader
from physrisk.kernel.hazard_model import Tile


def resource() -> HazardResource:
    return HazardResource(
        path="fire/{scenario}/{year}",
        hazard_type="Fire",
        indicator_id="fire_probability",
        indicator_model_gcm="combined",
        display_name="Fire probability",
        description="Fire probability",
        scenarios=[
            Scenario(id="historical", years=[1990, 2020]),
            Scenario(id="ssp245", years=[2030, 2050]),
        ],
        units="probability",
        map=MapInfo(
            path="fire_map/{scenario}/{year}",
            colormap=None,
            source="map_array_pyramid",
        ),
    )


def test_create_image_uses_resolver_and_forwards_rendering_arguments():
    hazard_resource = resource()
    reader = Mock(spec=ZarrReader)
    resolver = Mock(return_value=ScenarioYear("ssp245", 2050))
    creator = ResolvingImageCreator(
        Inventory([hazard_resource]),
        reader,
        resolver,
    )
    data = np.array([[0.5]])
    image = Image.new("RGBA", (1, 1))
    tile = Tile(x=0, y=0, z=2)
    with (
        patch(
            "physrisk.data.image_creator._read_map_array",
            return_value=data,
        ) as read_map_array,
        patch(
            "physrisk.data.image_creator._render_map_array",
            return_value=image,
        ) as render_map_array,
    ):
        result = creator.create_image(
            hazard_resource.key(),
            "ssp245",
            2040,
            format="PNG",
            colormap="flare",
            tile=tile,
            min_value=0.1,
            max_value=0.9,
            index_value=100,
            scaling="log",
        )

    resolver.assert_called_once_with(ScenarioYear("ssp245", 2040), hazard_resource)
    read_map_array.assert_called_once_with(
        reader,
        "fire_map/ssp245/2050/3",
        tile,
        100,
    )
    render_map_array.assert_called_once_with(
        data,
        "flare",
        min_value=0.1,
        max_value=0.9,
        scaling="log",
    )
    assert result.startswith(b"\x89PNG")


def test_get_info_uses_resolver_and_resolved_path():
    hazard_resource = resource()
    reader = Mock(spec=ZarrReader)
    resolver = Mock(return_value=ScenarioYear("historical", 2020))
    creator = ResolvingImageCreator(
        Inventory([hazard_resource]),
        reader,
        resolver,
    )
    expected = ([1.0], [1.0], "threshold", "probability", 1)
    with patch(
        "physrisk.data.image_creator._get_image_info",
        return_value=expected,
    ) as get_image_info:
        result = creator.get_info(hazard_resource.key(), "historical", 1990)

    resolver.assert_called_once_with(ScenarioYear("historical", 1990), hazard_resource)
    get_image_info.assert_called_once_with(
        reader,
        hazard_resource,
        "fire_map/historical/2020/1",
        hazard_resource.key(),
    )
    assert result == expected


def test_create_image_translates_missing_tile_to_tile_not_available():
    hazard_resource = resource()
    reader = Mock(spec=ZarrReader)
    resolver = Mock(return_value=ScenarioYear("ssp245", 2050))
    creator = ResolvingImageCreator(
        Inventory([hazard_resource]),
        reader,
        resolver,
    )

    with (
        patch(
            "physrisk.data.image_creator._read_map_array",
            side_effect=KeyError("missing tile"),
        ),
        pytest.raises(TileNotAvailableError, match="missing tile"),
    ):
        creator.create_image(
            hazard_resource.key(),
            "ssp245",
            2040,
            tile=Tile(x=0, y=0, z=2),
        )
