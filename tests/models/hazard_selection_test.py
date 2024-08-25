from typing import Optional
import numpy as np

from physrisk.data.hazard_data_provider import HazardDataHint
from physrisk.data.inventory import EmbeddedInventory, Inventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import (
    CoreFloodModels,
    CoreInventorySourcePaths,
    ResourceSubset,
)
from physrisk.kernel.hazard_model import HazardDataRequest, HazardEventDataResponse
from physrisk.kernel.hazards import RiverineInundation

from tests.data.hazard_model_store_test import (
    TestData,
    ZarrStoreMocker,
    inundation_return_periods,
)


def test_tudelft_selection():
    inventory = EmbeddedInventory()
    source_paths = CoreInventorySourcePaths(
        inventory, flood_model=CoreFloodModels.TUDelft
    ).source_paths()
    assert (
        source_paths[RiverineInundation](
            indicator_id="flood_depth", scenario="rcp8p5", year=2050
        )
        == "inundation/river_tudelft/v1/flood_depth_rcp8p5_2050"
    )
    assert (
        source_paths[RiverineInundation](
            indicator_id="flood_depth", scenario="historical", year=-1
        )
        == "inundation/river_tudelft/v1/flood_depth_historical_1971"
    )


def test_customize_hazard_selection():
    inventory = EmbeddedInventory()

    source_path_selectors = CoreInventorySourcePaths(inventory)

    def select_riverine_inundation_tudelft(
        candidates: ResourceSubset,
        scenario: str,
        year: int,
        hint: Optional[HazardDataHint] = None,
    ):
        return candidates.with_model_id("tudelft").first()

    # we can add selectors programmatically
    # test_tudelft_selection shows an example of using the options in CoreInventorySourcePaths
    source_path_selectors.add_selector(
        RiverineInundation, "flood_depth", select_riverine_inundation_tudelft
    )

    custom_source_paths = source_path_selectors.source_paths()

    def sp_riverine(scenario, year):
        return custom_source_paths[RiverineInundation](
            indicator_id="flood_depth", scenario=scenario, year=year
        )

    mocker = ZarrStoreMocker()
    return_periods = inundation_return_periods()
    flood_histo_curve = np.array(
        [0.0596, 0.333, 0.505, 0.715, 0.864, 1.003, 1.149, 1.163, 1.163]
    )

    for path in [sp_riverine("historical", 1980)]:
        mocker.add_curves_global(
            path,
            TestData.longitudes,
            TestData.latitudes,
            return_periods,
            flood_histo_curve,
        )

    hazard_model = ZarrHazardModel(source_paths=custom_source_paths, store=mocker.store)

    req = HazardDataRequest(
        RiverineInundation,
        TestData.longitudes[0],
        TestData.latitudes[0],
        indicator_id="flood_depth",
        scenario="historical",
        year=-1,
    )
    responses = hazard_model.get_hazard_events([req])
    resp = responses[req]
    assert isinstance(resp, HazardEventDataResponse)
    np.testing.assert_almost_equal(resp.intensities, flood_histo_curve)
