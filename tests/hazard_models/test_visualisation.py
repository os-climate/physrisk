import io
from typing import Tuple

import numpy as np
import pytest
import PIL.Image as Image

from physrisk.api.v1.hazard_image import HazardImageRequest
from physrisk.container import Container
from physrisk.kernel.hazard_model import Tile
from physrisk.hazard_models.jba_image_creator import JBAImageCreator


# https://jbavision.jbarisk.com/cog/tiles/9/265/176.png?LAYERS=853_WR30_202512_30m_4326:WR30_202512_FLRF_U_RP1500_RE_30m_4326

# Use the 30m Global Inland Flood Map by setting the country_code parameter to WR30, or
# Continue using individual country map layers by setting country_code to the relevant country.
# https://jbavision.jbarisk.com/cog/WMTS/WR30_202512_30m_4326


@pytest.mark.skip("Requires credentials")
def test_image_creator_with_jba(load_credentials):
    container = Container()
    container.override_providers(inventory_reader=None)
    container.override_providers(zarr_reader=None)
    requester = container.requester()
    requester.get_image(
        HazardImageRequest(
            resource="jba_riverine",
            scenario_id="ssp585",
            year=2050,
            index_value=100,
            min_value=0,
            max_value=6,
            tile=Tile(2, 2, 2),
        )
    )


@pytest.mark.skip("Requires credentials")
def test_jba_image_creator(load_credentials):
    creator = JBAImageCreator()
    image = creator.create_image("jba_riverine", "historical", -1, tile=Tile(0, 0, 0))
    assert image is not None
    # for country_code in ["WR30", "FR5C", "GB", "BE", "US"]:
    # Path("tile.png").write_bytes(resp.content)


@pytest.mark.skip("Requires credentials")
def test_get_legend(load_credentials):
    creator = JBAImageCreator()
    image = creator.create_image("jba_riverine", "historical", -1, tile=Tile(0, 0, 0))
    assert image is not None


@pytest.mark.skip("Requires credentials")
def test_get_jba_legend_png(load_credentials):
    creator = JBAImageCreator()
    legend_bytes = creator.get_legend("jba_riverine", return_period=1500)
    assert legend_bytes is not None
    image = Image.open(io.BytesIO(legend_bytes))
    assert image.format == "PNG"
    assert image.width > 0 and image.height > 0


def _locate_colorbar(image: Image.Image) -> Tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) bounding the colour swatches in a legend.

    Works for discrete stepped legends (e.g. JBA flood depth) where each swatch
    is a solid-colour rectangle separated by white/neutral gaps.  Detection uses
    chroma (max channel − min channel per pixel) rather than gradient scoring,
    so it is robust to both continuous and discrete colour bars.
    """
    arr = np.array(image.convert("RGB"))  # (H, W, 3) uint8

    # Mean colour per column; columns in the swatch band have noticeable chroma.
    col_mean = arr.mean(axis=0)  # (W, 3)
    col_chroma = col_mean.max(axis=1) - col_mean.min(axis=1)  # (W,)
    swatch_cols = np.where(col_chroma > 8)[0]

    # Mean colour per row within the swatch columns; swatch rows have chroma.
    x_min, x_max = int(swatch_cols[0]), int(swatch_cols[-1])
    band = arr[:, x_min : x_max + 1, :].astype(float)
    row_mean = band.mean(axis=1)  # (H, 3)
    row_chroma = row_mean.max(axis=1) - row_mean.min(axis=1)  # (H,)
    swatch_rows = np.where(row_chroma > 8)[0]

    return x_min, int(swatch_rows[0]), x_max, int(swatch_rows[-1])


@pytest.mark.skip("Requires credentials")
def test_locate_colorbar_in_jba_legend(load_credentials):
    creator = JBAImageCreator()
    legend_bytes = creator.get_legend("jba_riverine", return_period=1500)
    image = Image.open(io.BytesIO(legend_bytes))

    x_min, y_min, x_max, y_max = _locate_colorbar(image)
    print(f"\nColour bar bounding box: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]")

    assert x_max > x_min
    assert y_max > y_min
    # Sanity: bar must be at least 10 px in its long dimension
    assert max(x_max - x_min, y_max - y_min) >= 10
