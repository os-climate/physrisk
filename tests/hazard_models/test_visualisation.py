import io
from collections import Counter
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


def _extract_swatch_colours(
    image: Image.Image, x_min: int, x_max: int, min_run: int = 5
) -> list:
    """Return the fill colour of each legend swatch, reading down the centre column.

    Swatches are detected as runs of pixels that differ from the (near-)white
    background; each swatch's colour is the most common pixel value within its
    run, which is robust to the 1-2px anti-aliased bevel JBA renders at the top
    and bottom of every swatch. ``min_run`` filters out the legend's outer
    border, which is a few shades off white but only 1px thick.

    Uses a fixed x-column (shared across all JBA legend images, found once via
    ``_locate_colorbar`` on a colour legend) rather than re-detecting it per
    image, since chroma-based detection can't locate grey-on-white swatches
    (e.g. the 'jba_sop_riverine' legend).
    """
    arr = np.array(image.convert("RGB")).astype(int)
    xc = (x_min + x_max) // 2
    col = arr[:, xc, :]
    is_swatch = (255 - col.min(axis=1)) > 4

    runs = []
    start = None
    for i, v in enumerate(is_swatch):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(is_swatch)))
    runs = [(s, e) for s, e in runs if e - s >= min_run]

    colours = []
    for s, e in runs:
        rows = [tuple(col[i]) for i in range(s, e)]
        mode_colour, _ = Counter(rows).most_common(1)[0]
        colours.append(tuple(int(v) for v in mode_colour))
    return colours


@pytest.mark.skip("One-off: infers FLOOD_DEPTH_COLOURS/STANDARD_OF_PROTECTION_RIVERINE_COLOURS "
                   "constants in jba_image_creator.py from the live JBA legends; rerun "
                   "and transcribe by hand if JBA changes its legend colours.")
def test_infer_hazard_legend_colours(load_credentials):
    from physrisk.hazard_models.jba_image_creator import TileSet

    creator = JBAImageCreator(tileset=TileSet("WR", "202603", "5m", "4326"))

    riverine_bytes = creator.get_legend("jba_riverine", return_period=1500)
    riverine_image = Image.open(io.BytesIO(riverine_bytes))
    x_min, _, x_max, _ = _locate_colorbar(riverine_image)

    for resource_id in ["jba_riverine", "jba_coastal", "jba_pluvial", "jba_sop_riverine"]:
        legend_bytes = creator.get_legend(resource_id, return_period=1500)
        image = Image.open(io.BytesIO(legend_bytes))
        colours = _extract_swatch_colours(image, x_min, x_max)
        print(f"\n{resource_id} ({len(colours)} swatches):")
        for colour in colours:
            print(f"    {list(colour)},")

    # riverine is already hard-coded in jba_image_creator.FLOOD_DEPTH_COLOURS_RIVERINE; use
    # it to check the extraction method against a known-good reference.
    from physrisk.hazard_models.jba_image_creator import FLOOD_DEPTH_COLOURS_RIVERINE

    riverine_colours = _extract_swatch_colours(riverine_image, x_min, x_max)
    assert riverine_colours == [
        tuple(int(v) for v in row) for row in FLOOD_DEPTH_COLOURS_RIVERINE
    ]
