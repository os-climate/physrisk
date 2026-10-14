import asyncio
from dataclasses import dataclass
import io
import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import aiohttp
import numpy as np
import PIL.Image as Image
from lxml import etree

from physrisk.api.v1.hazard_image import TileNotAvailableError
from physrisk.data import colormap_provider
from physrisk.data.image_creator import ImageCreator, to_rgba
from physrisk.kernel.hazard_model import HazardImageCreator, Tile

from physrisk.utils.event_loop import get_loop, run
from physrisk.hazard_models.credentials_provider import (
    CredentialsProvider,
    EnvCredentialsProvider,
)

logger = logging.getLogger(__name__)

# Depth bins (and so FLOOD_DEPTH_UPPER/FLOOD_DEPTH_MID) are shared across the three flood
# hazard types; only the colour ramp (hue) differs between their legends.
FLOOD_DEPTH_UPPER = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, np.inf]

# Representative mid-point depth (metres) for each level; last bin uses 12 m.
FLOOD_DEPTH_MID = np.array(
    [0.005, 0.255, 0.75, 1.5, 2.5, 3.5, 4.5, 5.5, 8.0, 12.0],
    dtype=np.float32,
)

# RGB fill colours for each flood-depth level, sampled from the JBA legends (WR_202603_5m_4326
# tileset) for each undefended flood hazard type. Index 0 = shallowest (0-0.01m), index 9 =
# deepest (>10m) -- see FLOOD_DEPTH_UPPER.
FLOOD_DEPTH_COLOURS_RIVERINE = np.array(
    [
        [191, 232, 242],
        [178, 222, 232],
        [124, 195, 212],
        [106, 183, 202],
        [89, 172, 193],
        [71, 160, 183],
        [53, 149, 173],
        [36, 138, 163],
        [18, 126, 154],
        [1, 115, 144],
    ],
    dtype=np.float32,
)

FLOOD_DEPTH_COLOURS_COASTAL = np.array(
    [
        [224, 222, 200],
        [220, 218, 186],
        [215, 213, 171],
        [211, 208, 156],
        [207, 202, 139],
        [201, 196, 120],
        [196, 191, 103],
        [191, 185, 84],
        [186, 178, 65],
        [181, 172, 45],
    ],
    dtype=np.float32,
)

FLOOD_DEPTH_COLOURS_PLUVIAL = np.array(
    [
        [222, 194, 248],
        [214, 186, 239],
        [206, 177, 230],
        [197, 168, 220],
        [187, 158, 210],
        [178, 149, 200],
        [168, 138, 188],
        [158, 129, 178],
        [148, 118, 166],
        [137, 107, 154],
    ],
    dtype=np.float32,
)

# Riverine standard-of-protection (years defended), sampled from the 'jba_sop_riverine' legend
# (grey scale, 9 bins). Index 0 = least protected (0-20 yr), index 8 = most protected (>1500 yr).
# Coastal SoP will need its own STANDARD_OF_PROTECTION_COASTAL_* table once that layer exists.
# fmt: off
STANDARD_OF_PROTECTION_RIVERINE_UPPER = [20.0, 50.0, 75.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, np.inf]
# fmt: on

# Representative mid-point return period (years) for each level; last bin uses 1800 yr.
STANDARD_OF_PROTECTION_RIVERINE_MID = np.array(
    [10.0, 35.0, 62.5, 87.5, 150.0, 350.0, 750.0, 1250.0, 1800.0],
    dtype=np.float32,
)

STANDARD_OF_PROTECTION_RIVERINE_COLOURS = np.array(
    [
        [216, 216, 216],
        [188, 188, 188],
        [174, 174, 174],
        [160, 160, 160],
        [146, 146, 146],
        [132, 132, 132],
        [118, 118, 118],
        [104, 104, 104],
        [90, 90, 90],
    ],
    dtype=np.float32,
)


def _rgb_to_lightness(rgb_norm: np.ndarray) -> np.ndarray:
    """HLS lightness from a normalised RGB array (..., 3), values in [0, 1]."""
    return (rgb_norm.max(axis=-1) + rgb_norm.min(axis=-1)) * 0.5


@dataclass(frozen=True)
class _ResourceLegend:
    lightness: np.ndarray
    mid: np.ndarray


# Precomputed lightness for each resource's colour table, keyed by physrisk resource id.
_LEGEND_BY_RESOURCE = {
    "jba_riverine": _ResourceLegend(
        _rgb_to_lightness(FLOOD_DEPTH_COLOURS_RIVERINE / 255.0), FLOOD_DEPTH_MID
    ),
    "jba_coastal": _ResourceLegend(
        _rgb_to_lightness(FLOOD_DEPTH_COLOURS_COASTAL / 255.0), FLOOD_DEPTH_MID
    ),
    "jba_pluvial": _ResourceLegend(
        _rgb_to_lightness(FLOOD_DEPTH_COLOURS_PLUVIAL / 255.0), FLOOD_DEPTH_MID
    ),
    "jba_sop_riverine": _ResourceLegend(
        _rgb_to_lightness(STANDARD_OF_PROTECTION_RIVERINE_COLOURS / 255.0),
        STANDARD_OF_PROTECTION_RIVERINE_MID,
    ),
}


def image_to_flood_depth(
    img: Image.Image,
    resource_id: str,
    max_lightness_dist: float = 0.05,
) -> np.ndarray:
    """Convert an RGB(A) tile image to its approximate legend value, for purpose of
    changing colour map.

    Each pixel is matched to the nearest legend entry for *resource_id* by its HLS
    lightness value (the three flood hazard types and 'jba_sop_riverine' each have their own
    colour ramp -- see _LEGEND_BY_RESOURCE). Transparent pixels or pixels whose
    lightness differs from every reference by more than ``max_lightness_dist`` are
    assigned ``NaN``.

    Args:
        img: PIL image (RGB or RGBA).
        resource_id: physrisk resource identifier (e.g. ``"jba_riverine"``).
        max_lightness_dist: lightness threshold above which a pixel is treated
            as no-data.

    Returns:
        float32 array of shape (H, W): depth in metres for the flood hazard types,
        or return period in years for 'jba_sop_riverine'; NaN for no-data.
    """
    legend = _LEGEND_BY_RESOURCE[resource_id]
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    pixel_l = (
        rgb.max(axis=-1).astype(np.float32) + rgb.min(axis=-1).astype(np.float32)
    ) * (0.5 / 255.0)

    best_idx = np.zeros(pixel_l.shape, dtype=np.uint8)
    best_dist = np.full(pixel_l.shape, np.inf, dtype=np.float32)
    for i, ref_l in enumerate(legend.lightness):
        d = np.abs(pixel_l - ref_l)
        closer = d < best_dist
        best_dist[closer] = d[closer]
        best_idx[closer] = i

    depth = legend.mid[best_idx].copy()
    depth[alpha < 128] = np.nan
    depth[best_dist > max_lightness_dist] = np.nan
    return depth


@dataclass
class TileSet:
    name: str
    release_date: str
    resolution: str
    projection: str

    def identifier(self):
        return f"{self.name}_{self.release_date}_{self.resolution}_{self.projection}"


TileSpec = Tuple[int, int, int]  # (z, x, y)

_SUPPORTED_TILE_SIZES = (256, 512)
_DEFAULT_TILE_SIZE = 512

# JBA's WMTS natively serves 256px tiles up to this zoom. Requesting a larger output
# tile_size composites 256px tiles from one zoom level deeper, so the usable max zoom
# is correspondingly coarser (one level per doubling of tile_size).
_JBA_NATIVE_MAX_ZOOM = 16


def _resolve_tile_size(tile_size: Optional[int]) -> int:
    size = tile_size if tile_size is not None else _DEFAULT_TILE_SIZE
    if size not in _SUPPORTED_TILE_SIZES:
        raise ValueError(
            f"tile_size={size} is not supported; use one of {_SUPPORTED_TILE_SIZES}."
        )
    return size


def _jba_max_zoom(tile_size: int) -> int:
    f = tile_size // 256
    return _JBA_NATIVE_MAX_ZOOM - (f.bit_length() - 1)


class JBAImageCreator(HazardImageCreator):
    """Create images by calling out to JBA WMTS."""

    def __init__(
        self,
        credentials: Optional[CredentialsProvider] = None,
        tileset: TileSet = TileSet("WR", "202603", "5m", "4326"),
    ):
        self.credentials = (
            credentials if credentials is not None else EnvCredentialsProvider()
        )
        self.tileset = tileset
        # TileSet("WR30", "202512", "30m", "4326")
        # TileSet("WR30C", "202603", "30m", "4326")
        # TileSet("WR", "202603", "5m", "4326")
        templates_tiles, templates_legends = self._get_urls_from_capability()
        self.templates_tiles: dict[str, str] = templates_tiles
        self.templates_legends: dict[str, str] = templates_legends

    def create_image(
        self,
        resource_id: str,
        scenario: str,
        year: int,
        format="PNG",
        colormap: str = "heating",
        tile: Optional[Tile] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        index_value: Optional[Union[str, float]] = None,
        scaling: str = "linear",
        tile_size: Optional[int] = None,
    ):
        assert tile is not None
        size = _resolve_tile_size(tile_size)
        f = size // 256

        def expand(tile: Tile, f: int):
            return [
                Tile(tile.z + (f.bit_length() - 1), tile.x * f + dx, tile.y * f + dy)
                for dy in range(f)
                for dx in range(f)
            ]

        try:
            loop = get_loop()
            if index_value is None:
                index_value = 1500
            tiles = run(
                self._fetch_all_tiles(resource_id, int(index_value), expand(tile, f)),
                loop=loop,
            )
            stitched = self._stitch_tiles(tiles, grid=(f, f))
        except Exception as e:
            # we are creating a tile we let the error propagate
            # because many map controls expect an HTTPException in such cases.
            if isinstance(e, KeyError):
                raise TileNotAvailableError(e.args[0]) from e
            else:
                raise

        depth = image_to_flood_depth(stitched, resource_id)
        map_defn = colormap_provider.colormap(colormap)

        def get_colors(index: int):
            return map_defn[str(index)]

        rgba = to_rgba(
            depth, get_colors, min_value=min_value, max_value=max_value, scaling=scaling
        )
        image = Image.fromarray(rgba, mode="RGBA")

        image_bytes = io.BytesIO()
        image.save(image_bytes, format=format)
        return image_bytes.getvalue()

    def get_legend(self, resource_id: str, return_period: int = 1500) -> bytes:
        """Download the legend PNG for *resource_id* at *return_period* from the JBA WMTS.

        Args:
            resource_id: physrisk resource identifier (e.g. ``"jba_riverine"``).
            return_period: return period in years; selects the matching layer.

        Returns:
            Raw PNG bytes of the legend image.
        """
        identifier = self._identifier(self.tileset, resource_id, return_period)
        url = self.templates_legends[identifier]
        loop = get_loop()

        async def _fetch() -> bytes:
            async with aiohttp.ClientSession(
                proxy=self.credentials.proxies()["https"],
                auth=aiohttp.BasicAuth(
                    self.credentials.jba_vision_username(),
                    self.credentials.jba_vision_password(),
                ),
            ) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.read()

        return run(_fetch(), loop=loop)

    def get_info(
        self,
        resource_id: str,
        scenario: str,
        year: int,
        tile_size: Optional[int] = None,
    ) -> Tuple[Sequence[Any], Sequence[Any], str, str, Optional[int], int]:
        size = _resolve_tile_size(tile_size)
        index_values = [20, 50, 100, 200, 500, 1500]
        return (
            index_values,
            index_values,
            "return period",
            "years",
            _jba_max_zoom(size),
            size,
        )

    def _get_urls_from_capability(self):
        # async is not necessary, but we follow the same pattern
        loop = get_loop()
        with aiohttp.TCPConnector(loop=loop) as conn:
            identifiers, template_tiles, template_legends = {}, {}, {}

            async def get_capability():
                try:
                    async with aiohttp.ClientSession(
                        connector=conn, connector_owner=False
                    ) as session:
                        set_name = self.tileset.identifier()
                        url = f"https://jbavision.jbarisk.com/cog/WMTS/{set_name}?service=WMS&request=GetCapabilities&version=1.3.0"
                        async with session.get(
                            url=url,
                            proxy=self.credentials.proxies()["https"],
                            auth=aiohttp.BasicAuth(
                                self.credentials.jba_vision_username(),
                                self.credentials.jba_vision_password(),
                            ),
                        ) as resp:
                            resp.raise_for_status()
                            e_tree = etree.fromstring(await resp.text())
                            ns = {
                                "wmts": "http://www.opengis.net/wmts/1.0",
                                "ows": "http://www.opengis.net/ows/1.1",
                                "xlink": "http://www.w3.org/1999/xlink",
                            }
                            layers = e_tree.xpath("//wmts:Layer", namespaces=ns)
                            for layer in layers:
                                template_tile = layer.xpath(
                                    "wmts:ResourceURL[@format='image/png']",
                                    namespaces=ns,
                                )[0].get("template")
                                template_legend = layer.xpath(
                                    "wmts:Style/wmts:LegendURL[@format='image/png']",
                                    namespaces=ns,
                                )[0].get("{" + ns["xlink"] + "}href")
                                title_text = layer.xpath("ows:Title", namespaces=ns)[
                                    0
                                ].text
                                identifier = layer.xpath(
                                    "ows:Identifier", namespaces=ns
                                )[0].text
                                identifiers[title_text] = identifier
                                template_tiles[title_text] = template_tile
                                template_legends[title_text] = template_legend
                except Exception as e:
                    logger.exception(e)

            run(get_capability(), loop=loop)
            return template_tiles, template_legends

    async def _fetch_tile(
        self, session: aiohttp.ClientSession, url: str
    ) -> Image.Image:
        """Download a single tile and return it as a Pillow Image."""
        async with session.get(url) as resp:
            resp.raise_for_status()  # raise on HTTP errors
            data = await resp.read()  # raw bytes
            return Image.open(io.BytesIO(data)).convert("RGBA")  # ensure RGBA

    async def _fetch_all_tiles(
        self, resource_id: str, return_period: int, tile_specs: List[TileSpec]
    ):
        """Download all tiles concurrently and return them in the same order."""
        async with aiohttp.ClientSession(
            proxy=self.credentials.proxies()["https"],
            auth=aiohttp.BasicAuth(
                self.credentials.jba_vision_username(),
                self.credentials.jba_vision_password(),
            ),
        ) as session:
            tasks = []
            for z, x, y in tile_specs:
                url = self.templates_tiles[
                    self._identifier(self.tileset, resource_id, return_period)
                ].format(TileMatrix=z, TileCol=x, TileRow=y)
                tasks.append(self._fetch_tile(session, url))
            return await asyncio.gather(*tasks)

    def _stitch_tiles(self, tiles, grid=(2, 2)):
        """
        Assemble a list of Pillow images into one image.

        Parameters
        ----------
        tiles : list[Image.Image]
            Tiles ordered row‑wise (left → right, top → bottom).
        grid : tuple[int, int]
            (cols, rows) of the final mosaic.

        Returns
        -------
        Image.Image
            The combined image.
        """
        cols, rows = grid
        if len(tiles) != cols * rows:
            raise ValueError("Number of tiles does not match the grid size")

        # Assume all tiles have the same dimensions
        tile_w, tile_h = tiles[0].size
        combined = Image.new("RGBA", (cols * tile_w, rows * tile_h))

        for idx, tile in enumerate(tiles):
            col = idx % cols
            row = idx // cols
            combined.paste(tile, (col * tile_w, row * tile_h))

        return combined

    def _identifier(self, tile_set: TileSet, resource_id: str, return_period: int):
        if resource_id == "jba_coastal":  # undefended coastal
            return f"{tile_set.name}_{tile_set.release_date}_STSU_U_RP{return_period}_RD_{tile_set.resolution}_{tile_set.projection}"
        elif resource_id == "jba_riverine":  # undefended riverine
            return f"{tile_set.name}_{tile_set.release_date}_FLRF_U_RP{return_period}_RD_{tile_set.resolution}_{tile_set.projection}"
        elif resource_id == "jba_pluvial":  # undefended pluvial
            return f"{tile_set.name}_{tile_set.release_date}_FLSW_U_RP{return_period}_RD_{tile_set.resolution}_{tile_set.projection}"
        elif resource_id == "jba_sop_riverine":  # riverine standard of protection
            # the DRAS layer is suffixed "_VE_" in some tilesets (e.g. WR30) but not
            # others (e.g. WR); try both rather than hard-coding one.
            with_ve = f"{tile_set.name}_{tile_set.release_date}_DRAS_D_VE_{tile_set.resolution}_{tile_set.projection}"
            if with_ve in self.templates_tiles or with_ve in self.templates_legends:
                return with_ve
            return f"{tile_set.name}_{tile_set.release_date}_DRAS_D_{tile_set.resolution}_{tile_set.projection}"


class CombinedImageCreator(HazardImageCreator):
    def __init__(
        self, image_creator: "ImageCreator", jba_image_creator: "JBAImageCreator"
    ):
        self._image_creator = image_creator
        self._jba_image_creator = jba_image_creator

    def _creator(self, resource_id: str) -> HazardImageCreator:
        return (
            self._jba_image_creator
            if self._jba_image_creator and resource_id.startswith("jba_")
            else self._image_creator
        )

    def create_image(
        self,
        resource_id: str,
        scenario: str,
        year: int,
        format="PNG",
        colormap: str = "heating",
        tile: Optional[Tile] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        index_value: Optional[Union[str, float]] = None,
        scaling: str = "linear",
        tile_size: Optional[int] = None,
    ):
        return self._creator(resource_id).create_image(
            resource_id,
            scenario,
            year,
            format=format,
            colormap=colormap,
            tile=tile,
            min_value=min_value,
            max_value=max_value,
            index_value=index_value,
            scaling=scaling,
            tile_size=tile_size,
        )

    def get_info(
        self,
        resource_id: str,
        scenario: str,
        year: int,
        tile_size: Optional[int] = None,
    ):
        return self._creator(resource_id).get_info(
            resource_id, scenario, year, tile_size=tile_size
        )
