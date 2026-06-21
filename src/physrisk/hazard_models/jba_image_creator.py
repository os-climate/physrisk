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
from physrisk.data.image_creator import ImageCreator
from physrisk.kernel.hazard_model import HazardImageCreator, Tile

from physrisk.utils.event_loop import get_loop, run
from physrisk.hazard_models.credentials_provider import (
    CredentialsProvider,
    EnvCredentialsProvider,
)

logger = logging.getLogger(__name__)

# RGB fill colours for each flood-depth level, sampled from the JBA legend
# (WR30_202512 tileset).  Index 0 = shallowest, index 9 = deepest.
FLOOD_DEPTH_COLOURS = np.array(
    [
        [191, 232, 242],  # 0: 0 – 0.01 m
        [178, 222, 232],  # 1: 0.01 – 0.5 m
        [124, 195, 212],  # 2: 0.5 – 1 m
        [106, 183, 202],  # 3: 1 – 2 m
        [89, 172, 193],  # 4: 2 – 3 m
        [71, 160, 183],  # 5: 3 – 4 m
        [53, 149, 173],  # 6: 4 – 5 m
        [36, 138, 163],  # 7: 5 – 6 m
        [18, 126, 154],  # 8: 6 – 10 m
        [1, 115, 144],  # 9: > 10 m
    ],
    dtype=np.float32,
)

FLOOD_DEPTH_UPPER = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, np.inf]

# Representative mid-point depth (metres) for each level; last bin uses 12 m.
FLOOD_DEPTH_MID = np.array(
    [0.005, 0.255, 0.75, 1.5, 2.5, 3.5, 4.5, 5.5, 8.0, 12.0],
    dtype=np.float32,
)


def _rgb_to_lightness(rgb_norm: np.ndarray) -> np.ndarray:
    """HLS lightness from a normalised RGB array (..., 3), values in [0, 1]."""
    return (rgb_norm.max(axis=-1) + rgb_norm.min(axis=-1)) * 0.5


# Precomputed lightness for each FLOOD_DEPTH_COLOURS entry.
_FLOOD_DEPTH_LIGHTNESS = _rgb_to_lightness(FLOOD_DEPTH_COLOURS / 255.0)


def image_to_flood_depth(
    img: Image.Image,
    max_lightness_dist: float = 0.05,
) -> np.ndarray:
    """Convert an RGB(A) tile image to approximate flood depth in metres for
    purpose of changing colour map.

    Each pixel is matched to the nearest entry in ``FLOOD_DEPTH_COLOURS`` by
    its HLS lightness value. Transparent pixels or pixels whose lightness
    differs from every reference by more than ``max_lightness_dist`` are
    assigned ``NaN``.

    Args:
        img: PIL image (RGB or RGBA).
        max_lightness_dist: lightness threshold above which a pixel is treated
            as no-data.

    Returns:
        float32 array of shape (H, W) with depth in metres, NaN for no-data.
    """
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    pixel_l = (
        rgb.max(axis=-1).astype(np.float32) + rgb.min(axis=-1).astype(np.float32)
    ) * (0.5 / 255.0)

    best_idx = np.zeros(pixel_l.shape, dtype=np.uint8)
    best_dist = np.full(pixel_l.shape, np.inf, dtype=np.float32)
    for i, ref_l in enumerate(_FLOOD_DEPTH_LIGHTNESS):
        d = np.abs(pixel_l - ref_l)
        closer = d < best_dist
        best_dist[closer] = d[closer]
        best_idx[closer] = i

    depth = FLOOD_DEPTH_MID[best_idx].copy()
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


class JBAImageCreator(HazardImageCreator):
    """Create images by calling out to JBA WMTS."""

    def __init__(
        self,
        credentials: Optional[CredentialsProvider] = None,
    ):
        self.credentials = (
            credentials if credentials is not None else EnvCredentialsProvider()
        )
        # set_name = "WR30_202512_30m_4326"
        # set_name = "WR30C_202603_30m_4326"
        self.tileset = TileSet("WR30C", "202603", "30m", "4326")
        self.tileset = TileSet("WR30", "202512", "30m", "4326")
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
    ):
        assert tile is not None

        def expand(tile: Tile, size=512):
            f = size // 256
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
                self._fetch_all_tiles(resource_id, int(index_value), expand(tile)),
                loop=loop,
            )
            stitched = self._stitch_tiles(tiles)
        except Exception as e:
            # we are creating a tile we let the error propagate
            # because many map controls expect an HTTPException in such cases.
            if isinstance(e, KeyError):
                raise TileNotAvailableError(e.args[0]) from e
            else:
                raise

        depth = image_to_flood_depth(stitched)
        map_defn = colormap_provider.colormap(colormap)

        def get_colors(index: int):
            return map_defn[str(index)]

        rgba = ImageCreator.to_rgba(
            depth, get_colors, min_value=min_value, max_value=max_value
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
        self, resource_id: str, scenario: str, year: int
    ) -> Tuple[Sequence[Any], Sequence[Any], str, str, Optional[int]]:
        index_values = [20, 50, 100, 200, 500, 1500]
        return (index_values, index_values, "return period", "years", 12)

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
        try:
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
        except Exception as e:
            logger.exception(e)

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
        elif resource_id == "jba_sop":
            return f"{tile_set.name}_{tile_set.release_date}_DRAS_D_VE_{tile_set.resolution}_{tile_set.projection}"


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
        )

    def get_info(self, resource_id: str, scenario: str, year: int):
        return self._creator(resource_id).get_info(resource_id, scenario, year)
