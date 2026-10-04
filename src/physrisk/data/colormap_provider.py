# Colour maps for hazard overlays rendered with transparency on top of a (light) basemap.
# Colours are chosen for distinguishability, not to match the hazard type (e.g. not blue for
# water, red for fire).
#
# Each colormap is defined by a small set of RGB control points, interpolated to 256 entries by
# _lerp_rgb. Transparency is applied programmatically by colormap(): index 0 ('no data') is
# always fully transparent, and index 1 (the minimum/zero value bin) is transparent too unless
# the colormap is listed in _OPAQUE_MIN_BIN.

Rgb = tuple[int, int, int]

_DEFAULT_ALPHA = 200


def _lerp_rgb(control_points: list[Rgb], n: int = 256) -> list[Rgb]:
    """Linearly interpolate a small set of RGB control points up to n samples.

    This lets a colormap be defined by a couple of dozen anchor colours, sampled from the
    reference colormap, instead of a hard-coded 256-entry lookup table.
    """
    n_ctrl = len(control_points)
    result = []
    for i in range(n):
        pos = i * (n_ctrl - 1) / (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n_ctrl - 1)
        frac = pos - lo
        r, g, b = (
            round(
                control_points[lo][c]
                + frac * (control_points[hi][c] - control_points[lo][c])
            )
            for c in range(3)
        )
        result.append((r, g, b))
    return result


# control points sampled directly from Seaborn's 'flare' colormap, and from the warming half
# (0.5-1.0) of Seaborn's 'coolwarm' colormap for 'heating'.
# fmt: off
_CP_FLARE: list[Rgb] = [
    (237, 176, 129), (236, 168, 124), (236, 160, 119), (235, 152, 114), (234, 144, 109), (233, 136, 104), (232, 128, 100), (230, 119, 96),
    (228, 111, 94), (226, 103, 92), (224, 95, 92), (220, 88, 92), (215, 81, 94), (210, 75, 96), (204, 70, 99), (198, 67, 102),
    (190, 63, 105), (182, 61, 107), (175, 59, 109), (167, 57, 110), (159, 55, 111), (151, 53, 112), (144, 51, 113), (136, 49, 113),
    (129, 47, 112), (120, 46, 111), (113, 44, 110), (105, 43, 108), (98, 41, 106), (89, 39, 103), (82, 37, 100), (75, 35, 98),
]
# fmt: on

# fmt: off
_CP_HEATING: list[Rgb] = [
    (221, 220, 220), (225, 218, 214), (229, 216, 209), (233, 213, 203), (236, 211, 197), (239, 207, 191), (241, 204, 184), (243, 199, 177),
    (245, 194, 170), (246, 190, 164), (247, 185, 158), (247, 180, 151), (247, 175, 145), (247, 169, 139), (246, 163, 133), (245, 157, 126),
    (243, 150, 119), (241, 143, 113), (239, 136, 107), (236, 129, 101), (233, 122, 95), (230, 114, 89), (227, 107, 84), (223, 99, 78),
    (218, 89, 72), (213, 81, 67), (208, 71, 61), (203, 62, 56), (197, 51, 52), (192, 40, 47), (186, 22, 43), (180, 4, 38),
]
# fmt: on

# control points sampled from matplotlib's viridis, magma and turbo, and from Fabio Crameri's
# scientific colour map 'batlow' (https://www.fabiocrameri.ch/colourmaps/). All are perceptually
# uniform and colourblind-friendly except turbo, chosen for distinguishability rather than
# hazard-matched hue. viridis, magma and batlow are reversed from their native direction so that
# low values are light (blend into a light basemap) and high values are dark, matching
# flare/heating; turbo isn't reversed since its lightness is non-monotonic either way.
# fmt: off
_CP_VIRIDIS: list[Rgb] = [
    (253, 231, 37), (233, 229, 26), (212, 226, 26), (190, 223, 38), (168, 219, 52), (147, 215, 65), (126, 211, 79), (106, 205, 91),
    (88, 199, 101), (72, 193, 110), (57, 186, 118), (46, 179, 124), (37, 171, 130), (32, 163, 134), (30, 156, 137), (31, 148, 140),
    (34, 140, 141), (37, 132, 142), (40, 125, 142), (43, 117, 142), (46, 109, 142), (50, 101, 142), (54, 93, 141), (58, 84, 140),
    (62, 75, 138), (66, 65, 134), (69, 56, 130), (71, 46, 124), (72, 36, 117), (72, 25, 107), (71, 13, 96), (68, 1, 84),
]
# fmt: on

# fmt: off
_CP_MAGMA: list[Rgb] = [
    (252, 253, 191), (252, 238, 176), (253, 223, 161), (254, 208, 147), (254, 193, 133), (254, 178, 122), (254, 162, 111), (253, 147, 102),
    (251, 131, 95), (248, 116, 92), (243, 101, 92), (236, 88, 96), (226, 77, 102), (214, 69, 108), (202, 62, 114), (189, 57, 119),
    (175, 52, 123), (161, 48, 126), (148, 44, 128), (135, 39, 129), (122, 34, 130), (109, 29, 129), (96, 24, 128), (84, 19, 125),
    (70, 16, 120), (56, 16, 109), (42, 17, 92), (30, 17, 73), (20, 14, 53), (10, 8, 35), (3, 3, 18), (0, 0, 4),
]
# fmt: on

# fmt: off
_CP_BATLOW: list[Rgb] = [
    (250, 204, 250), (251, 198, 231), (252, 191, 213), (253, 185, 195), (253, 179, 177), (253, 173, 160), (252, 167, 141), (247, 161, 122),
    (239, 155, 103), (228, 151, 86), (215, 148, 71), (201, 146, 60), (186, 143, 50), (169, 140, 44), (153, 136, 44), (137, 132, 47),
    (122, 128, 53), (108, 124, 60), (95, 120, 67), (81, 116, 74), (68, 112, 82), (57, 108, 88), (45, 103, 93), (36, 97, 96),
    (29, 91, 98), (23, 83, 98), (20, 76, 98), (17, 68, 96), (15, 60, 95), (13, 49, 93), (8, 37, 91), (1, 25, 89),
]
# fmt: on

# fmt: off
_CP_TURBO: list[Rgb] = [
    (48, 18, 59), (57, 43, 116), (64, 65, 164), (69, 88, 202), (71, 110, 230), (70, 130, 248), (65, 151, 255), (52, 172, 247),
    (37, 192, 231), (26, 210, 210), (24, 225, 188), (35, 235, 169), (59, 245, 143), (89, 251, 115), (122, 254, 89), (151, 254, 67),
    (174, 250, 55), (195, 241, 52), (215, 229, 53), (232, 214, 57), (245, 198, 58), (252, 180, 54), (254, 158, 47), (252, 134, 37),
    (247, 110, 26), (238, 86, 16), (226, 67, 10), (212, 51, 5), (194, 36, 3), (173, 23, 1), (150, 13, 1), (122, 4, 3),
]
# fmt: on

# used in tests only; value equals index so round-tripping can be checked, colour is irrelevant
_RGB_TEST: list[Rgb] = [(i, 0, 0) for i in range(256)]

_RGB_COLORMAPS: dict[str, list[Rgb]] = {
    "flare": _lerp_rgb(_CP_FLARE),
    "heating": _lerp_rgb(_CP_HEATING),
    "heating_2": _lerp_rgb(_CP_HEATING),
    "viridis": _lerp_rgb(_CP_VIRIDIS),
    "magma": _lerp_rgb(_CP_MAGMA),
    "batlow": _lerp_rgb(_CP_BATLOW),
    "turbo": _lerp_rgb(_CP_TURBO),
    "test": _RGB_TEST,
}

# colormaps where index 1 (the minimum/zero value bin) is kept opaque instead of transparent
_OPAQUE_MIN_BIN = {"heating_2"}

# dummy colormap used only in unit tests; alpha is irrelevant so every index is transparent
_FULLY_TRANSPARENT = {"test"}


def colormap(id: str) -> dict[str, list[int]]:
    """Return the 256-entry RGBA lookup table for a colormap, keyed by string index.

    Index 0 ('no data') is always fully transparent; index 1 (the minimum/zero value bin)
    is transparent too unless id is in _OPAQUE_MIN_BIN.
    """
    rgb = _RGB_COLORMAPS[id]
    if id in _FULLY_TRANSPARENT:
        return {str(i): [*rgb[i], 0] for i in range(256)}
    transparent_indexes = {0} if id in _OPAQUE_MIN_BIN else {0, 1}
    return {
        str(i): [*rgb[i], 0 if i in transparent_indexes else _DEFAULT_ALPHA]
        for i in range(256)
    }


colormaps = {name: colormap(name) for name in _RGB_COLORMAPS}
