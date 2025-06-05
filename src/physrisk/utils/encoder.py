import json
import math
from typing import Union
import numpy as np


def sig_figures(x: Union[np.ndarray, float], sf: int):
    """Round the array or float to sf significant figures.
    Why do we support this: is it not better to keep all numbers as floats
    until the end and then use formatting like {0:.6g} or similar only when displaying?
    cf also discussions as to whether floats can _really_ be rounded.
    Probably it is, but Python JSON encoders often do not support use of format strings,
    hence this option is also given as it can provide desired behaviour in a performant way.

    Args:
        x (Union[np.ndarray, float]): Input.
        sf (int): Number of significant figures

    Returns:
        Union[np.ndarray, float]: Rounded data.
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (sf - 1))
    mags = 10 ** (sf - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def nans_and_infs(obj):
    if isinstance(obj, dict):
        return {k: nans_and_infs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nans_and_infs(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return nans_and_infs(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
    return obj


class PhysriskDefaultEncoder(json.JSONEncoder):
    """Encoder that will convert NaN in arrays to null and infinities to
    "inf" or "-inf".
    """

    def encode(self, obj, *args, **kwargs):
        return super().encode(nans_and_infs(obj), *args, **kwargs)
