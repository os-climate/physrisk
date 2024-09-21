import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import numpy as np

from physrisk.data.hazard_data_provider import HazardDataHint
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import (
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import (
    ChronicHeat,
    CombinedInundation,
    Drought,
    Fire,
    Hail,
    Wind,
)
from physrisk.kernel.impact import _request_consolidated
from physrisk.kernel.vulnerability_model import DataRequester
from physrisk.utils.helpers import get_iterable


class Category(Enum):
    LOWEST = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    HIGHEST = 5
    NODATA = 6


@dataclass
class Bounds:
    """Category applies if lower <= value < upper"""

    category: str
    lower: float
    upper: float


@dataclass
class AssetExposureResult:
    hazard_categories: Dict[type, Tuple[Category, float, str]]


class ExposureMeasure(DataRequester):
    @abstractmethod
    def get_exposures(
        self, asset: Asset, data_responses: Sequence[HazardDataResponse]
    ) -> Dict[type, Tuple[Category, float, str]]: ...


class JupterExposureMeasure(ExposureMeasure):
    def __init__(self):
        self.exposure_bins = self.get_exposure_bins()

    def get_data_requests(
        self, asset: Asset, *, scenario: str, year: int
    ) -> Sequence[HazardDataRequest]:
        return [
            HazardDataRequest(
                hazard_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                indicator_id=indicator_id,
                # select specific model for wind for consistency with thresholds
                hint=HazardDataHint(path="wind/jupiter/v1/max_1min_{scenario}_{year}")
                if hazard_type == Wind
                else None,
            )
            for (hazard_type, indicator_id) in self.exposure_bins.keys()
        ]

    def get_exposures(self, asset: Asset, data_responses: Sequence[HazardDataResponse]):
        result: Dict[type, Tuple[Category, float, str]] = {}
        for (k, v), resp in zip(self.exposure_bins.items(), data_responses):
            if isinstance(resp, HazardParameterDataResponse):
                param = resp.parameter
                hazard_path = resp.path
            elif isinstance(resp, HazardEventDataResponse):
                if len(resp.intensities) > 1:
                    raise ValueError("single-value curve expected")
                param = resp.intensities[0]
                hazard_path = resp.path
            (hazard_type, _) = k
            (lower_bounds, categories) = v
            if math.isnan(param):
                result[hazard_type] = (Category.NODATA, float(param), hazard_path)
            else:
                index = np.searchsorted(lower_bounds, param, side="right") - 1
                result[hazard_type] = (categories[index], float(param), hazard_path)
        return result

    def get_exposure_bins(self):
        categories = {}
        # specify exposure bins as dataclass in case desirable to use JSON in future
        categories[(CombinedInundation, "flooded_fraction")] = self.bounds_to_lookup(
            [
                Bounds(category=Category.LOWEST, lower=float("-inf"), upper=0.01),
                Bounds(category=Category.LOW, lower=0.01, upper=0.04),
                Bounds(category=Category.MEDIUM, lower=0.04, upper=0.1),
                Bounds(category=Category.HIGH, lower=0.1, upper=0.2),
                Bounds(category=Category.HIGHEST, lower=0.2, upper=float("inf")),
            ]
        )
        categories[(ChronicHeat, "days/above/35c")] = self.bounds_to_lookup(
            [
                Bounds(category=Category.LOWEST, lower=float("-inf"), upper=5),
                Bounds(category=Category.LOW, lower=5, upper=10),
                Bounds(category=Category.MEDIUM, lower=10, upper=20),
                Bounds(category=Category.HIGH, lower=20, upper=30),
                Bounds(category=Category.HIGHEST, lower=30, upper=float("inf")),
            ]
        )
        categories[(Wind, "max_speed")] = self.bounds_to_lookup(
            [
                Bounds(category=Category.LOWEST, lower=float("-inf"), upper=63),
                Bounds(category=Category.LOW, lower=63, upper=90),
                Bounds(category=Category.MEDIUM, lower=90, upper=119),
                Bounds(category=Category.HIGH, lower=119, upper=178),
                Bounds(category=Category.HIGHEST, lower=178, upper=float("inf")),
            ]
        )
        categories[(Drought, "months/spei3m/below/-2")] = self.bounds_to_lookup(
            [
                Bounds(category=Category.LOWEST, lower=float("-inf"), upper=0.1),
                Bounds(category=Category.LOW, lower=0.1, upper=0.25),
                Bounds(category=Category.MEDIUM, lower=0.25, upper=0.5),
                Bounds(category=Category.HIGH, lower=0.5, upper=1.0),
                Bounds(category=Category.HIGHEST, lower=1.0, upper=float("inf")),
            ]
        )
        categories[(Hail, "days/above/5cm")] = self.bounds_to_lookup(
            [
                Bounds(category=Category.LOWEST, lower=float("-inf"), upper=0.2),
                Bounds(category=Category.LOW, lower=0.2, upper=1.0),
                Bounds(category=Category.MEDIUM, lower=1.0, upper=2.0),
                Bounds(category=Category.HIGH, lower=2.0, upper=3.0),
                Bounds(category=Category.HIGHEST, lower=3.0, upper=float("inf")),
            ]
        )
        categories[(Fire, "fire_probability")] = self.bounds_to_lookup(
            [
                Bounds(category=Category.LOWEST, lower=float("-inf"), upper=0.1),
                Bounds(category=Category.LOW, lower=0.1, upper=0.2),
                Bounds(category=Category.MEDIUM, lower=0.2, upper=0.35),
                Bounds(category=Category.HIGH, lower=0.35, upper=0.5),
                Bounds(category=Category.HIGHEST, lower=0.5, upper=float("inf")),
            ]
        )
        return categories

    def bounds_to_lookup(self, bounds: Sequence[Bounds]):
        lower_bounds = np.array([b.lower for b in bounds])
        categories = np.array([b.category for b in bounds])
        return (lower_bounds, categories)


def calculate_exposures(
    assets: List[Asset],
    hazard_model: HazardModel,
    exposure_measure: ExposureMeasure,
    scenario: str,
    year: int,
) -> Dict[Asset, AssetExposureResult]:
    requester_assets: Dict[DataRequester, List[Asset]] = {exposure_measure: assets}
    asset_requests, responses = _request_consolidated(
        hazard_model, requester_assets, scenario, year
    )

    logging.info(
        "Applying exposure measure {0} to {1} assets of type {2}".format(
            type(exposure_measure).__name__, len(assets), type(assets[0]).__name__
        )
    )
    result: Dict[Asset, AssetExposureResult] = {}

    for asset in assets:
        requests = asset_requests[
            (exposure_measure, asset)
        ]  # (ordered) requests for a given asset
        hazard_data = [responses[req] for req in get_iterable(requests)]
        result[asset] = AssetExposureResult(
            hazard_categories=exposure_measure.get_exposures(asset, hazard_data)
        )

    return result
