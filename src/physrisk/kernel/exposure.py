from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Tuple

import numpy as np

from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import HazardDataRequest, HazardDataResponse, HazardParameterDataResponse
from physrisk.kernel.hazards import ChronicHeat, CombinedInundation, Drought, Fire, Hail, Wind
from physrisk.kernel.vulnerability_model import DataRequester


class Category(Enum):
    LOWEST = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    HIGHEST = 5


@dataclass
class Bounds:
    """Category applies if lower <= value < upper"""

    category: str
    lower: float
    upper: float


class ExposureMeasure(DataRequester):
    @abstractmethod
    def get_exposures(self, asset: Asset, data_responses: Iterable[HazardDataResponse]) -> Dict[type, Category]:
        ...


class JupterExposureMeasure(ExposureMeasure):
    def __init__(self):
        self.exposure_bins: Dict[Tuple[type, str], Tuple[np.ndarray, np.ndarray]] = self.get_exposure_bins()

    def get_data_requests(self, asset: Asset, *, scenario: str, year: int) -> Iterable[HazardDataRequest]:
        return [
            HazardDataRequest(
                hazard_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                model=model,
            )
            for (hazard_type, model) in self.exposure_bins.keys()
        ]

    def get_exposures(self, asset: Asset, data_responses: Iterable[HazardDataResponse]):
        result: Dict[type, Category] = {}
        for (k, v), resp in zip(self.exposure_bins.items(), data_responses):
            assert isinstance(resp, HazardParameterDataResponse)  # should all be parameters
            (hazard_type, _) = k
            (lower_bounds, categories) = v
            index = np.searchsorted(lower_bounds, resp.parameter, side="right") - 1
            result[hazard_type] = categories[index]
        return result

    def get_exposure_bins(self):
        categories: Dict[Tuple[type, str], Tuple[np.ndarray, np.ndarray]] = {}
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
        categories[(Wind, "max/1min")] = self.bounds_to_lookup(
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

    def bounds_to_lookup(self, bounds: Iterable[Bounds]):
        lower_bounds = np.array([b.lower for b in bounds])
        categories = np.array([b.category for b in bounds])
        return (lower_bounds, categories)
