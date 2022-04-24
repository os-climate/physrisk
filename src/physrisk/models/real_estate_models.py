from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import scipy.stats as stats

from physrisk.data_objects.vulnerability_curve import VulnerabilityCurve, VulnerabilityCurves
from physrisk.kernel.assets import RealEstateAsset
from physrisk.kernel.impact_curve import ImpactCurve
from physrisk.kernel.vulnerability_model import VulnerabilityModel

from ..kernel.events import RiverineInundation
from ..kernel.vulnerability_model import applies_to_events, get_vulnerability_curves_from_resource


@applies_to_events([RiverineInundation])
class RealEstateInundationModel(VulnerabilityModel):
    def __init__(
        self,
        *,
        resource: str = "EU JRC global flood depth-damage functions",
        impact_bin_edges=np.array([0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ):
        curve_set: VulnerabilityCurves = get_vulnerability_curves_from_resource(resource)

        # for this model, key for looking up curves is (location, asset_type), e.g. ('Asian', 'Building/Industrial')
        self.vulnerability_curves = dict(((c.location, c.asset_type), c) for c in curve_set.items)
        self.vuln_curves_by_type = defaultdict(list)
        self.proxy_curves: Dict[Tuple[str, str], VulnerabilityCurve] = {}
        for item in curve_set.items:
            self.vuln_curves_by_type[item.asset_type].append(item)

        super().__init__(model="MIROC-ESM-CHEM", event_type=RiverineInundation, impact_bin_edges=impact_bin_edges)

    def get_impact_curve(self, intensities, asset: RealEstateAsset):
        # we interpolate the mean and standard deviation and use this to construct distributions
        # assert asset is RealEstateAsset

        key = (asset.location, asset.type)
        curve = self.vulnerability_curves[key]

        std_curve = curve
        if len(curve.impact_std) == 0:
            if key not in self.proxy_curves:
                self.proxy_curves[key] = self.closest_curve_of_type(curve, asset)
            std_curve = self.proxy_curves[key]

        impact_means = np.interp(intensities, curve.intensity, curve.impact_mean)
        impact_stddevs = np.interp(intensities, std_curve.intensity, std_curve.impact_std)

        return ImpactCurve(
            intensities, impact_cdfs=[checked_beta_distrib(m, s) for m, s in zip(impact_means, impact_stddevs)]
        )

    def closest_curve_of_type(self, curve: VulnerabilityCurve, asset: RealEstateAsset):
        # we return the standard deviations of the damage curve most similar to the asset location
        candidate_set = list(cand for cand in self.vuln_curves_by_type[asset.type] if (len(cand.impact_std) > 0))
        sum_square_diff = (self.sum_square_diff(curve, cand) for cand in candidate_set)
        lowest = np.argmin(np.array(list(sum_square_diff)))
        return candidate_set[lowest]

    def sum_square_diff(self, curve1: VulnerabilityCurve, curve2: VulnerabilityCurve):
        return np.sum((curve1.impact_mean - np.interp(curve1.intensity, curve2.intensity, curve2.impact_mean)) ** 2)


def delta_cdf(y):
    return lambda x: np.where(x >= y, 1, 0)


def checked_beta_distrib(mean, std):
    if mean == 0:
        return delta_cdf(0)
    if mean == 1.0:
        return delta_cdf(1)
    else:
        return beta_distrib(mean, std)


def beta_distrib(mean, std):
    cv = std / mean
    a = (1 - mean) / (cv * cv) - mean
    b = a * (1 - mean) / mean
    return lambda x, a=a, b=b: stats.beta.cdf(x, a, b)
