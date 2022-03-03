from types import SimpleNamespace

import numpy as np
import scipy.stats as stats

from physrisk.kernel.impact_curve import ImpactCurve
from physrisk.kernel.vulnerability_model import VulnerabilityModel

from ..kernel.events import RiverineInundation
from ..kernel.vulnerability_model import applies_to_events

# following example for flood-depth daamge curve on industry buildings in Asia 

@applies_to_events([RiverineInundation])
class ExampleCdfBasedVulnerabilityModel(VulnerabilityModel):
    def __init__(self, *, model: str, event_type: type):
        self.intensities = np.array([0, 0.5, 1.0, 1.5, 2, 3, 4, 5, 6])
        self.impact_means = np.array([0, 0.28, 0.48, 0.63, 0.72, 0.86, 0.91, 0.96, 1.0])
        self.impact_stddevs = np.array([0, 0.17, 0.14, 0.14, 0.17, 0.14, 0.10, 0.06, 0]) # completely random at the moment
        impact_bin_edges = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        super().__init__(model=model, event_type=event_type, impact_bin_edges=impact_bin_edges)

    def get_impact_curve(self, intensities):
        # we interpolate the mean and standard deviation and use this to construct distributions
        impact_means = np.interp(intensities, self.intensities, self.impact_means)
        impact_stddevs = np.interp(intensities, self.intensities, self.impact_stddevs)
        return ImpactCurve(
            intensities, impact_distribs=[checked_beta_distrib(m, s) for m, s in zip(impact_means, impact_stddevs)]
        )


def delta_cdf(y):
    return SimpleNamespace(pdf=lambda x: np.where(x >= y, 1, 0))


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
    return SimpleNamespace(cdf=lambda x, a=a, b=b: stats.beta.cdf(x, a, b))
