from typing import Iterable, Union

import numpy as np
from scipy.stats import norm

from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import HazardDataRequest, HazardDataResponse
from physrisk.kernel.hazards import ChronicHeat
from physrisk.kernel.impact_distrib import ImpactDistrib, ImpactType
from physrisk.kernel.vulnerability_model import VulnerabilityModelBase


class ChronicHeatGZN(VulnerabilityModelBase):
    """Model which estiamtes the labour productivity impact based on chronic heat based on the paper "Neidell M,
    Graff Zivin J, Sheahan M,  Willwerth J, Fant C, Sarofim M, et al. (2021) Temperature and work:
    Time allocated to work under varying climate and labor market conditions."
    Average annual work hours are based on USA values reported by the OECD for 2021."""

    def __init__(self, model: str = "mean_degree_days_above_32c", delta=True):
        super().__init__(model, ChronicHeat)  # opportunity to give a model hint, but blank here
        self.time_lost_per_degree_day = 4.671  # This comes from the paper converted to celsius
        self.time_lost_per_degree_day_se = 2.2302  # This comes from the paper converted to celsius
        self.total_labour_hours = 107460
        self.delta = delta

    def get_data_requests(
        self, asset: Asset, *, scenario: str, year: int
    ) -> Union[HazardDataRequest, Iterable[HazardDataRequest]]:
        """Request the hazard data needed by the vulnerability model for a specific asset
        (this is a Google-style doc string)

        Args:
            asset: Asset for which data is requested.
            scenario: Climate scenario of calculation.
            year: Projection year of calculation.

        Returns:
            Single data requests.
        """

        # specify hazard data needed. Model string is hierarchical and '/' separated.
        model = "mean_degree_days/above/32c"

        return [
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario="historical",
                year=1980,
                model=model,
            ),
            HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                scenario=scenario,
                year=year,
                model=model,
            ),
        ]

    def get_impact(self, asset: Asset, data_responses: Iterable[HazardDataResponse]) -> ImpactDistrib:
        """Calcaulate impact (disruption) of asset based on the hazard data returned.

        Args:
            asset: Asset for which impact is calculated.
            data_responses: responses to the hazard data requests generated in get_data_requests.

        Returns:
            Probability distribution of impacts.
        """

        baseline_dd_above_mean, scenario_dd_above_mean = data_responses

        # assert isinstance(baseline_dd_above_mean, HazardParameterDataResponse)
        # assert isinstance(scenario_dd_above_mean, HazardParameterDataResponse)

        delta_dd_above_mean: float = scenario_dd_above_mean.parameter - baseline_dd_above_mean.parameter * self.delta

        hours_worked = self.total_labour_hours
        fraction_loss_mean = (delta_dd_above_mean * self.time_lost_per_degree_day) / hours_worked
        fraction_loss_std = (delta_dd_above_mean * self.time_lost_per_degree_day_se) / hours_worked

        return get_impact_distrib(fraction_loss_mean, fraction_loss_std, ChronicHeat, ImpactType.disruption)


def get_impact_distrib(
    fraction_loss_mean: float, fraction_loss_std: float, hazard_type: type, impact_type: ImpactType
) -> ImpactDistrib:
    """Calculate impact (disruption) of asset based on the hazard data returned.

    Args:
        fraction_loss_mean: mean of the impact distribution
        fraction_loss_std: standard deviation of the impact distribution
        hazard_type: Hazard Type.
        impact_type: Impact Type.

    Returns:
        Probability distribution of impacts.
    """
    impact_bins = np.concatenate(
        [
            np.linspace(-0.001, 0.001, 1, endpoint=False),
            np.linspace(0.001, 0.01, 9, endpoint=False),
            np.linspace(0.01, 0.1, 10, endpoint=False),
            np.linspace(0.1, 0.999, 10, endpoint=False),
            np.linspace(0.999, 1.001, 2),
        ]
    )

    probs_cumulative = np.vectorize(lambda x: norm.cdf(x, loc=fraction_loss_mean, scale=max(1e-12, fraction_loss_std)))(
        impact_bins
    )
    probs_cumulative[-1] = np.maximum(probs_cumulative[-1], 1.0)
    probs = np.diff(probs_cumulative)

    probs_norm = np.sum(probs)
    prob_differential = 1 - probs_norm
    if probs_norm < 1e-8:
        if fraction_loss_mean <= 0.0:
            probs = np.concatenate((np.array([1.0]), np.zeros(len(impact_bins) - 2)))
        elif fraction_loss_mean >= 1.0:
            probs = np.concatenate((np.zeros(len(impact_bins) - 2), np.array([1.0])))
    else:
        probs[0] = probs[0] + prob_differential

    return ImpactDistrib(hazard_type, impact_bins, probs, impact_type)
