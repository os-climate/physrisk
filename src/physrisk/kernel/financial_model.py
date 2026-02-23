from typing import Protocol, Sequence

import numpy as np

from physrisk.vulnerability_models.config_based_impact_curves import DowntimeConfigItem
from physrisk.vulnerability_models.downtime import DowntimeModels

from .assets import Asset


class FinancialDataProvider(Protocol):
    """Financial data sufficient to convert impacts into financial quantities that can be aggregated.

    Damage, specified as a fraction of total insurable value (TIV), is multiplied by TIV in order to
    compute the cost of asset restoration in the currency specified -- which can then be aggregated across assets.

    Disruption, specified as a fraction of annual revenue, is multiplied by the portion of the annual revenue that
    is attributable to that asset: that which would be lost if the asset were completely non-productive.

    Dealing with productivity loss is more complex, as a decrease in worker productivity does not necessarily result in a
    proportionate loss in revenue. It is assumed, however, that productivity loss models (e.g. related to heat), provide
    figures that are already adjusted for this effect, and can be multiplied by annual revenue in order to aggregate.

    What might this adjustment look like? Possibly the simplest approach is to account for the fact that labour costs are only
    a fraction of value-added, which is itself a fraction of revenue, i.e.:

    Revenue = Intermediate inputs + Value added
    Value added = Labour costs + EBITDA
    """

    def revenue_attributable_to_asset(self, asset: Asset, currency: str) -> float:
        """Annual revenue that would be lost if the asset were completely non-productive.

        Args:
            asset (Asset): Asset.
            currency (str): Currency (3-letter code).

        Returns:
            float: Revenue attributable to the asset in specified currency.
        """
        ...

    def total_insurable_value(self, asset: Asset, currency: str) -> float:
        """Total insurable value of the asset.

        Args:
            asset (Asset): Asset.
            currency (str): Currency (3-letter code).

        Returns:
            float: Total insurable value of the asset in specified currency.
        """
        ...


class FinancialModel(Protocol):
    """ "Financial Model using a FinancialDataProvider as source of information."""

    def financial_data_provider(self) -> FinancialDataProvider:
        """Get the financial data provider.

        Returns:
            FinancialDataProvider: Financial data provider.
        """
        ...

    def frac_damage_to_restoration_cost_and_revenue_disruption(
        self, asset: Asset, impact: np.ndarray, currency: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert damage, specified as a fraction of total insurable value, to cost
        of asset restoration and loss of revenue during downtime.

        Args:
            asset (Asset): Asset.
            impact (np.ndarray): Damage as a fraction of total insurable value.
            currency (str): Currency (3-letter code).

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Cost of asset restoration in specified currency.
                - Annual loss of revenue from downtime in specified currency.
        """

    def frac_disruption_to_revenue_loss(
        self, asset: Asset, impact: np.ndarray, year: int, currency: str
    ):
        """Convert disruption, specified as a fraction of annual revenue, to loss of
        revenue.

        Args:
            asset (Asset): Asset.
            impact (np.ndarray): Disruption as a fraction of annual revenue.
            year (int): Year of disruption.
            currency (str): Currency (3-letter code).
        """
        ...


class DefaultFinancialModel(FinancialModel):
    """ "Financial Model using a FinancialDataProvider as source of information."""

    def __init__(
        self,
        data_provider: FinancialDataProvider,
        downtime_config: Sequence[DowntimeConfigItem],
    ):
        self.data_provider = data_provider
        self.downtime_config = DowntimeModels(downtime_config)

    def frac_damage_to_restoration_cost_and_revenue_disruption(
        self, asset: Asset, impact: np.ndarray, currency: str
    ):
        damage = self.data_provider.total_insurable_value(asset, currency) * impact
        revenue_loss = self._downtime_to_revenue_loss(asset, impact, currency)
        return damage, revenue_loss

    def frac_disruption_to_revenue_loss(
        self, asset: Asset, impact: np.ndarray, year: int, currency: str
    ):
        return (
            self.data_provider.revenue_attributable_to_asset(asset, currency) * impact
        )

    def _downtime_to_revenue_loss(
        self, asset: Asset, impact: np.ndarray, currency: str
    ):
        downtime_model = self.downtime_config.downtime_model_for_asset_of_type(
            type(asset)
        )
        if downtime_model and len(downtime_model) == 1:
            return downtime_model[0].get_impact(
                asset, impact
            ) * self.data_provider.revenue_attributable_to_asset(asset, currency)
        else:
            return np.zeros_like(impact)
