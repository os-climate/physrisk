from typing import NamedTuple

import numpy as np
from typing_extensions import Protocol

from physrisk.kernel.assets import Asset
from physrisk.kernel.financial_model import FinancialDataProvider
from physrisk.kernel.hazards import Hazard
from physrisk.kernel.risk import QuantityType


class SectoralInsuranceData(NamedTuple):
    uptake: float  # statistical insurance uptake: probability that asset in sector is insured
    deductable: float  # deductable as a fraction of total insured value
    limit: float  # limit as a fraction of total insured value


class InsuranceDataProvider(Protocol):
    def __call__(
        self, asset: Asset, hazard_type: type[Hazard], impact_type: QuantityType
    ) -> SectoralInsuranceData:
        """Provide insurance data, including state support.

        Args:
            asset (Asset): Asset
            hazard_type (type[Hazard]): Hazard type
            impact_type (QuantityType): `QuantityType.DAMAGE` or `QuantityType.REVENUE_LOSS`
        """
        ...


class InsuranceModel(Protocol):
    """ "Insurance Model using a FinancialDataProvider as source of information."""

    @property
    def financial_data_provider(self) -> FinancialDataProvider:
        """Get the financial data provider."""

    def claims_payment_from_restoration_cost(
        self,
        sectoral_insurance_data: SectoralInsuranceData,
        restoration_cost: np.ndarray,
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
