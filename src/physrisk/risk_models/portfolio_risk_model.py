from collections import defaultdict
import statistics
from typing import Optional

from physrisk.api.v1.impact_req_resp import (
    Category,
    RiskMeasureDefinition,
    ScoreBasedRiskMeasureDefinition,
)
from physrisk.kernel.hazards import Hazard
from physrisk.kernel.risk import Measure, MeasureKey, RiskQuantity, RiskQuantityKey


class AveragingAssetBasedPortfolioRiskMeasureCalculator:
    """Calculates portfolio score-based risk measures from asset-level score-based risk measures only."""

    def __init__(self):
        self._definition = ScoreBasedRiskMeasureDefinition(
            hazard_types=[],
            values=[],
            underlying_measures=[
                RiskMeasureDefinition(
                    measure_id="average_asset_score",
                    label="Average asset score.",
                    description=("Average asset score."),
                    units="",
                )
            ],
        )

    def get_definition(self, hazard_type: Optional[type[Hazard]] = None):
        return self._definition

    def calculate_risk_measures(
        self,
        asset_level_measures: dict[MeasureKey, Measure] = {},
        portfolio_quantities: dict[RiskQuantityKey, RiskQuantity] = {},
    ) -> dict[MeasureKey, Measure]:
        portfolio_measures: dict[MeasureKey, Measure] = {}
        measure_by_year_scen: dict[tuple[str, int | None], list[MeasureKey]] = (
            defaultdict(list)
        )
        for mk in asset_level_measures.keys():
            measure_by_year_scen[(mk.prosp_scen, mk.year)].append(mk)
        for k, v in measure_by_year_scen.items():
            # Calculate portfolio score-based risk measures for this year/scenario
            scores = [float(asset_level_measures[m].score) for m in v]
            average_score = statistics.mean(s for s in scores if s > 0)
            portfolio_measures[MeasureKey(None, k[0], k[1], None, None)] = Measure(
                score=Category(round(average_score)),
                measure_0=average_score,
                definition=self._definition,
            )
        return portfolio_measures

    def asset_level_measures_required(self) -> bool:
        return True

    def portfolio_quantities_required(self) -> bool:
        return False
