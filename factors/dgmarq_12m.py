"""DGMARQ 12M factor."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class DGMARQ12M(BaseFactor):
    name = "dgmarq_12m"
    description = "Year-over-year gross profit change divided by prior revenue."

    def __init__(self, year_days: int = 252) -> None:
        self.year_days = year_days

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"revenue", "cost_revenue"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"DGMARQ12M missing data columns: {sorted(missing)}")

        revenue = data["revenue"].unstack("symbol").astype(float).sort_index()
        cost_revenue = data["cost_revenue"].unstack("symbol").astype(float).sort_index()

        gross_profit = revenue - cost_revenue
        prior_gross_profit = gross_profit.shift(self.year_days)
        prior_revenue = revenue.shift(self.year_days)

        signals = (
            gross_profit - prior_gross_profit
        ) / prior_revenue.where(prior_revenue > 0)
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
