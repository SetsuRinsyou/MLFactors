"""DGPOAQ 12M factor."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class DGPOAQ12M(BaseFactor):
    name = "dgpoaq_12m"
    description = "Year-over-year gross profit change divided by prior total assets."

    def __init__(self, year_days: int = 252) -> None:
        self.year_days = year_days

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"total_assets", "revenue", "cost_revenue"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"DGPOAQ12M missing data columns: {sorted(missing)}")

        total_assets = data["total_assets"].unstack("symbol").astype(float).sort_index()
        revenue = data["revenue"].unstack("symbol").astype(float).sort_index()
        cost_revenue = data["cost_revenue"].unstack("symbol").astype(float).sort_index()

        gross_profit = revenue - cost_revenue
        prior_gross_profit = gross_profit.shift(self.year_days)
        prior_total_assets = total_assets.shift(self.year_days)

        signals = (
            gross_profit - prior_gross_profit
        ) / prior_total_assets.where(prior_total_assets > 0)
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
