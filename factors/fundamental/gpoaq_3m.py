"""GPOAQ 3M factor."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class GPOAQ3M(BaseFactor):
    name = "gpoaq_3m"
    description = "Quarterly gross profitability on average total assets."

    def __init__(self, quarter_days: int = 63) -> None:
        self.quarter_days = quarter_days

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"total_assets", "revenue", "cost_revenue"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"GPOAQ3M missing data columns: {sorted(missing)}")

        total_assets = data["total_assets"].unstack("symbol").astype(float).sort_index()
        revenue = data["revenue"].unstack("symbol").astype(float).sort_index()
        cost_revenue = data["cost_revenue"].unstack("symbol").astype(float).sort_index()

        gross_profit = revenue - cost_revenue
        beginning_assets = total_assets.shift(self.quarter_days)
        average_assets = (beginning_assets + total_assets) / 2.0

        signals = gross_profit / average_assets.where(average_assets > 0)
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
