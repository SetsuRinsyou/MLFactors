"""Price-adjusted EPS growth acceleration."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EPSGrowthAccelPriceAdj3M(BaseFactor):
    name = "eps_growth_accel_price_adj_3m"
    description = "EAP: current price-adjusted EPS growth minus previous-quarter value."

    def __init__(self, year_days: int = 252, period_days: int = 63) -> None:
        self.year_days = year_days
        self.period_days = period_days

    def generate_signals(self, data: pd.DataFrame, constituents=None) -> pd.DataFrame:
        required = {"eps", "adj_close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                f"EPSGrowthAccelPriceAdj3M missing data columns: {sorted(missing)}"
            )

        eps = data["eps"].unstack("symbol").astype(float).sort_index()
        close = data["adj_close"].unstack("symbol").astype(float).sort_index()

        current_growth = (
            eps - eps.shift(self.year_days)
        ) / close.shift(self.period_days).where(close.shift(self.period_days) > 0)

        previous_growth = (
            eps.shift(self.period_days)
            - eps.shift(self.period_days + self.year_days)
        ) / close.shift(2 * self.period_days).where(
            close.shift(2 * self.period_days) > 0
        )

        signals = current_growth - previous_growth
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
