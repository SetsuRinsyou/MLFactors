"""Price-adjusted EPS growth."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EPSGrowthPriceAdj3M(BaseFactor):
    name = "eps_growth_price_adj_3m"
    description = "EGP: EPS year-over-year change divided by prior-period stock price."

    def __init__(self, year_days: int = 252, period_days: int = 63) -> None:
        self.year_days = year_days
        self.period_days = period_days

    def generate_signals(self, data: pd.DataFrame, constituents=None) -> pd.DataFrame:
        required = {"eps", "adj_close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"EPSGrowthPriceAdj3M missing data columns: {sorted(missing)}")

        eps = data["eps"].unstack("symbol").astype(float).sort_index()
        close = data["adj_close"].unstack("symbol").astype(float).sort_index()

        prior_year_eps = eps.shift(self.year_days)
        prior_period_price = close.shift(self.period_days).where(
            close.shift(self.period_days) > 0
        )

        signals = (eps - prior_year_eps) / prior_period_price
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
