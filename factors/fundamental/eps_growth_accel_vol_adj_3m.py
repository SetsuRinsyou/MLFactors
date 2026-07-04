"""Volatility-adjusted EPS growth acceleration."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EPSGrowthAccelVolAdj3M(BaseFactor):
    name = "eps_growth_accel_vol_adj_3m"
    description = "EAV: current EGV minus previous-quarter EGV."

    def __init__(
        self,
        year_days: int = 252,
        period_days: int = 63,
        quarter_days: int = 63,
        n_quarters: int = 8,
    ) -> None:
        self.year_days = year_days
        self.period_days = period_days
        self.quarter_days = quarter_days
        self.n_quarters = n_quarters

    def generate_signals(self, data: pd.DataFrame, constituents=None) -> pd.DataFrame:
        if "eps" not in data.columns:
            raise ValueError("EPSGrowthAccelVolAdj3M missing data column: eps")

        eps = data["eps"].unstack("symbol").astype(float).sort_index()

        current_quarterly_values = [
            eps.shift(k * self.quarter_days) for k in range(self.n_quarters)
        ]
        current_panel = pd.concat(
            current_quarterly_values,
            axis=1,
            keys=range(self.n_quarters),
        )
        current_std = current_panel.T.groupby(level=1).std(ddof=1).T
        current_std = current_std.where(current_std > 1e-12)
        current_growth = (eps - eps.shift(self.year_days)) / current_std

        previous_eps = eps.shift(self.period_days)
        previous_quarterly_values = [
            previous_eps.shift(k * self.quarter_days)
            for k in range(self.n_quarters)
        ]
        previous_panel = pd.concat(
            previous_quarterly_values,
            axis=1,
            keys=range(self.n_quarters),
        )
        previous_std = previous_panel.T.groupby(level=1).std(ddof=1).T
        previous_std = previous_std.where(previous_std > 1e-12)
        previous_growth = (
            previous_eps - eps.shift(self.period_days + self.year_days)
        ) / previous_std

        signals = current_growth - previous_growth
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
