"""Volatility-adjusted EPS growth."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EPSGrowthVolAdj3M(BaseFactor):
    name = "eps_growth_vol_adj_3m"
    description = "EGV: EPS year-over-year change divided by recent EPS standard deviation."

    def __init__(
        self,
        year_days: int = 252,
        quarter_days: int = 63,
        n_quarters: int = 8,
    ) -> None:
        self.year_days = year_days
        self.quarter_days = quarter_days
        self.n_quarters = n_quarters

    def generate_signals(self, data: pd.DataFrame, constituents=None) -> pd.DataFrame:
        if "eps" not in data.columns:
            raise ValueError("EPSGrowthVolAdj3M missing data column: eps")

        eps = data["eps"].unstack("symbol").astype(float).sort_index()

        quarterly_values = [
            eps.shift(k * self.quarter_days) for k in range(self.n_quarters)
        ]
        quarterly_panel = pd.concat(
            quarterly_values,
            axis=1,
            keys=range(self.n_quarters),
        )
        eps_std = quarterly_panel.T.groupby(level=1).std(ddof=1).T
        eps_std = eps_std.where(eps_std > 1e-12)

        signals = (eps - eps.shift(self.year_days)) / eps_std
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
