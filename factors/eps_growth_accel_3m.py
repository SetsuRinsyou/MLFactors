"""EPS growth acceleration over one quarter."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EPSGrowthAccel3M(BaseFactor):
    name = "eps_growth_accel_3m"
    description = "EAA: current EPS YoY growth minus previous-quarter EPS YoY growth."

    def __init__(self, year_days: int = 252, period_days: int = 63) -> None:
        self.year_days = year_days
        self.period_days = period_days

    def generate_signals(self, data: pd.DataFrame, constituents=None) -> pd.DataFrame:
        if "eps" not in data.columns:
            raise ValueError("EPSGrowthAccel3M missing data column: eps")

        eps = data["eps"].unstack("symbol").astype(float).sort_index()

        prior_year_eps = eps.shift(self.year_days)
        current_growth = (eps - prior_year_eps) / prior_year_eps.abs().where(
            prior_year_eps.abs() > 1e-12
        )

        previous_eps = eps.shift(self.period_days)
        previous_year_eps = eps.shift(self.period_days + self.year_days)
        previous_growth = (
            previous_eps - previous_year_eps
        ) / previous_year_eps.abs().where(previous_year_eps.abs() > 1e-12)

        signals = current_growth - previous_growth
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
