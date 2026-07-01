"""EPS growth over 12 months."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EPSGrowth12M(BaseFactor):
    name = "eps_growth_12m"
    description = "EPS year-over-year growth: (EPS_t - EPS_t-12m) / abs(EPS_t-12m)."

    def __init__(self, year_days: int = 252) -> None:
        self.year_days = year_days

    def generate_signals(self, data: pd.DataFrame, constituents=None) -> pd.DataFrame:
        if "eps" not in data.columns:
            raise ValueError("EPSGrowth12M missing data column: eps")

        eps = data["eps"].unstack("symbol").astype(float).sort_index()
        prior_eps = eps.shift(self.year_days)
        denominator = prior_eps.abs().where(prior_eps.abs() > 1e-12)

        signals = (eps - prior_eps) / denominator
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
