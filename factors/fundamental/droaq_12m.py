"""DROAQ 12M factor."""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class DROAQ12M(BaseFactor):
    name = "droaq_12m"
    description = "Year-over-year ROA improvement."

    def __init__(self, year_days: int = 252) -> None:
        self.year_days = year_days

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if "roa" not in data.columns:
            raise ValueError("DROAQ12M missing data column: roa")

        roa = data["roa"].unstack("symbol").astype(float).sort_index()

        signals = roa - roa.shift(self.year_days)
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
