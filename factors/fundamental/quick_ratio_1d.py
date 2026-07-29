"""Quick ratio 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class QuickRatioFactor1D(BaseFactor):
    """Quick ratio 1D factor."""

    name = "quick_ratio_1d"
    description = "Quick ratio 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = (data["current_assets"] - data["inventory"]) / data["current_liabilities"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
