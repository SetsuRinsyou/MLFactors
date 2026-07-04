"""ROE growth over 63 trading days."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class RoeGrowthFactor63D(BaseFactor):
    """ROE growth over 63 trading days."""

    name = "roe_growth_63d"
    description = "ROE growth over 63 trading days."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        roe = data["roe"]
        roe_prev = roe.groupby(level="symbol").shift(63)
        factor = (roe - roe_prev) / roe_prev.abs().replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
