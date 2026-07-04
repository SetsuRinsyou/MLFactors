"""Revenue growth over 63 trading days."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class RevenueGrowthFactor63D(BaseFactor):
    """Revenue growth over 63 trading days."""

    name = "revenue_growth_63d"
    description = "Revenue growth over 63 trading days."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        revenue = data["revenue"]
        revenue_prev = revenue.groupby(level="symbol").shift(63)
        factor = (revenue - revenue_prev) / revenue_prev.abs().replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
