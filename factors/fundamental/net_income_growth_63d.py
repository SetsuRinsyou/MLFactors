"""Net income growth over 63 trading days."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class NetIncomeGrowthFactor63D(BaseFactor):
    """Net income growth over 63 trading days."""

    name = "net_income_growth_63d"
    description = "Net income growth over 63 trading days."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        net_income = data["net_income"]
        net_income_prev = net_income.groupby(level="symbol").shift(63)
        factor = (net_income - net_income_prev) / net_income_prev.abs().replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
