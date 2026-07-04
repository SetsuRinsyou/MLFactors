"""Shareholder equity growth over 63 trading days."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EquityGrowthFactor63D(BaseFactor):
    """Shareholder equity growth over 63 trading days."""

    name = "equity_growth_63d"
    description = "Shareholder equity growth over 63 trading days."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        equity = data["shareholder_equity"]
        equity_prev = equity.groupby(level="symbol").shift(63)
        factor = (equity - equity_prev) / equity_prev.abs().replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
