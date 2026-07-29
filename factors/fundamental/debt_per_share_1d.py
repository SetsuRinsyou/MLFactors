"""Debt per share 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class DebtPerShareFactor1D(BaseFactor):
    """Debt per share 1D factor."""

    name = "debt_per_share_1d"
    description = "Debt per share 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = data["total_debt"] / data["shares_basic"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
