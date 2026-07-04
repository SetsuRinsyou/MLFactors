"""Long-term debt ratio 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class LongTermDebtRatioFactor1D(BaseFactor):
    """Long-term debt ratio 1D factor."""

    name = "long_term_debt_ratio_1d"
    description = "Long-term debt ratio 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = data["non_current_debt"] / data["total_assets"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
