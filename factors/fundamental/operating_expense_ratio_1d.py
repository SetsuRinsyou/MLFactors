"""Operating expense ratio 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class OperatingExpenseRatioFactor1D(BaseFactor):
    """Operating expense ratio 1D factor."""

    name = "operating_expense_ratio_1d"
    description = "Operating expense ratio 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = data["operating_expense"] / data["revenue"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
