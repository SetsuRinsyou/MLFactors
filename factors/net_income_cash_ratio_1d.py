"""Operating cash flow divided by net income 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class NetIncomeCashRatioFactor1D(BaseFactor):
    """Operating cash flow divided by net income 1D factor."""

    name = "net_income_cash_ratio_1d"
    description = "Operating cash flow divided by net income 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = data["operating_cash_flow"] / data["net_income"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
