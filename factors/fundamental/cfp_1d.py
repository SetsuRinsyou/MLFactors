"""Cash-flow-to-price proxy 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class CfpFactor1D(BaseFactor):
    """Cash-flow-to-price proxy 1D factor."""

    name = "cfp_1d"
    description = "Cash-flow-to-price proxy 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = data["operating_cash_flow"] / data["market_cap"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
