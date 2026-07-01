"""Sales-to-price proxy 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class SpFactor1D(BaseFactor):
    """Sales-to-price proxy 1D factor."""

    name = "sp_1d"
    description = "Sales-to-price proxy 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = data["market_cap"] / data["revenue"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
