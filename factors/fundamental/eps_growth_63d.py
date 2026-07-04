"""EPS growth over 63 trading days."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EpsGrowthFactor63D(BaseFactor):
    """EPS growth over 63 trading days."""

    name = "eps_growth_63d"
    description = "EPS growth over 63 trading days."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        eps = data["eps"]
        eps_prev = eps.groupby(level="symbol").shift(63)
        factor = (eps - eps_prev) / eps_prev.abs().replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
