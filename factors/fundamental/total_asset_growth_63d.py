"""Total asset growth over 63 trading days."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class TotalAssetGrowthFactor63D(BaseFactor):
    """Total asset growth over 63 trading days."""

    name = "total_asset_growth_63d"
    description = "Total asset growth over 63 trading days."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        total_assets = data["total_assets"]
        total_assets_prev = total_assets.groupby(level="symbol").shift(63)
        factor = (total_assets - total_assets_prev) / total_assets_prev.abs().replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
