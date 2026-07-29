"""Total asset turnover 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class TotalAssetTurnoverFactor1D(BaseFactor):
    """Total asset turnover 1D factor."""

    name = "total_asset_turnover_1d"
    description = "Total asset turnover 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = data["revenue"] / data["total_assets"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
