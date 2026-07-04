"""Inventory turnover 1D factor."""
from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class InventoryTurnoverFactor1D(BaseFactor):
    """Inventory turnover 1D factor."""

    name = "inventory_turnover_1d"
    description = "Inventory turnover 1D factor."

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor = data["cost_revenue"] / data["inventory"].replace(0, pd.NA)

        result = factor.unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
