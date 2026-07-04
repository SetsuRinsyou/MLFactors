"""海通54：资产负债率变化因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class DebtToAssetChange12m(BaseFactor):
    """资产负债率的同比变化。"""

    name = "debt_to_asset_change_12m"
    description = "海通54：资产负债率同比变化"

    def __init__(self, min_abs_denominator: float = 1e-12) -> None:
        self.min_abs_denominator = min_abs_denominator
        self.year_days = 21 * 12

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"total_liabilities", "total_assets"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"{self.name} 缺少字段: {sorted(missing)}")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        total_liabilities = data["total_liabilities"].unstack("symbol").astype(float).sort_index()
        total_assets = data["total_assets"].unstack("symbol").astype(float).reindex_like(total_liabilities)
        ratio = total_liabilities.divide(total_assets.where(total_assets > self.min_abs_denominator))
        result = ratio - ratio.shift(self.year_days)

        result.index.name = "date"
        result.columns.name = "symbol"
        return result.replace([np.inf, -np.inf], np.nan)
