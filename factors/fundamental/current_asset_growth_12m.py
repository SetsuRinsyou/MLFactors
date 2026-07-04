"""海通54：流动资产增长因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class CurrentAssetGrowth12m(BaseFactor):
    """流动资产同比增长率。"""

    name = "current_asset_growth_12m"
    description = "海通54：流动资产同比增长"

    def __init__(self, min_abs_denominator: float = 1e-12) -> None:
        self.min_abs_denominator = min_abs_denominator
        self.year_days = 21 * 12

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if "current_assets" not in data.columns:
            raise KeyError(f"{self.name} 缺少字段: ['current_assets']")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        current_assets = data["current_assets"].unstack("symbol").astype(float).sort_index()
        lag = current_assets.shift(self.year_days)
        result = current_assets.divide(lag.where(lag > self.min_abs_denominator)) - 1.0

        result.index.name = "date"
        result.columns.name = "symbol"
        return result.replace([np.inf, -np.inf], np.nan)
