"""海通54：总资产增长波动率因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class TotalAssetGrowthVol24m(BaseFactor):
    """总资产同比增长率的 24 个月滚动波动率。"""

    name = "total_asset_growth_vol_24m"
    description = "海通54：总资产增长稳定性，值越低越稳定"

    def __init__(self, min_abs_denominator: float = 1e-12) -> None:
        self.min_abs_denominator = min_abs_denominator
        self.year_days = 21 * 12
        self.window = 21 * 24

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if "total_assets" not in data.columns:
            raise KeyError(f"{self.name} 缺少字段: ['total_assets']")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        total_assets = data["total_assets"].unstack("symbol").astype(float).sort_index()
        lag = total_assets.shift(self.year_days)
        growth = total_assets.divide(lag.where(lag > self.min_abs_denominator)) - 1.0
        result = growth.rolling(self.window, min_periods=self.year_days).std()

        result.index.name = "date"
        result.columns.name = "symbol"
        return result.replace([np.inf, -np.inf], np.nan)
