"""海通18价格形态：盘低回升因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor

@register_factor
class DipRebound10(BaseFactor):
    """过去 10 日从日内最低价到收盘价的平均回升幅度。"""

    name = "dip_rebound_10d"
    description = "海通18价格形态因子：10日平均 log(close/low)"

    def __init__(
        self,
        window: int = 10,
        min_periods: int = 5,
        shift_days: int = 0,
    ) -> None:
        self.window = window
        self.min_periods = min_periods
        self.shift_days = shift_days

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        """计算最近 window 个交易日 log(close / low) 的均值。"""
        if "low" not in data.columns:
            raise ValueError("字段 low 缺失")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")
        if "close" not in data.columns:
            raise ValueError("字段 close 缺失")

        low = data["low"].unstack("symbol").astype(float).sort_index()
        close = data["close"].unstack("symbol").astype(float).sort_index()
        close = close.reindex(index=low.index, columns=low.columns)
        valid = (low > 0) & (close > 0) & (close >= low)
        daily_value = np.log(close.where(valid) / low.where(valid))
        factor = daily_value.rolling(
            window=self.window,
            min_periods=self.min_periods,
        ).mean()
        if self.shift_days:
            factor = factor.shift(self.shift_days)

        factor.index.name = "date"
        factor.columns.name = "symbol"
        if not constituents:
            return factor

        mask = pd.DataFrame(False, index=factor.index, columns=factor.columns)
        for date in factor.index:
            allowed = constituents.get(str(pd.Timestamp(date).date()), set())
            if allowed:
                present = factor.columns.intersection(allowed)
                mask.loc[date, present] = True
        return factor.where(mask)
