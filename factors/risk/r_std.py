"""收益率波动率因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class RStd84D(BaseFactor):
    """过去 84 个交易日的个股收益率标准差。"""

    name = "r_std_84d"
    description = "84 日收益率波动率"

    def __init__(self, lookback: int = 84, min_periods: int | None = None) -> None:
        if lookback != 84:
            raise ValueError("r_std_84d 的 lookback 必须为 84")
        if min_periods is None:
            min_periods = int(np.ceil(lookback * 0.8))
        if lookback <= 0 or min_periods <= 0 or min_periods > lookback:
            raise ValueError("min_periods 必须在 1 到 lookback 之间")

        self.lookback = lookback
        self.min_periods = min_periods

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"adj_close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"r_std_84d 缺少字段: {sorted(missing)}")

        close = data["adj_close"].unstack("symbol").astype(float).sort_index()
        volume = data["volume"].unstack("symbol").reindex_like(close).astype(float)
        close = close.where((close > 0) & (volume > 0))
        returns = close.pct_change(fill_method=None)
        result = returns.rolling(
            self.lookback,
            min_periods=self.min_periods,
        ).std()
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
