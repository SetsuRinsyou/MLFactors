"""日内高低收益率波动率因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


class HighLowVolatility(BaseFactor):
    """日内相对前收盘收益率波动率的公共计算逻辑。"""

    component: str = ""
    expected_lookback: int | None = None

    def __init__(self, lookback: int = 84, min_periods: int | None = None) -> None:
        if self.expected_lookback is not None and lookback != self.expected_lookback:
            raise ValueError(
                f"{self.name} 的 lookback 必须为 {self.expected_lookback}"
            )
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
        required = {"adj_close", "adj_high", "adj_low", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"{self.name} 缺少字段: {sorted(missing)}")

        close = data["adj_close"].unstack("symbol").astype(float).sort_index()
        high = data["adj_high"].unstack("symbol").reindex_like(close).astype(float)
        low = data["adj_low"].unstack("symbol").reindex_like(close).astype(float)
        volume = data["volume"].unstack("symbol").reindex_like(close).astype(float)
        valid_close = (close > 0) & (volume > 0)
        previous_close = close.where(valid_close).shift(1)
        high_return = high.where(valid_close & (high > 0)).div(previous_close).sub(1)
        low_return = low.where(valid_close & (low > 0)).div(previous_close).sub(1)

        high_std = high_return.rolling(
            self.lookback,
            min_periods=self.min_periods,
        ).std()
        if self.component == "high":
            result = high_std
        else:
            low_std = low_return.rolling(
                self.lookback,
                min_periods=self.min_periods,
            ).std()
            result = high_std - low_std

        result.index.name = "date"
        result.columns.name = "symbol"
        return result


@register_factor
class HighRStd84D(HighLowVolatility):
    """过去 84 个交易日的日内最高涨幅标准差。"""

    name = "high_r_std_84d"
    description = "84 日日内最高涨幅波动率"
    component = "high"
    expected_lookback = 84


@register_factor
class HMLRStd84D(HighLowVolatility):
    """日内最高涨幅波动率减最低跌幅波动率。"""

    name = "hml_r_std_84d"
    description = "84 日日内高低收益率波动率之差"
    component = "hml"
    expected_lookback = 84
