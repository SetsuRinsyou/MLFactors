"""波动率因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class Volatility20(BaseFactor):
    name = "volatility_20"
    description = "20日收益率标准差"
    category = "risk"

    def compute(self, market_data: pd.DataFrame, fundamental_data=None) -> pd.Series:
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        ret = close.pct_change()
        vol = ret.rolling(20).std()
        return vol.stack().rename(self.name)


@register_factor
class Volatility5(BaseFactor):
    name = "volatility_5"
    description = "5日收益率标准差"
    category = "risk"

    def compute(self, market_data: pd.DataFrame, fundamental_data=None) -> pd.Series:
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        ret = close.pct_change()
        vol = ret.rolling(5).std()
        return vol.stack().rename(self.name)


@register_factor
class HighLowSpread20(BaseFactor):
    name = "highlow_spread_20"
    description = "20日最高最低价振幅均值"
    category = "risk"

    def compute(self, market_data: pd.DataFrame, fundamental_data=None) -> pd.Series:
        high = market_data[Col.HIGH].unstack(Col.SYMBOL)
        low = market_data[Col.LOW].unstack(Col.SYMBOL)
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        spread = (high - low) / close
        avg_spread = spread.rolling(20).mean()
        return avg_spread.stack().rename(self.name)
