"""行业基础量价特征因子"""

import pandas as pd
from factors.base import BaseFactor
from factors.registry import register_factor

@register_factor
class Reversal1M(BaseFactor):
    name = "reversal_1m"
    description = "一个月股价反转"

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        close = data["close"].unstack("symbol").astype(float).sort_index()
        # 过去20日涨幅取负
        return -close.pct_change(20, fill_method=None)

@register_factor
class Reversal3M(BaseFactor):
    name = "reversal_3m"
    description = "三个月股价反转"

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        close = data["close"].unstack("symbol").astype(float).sort_index()
        # 过去60日涨幅取负
        return -close.pct_change(60, fill_method=None)

@register_factor
class Reversal6M(BaseFactor):
    name = "reversal_6m"
    description = "六个月股价反转"

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        close = data["close"].unstack("symbol").astype(float).sort_index()
        # 过去120日涨幅取负
        return -close.pct_change(120, fill_method=None)

@register_factor
class Volume3M(BaseFactor):
    name = "volume_3m"
    description = "最近3月成交量均值"

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        volume = data["volume"].unstack("symbol").astype(float).sort_index()
        # 过去60天平均成交量
        return volume.rolling(60, min_periods=20).mean()

@register_factor
class TradingAmount(BaseFactor):
    name = "trading_amount_20d"
    description = "20日成交金额均值"

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        close = data["close"].unstack("symbol").astype(float).sort_index()
        volume = data["volume"].unstack("symbol").astype(float).sort_index()
        
        amount = close * volume
        # 过去20天平均成交金额
        return amount.rolling(20, min_periods=5).mean()