"""双均线交叉择时因子。

信号逻辑：
- 快线（MA_fast）在慢线（MA_slow）上方 → +1（多头方向）
- 快线在慢线下方 → -1（空头方向）
- 快线等于慢线 → 0（空仓，极少发生）
"""

from __future__ import annotations

import pandas as pd

from data.schema import Col
from factors.base import BaseTimingFactor
from factors.registry import register_factor


@register_factor
class MACrossTiming(BaseTimingFactor):
    """双均线交叉择时因子（快线/慢线）。

    Parameters
    ----------
    fast : 快线周期（默认 5 日）
    slow : 慢线周期（默认 20 日）
    """

    name = "ma_cross_timing"
    description = "双均线交叉择时（快线在慢线上方=+1多头，快线在慢线下方=-1空头）"
    category = "timing"

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        self.fast = fast
        self.slow = slow

    def compute_timing(self, market_data: pd.DataFrame, symbol: str) -> pd.Series:
        close = self._get_symbol_close(market_data, symbol)

        ma_fast = close.rolling(self.fast, min_periods=self.fast).mean()
        ma_slow = close.rolling(self.slow, min_periods=self.slow).mean()

        diff = ma_fast - ma_slow
        signal = pd.Series(0.0, index=close.index, name=self.name)
        signal[diff > 0] = 1.0
        signal[diff < 0] = -1.0

        return signal

