"""RSI 择时因子。

信号逻辑（超买超卖反转策略）：
- RSI < oversold（默认 30）→ +1（超卖，多头方向）
- RSI > overbought（默认 70）→ -1（超买，空头方向）
- 其余 → 0（空仓）
"""

from __future__ import annotations

import pandas as pd

from data.schema import Col
from factors.base import BaseTimingFactor
from factors.registry import register_factor


@register_factor
class RSITiming(BaseTimingFactor):
    """RSI 超买超卖择时因子。

    Parameters
    ----------
    period : RSI 计算周期（默认 14 日）
    overbought : 超买阈值（默认 70）
    oversold : 超卖阈值（默认 30）
    """

    name = "rsi_timing"
    description = "RSI 超买超卖择时（RSI<oversold=+1多头，RSI>overbought=-1空头，其余=0空仓）"
    category = "timing"

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> None:
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def compute_timing(self, market_data: pd.DataFrame, symbol: str) -> pd.Series:
        close = self._get_symbol_close(market_data, symbol)
        rsi = self._calc_rsi(close)

        signal = pd.Series(0.0, index=close.index, name=self.name)
        signal[rsi < self.oversold] = 1.0    # 超卖 → 多头
        signal[rsi > self.overbought] = -1.0  # 超买 → 空头

        return signal

    # ------------------------------------------------------------------ #
    #  内部工具
    # ------------------------------------------------------------------ #

    def _calc_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # 指数加权移动平均（Wilder 平滑）
        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

        # 避免除以0：avg_loss=0 时 RS 设为无穷大，RSI→100
        rs = avg_gain / avg_loss.replace(0.0, float("inf"))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.rename("rsi")

    @staticmethod
    def _extract_close(market_data: pd.DataFrame, symbol: str) -> pd.Series:
        # Kept for backward compatibility — delegates to base class helper
        from data.schema import Col
        return BaseTimingFactor._get_symbol_close(market_data, symbol, Col.CLOSE)
