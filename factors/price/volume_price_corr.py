"""量价相关性因子——基于海通选股因子系列研究12的量价结合策略。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class VolumePriceCorr(BaseFactor):
    """量价相关性因子：捕捉量价背离/同向对收益的预测能力。

    核心思想：放量上涨的股票在存量资金博弈市场中难以持续，
    未来大概率下跌；放量下跌的股票风险已释放，未来表现更好。
    通过计算收盘价与换手率的Pearson相关系数来衡量量价关系。

    Parameters
    ----------
    window : int, default 10
        计算相关系数的时间窗口（交易日），默认10天（半个月）。
    """

    name = "volume_price_corr_15d"
    description = "量价相关性因子"

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        close = data["adj_close"].unstack("symbol").astype(float).sort_index()
        volume = data["volume"].unstack("symbol").astype(float).sort_index()
        market_cap = data["market_cap"].unstack("symbol").astype(float).sort_index()

        # 计算近似换手率: turnover = volume × close / market_cap
        turnover = volume * close / market_cap

        # 计算滚动相关系数
        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        for symbol in close.columns:
            close_col = close[symbol]
            turnover_col = turnover[symbol]

            # 滚动计算Pearson相关系数
            corr = close_col.rolling(window=self.window).corr(turnover_col)
            signals[symbol] = corr

        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
