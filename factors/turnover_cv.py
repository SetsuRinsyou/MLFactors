"""换手率变异系数因子——基于海通选股因子系列研究14的交易行为波动策略。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class TurnoverCV(BaseFactor):
    """换手率变异系数因子：捕捉交易行为波动对收益的预测能力。

    核心思想：成交量波动大的股票，说明投资者分歧大，不确定性高，
    根据流动性溢价理论，这种股票需要更高的风险补偿，预期收益更低。
    通过计算过去20日换手率的变异系数（标准差/均值）来衡量交易行为波动性。

    Parameters
    ----------
    window : int, default 20
        计算变异系数的时间窗口（交易日），默认20天。
    min_periods : int, default 15
        最少需要的交易日数量，默认15天。
    """

    name = "turnover_cv_20d"
    description = "换手率变异系数因子"

    def __init__(self, window: int = 20, min_periods: int = 15) -> None:
        self.window = window
        self.min_periods = min_periods

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        close = data["close"].unstack("symbol").astype(float).sort_index()
        volume = data["volume"].unstack("symbol").astype(float).sort_index()
        market_cap = data["market_cap"].unstack("symbol").astype(float).sort_index()

        # 计算近似换手率: turnover = volume × close / market_cap
        turnover = volume * close / market_cap

        # 计算滚动变异系数: CV = std / mean
        rolling_mean = turnover.rolling(window=self.window, min_periods=self.min_periods).mean()
        rolling_std = turnover.rolling(window=self.window, min_periods=self.min_periods).std()

        # 变异系数
        cv = rolling_std / rolling_mean

        # 因子值 = CV（CV越大，未来收益越低）
        signals = cv.copy()

        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
