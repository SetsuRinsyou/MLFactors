"""多期限均线因子（日频版）——基于广发多因子系列34的MA多期限选股策略。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class MultiHorizonMADaily(BaseFactor):
    """多期限均线因子（日频版）：综合多条不同期限MA均线捕捉动量/反转效应。

    与MultiHorizonMA的区别：
    - MultiHorizonMA是周频因子，每周五输出一个信号
    - 本因子是日频因子，每个交易日输出一个信号
    - 用10日滚动回归替代25周滚动回归

    Parameters
    ----------
    periods : list[int], default None
        均线期限列表，默认为 [3,5,10,20,30,60,90,120,180,240,270,300]。
    lookback_days : int, default 50
        回归系数计算的回溯天数，默认50天（约10周）。
    """

    name = "multi_horizon_ma_50d"
    description = "多期限均线因子（日频版）"

    def __init__(
        self,
        periods: list[int] | None = None,
        lookback_days: int = 50,
    ) -> None:
        self.periods = periods or [3, 5, 10, 20, 30, 60, 90, 120, 180, 240, 270, 300]
        self.lookback_days = lookback_days

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        close = data["close"].unstack("symbol").astype(float).sort_index()

        # 计算多期限MA均线
        ma_dict = {}
        for period in self.periods:
            ma_dict[period] = close.rolling(period, min_periods=period).mean()

        # 标准化因子: Ã = MA_L / P
        a_dict = {}
        for period in self.periods:
            a_dict[period] = ma_dict[period] / close

        # 计算日收益率
        daily_ret = close.pct_change(fill_method=None)

        # 初始化因子结果矩阵
        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        # 滚动回归预测（日频）
        for i in range(self.lookback_days, len(close.index)):
            current_date = close.index[i]

            # 获取过去lookback_days天的收益率和因子值
            hist_dates = close.index[i - self.lookback_days:i]

            # 对每只股票计算回归系数
            for symbol in close.columns:
                # 获取该股票的历史收益率
                y = daily_ret.loc[hist_dates, symbol].values

                # 获取历史因子值 (各期限的Ã值)
                X_list = []
                for period in self.periods:
                    X_list.append(a_dict[period].loc[hist_dates, symbol].values)
                X = np.column_stack(X_list)

                # 去除NaN
                valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                if valid.sum() < self.lookback_days * 0.5:
                    continue

                y_valid = y[valid]
                X_valid = X[valid]

                # 回归: r = β0 + Σ βi * Ãi
                X_with_const = np.column_stack([np.ones(len(X_valid)), X_valid])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_with_const, y_valid, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                # 用最新因子值计算预期收益率
                latest_a = np.array([
                    a_dict[period].at[current_date, symbol]
                    if pd.notna(a_dict[period].at[current_date, symbol])
                    else np.nan
                    for period in self.periods
                ])

                if np.any(np.isnan(latest_a)):
                    continue

                # E[r] = Σ βi * Ãi (不含常数项的贡献)
                expected_return = np.dot(beta[1:], latest_a)
                signals.at[current_date, symbol] = expected_return

        signals.index.name = "date"
        signals.columns.name = "symbol"
        return signals
