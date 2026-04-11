"""因子评估汇总报告。"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from evaluation.ic import (
    calc_forward_returns,
    calc_ic_series,
    calc_icir,
    calc_t_stat,
    calc_turnover,
    calc_ic_decay,
)
from evaluation.layered import LayeredResult, layered_backtest


class FactorReport:
    """聚合全部评估指标，生成因子测试报告。

    Parameters
    ----------
    factor_values : 因子值 pd.Series，MultiIndex(date, symbol)
    market_data : 行情数据 pd.DataFrame，MultiIndex(date, symbol)
    forward_periods : 前向收益计算周期列表
    ic_method : "rank" (Spearman) 或 "pearson"
    n_groups : 分层回测分组数
    open_col : 开盘价列名，用于计算前向收益（T+1 开盘买入）
    """

    def __init__(
        self,
        factor_values: pd.Series,
        market_data: pd.DataFrame,
        forward_periods: list[int] | None = None,
        ic_method: str = "rank",
        n_groups: int = 5,
        open_col: str = "open"
    ) -> None:
        self.factor_values = factor_values
        self.market_data = market_data
        self.forward_periods = forward_periods or [1, 5, 10, 20]
        self.ic_method = ic_method
        self.n_groups = n_groups
        self.open_col = open_col

        # 计算前向收益（开盘价口径，T+1 开盘买入，统一用于 IC/ICIR 及分层回测）
        self._fwd_returns = calc_forward_returns(
            market_data, self.forward_periods, self.open_col
        )

        # 缓存结果
        self._ic_series: dict[int, pd.Series] = {}
        self._layered: dict[int, LayeredResult] = {}
        self._summary: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    #  IC 相关
    # ------------------------------------------------------------------ #

    def ic_series(self, period: int = 5) -> pd.Series:
        if period not in self._ic_series:
            self._ic_series[period] = calc_ic_series(
                self.factor_values, self._fwd_returns[period], self.ic_method
            )
        return self._ic_series[period]

    def icir(self, period: int = 5) -> float:
        return calc_icir(self.ic_series(period))

    def t_stat(self, period: int = 5) -> tuple[float, float]:
        return calc_t_stat(self.ic_series(period))

    def turnover(self) -> pd.Series:
        return calc_turnover(self.factor_values)

    def ic_decay(self, max_lag: int = 20) -> pd.Series:
        return calc_ic_decay(
            self.factor_values,
            self._fwd_returns.get(1, next(iter(self._fwd_returns.values()))),
            max_lag=max_lag,
            method=self.ic_method,
        )

    # ------------------------------------------------------------------ #
    #  分层回测
    # ------------------------------------------------------------------ #

    def layered(self, period: int = 5) -> LayeredResult:
        if period not in self._layered:
            self._layered[period] = layered_backtest(
                self.factor_values, self._fwd_returns[period], self.n_groups
            )
        return self._layered[period]

    # ------------------------------------------------------------------ #
    #  汇总
    # ------------------------------------------------------------------ #

    def summary(self) -> pd.DataFrame:
        """生成各周期的汇总指标表。"""
        if self._summary is not None:
            return self._summary

        records = []
        turnover_mean = float(self.turnover().mean())

        for p in self.forward_periods:
            ic_s = self.ic_series(p)
            ic_mean = ic_s.mean()
            ic_std = ic_s.std()
            icir = calc_icir(ic_s)
            t, pval = calc_t_stat(ic_s)
            ic_positive_ratio = (ic_s > 0).mean()

            lr = self.layered(p)

            records.append({
                "period": p,
                "IC_mean": round(ic_mean, 4),
                "IC_std": round(ic_std, 4),
                "ICIR": round(icir, 4),
                "t_stat": round(t, 4),
                "p_value": round(pval, 6),
                "IC>0_ratio": round(ic_positive_ratio, 4),
                "turnover": round(turnover_mean, 4),
                "LS_annual_ret": round(lr.long_short_annual, 4),
                "LS_sharpe": round(lr.long_short_sharpe, 4),
                "LS_max_drawdown": round(lr.long_short_max_drawdown, 4),
                "top_excess_annual": round(lr.top_excess_annual, 4),
                "top_excess_max_dd": round(lr.top_excess_max_drawdown, 4),
                "top_excess_calmar": round(lr.top_excess_calmar, 4),
            })

        self._summary = pd.DataFrame(records).set_index("period")
        return self._summary

    def to_dict(self) -> dict:
        return self.summary().to_dict(orient="index")

    def print(self) -> None:
        logger.info("=" * 60)
        logger.info("因子评估报告: {}", getattr(self.factor_values, "name", "unknown"))
        logger.info("=" * 60)
        summary = self.summary()
        print(summary.to_string())
        logger.info("-" * 60)
