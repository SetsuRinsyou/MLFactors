"""qlib 回测适配器。

需要安装可选依赖:  pip install qlib

使用前需初始化 qlib:
    import qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def _ensure_qlib():
    try:
        import qlib
        return qlib
    except ImportError:
        raise ImportError("请先安装 qlib:  pip install qlib")


@dataclass
class BacktestResult:
    """回测结果容器。"""
    portfolio_returns: pd.Series
    cumulative_returns: pd.Series
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    turnover: float
    report_df: pd.DataFrame | None = None


class QlibAdapter:
    """将因子信号接入 qlib 回测框架。

    Parameters
    ----------
    signal : 因子预测值, MultiIndex(date, symbol)
    topk : 持仓股票数量
    n_drop : 每期卖出股票数（控制换手）
    benchmark : 基准指数代码
    account : 初始资金
    """

    def __init__(
        self,
        signal: pd.Series | pd.DataFrame,
        topk: int = 50,
        n_drop: int | None = None,
        benchmark: str = "SH000300",
        account: float = 1e8,
    ) -> None:
        _ensure_qlib()
        if isinstance(signal, pd.DataFrame):
            signal = signal.iloc[:, 0]
        self.signal = signal
        self.topk = topk
        self.n_drop = n_drop
        self.benchmark = benchmark
        self.account = account

    def _to_qlib_signal(self) -> pd.DataFrame:
        """将因子 Series 转为 qlib 期望的 signal DataFrame。"""
        df = self.signal.to_frame("score")
        # qlib 要求索引名: datetime, instrument
        if df.index.names != ["datetime", "instrument"]:
            df.index = df.index.set_names(["datetime", "instrument"])
        return df

    def run(self) -> BacktestResult:
        """执行 qlib 回测，返回 BacktestResult。"""
        from qlib.contrib.evaluate import backtest_daily
        from qlib.contrib.strategy import TopkDropoutStrategy

        signal_df = self._to_qlib_signal()

        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": signal_df,
                "topk": self.topk,
                "n_drop": self.n_drop or self.topk // 5,
            },
        }

        backtest_config = {
            "start_time": str(signal_df.index.get_level_values(0).min().date()),
            "end_time": str(signal_df.index.get_level_values(0).max().date()),
            "account": self.account,
            "benchmark": self.benchmark,
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }

        logger.info("启动 qlib 回测...")
        portfolio_metric_dict, indicator_dict = backtest_daily(
            strategy=strategy_config,
            **backtest_config,
        )

        # 提取组合收益
        report_normal = portfolio_metric_dict.get("1day", [None, None])
        if isinstance(report_normal, (list, tuple)) and len(report_normal) >= 1:
            report_df = report_normal[0]
        else:
            report_df = pd.DataFrame()

        if report_df is not None and "return" in report_df.columns:
            port_returns = report_df["return"]
        else:
            port_returns = pd.Series(dtype=float)

        cumulative = (1 + port_returns).cumprod() - 1 if len(port_returns) > 0 else pd.Series(dtype=float)

        # 性能指标
        total_ret = float((1 + port_returns).prod() - 1) if len(port_returns) > 0 else 0.0
        n_days = len(port_returns)
        annual_ret = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)

        std = port_returns.std()
        sharpe = float(port_returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        # 最大回撤
        cum_peak = (1 + cumulative).cummax()
        drawdown = (1 + cumulative) / cum_peak - 1
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        calmar = float(annual_ret / abs(max_dd)) if max_dd != 0 else 0.0

        # 换手率
        turnover_val = float(report_df.get("turnover", pd.Series([0])).mean()) if report_df is not None else 0.0

        result = BacktestResult(
            portfolio_returns=port_returns,
            cumulative_returns=cumulative,
            total_return=total_ret,
            annual_return=annual_ret,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            turnover=turnover_val,
            report_df=report_df,
        )

        logger.info(
            "回测完成: total_ret={:.2%}, annual_ret={:.2%}, sharpe={:.2f}, max_dd={:.2%}",
            result.total_return, result.annual_return, result.sharpe_ratio, result.max_drawdown,
        )
        return result
