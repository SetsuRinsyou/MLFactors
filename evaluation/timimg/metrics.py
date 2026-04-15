"""择时因子评估指标 — 纯函数层。

所有函数均为无状态纯函数，供 TimingReport 及上层代码调用。

信号约定
--------
- signal > 0 : 多头方向（做多标的）
- signal < 0 : 空头方向（做空标的）
- signal = 0 : 空仓（不持仓）
"""

from __future__ import annotations

import pandas as pd


def calc_direction_win_rate(signal: pd.Series, returns: pd.Series) -> float:
    """方向胜率：信号方向与实际收益方向一致的比例。

    仅统计有明确方向信号（signal != 0）的时段，空仓期不计入。

    Parameters
    ----------
    signal : 信号序列，索引为 date
    returns : 对应期间的日收益序列，索引为 date

    Returns
    -------
    方向胜率，取值 [0, 1]；有效样本不足时返回 ``np.nan``
    """
    aligned = pd.DataFrame({"signal": signal, "returns": returns}).dropna()
    active = aligned[aligned["signal"] != 0]
    if len(active) == 0:
        return float("nan")
    correct = (
        ((active["signal"] > 0) & (active["returns"] > 0))
        | ((active["signal"] < 0) & (active["returns"] < 0))
    )
    return float(correct.mean())


def calc_excess_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> pd.Series:
    """计算超额收益序列。

    Parameters
    ----------
    strategy_returns : 策略日收益序列
    benchmark_returns : 基准日收益序列；``None`` 时超额等同于绝对收益（以 0 为基准）

    Returns
    -------
    超额日收益 pd.Series
    """
    if benchmark_returns is None:
        return strategy_returns.copy().rename("excess_returns")
    aligned = pd.DataFrame(
        {"strat": strategy_returns, "bench": benchmark_returns}
    ).dropna()
    return (aligned["strat"] - aligned["bench"]).rename("excess_returns")



