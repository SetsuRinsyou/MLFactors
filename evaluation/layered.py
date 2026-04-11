"""分层回测 — 按因子值分组计算收益。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LayeredResult:
    """分层回测结果。"""
    group_returns: pd.DataFrame          # 各组每期收益 (date × group)
    cumulative_returns: pd.DataFrame     # 各组累计收益
    annual_returns: pd.Series            # 各组年化收益
    sharpe_ratios: pd.Series             # 各组夏普比率
    long_short_returns: pd.Series        # 多空对冲每期收益
    long_short_cumulative: pd.Series     # 多空对冲累计收益
    long_short_annual: float = 0.0       # 多空年化收益
    long_short_sharpe: float = 0.0       # 多空夏普比率
    long_short_max_drawdown: float = 0.0 # 多空最大回撤
    top_excess_annual: float = 0.0       # 多头超额年化收益
    top_excess_max_drawdown: float = 0.0 # 多头超额最大回撤
    top_excess_calmar: float = 0.0       # 多头超额卡玛比率
    n_groups: int = 5


def layered_backtest(
    factor_df: pd.DataFrame | pd.Series,
    returns_df: pd.DataFrame | pd.Series,
    n_groups: int = 5,
    annual_trading_days: int = 252,
) -> LayeredResult:
    """按因子值分 N 组，计算各组收益表现。

    Parameters
    ----------
    factor_df : 因子值, MultiIndex(date, symbol)
    returns_df : 前向收益, 相同索引
    n_groups : 分组数
    annual_trading_days : 年化换算天数

    Returns
    -------
    LayeredResult
    """
    if isinstance(factor_df, pd.DataFrame):
        factor_df = factor_df.iloc[:, 0]
    if isinstance(returns_df, pd.DataFrame):
        returns_df = returns_df.iloc[:, 0]

    combined = pd.DataFrame({"factor": factor_df, "returns": returns_df}).dropna()

    # 每个截面日期分组
    dates = combined.index.get_level_values(0).unique().sort_values()
    group_ret_records: list[dict] = []

    for dt in dates:
        cross = combined.loc[dt].copy()
        if len(cross) < n_groups:
            continue
        # quantile 分组 (1=最小, n_groups=最大)
        cross["group"] = pd.qcut(cross["factor"], n_groups, labels=False, duplicates="drop") + 1
        for g in range(1, n_groups + 1):
            g_mask = cross["group"] == g
            if g_mask.sum() == 0:
                continue
            group_ret_records.append({
                "date": dt,
                "group": g,
                "returns": cross.loc[g_mask, "returns"].mean(),
            })

    if not group_ret_records:
        return LayeredResult(
            group_returns=pd.DataFrame(),
            cumulative_returns=pd.DataFrame(),
            annual_returns=pd.Series(dtype=float),
            sharpe_ratios=pd.Series(dtype=float),
            long_short_returns=pd.Series(dtype=float),
            long_short_cumulative=pd.Series(dtype=float),
            n_groups=n_groups,
        )

    ret_df = pd.DataFrame(group_ret_records)
    group_returns = ret_df.pivot(index="date", columns="group", values="returns").sort_index()

    # 累计收益
    cumulative = (1 + group_returns).cumprod() - 1

    # 年化收益
    n_periods = len(group_returns)
    total_ret = (1 + group_returns).prod()
    annual_ret = total_ret ** (annual_trading_days / max(n_periods, 1)) - 1
    annual_ret.name = "annual_return"

    # 夏普比率
    sharpe = (group_returns.mean() / group_returns.std()) * np.sqrt(annual_trading_days)
    sharpe.name = "sharpe_ratio"

    # 多空对冲 (top - bottom)
    top_group = n_groups
    bottom_group = 1
    ls_returns = group_returns[top_group] - group_returns[bottom_group]
    ls_returns.name = "long_short"
    ls_cumulative = (1 + ls_returns).cumprod() - 1
    ls_total = (1 + ls_returns).prod()
    ls_annual = float(ls_total ** (annual_trading_days / max(n_periods, 1)) - 1)
    ls_std = ls_returns.std()
    ls_sharpe = float((ls_returns.mean() / ls_std) * np.sqrt(annual_trading_days)) if ls_std > 0 else 0.0
    
    # 多空最大回撤 (Max Drawdown)
    ls_wealth = 1 + ls_cumulative
    ls_peak = ls_wealth.cummax()  # 历史最高点序列
    ls_drawdown = (ls_wealth - ls_peak) / ls_peak
    ls_max_drawdown = float(ls_drawdown.min()) if not ls_drawdown.empty else 0.0

    # 多头相对全市场的超额年化 (Top Group Excess Return)
    benchmark_returns = group_returns.mean(axis=1)
    top_excess_returns = group_returns[top_group] - benchmark_returns
    
    top_excess_total = (1 + top_excess_returns).prod()
    top_excess_annual = float(top_excess_total ** (annual_trading_days / max(n_periods, 1)) - 1)
    
    # 多头超额最大回撤及卡玛比率
    top_excess_wealth = (1 + top_excess_returns).cumprod()
    top_excess_peak = top_excess_wealth.cummax()
    top_excess_drawdown = (top_excess_wealth - top_excess_peak) / top_excess_peak
    top_excess_max_drawdown = float(top_excess_drawdown.min()) if not top_excess_drawdown.empty else 0.0
    if top_excess_max_drawdown < 0:
        top_excess_calmar = float(top_excess_annual / abs(top_excess_max_drawdown))
    else:
        top_excess_calmar = 0.0  # 如果没有回撤（极少见），设为0或无穷大
    
    return LayeredResult(
        group_returns=group_returns,
        cumulative_returns=cumulative,
        annual_returns=annual_ret,
        sharpe_ratios=sharpe,
        long_short_returns=ls_returns,
        long_short_cumulative=ls_cumulative,
        long_short_annual=ls_annual,
        long_short_sharpe=ls_sharpe,
        long_short_max_drawdown=ls_max_drawdown,
        top_excess_annual=top_excess_annual,
        top_excess_max_drawdown=top_excess_max_drawdown,
        top_excess_calmar=top_excess_calmar,
        n_groups=n_groups,
    )
