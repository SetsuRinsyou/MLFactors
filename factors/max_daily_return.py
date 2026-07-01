"""海通15博彩型股票：每 21 日前 5 大单日收益均值因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class MaxDailyReturn5(BaseFactor):
    """每 21 日前 5 大单日收益均值，值越高表示博彩型极端上涨特征越强。"""

    name = "max_daily_return_5d"
    description = "海通15博彩型因子：每21日前5大单日收益均值"

    def __init__(
        self,
        top_n: int = 5,
        min_obs: int | None = None,
        lookback: int = 21,
        rebalance_step: int = 21,
    ) -> None:
        self.top_n = top_n
        self.min_obs = min_obs
        self.lookback = lookback
        self.rebalance_step = rebalance_step

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        """每 rebalance_step 个交易日输出一次过去 lookback 日前 top_n 大日收益均值。"""
        min_obs = self.top_n if self.min_obs is None else self.min_obs
        if self.top_n <= 0:
            raise ValueError("top_n 必须为正整数")
        if min_obs <= 0:
            raise ValueError("min_obs 必须为正整数")
        if self.lookback <= 0:
            raise ValueError("lookback 必须为正整数")
        if self.rebalance_step <= 0:
            raise ValueError("rebalance_step 必须为正整数")
        if "close" not in data.columns:
            raise ValueError("字段 close 缺失")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        close = data["close"].unstack("symbol").astype(float).sort_index()
        returns = close.pct_change(fill_method=None)
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        rebalance_positions = range(self.lookback, len(returns), self.rebalance_step)
        for position in rebalance_positions:
            window_ret = returns.iloc[position - self.lookback + 1 : position + 1]
            factor_date = returns.index[position]
            clean = window_ret.replace([np.inf, -np.inf], np.nan)
            result.loc[factor_date] = clean.apply(
                lambda values: values.dropna().nlargest(self.top_n).mean()
                if values.notna().sum() >= min_obs
                else np.nan
            )

        result.index.name = "date"
        result.columns.name = "symbol"
        if not constituents:
            return result

        mask = pd.DataFrame(False, index=result.index, columns=result.columns)
        for date in result.index:
            allowed = constituents.get(str(pd.Timestamp(date).date()), set())
            if allowed:
                present = result.columns.intersection(allowed)
                mask.loc[date, present] = True
        return result.where(mask)
