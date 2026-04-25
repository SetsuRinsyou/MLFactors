"""风控与仓位管理基类及最小示例实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from data.schema import Col


class BasePortfolioManager(ABC):
    """仓位与风控管理基类。"""

    @abstractmethod
    def generate_target_weights(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """将原始信号转换为目标持仓权重矩阵。"""
        ...


class SimpleTopKPortfolioManager(BasePortfolioManager):
    """最简单的长仓等权组合管理器。

    每个交易日选出信号最高的 ``top_k`` 个标的，做多且等权配置。
    若当日无正信号，则空仓。
    """

    def __init__(
        self,
        top_k: int = 5,
        min_signal: float = 0.0,
        rebalance_frequency: str = "W-FRI",
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        self.top_k = top_k
        self.min_signal = min_signal
        self.rebalance_frequency = rebalance_frequency

    def _build_tradable_mask(self, market_data: pd.DataFrame) -> pd.DataFrame:
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
        if Col.VOLUME in market_data.columns:
            volume = market_data[Col.VOLUME].unstack(Col.SYMBOL).sort_index()
            return close.notna() & volume.fillna(0).gt(0)
        return close.notna()

    def generate_target_weights(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        tradable = self._build_tradable_mask(market_data)
        aligned_signals = signals.reindex(index=tradable.index, columns=tradable.columns)
        weights = pd.DataFrame(np.nan, index=tradable.index, columns=tradable.columns)

        rebalance_dates = (
            pd.Series(aligned_signals.index, index=aligned_signals.index)
            .resample(self.rebalance_frequency)
            .last()
            .dropna()
            .tolist()
        )

        for date in rebalance_dates:
            row = aligned_signals.loc[date]
            tradable_row = tradable.loc[date]
            weights.loc[date] = 0.0
            eligible = row.where(tradable_row).replace([np.inf, -np.inf], np.nan).dropna()
            eligible = eligible[eligible > self.min_signal]
            if eligible.empty:
                continue

            selected = eligible.nlargest(self.top_k).index
            weights.loc[date, selected] = 1.0 / len(selected)

        weights = weights.ffill().fillna(0.0)
        weights.index.name = Col.DATE
        weights.columns.name = Col.SYMBOL
        return weights
