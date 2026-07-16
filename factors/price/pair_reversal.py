"""个股配对反转因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class PairReversal(BaseFactor):
    """
    个股配对反转因子 (Pair Reversal)
    基于过去 N 天的收益率相关性寻找孪生股票，并基于近期 M 天的收益率差计算反转信号。
    """

    name = "pair_reversal_60d"
    description = "个股配对反转因子"

    def __init__(
        self,
        formation_period: int = 60,
        reversal_period: int = 5,
    ) -> None:
        self.formation_period = formation_period
        self.reversal_period = reversal_period

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        # 获取收盘价并转为矩阵格式
        close = data["adj_close"].unstack("symbol").astype(float).sort_index()
        
        # 计算每日收益率（用于计算相关性）和短期反转收益率（用于计算价差偏离）
        daily_returns = close.pct_change(fill_method=None)
        rev_returns = close.pct_change(self.reversal_period, fill_method=None)

        # 初始化信号矩阵
        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        # 滚动寻找孪生股票并计算偏离度
        # 注：为了适应动态的时间序列截面，这里采用按日循环截面的方式计算相关性
        for i in range(self.formation_period, len(close)):
            # 提取过去 60 天的收益率窗口
            window_ret = daily_returns.iloc[i - self.formation_period + 1 : i + 1]
            
            # 只有当窗口内有足够数据时才计算
            if window_ret.isna().all().all():
                continue
                
            # 计算截面相关性矩阵
            corr_matrix = window_ret.corr().values
            # 将对角线（自己与自己）设为 -inf，排除自身
            np.fill_diagonal(corr_matrix, -np.inf)
            
            # 当前日的短期收益率向量
            current_rev_ret = rev_returns.iloc[i].values
            
            # 为每一只股票寻找相关性最高的孪生股，并计算收益率背离
            row_signals = np.full(len(close.columns), np.nan)
            for col_idx in range(len(close.columns)):
                corrs = corr_matrix[col_idx]
                # 排除全是 nan 的情况（如停牌或新股）
                if np.isnan(corrs).all():
                    continue
                
                # 找到最相似的“孪生股”索引
                best_peer_idx = np.nanargmax(corrs)
                
                # 因子值 = 孪生股的近期收益率 - 目标股的近期收益率
                # (孪生股涨得多而你跌了，差值为正，预期你补涨，产生多头信号)
                row_signals[col_idx] = current_rev_ret[best_peer_idx] - current_rev_ret[col_idx]
                
            signals.iloc[i] = row_signals

        return signals
