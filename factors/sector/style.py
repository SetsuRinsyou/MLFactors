"""风格分类与动量溢出因子"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor

@register_factor
class IndustryMomentum(BaseFactor):
    name = "industry_momentum_1d"
    description = "行业动量因子 (剔除自身的同行业平均收益)"

    def __init__(self, period: int = 1) -> None:
        self.period = period

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        close = data["adj_close"].unstack("symbol").astype(float).sort_index()
        sector = data["sector"].unstack("symbol")
        
        returns = close.pct_change(self.period, fill_method=None)
        
        # 展平为长表以进行高效的 Groupby 运算
        df = pd.DataFrame({
            "ret": returns.stack(),
            "sector": sector.stack()
        }).dropna(subset=["sector"])
        
        # 计算每个行业在每一天的总收益和股票数量
        grp = df.groupby(["date", "sector"])["ret"]
        df["grp_sum"] = grp.transform("sum")
        df["grp_count"] = grp.transform("count")
        
        # 剔除自身收益后的平均值：(行业总收益 - 自身收益) / (行业总数 - 1)
        df["peer_mean"] = (df["grp_sum"] - df["ret"].fillna(0)) / (df["grp_count"] - 1).clip(lower=1)
        df.loc[df["grp_count"] <= 1, "peer_mean"] = np.nan
        
        return df["peer_mean"].unstack("symbol")
