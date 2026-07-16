"""风格分类与动量溢出因子"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

@register_factor
class StyleCategoryMomentum(BaseFactor):
    name = "style_category_momentum_1d"
    description = "风格分类动量因子 (基于 K-means 聚类的簇内动量溢出)"

    def __init__(self, period: int = 1, n_clusters: int = 30) -> None:
        self.period = period
        self.n_clusters = n_clusters # 研报中对标中信一级行业数量，设为 30

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        close = data["adj_close"].unstack("symbol").astype(float).sort_index()
        returns = close.pct_change(self.period, fill_method=None)
        
        # 提取用于聚类的特征矩阵 (市值、估值、流动性等)
        mc = data.get("market_cap")
        pe = data.get("pe_ratio")
        
        if mc is None or pe is None:
            raise ValueError("StyleCategoryMomentum 需要 market_cap 和 pe_ratio 数据列进行聚类。")
            
        mc = mc.unstack("symbol").astype(float)
        pe = pe.unstack("symbol").astype(float)
        
        signals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        
        # 逐日进行截面聚类计算
        for dt in returns.index:
            row_ret = returns.loc[dt]
            if row_ret.isna().all():
                continue
                
            # 构建当天的截面特征集
            feat_df = pd.DataFrame({
                "mc": np.log(mc.loc[dt].replace(0, np.nan)),
                "pe": pe.loc[dt],
            }).dropna()
            
            # 如果特征不足以支撑聚类数量，跳过该日
            if len(feat_df) <= self.n_clusters:
                continue
                
            # 数据标准化与 K-means 聚类
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(feat_df)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            feat_df["cluster"] = kmeans.fit_predict(X_scaled)
            feat_df["ret"] = row_ret
            
            # 计算风格簇内的 peer_mean (剔除自身)
            grp = feat_df.groupby("cluster")["ret"]
            feat_df["grp_sum"] = grp.transform("sum")
            feat_df["grp_count"] = grp.transform("count")
            
            feat_df["peer_mean"] = (feat_df["grp_sum"] - feat_df["ret"].fillna(0)) / (feat_df["grp_count"] - 1).clip(lower=1)
            feat_df.loc[feat_df["grp_count"] <= 1, "peer_mean"] = np.nan
            
            signals.loc[dt, feat_df.index] = feat_df["peer_mean"]
            
        return signals
