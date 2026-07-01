"""海通54：行业流动资产增长稳定性因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class IndustryCurrentAssetGrowthStability24m(BaseFactor):
    """行业内流动资产增长波动率中位数的相反数。"""

    name = "industry_current_asset_growth_stability_24m"
    description = "海通54：行业流动资产增长稳定性，值越高代表行业越稳定"

    def __init__(
        self,
        min_abs_denominator: float = 1e-12,
        min_industry_members: int = 3,
    ) -> None:
        self.min_abs_denominator = min_abs_denominator
        self.min_industry_members = min_industry_members
        self.year_days = 21 * 12
        self.window = 21 * 24

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"current_assets", "sector"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"{self.name} 缺少字段: {sorted(missing)}")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        current_assets = data["current_assets"].unstack("symbol").astype(float).sort_index()
        lag = current_assets.shift(self.year_days)
        growth = current_assets.divide(lag.where(lag > self.min_abs_denominator)) - 1.0
        stock_vol = growth.rolling(self.window, min_periods=self.year_days).std()
        sector = data["sector"].unstack("symbol").reindex_like(stock_vol)
        result = pd.DataFrame(np.nan, index=stock_vol.index, columns=stock_vol.columns)

        for date in stock_vol.index:
            cross_section = pd.DataFrame({
                "vol": stock_vol.loc[date],
                "sector": sector.loc[date],
            }).dropna()
            if cross_section.empty:
                continue
            count = cross_section.groupby("sector")["vol"].count()
            median = cross_section.groupby("sector")["vol"].median()
            stability = (-median).where(count >= self.min_industry_members)
            result.loc[date] = sector.loc[date].map(stability)

        result.index.name = "date"
        result.columns.name = "symbol"
        return result.replace([np.inf, -np.inf], np.nan)
