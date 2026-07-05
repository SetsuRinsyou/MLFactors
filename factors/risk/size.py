import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor

@register_factor
class SizeFactor(BaseFactor):
    name = "size_factor_1d"
    description = "市值因子 (ln(MarketCap))"

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        market_cap = data["market_cap"].unstack("symbol").astype(float)
        # 取对数
        return np.log(market_cap.replace(0, np.nan))

@register_factor
class SizeSquaredFactor(BaseFactor):
    name = "size_squared_1d"
    description = "市值平方因子 (ln(MarketCap)^2)"

    def generate_signals(self, data: pd.DataFrame, constituents: dict[str, set[str]] | None = None) -> pd.DataFrame:
        market_cap = data["market_cap"].unstack("symbol").astype(float)
        # 取对数的平方
        return np.log(market_cap.replace(0, np.nan)) ** 2