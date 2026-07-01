"""海通39：GP 同比增长率因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class GrossProfitabilityGrowth12m(BaseFactor):
    """最新 GP 相对去年同期的增长率。"""

    name = "gross_profitability_growth_12m"
    description = "海通39：GP 同比增长率"

    def __init__(self, min_abs_denominator: float = 1e-12) -> None:
        self.min_abs_denominator = min_abs_denominator
        self.year_days = 21 * 12

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"gross_profit", "total_assets"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"{self.name} 缺少字段: {sorted(missing)}")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        gross_profit = data["gross_profit"].unstack("symbol").astype(float).sort_index()
        total_assets = data["total_assets"].unstack("symbol").astype(float).reindex_like(gross_profit)
        gp = gross_profit.divide(total_assets.where(total_assets > self.min_abs_denominator))
        lag = gp.shift(self.year_days)
        result = gp.divide(lag.where(lag > self.min_abs_denominator)) - 1.0

        result.index.name = "date"
        result.columns.name = "symbol"
        return result.replace([np.inf, -np.inf], np.nan)
