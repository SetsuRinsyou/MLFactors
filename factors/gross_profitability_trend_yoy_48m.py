"""海通39：GP 同比趋势因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class GrossProfitabilityTrendYoy48m(BaseFactor):
    """最近 4 个年度近似点的 GP 截面标准分线性趋势。"""

    name = "gross_profitability_trend_yoy_48m"
    description = "海通39：GP 同比趋势，使用 4 个年度近似点计算斜率"

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

        low = gp.quantile(0.01, axis=1)
        high = gp.quantile(0.99, axis=1)
        gp = gp.clip(lower=low, upper=high, axis=0)
        mean = gp.mean(axis=1)
        std = gp.std(axis=1, ddof=0).replace(0, np.nan)
        gp = gp.sub(mean, axis=0).div(std, axis=0)

        lags = [self.year_days * 3, self.year_days * 2, self.year_days, 0]
        values = np.stack([gp.shift(lag).to_numpy(dtype=float) for lag in lags], axis=0)
        valid = np.isfinite(values).all(axis=0)
        x = np.arange(1, 5, dtype=float)
        x = x - x.mean()
        y = values - values.mean(axis=0, keepdims=True)
        result = pd.DataFrame(
            np.where(valid, (x[:, None, None] * y).sum(axis=0) / (x * x).sum(), np.nan),
            index=gp.index,
            columns=gp.columns,
        )

        result.index.name = "date"
        result.columns.name = "symbol"
        return result.replace([np.inf, -np.inf], np.nan)
