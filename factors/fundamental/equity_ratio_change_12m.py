"""海通54：股东权益比率变化因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class EquityRatioChange12m(BaseFactor):
    """股东权益占总资产比例的同比变化。"""

    name = "equity_ratio_change_12m"
    description = "海通54：股东权益比率同比变化"

    def __init__(self, min_abs_denominator: float = 1e-12) -> None:
        self.min_abs_denominator = min_abs_denominator
        self.year_days = 21 * 12

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"shareholder_equity", "total_assets"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"{self.name} 缺少字段: {sorted(missing)}")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        shareholder_equity = data["shareholder_equity"].unstack("symbol").astype(float).sort_index()
        total_assets = data["total_assets"].unstack("symbol").astype(float).reindex_like(shareholder_equity)
        ratio = shareholder_equity.divide(total_assets.where(total_assets > self.min_abs_denominator))
        result = ratio - ratio.shift(self.year_days)

        result.index.name = "date"
        result.columns.name = "symbol"
        return result.replace([np.inf, -np.inf], np.nan)
