"""海通54：长期债务比率变化因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class LongTermDebtRatioChange12m(BaseFactor):
    """长期债务占总资产比例的同比变化。"""

    name = "long_term_debt_ratio_change_12m"
    description = "海通54：长期债务比率同比变化"

    def __init__(self, min_abs_denominator: float = 1e-12) -> None:
        self.min_abs_denominator = min_abs_denominator
        self.year_days = 21 * 12

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"non_current_debt", "total_assets"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"{self.name} 缺少字段: {sorted(missing)}")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        non_current_debt = data["non_current_debt"].unstack("symbol").astype(float).sort_index()
        total_assets = data["total_assets"].unstack("symbol").astype(float).reindex_like(non_current_debt)
        ratio = non_current_debt.divide(total_assets.where(total_assets > self.min_abs_denominator))
        result = ratio - ratio.shift(self.year_days)

        result.index.name = "date"
        result.columns.name = "symbol"
        return result.replace([np.inf, -np.inf], np.nan)
