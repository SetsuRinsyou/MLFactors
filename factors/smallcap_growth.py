"""小市值成长弹性因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class SmallCapGrowth(BaseFactor):
    """海通83小市值成长弹性近似因子：市值越小，因子值越高。"""

    name = "smallcap_growth_1d"
    description = "小市值成长弹性因子：-log(market_cap) 横截面标准化"

    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 0.99,
        use_rank: bool = False,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.use_rank = use_rank

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if "market_cap" not in data.columns:
            raise ValueError("字段 market_cap 缺失")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        market_cap = data["market_cap"].unstack("symbol").astype(float).sort_index()
        market_cap.index.name = "date"
        market_cap.columns.name = "symbol"
        log_cap = np.log(market_cap.where(market_cap > 0))
        raw_score = -log_cap

        if self.use_rank:
            score = raw_score.rank(axis=1, pct=True, ascending=True)
            score = score.where(raw_score.notna())
        else:
            low = raw_score.quantile(self.lower, axis=1)
            high = raw_score.quantile(self.upper, axis=1)
            winsorized = raw_score.clip(lower=low, upper=high, axis=0)
            mean = winsorized.mean(axis=1)
            std = winsorized.std(axis=1, ddof=0).replace(0, np.nan)
            score = winsorized.sub(mean, axis=0).div(std, axis=0)

        score.index.name = "date"
        score.columns.name = "symbol"
        if not constituents:
            return score

        mask = pd.DataFrame(False, index=score.index, columns=score.columns)
        for date in score.index:
            allowed = constituents.get(str(pd.Timestamp(date).date()), set())
            if allowed:
                present = score.columns.intersection(allowed)
                mask.loc[date, present] = True
        return score.where(mask)
