"""基于市值加权市场组合收益的 SemiBeta 因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


class SemiBeta(BaseFactor):
    """SemiBeta 的公共计算逻辑。"""

    component: str = ""

    def __init__(
        self,
        window: int = 252,
        min_obs: int | None = 120,
        min_market_obs: int = 100,
    ) -> None:
        if min_obs is None:
            min_obs = window
        if min_obs <= 0 or min_obs > window:
            raise ValueError("min_obs 必须在 1 到 window 之间")
        if min_market_obs <= 0:
            raise ValueError("min_market_obs 必须为正整数")

        self.window = window
        self.min_obs = min_obs
        self.min_market_obs = min_market_obs

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        close = data["close"].unstack("symbol").astype(float).sort_index()
        market_cap = data["market_cap"].unstack("symbol").astype(float).sort_index()
        market_cap = market_cap.reindex(index=close.index, columns=close.columns)

        stock_ret = close.pct_change(fill_method=None)
        weights = market_cap.shift(1).where(stock_ret.notna())
        weights = weights.where(weights > 0)
        constituent_mask = None
        if constituents:
            constituent_mask = pd.DataFrame(
                [
                    [symbol in constituents.get(str(date.date()), set()) for symbol in close.columns]
                    for date in close.index
                ],
                index=close.index,
                columns=close.columns,
            )
            weights = weights.where(constituent_mask)

        valid_market_count = weights.notna().sum(axis=1)
        weights = weights.div(weights.sum(axis=1), axis=0)
        market_ret = (stock_ret * weights).sum(axis=1, min_count=1)
        market_ret = market_ret.where(valid_market_count >= self.min_market_obs)

        market_ret_frame = pd.DataFrame(
            np.repeat(market_ret.to_numpy()[:, None], len(stock_ret.columns), axis=1),
            index=stock_ret.index,
            columns=stock_ret.columns,
        )
        valid = stock_ret.notna() & market_ret_frame.notna()
        stock_ret = stock_ret.where(valid)
        market_ret_frame = market_ret_frame.where(valid)

        stock_pos = stock_ret.clip(lower=0)
        stock_neg = stock_ret.clip(upper=0)
        market_pos = market_ret_frame.clip(lower=0)
        market_neg = market_ret_frame.clip(upper=0)

        products = {
            "N": stock_neg * market_neg,
            "P": stock_pos * market_pos,
            "MN": -(stock_pos * market_neg),
            "MP": -(stock_neg * market_pos),
        }
        numerator = products[self.component].rolling(
            self.window,
            min_periods=self.min_obs,
        ).sum()
        denominator = (market_ret_frame**2).rolling(
            self.window,
            min_periods=self.min_obs,
        ).sum()
        semi_beta = numerator.div(denominator.where(denominator != 0))

        if constituent_mask is not None:
            semi_beta = semi_beta.where(constituent_mask)
        semi_beta.index.name = "date"
        semi_beta.columns.name = "symbol"
        return semi_beta.replace([np.inf, -np.inf], np.nan)


@register_factor
class BetaN(SemiBeta):
    """市场下跌、个股下跌时的 SemiBeta。"""

    name = "beta_n_252d"
    description = "SemiBeta：市场跌、股票跌"
    component = "N"


@register_factor
class BetaP(SemiBeta):
    """市场上涨、个股上涨时的 SemiBeta。"""

    name = "beta_p_252d"
    description = "SemiBeta：市场涨、股票涨"
    component = "P"


@register_factor
class BetaMN(SemiBeta):
    """市场下跌、个股上涨时的 SemiBeta。"""

    name = "beta_mn_252d"
    description = "SemiBeta：市场跌、股票涨"
    component = "MN"


@register_factor
class BetaMP(SemiBeta):
    """市场上涨、个股下跌时的 SemiBeta。"""

    name = "beta_mp_252d"
    description = "SemiBeta：市场涨、股票跌"
    component = "MP"
