"""基于 Fama-French 三因子残差的波动率因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


class VFF3Volatility(BaseFactor):
    """VFF3 残差波动率的公共计算逻辑。"""

    component: str = ""
    expected_lookback: int | None = None

    def __init__(
        self,
        lookback: int = 63,
        min_periods: int = 40,
        min_residuals: int = 10,
    ) -> None:
        if self.expected_lookback is not None and lookback != self.expected_lookback:
            raise ValueError(
                f"{self.name} 的 lookback 必须为 {self.expected_lookback}"
            )
        if lookback <= 0 or min_periods < 4 or min_periods > lookback:
            raise ValueError("min_periods 必须在 4 到 lookback 之间")
        if min_residuals <= 0 or min_residuals > lookback:
            raise ValueError("min_residuals 必须在 1 到 lookback 之间")

        self.lookback = lookback
        self.min_periods = min_periods
        self.min_residuals = min_residuals

    @staticmethod
    def _portfolio_return(
        stock_ret: pd.Series,
        market_cap: pd.Series,
        mask: pd.Series,
    ) -> float:
        weights = market_cap.where(mask & stock_ret.notna() & (market_cap > 0))
        if weights.empty or weights.notna().sum() == 0:
            return np.nan
        weights = weights / weights.sum()
        return float((stock_ret * weights).sum(min_count=1))

    def _ff3_returns(
        self,
        stock_ret: pd.DataFrame,
        market_cap: pd.DataFrame,
        pb_ratio: pd.DataFrame,
    ) -> pd.DataFrame:
        lagged_cap = market_cap.shift(1)
        bp_ratio = 1.0 / pb_ratio.shift(1).where(pb_ratio.shift(1) > 0)
        factor_returns = []

        for date in stock_ret.index:
            daily_return = stock_ret.loc[date]
            cap = lagged_cap.loc[date]
            bp = bp_ratio.loc[date]
            market = self._portfolio_return(
                daily_return,
                cap,
                pd.Series(True, index=daily_return.index),
            )

            eligible = (cap > 0) & bp.notna()
            eligible_symbols = eligible[eligible].index
            if len(eligible_symbols) < 6:
                factor_returns.append((market, np.nan, np.nan))
                continue

            size_rank = cap.loc[eligible_symbols].rank(method="first")
            bp_rank = bp.loc[eligible_symbols].rank(method="first")
            size_group = size_rank <= len(eligible_symbols) * 0.5
            bp_group = pd.cut(
                bp_rank,
                bins=[0, len(eligible_symbols) * 0.3, len(eligible_symbols) * 0.7, np.inf],
                labels=["L", "N", "H"],
                include_lowest=True,
            )
            small = pd.Series(False, index=daily_return.index)
            small.loc[eligible_symbols] = size_group.to_numpy()
            value_group = pd.Series(index=daily_return.index, dtype=object)
            value_group.loc[eligible_symbols] = bp_group.astype(object).to_numpy()

            portfolios = {
                f"{size}/{value}": self._portfolio_return(
                    daily_return,
                    cap,
                    (small if size == "S" else ~small) & (value_group == value),
                )
                for size in ("S", "B")
                for value in ("L", "N", "H")
            }
            if any(pd.isna(value) for value in portfolios.values()):
                factor_returns.append((market, np.nan, np.nan))
                continue
            smb = (
                portfolios["S/L"] + portfolios["S/N"] + portfolios["S/H"]
            ) / 3 - (
                portfolios["B/L"] + portfolios["B/N"] + portfolios["B/H"]
            ) / 3
            hml = (portfolios["S/H"] + portfolios["B/H"]) / 2 - (
                portfolios["S/L"] + portfolios["B/L"]
            ) / 2
            factor_returns.append((market, smb, hml))

        return pd.DataFrame(
            factor_returns,
            index=stock_ret.index,
            columns=["market_ret", "smb_ret", "hml_ret"],
        )

    def _residual_volatility(
        self,
        stock_ret: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        result = pd.DataFrame(np.nan, index=stock_ret.index, columns=stock_ret.columns)
        factors = factor_returns.to_numpy(dtype=float)

        for symbol in stock_ret.columns:
            returns = stock_ret[symbol].to_numpy(dtype=float)
            for end in range(self.min_periods - 1, len(returns)):
                start = max(0, end - self.lookback + 1)
                y = returns[start : end + 1]
                x = factors[start : end + 1]
                valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
                if valid.sum() < self.min_periods:
                    continue

                design = np.column_stack((np.ones(valid.sum()), x[valid]))
                coefficients, _, rank, _ = np.linalg.lstsq(design, y[valid], rcond=None)
                if rank < design.shape[1]:
                    continue
                residuals = y[valid] - design @ coefficients
                if self.component == "all":
                    result.iat[end, result.columns.get_loc(symbol)] = residuals.std(ddof=1)
                elif self.component == "up":
                    positive = residuals[residuals > 0]
                    if len(positive) >= self.min_residuals:
                        result.iat[end, result.columns.get_loc(symbol)] = positive.std(ddof=1)
                else:
                    positive = residuals[residuals > 0]
                    negative = residuals[residuals < 0]
                    if len(positive) >= self.min_residuals and len(negative) >= self.min_residuals:
                        result.iat[end, result.columns.get_loc(symbol)] = (
                            positive.std(ddof=1) + negative.std(ddof=1)
                        )
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        required = {"close", "volume", "market_cap", "pb_ratio"}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"{self.name} 缺少字段: {sorted(missing)}")

        close = data["close"].unstack("symbol").astype(float).sort_index()
        volume = data["volume"].unstack("symbol").reindex_like(close).astype(float)
        market_cap = data["market_cap"].unstack("symbol").reindex_like(close).astype(float)
        pb_ratio = data["pb_ratio"].unstack("symbol").reindex_like(close).astype(float)
        close = close.where((close > 0) & (volume > 0))
        stock_ret = close.pct_change(fill_method=None)
        factor_returns = self._ff3_returns(stock_ret, market_cap, pb_ratio)
        result = self._residual_volatility(stock_ret, factor_returns)
        result.index.name = "date"
        result.columns.name = "symbol"
        return result


@register_factor
class VFF363D(VFF3Volatility):
    """过去 63 个交易日的 VFF3 残差标准差。"""

    name = "vff3_63d"
    description = "63 日 Fama-French 三因子残差波动率"
    component = "all"
    expected_lookback = 63


@register_factor
class VFF3Up63D(VFF3Volatility):
    """过去 63 个交易日的正 VFF3 残差标准差。"""

    name = "vff3_up_63d"
    description = "63 日 Fama-French 三因子正残差波动率"
    component = "up"
    expected_lookback = 63


@register_factor
class VFF3UpD63D(VFF3Volatility):
    """过去 63 个交易日的 VFF3 正负残差标准差之和。"""

    name = "vff3_upd_63d"
    description = "63 日 Fama-French 三因子正负残差波动率之和"
    component = "upd"
    expected_lookback = 63
