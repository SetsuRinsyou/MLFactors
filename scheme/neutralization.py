"""Neutralization utilities for scheme factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _exposure_frame(
    factor_values: pd.Series,
    market_cap: pd.Series,
    sector: pd.Series,
) -> pd.DataFrame:
    """Build one-date regression frame for market-cap and sector neutralization."""
    frame = pd.DataFrame(
        {
            "factor": factor_values,
            "log_market_cap": np.log(market_cap.replace(0, np.nan)),
            "sector": sector,
        }
    ).replace([np.inf, -np.inf], np.nan)
    return frame.dropna()


def _neutralize_cross_section(
    factor_values: pd.Series,
    market_cap: pd.Series,
    sector: pd.Series,
) -> pd.Series:
    """Return OLS residuals from factor ~ log(market_cap) + sector dummies."""
    frame = _exposure_frame(factor_values, market_cap, sector)
    if len(frame) < 3 or frame["factor"].nunique(dropna=True) <= 1:
        return pd.Series(np.nan, index=factor_values.index)

    sector_dummies = pd.get_dummies(
        frame["sector"].astype(str),
        prefix="sector",
        drop_first=True,
        dtype=float,
    )
    x = pd.concat(
        [
            pd.Series(1.0, index=frame.index, name="const"),
            frame[["log_market_cap"]].astype(float),
            sector_dummies,
        ],
        axis=1,
    )
    y = frame["factor"].astype(float)
    if len(frame) <= x.shape[1]:
        return pd.Series(np.nan, index=factor_values.index)

    beta, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None)
    residuals = y - x.to_numpy(dtype=float) @ beta
    return pd.Series(residuals, index=frame.index).reindex(factor_values.index)


def neutralize_market_cap_sector(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    market_cap_col: str = "market_cap",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Neutralize date x symbol signals by market cap and sector.

    The input ``signals`` must be a date x symbol factor matrix. ``market_data`` must
    use a ``(date, symbol)`` MultiIndex and include market cap and sector columns.
    """
    if market_cap_col not in market_data.columns:
        raise ValueError(f"组合因子中性化缺少字段: {market_cap_col}")
    if sector_col not in market_data.columns:
        raise ValueError(f"组合因子中性化缺少字段: {sector_col}")

    market_cap = (
        market_data[market_cap_col]
        .unstack("symbol")
        .reindex(index=signals.index, columns=signals.columns)
    )
    sector = (
        market_data[sector_col]
        .unstack("symbol")
        .reindex(index=signals.index, columns=signals.columns)
    )

    result = pd.DataFrame(np.nan, index=signals.index, columns=signals.columns)
    for date in signals.index:
        result.loc[date] = _neutralize_cross_section(
            signals.loc[date],
            market_cap.loc[date],
            sector.loc[date],
        )

    result.index.name = signals.index.name
    result.columns.name = signals.columns.name
    return result
