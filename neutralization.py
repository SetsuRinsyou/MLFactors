"""Cross-sectional factor neutralization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from settings import SETTINGS


def neutralize_cross_section(
    factor: pd.Series,
    market_cap: pd.Series,
    industry: pd.Series,
) -> pd.Series:
    """Return OLS residuals after controlling for size and SW L1 industry.

    The regression for one signal date is::

        factor = intercept + beta * log(market_cap) + industry_dummies + error

    Only observations with a finite factor, positive finite market cap and a
    non-empty industry are included.  Industry is one-hot encoded with the
    first category dropped so that the intercept and dummy columns are not
    collinear.  Observations excluded from the regression remain missing.
    """
    aligned = pd.concat(
        [
            pd.to_numeric(factor, errors="coerce").rename("factor"),
            pd.to_numeric(market_cap, errors="coerce").rename("market_cap"),
            industry.astype("string").rename("industry"),
        ],
        axis=1,
    )
    valid = (
        np.isfinite(aligned["factor"])
        & np.isfinite(aligned["market_cap"])
        & aligned["market_cap"].gt(0)
        & aligned["industry"].notna()
        & aligned["industry"].str.strip().ne("")
    )
    sample = aligned.loc[valid].copy()
    residuals = pd.Series(np.nan, index=aligned.index, dtype=float, name=factor.name)
    if sample.empty:
        return residuals

    industry_dummies = pd.get_dummies(
        sample["industry"],
        prefix="industry",
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=sample.index, name="intercept"),
            np.log(sample["market_cap"]).rename("log_market_cap"),
            industry_dummies,
        ],
        axis=1,
    )
    x = design.to_numpy(dtype=float)
    y = sample["factor"].to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(x)
    if len(sample) <= rank:
        return residuals

    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals.loc[sample.index] = y - x @ coefficients
    return residuals


def neutralize_factor_values(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    market_cap_column: str | None = None,
    industry_column: str | None = None,
) -> pd.DataFrame:
    """Neutralize a ``date x symbol`` factor table date by date.

    ``market_data`` must use a ``(date, symbol)`` MultiIndex.  The returned
    frame has exactly the same index and columns as ``signals``.
    """
    market_cap_column = market_cap_column or SETTINGS.data.market_cap_column
    industry_column = industry_column or SETTINGS.data.industry_column

    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals 必须是 date × symbol DataFrame")
    if not isinstance(market_data.index, pd.MultiIndex):
        raise ValueError("market_data 必须使用 (date, symbol) MultiIndex")
    if market_data.index.names[:2] != ["date", "symbol"]:
        raise ValueError("market_data 索引前两级必须命名为 date、symbol")

    required = {market_cap_column, industry_column}
    missing = required.difference(market_data.columns)
    if missing:
        raise ValueError(f"中性化缺少字段: {', '.join(sorted(missing))}")

    normalized = signals.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized.index.name = "date"
    normalized.columns.name = "symbol"

    result = pd.DataFrame(
        np.nan,
        index=normalized.index,
        columns=normalized.columns,
        dtype=float,
    )
    signal_dates = set(normalized.index)
    exposures = market_data[[market_cap_column, industry_column]]
    for signal_date, cross_section in exposures.groupby(level="date", sort=False):
        signal_date = pd.Timestamp(signal_date)
        if signal_date not in signal_dates:
            continue
        cross_section = cross_section.droplevel("date")
        symbols = normalized.columns.intersection(cross_section.index)
        if symbols.empty:
            continue
        residuals = neutralize_cross_section(
            normalized.loc[signal_date, symbols],
            cross_section.loc[symbols, market_cap_column],
            cross_section.loc[symbols, industry_column],
        )
        result.loc[signal_date, symbols] = residuals.reindex(symbols).to_numpy()

    result.index.name = signals.index.name
    result.columns.name = signals.columns.name
    return result
