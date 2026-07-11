"""Haitong report-26 factor weighting, orthogonalization and timing schemes.

The report proves three relationships rather than prescribing an empirical
parameter set. This module uses an explicit, point-in-time implementation:

* daily cross-sectional Fama-MacBeth factor-premium estimation;
* rolling 24-month mean premiums as static factor weights;
* sequential orthogonalization as an alternative factor representation; and
* Qian (2012) timing, which predicts premiums from market-state variables.

All final scores are jointly neutralized by market capitalization and sector.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor
from scheme.MaxICWeight import ALL_FACTOR_COLUMNS
from scheme.neutralization import neutralize_market_cap_sector


DEFAULT_PREMIUM_WINDOW = 24 * 21
DEFAULT_MIN_STOCKS = 30
DEFAULT_MIN_FACTOR_COVERAGE = 0.5


class CombinationMethod(str, Enum):
    """Report-26 combination methods implemented by this module."""

    FAMA_MACBETH = "fama_macbeth"
    FAMA_MACBETH_ORTH = "fama_macbeth_orth"
    FAMA_MACBETH_TIMING = "fama_macbeth_timing"


def _zscore_by_date(values: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally standardize a date x symbol factor panel."""
    mean = values.mean(axis=1, skipna=True)
    std = values.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
    return values.sub(mean, axis=0).div(std, axis=0)


def _one_day_forward_return(close: pd.DataFrame) -> pd.DataFrame:
    """Return from t+1 to t+2, matching the framework's one-day evaluation."""
    return close.shift(-2).div(close.shift(-1)).sub(1.0)


def _orthogonalize_matrix(values: np.ndarray) -> np.ndarray:
    """Sequentially orthogonalize factor columns, including an intercept.

    QR decomposition is numerically equivalent to residualizing every new
    factor against all preceding factors. Each residual is rescaled to unit
    cross-sectional variance and oriented consistently with its source factor.
    """
    n_stocks, n_factors = values.shape
    if n_stocks <= n_factors + 1:
        return np.zeros_like(values)

    design = np.column_stack([np.ones(n_stocks), values])
    try:
        q, r = np.linalg.qr(design, mode="reduced")
    except np.linalg.LinAlgError:
        return np.zeros_like(values)

    diagonal = np.abs(np.diag(r))
    threshold = np.finfo(float).eps * max(design.shape) * max(float(diagonal.max()), 1.0)
    result = np.zeros_like(values)
    scale = np.sqrt(n_stocks)
    for factor_position in range(n_factors):
        qr_position = factor_position + 1
        if qr_position >= len(diagonal) or diagonal[qr_position] <= threshold:
            continue
        residual = q[:, qr_position] * scale
        if float(np.dot(residual, values[:, factor_position])) < 0.0:
            residual = -residual
        result[:, factor_position] = residual
    return result


class HaitongFactorWeightOrthTiming(BaseFactor):
    """Base class for the report-26 factor-combination methods."""

    name = "haitong_fmb_24m"
    description = "海通选股因子系列研究26：Fama-MacBeth 因子加权"
    method = CombinationMethod.FAMA_MACBETH
    factor_columns = ALL_FACTOR_COLUMNS

    def __init__(
        self,
        premium_window: int = DEFAULT_PREMIUM_WINDOW,
        min_cross_section_stocks: int = DEFAULT_MIN_STOCKS,
        min_factor_coverage: float = DEFAULT_MIN_FACTOR_COVERAGE,
        benchmark_col: str = "hs300_close",
        timing_return_windows: tuple[int, ...] | list[int] = (5, 21),
        timing_volatility_window: int = 21,
    ) -> None:
        if premium_window <= 0:
            raise ValueError("premium_window 必须为正整数")
        if min_cross_section_stocks < 3:
            raise ValueError("min_cross_section_stocks 至少为 3")
        if not 0.0 < min_factor_coverage <= 1.0:
            raise ValueError("min_factor_coverage 必须在 (0, 1] 内")
        if not timing_return_windows or any(window <= 0 for window in timing_return_windows):
            raise ValueError("timing_return_windows 必须包含正整数")
        if timing_volatility_window <= 1:
            raise ValueError("timing_volatility_window 必须大于 1")

        self.premium_window = premium_window
        self.min_cross_section_stocks = min_cross_section_stocks
        self.min_factor_coverage = min_factor_coverage
        self.benchmark_col = benchmark_col
        self.timing_return_windows = tuple(timing_return_windows)
        self.timing_volatility_window = timing_volatility_window

    @property
    def use_orthogonal_factors(self) -> bool:
        return self.method == CombinationMethod.FAMA_MACBETH_ORTH

    @property
    def use_timing(self) -> bool:
        return self.method == CombinationMethod.FAMA_MACBETH_TIMING

    def _prepare_factor_panels(
        self,
        data: pd.DataFrame,
        close: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        missing = [column for column in self.factor_columns if column not in data.columns]
        if missing:
            raise ValueError(f"factor_result 字段缺失: {', '.join(missing)}")

        panels: dict[str, pd.DataFrame] = {}
        available_count = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=int)
        for column in self.factor_columns:
            raw = (
                data[column]
                .unstack("symbol")
                .astype(float)
                .reindex(index=close.index, columns=close.columns)
                .replace([np.inf, -np.inf], np.nan)
            )
            standardized = _zscore_by_date(raw)
            available_count = available_count.add(standardized.notna().astype(int), fill_value=0).astype(int)
            # Missing values represent unavailable signals. After z-score, the
            # cross-sectional mean is zero and is the neutral imputation value.
            panels[column] = standardized.fillna(0.0)
        return panels, available_count

    def _factor_matrix(
        self,
        panels: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> np.ndarray:
        matrix = np.column_stack(
            [panels[column].loc[date].to_numpy(dtype=float) for column in self.factor_columns]
        )
        if self.use_orthogonal_factors:
            return _orthogonalize_matrix(matrix)
        return matrix

    def _estimate_premiums(
        self,
        panels: dict[str, pd.DataFrame],
        forward_return: pd.DataFrame,
    ) -> pd.DataFrame:
        premiums = pd.DataFrame(
            np.nan,
            index=forward_return.index,
            columns=self.factor_columns,
            dtype=float,
        )
        minimum_observations = max(
            self.min_cross_section_stocks,
            len(self.factor_columns) + 2,
        )

        for date in forward_return.index:
            returns = forward_return.loc[date].replace([np.inf, -np.inf], np.nan)
            valid = returns.notna().to_numpy()
            if int(valid.sum()) < minimum_observations:
                continue

            y = returns.loc[returns.notna()].to_numpy(dtype=float)
            y_std = y.std(ddof=0)
            if y_std <= 0.0 or not np.isfinite(y_std):
                continue
            y = (y - y.mean()) / y_std
            x = self._factor_matrix(panels, date)[valid]
            design = np.column_stack([np.ones(len(y)), x])
            try:
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            premiums.loc[date] = beta[1:]
        return premiums

    def _market_conditions(
        self,
        data: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        if self.benchmark_col not in data.columns:
            raise ValueError(f"因子择时缺少指数价格字段: {self.benchmark_col}")

        benchmark = (
            data[self.benchmark_col]
            .groupby(level="date")
            .first()
            .reindex(dates)
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        daily_return = benchmark.pct_change(fill_method=None)
        conditions = {
            f"market_return_{window}d": benchmark.pct_change(window, fill_method=None)
            for window in self.timing_return_windows
        }
        conditions[f"market_volatility_{self.timing_volatility_window}d"] = (
            daily_return.rolling(self.timing_volatility_window).std(ddof=0)
        )
        return pd.DataFrame(conditions, index=dates).replace([np.inf, -np.inf], np.nan)

    def _timed_weights(
        self,
        premiums: pd.DataFrame,
        conditions: pd.DataFrame,
        position: int,
    ) -> pd.Series:
        current_condition = conditions.iloc[position]
        if current_condition.isna().any():
            return pd.Series(np.nan, index=self.factor_columns, dtype=float)

        history_start = position - self.premium_window - 1
        history_end = position - 1
        condition_history = conditions.iloc[history_start:history_end]
        minimum_observations = max(
            min(60, self.premium_window),
            len(conditions.columns) + 2,
        )
        weights = pd.Series(np.nan, index=self.factor_columns, dtype=float)

        for column in self.factor_columns:
            history = pd.concat(
                [premiums[column].iloc[history_start:history_end], condition_history],
                axis=1,
            ).dropna()
            if len(history) < minimum_observations:
                continue
            y = history.iloc[:, 0].to_numpy(dtype=float)
            x = history.iloc[:, 1:].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            try:
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            weights.loc[column] = beta[0] + float(current_condition.to_numpy(float) @ beta[1:])
        return weights

    def _weights_at_date(
        self,
        premiums: pd.DataFrame,
        conditions: pd.DataFrame | None,
        position: int,
    ) -> pd.Series:
        if self.use_timing:
            if conditions is None:
                raise RuntimeError("因子择时缺少条件变量")
            return self._timed_weights(premiums, conditions, position)

        history_start = position - self.premium_window - 1
        history_end = position - 1
        return premiums.iloc[history_start:history_end].mean(axis=0, skipna=True)

    def _combine(
        self,
        panels: dict[str, pd.DataFrame],
        available_count: pd.DataFrame,
        premiums: pd.DataFrame,
        conditions: pd.DataFrame | None,
    ) -> pd.DataFrame:
        dates = premiums.index
        symbols = available_count.columns
        result = pd.DataFrame(np.nan, index=dates, columns=symbols, dtype=float)
        required_factor_count = int(np.ceil(len(self.factor_columns) * self.min_factor_coverage))

        # At t, premium[t-1] still needs the t+1 price. Excluding it ensures
        # that both static and timed weights only use information available at t.
        for position in range(self.premium_window + 1, len(dates)):
            weights = self._weights_at_date(premiums, conditions, position)
            if weights.notna().sum() == 0:
                continue
            matrix = self._factor_matrix(panels, dates[position])
            score = matrix @ weights.fillna(0.0).to_numpy(dtype=float)
            eligible = available_count.loc[dates[position]].to_numpy() >= required_factor_count
            result.iloc[position, eligible] = score[eligible]

        result.index.name = "date"
        result.columns.name = "symbol"
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")
        for column in ("adj_close", "market_cap", "sector"):
            if column not in data.columns:
                raise ValueError(f"组合因子缺少字段: {column}")

        close = data["adj_close"].unstack("symbol").astype(float).sort_index()
        close = close.replace([np.inf, -np.inf], np.nan)
        panels, available_count = self._prepare_factor_panels(data, close)
        forward_return = _one_day_forward_return(close)
        premiums = self._estimate_premiums(panels, forward_return)
        conditions = self._market_conditions(data, close.index) if self.use_timing else None
        result = self._combine(panels, available_count, premiums, conditions)
        return neutralize_market_cap_sector(result, data)


@register_factor
class HaitongFamaMacBeth(HaitongFactorWeightOrthTiming):
    name = "haitong_fmb_24m"
    description = "海通选股因子系列研究26：Fama-MacBeth 因子溢价加权"
    method = CombinationMethod.FAMA_MACBETH


@register_factor
class HaitongFamaMacBethOrth(HaitongFactorWeightOrthTiming):
    name = "haitong_fmb_orth_24m"
    description = "海通选股因子系列研究26：正交因子 Fama-MacBeth 加权"
    method = CombinationMethod.FAMA_MACBETH_ORTH


@register_factor
class HaitongFamaMacBethTiming(HaitongFactorWeightOrthTiming):
    name = "haitong_fmb_timing_24m"
    description = "海通选股因子系列研究26：条件变量驱动的因子溢价择时"
    method = CombinationMethod.FAMA_MACBETH_TIMING
