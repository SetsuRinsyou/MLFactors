"""Haitong factor-dimension reduction scheme factors.

The methods follow Haitong Securities factor-library governance report 29:
bottom factor dimension-reduction method comparison. Each class first reduces
highly related bottom factors within categories, then combines reduced category
factors by a Max-IC style weight. The final scheme signal is neutralized by
market cap and sector.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor
from scheme.MaxICWeight import (
    FUNDAMENTAL_FACTOR_COLUMNS,
    PRICE_FACTOR_COLUMNS,
    RISK_FACTOR_COLUMNS,
)
from scheme.neutralization import neutralize_market_cap_sector


class ReductionMethod(str, Enum):
    """Supported bottom-factor dimension-reduction methods."""

    BEST_IC = "best_ic"
    BEST_IC_ORTH = "best_ic_orth"
    IC_WEIGHT = "ic_weight"
    IC_WEIGHT_ORTH = "ic_weight_orth"
    PCA_IC = "pca_ic"
    PCA_IC_ORTH = "pca_ic_orth"


REPORT_FACTOR_GROUPS = OrderedDict(
    [
        ("size", ["size_factor_1d"]),
        ("nonlinear_size", ["size_squared_1d", "smallcap_growth_1d"]),
        (
            "liquidity",
            [
                "trading_amount_20d",
                "turnover_cv_20d",
                "volume_3m",
                "volume_price_corr_15d",
            ],
        ),
        (
            "reversal",
            [
                "reversal_1m",
                "reversal_3m",
                "reversal_6m",
                "slp_reversal_20d",
                "pair_reversal_60d",
                "dip_rebound_10d",
                "open_high_surge_10d",
            ],
        ),
        (
            "risk",
            [
                "beta_252d",
                "beta_mn_252d",
                "beta_mp_252d",
                "beta_n_252d",
                "beta_p_252d",
                "high_r_std_84d",
                "hml_r_std_84d",
                "r_std_84d",
                "vff3_63d",
                "vff3_up_63d",
                "vff3_upd_63d",
            ],
        ),
        ("valuation", ["cfp_1d", "dividend_yield_1d", "sp_1d"]),
        (
            "growth",
            [
                "current_asset_growth_12m",
                "current_asset_growth_vol_24m",
                "debt_to_asset_change_12m",
                "dgmarq_12m",
                "dgpoaq_12m",
                "eps_growth_12m",
                "eps_growth_63d",
                "eps_growth_accel_3m",
                "eps_growth_accel_price_adj_3m",
                "eps_growth_accel_vol_adj_3m",
                "eps_growth_price_adj_3m",
                "eps_growth_vol_adj_3m",
                "equity_growth_63d",
                "equity_ratio_change_12m",
                "equity_to_fixed_asset_change_12m",
                "gross_profitability_growth_12m",
                "gross_profitability_trend_qoq_12m",
                "gross_profitability_trend_yoy_48m",
                "net_income_growth_63d",
                "non_current_asset_growth_12m",
                "non_current_asset_growth_vol_24m",
                "revenue_growth_63d",
                "roe_growth_63d",
                "total_asset_growth_63d",
                "total_asset_growth_vol_24m",
            ],
        ),
        (
            "profitability",
            [
                "droaq_12m",
                "gross_margin_1d",
                "gpoaq_3m",
                "net_income_cash_ratio_1d",
                "net_profit_margin_1d",
                "operating_expense_ratio_1d",
                "total_asset_turnover_1d",
            ],
        ),
        (
            "solvency",
            [
                "current_liability_ratio_1d",
                "current_ratio_1d",
                "debt_per_share_1d",
                "debt_to_asset_1d",
                "financial_expense_ratio_1d",
                "fixed_ratio_1d",
                "inventory_turnover_1d",
                "long_term_debt_ratio_1d",
                "long_term_debt_ratio_change_12m",
                "quick_ratio_1d",
            ],
        ),
    ]
)

ALL_DIMENSION_REDUCTION_COLUMNS = list(
    dict.fromkeys(
        FUNDAMENTAL_FACTOR_COLUMNS + PRICE_FACTOR_COLUMNS + RISK_FACTOR_COLUMNS
    )
)
DEFAULT_LOOKBACK_MONTHS = 24
TRADING_DAYS_PER_MONTH = 21
DEFAULT_IC_WINDOW = DEFAULT_LOOKBACK_MONTHS * TRADING_DAYS_PER_MONTH


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score by date."""
    mean = frame.mean(axis=1, skipna=True)
    std = frame.std(axis=1, skipna=True, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _cross_sectional_ic(factor: pd.DataFrame, forward_return: pd.DataFrame) -> pd.Series:
    """Calculate daily rank IC for one factor matrix."""
    result = {}
    for date in factor.index.intersection(forward_return.index):
        aligned = pd.DataFrame(
            {
                "factor": factor.loc[date],
                "return": forward_return.loc[date],
            }
        ).dropna()
        if len(aligned) < 3:
            result[date] = np.nan
        else:
            result[date] = aligned["factor"].rank().corr(aligned["return"].rank())
    return pd.Series(result, dtype=float).sort_index()


def _residualize_panel(target: pd.DataFrame, controls: list[pd.DataFrame]) -> pd.DataFrame:
    """Residualize a date x symbol panel against control panels date by date."""
    if not controls:
        return target

    result = pd.DataFrame(np.nan, index=target.index, columns=target.columns)
    for date in target.index:
        frame = pd.DataFrame({"target": target.loc[date]})
        for idx, control in enumerate(controls):
            frame[f"control_{idx}"] = control.loc[date]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) <= len(controls) + 1:
            continue
        x = pd.concat(
            [
                pd.Series(1.0, index=frame.index, name="const"),
                frame[[f"control_{idx}" for idx in range(len(controls))]],
            ],
            axis=1,
        )
        y = frame["target"]
        beta, *_ = np.linalg.lstsq(x.to_numpy(float), y.to_numpy(float), rcond=None)
        residuals = y - x.to_numpy(float) @ beta
        result.loc[date, residuals.index] = residuals
    return result


def _normalized_signed_weights(values: pd.Series) -> pd.Series:
    """Normalize signed values by sum of absolute values."""
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = values.abs().sum()
    if denom <= 0:
        return pd.Series(1.0 / len(values), index=values.index)
    return values / denom


def _pca_first_component_weights(ic_window: pd.DataFrame) -> pd.Series:
    """Get first principal-component weights from an IC history matrix."""
    valid_columns = ic_window.columns
    matrix = ic_window.fillna(0.0).to_numpy(dtype=float)
    if matrix.size == 0 or len(valid_columns) == 0:
        return pd.Series(dtype=float)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        weights = pd.Series(vh[0], index=valid_columns, dtype=float)
    except np.linalg.LinAlgError:
        weights = pd.Series(1.0, index=valid_columns, dtype=float)

    mean_ic = ic_window.mean(axis=0, skipna=True).reindex(valid_columns).fillna(0.0)
    if float(weights.mul(mean_ic).sum()) < 0:
        weights = -weights
    return _normalized_signed_weights(weights)


class HaitongDimensionReduction(BaseFactor):
    """Base class for Haitong bottom-factor dimension-reduction methods."""

    name = "haitong_dimension_reduction"
    description = "海通因子库治理29：底层因子降维"
    method = ReductionMethod.BEST_IC

    def __init__(self, ic_window: int = DEFAULT_IC_WINDOW) -> None:
        if ic_window <= 0:
            raise ValueError("ic_window 必须为正整数")
        self.ic_window = ic_window

    @property
    def factor_columns(self) -> list[str]:
        """Expected bottom-factor columns."""
        return ALL_DIMENSION_REDUCTION_COLUMNS

    @property
    def use_orthogonal_ic(self) -> bool:
        """Whether this method uses orthogonalized bottom-factor IC."""
        return self.method in {
            ReductionMethod.BEST_IC_ORTH,
            ReductionMethod.IC_WEIGHT_ORTH,
            ReductionMethod.PCA_IC_ORTH,
        }

    def _factor_groups(self, data: pd.DataFrame) -> OrderedDict[str, list[str]]:
        """Return available factor groups in configured order."""
        groups: OrderedDict[str, list[str]] = OrderedDict()
        available = set(data.columns)
        for group_name, columns in REPORT_FACTOR_GROUPS.items():
            present = [column for column in columns if column in available]
            if present:
                groups[group_name] = present
        missing = sorted(set(self.factor_columns) - available)
        if missing:
            raise ValueError(f"factor_result 字段缺失: {', '.join(missing)}")
        return groups

    def _prepare_factor_panels(
        self,
        data: pd.DataFrame,
        close: pd.DataFrame,
        groups: OrderedDict[str, list[str]],
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Build z-scored bottom-factor panels by group."""
        panels: dict[str, dict[str, pd.DataFrame]] = {}
        for group_name, columns in groups.items():
            panels[group_name] = {
                column: _zscore(
                    data[column]
                    .unstack("symbol")
                    .astype(float)
                    .reindex(index=close.index, columns=close.columns)
                    .replace([np.inf, -np.inf], np.nan)
                )
                for column in columns
            }
        return panels

    def _group_weights(self, ic_window: pd.DataFrame) -> pd.Series:
        """Calculate bottom-factor reduction weights for one category."""
        mean_ic = ic_window.mean(axis=0, skipna=True)
        if self.method in {ReductionMethod.BEST_IC, ReductionMethod.BEST_IC_ORTH}:
            weights = pd.Series(0.0, index=ic_window.columns)
            if mean_ic.notna().any():
                weights.loc[mean_ic.abs().idxmax()] = 1.0
            return weights
        if self.method in {ReductionMethod.IC_WEIGHT, ReductionMethod.IC_WEIGHT_ORTH}:
            return _normalized_signed_weights(mean_ic)
        return _pca_first_component_weights(ic_window)

    def _reduce_one_group(
        self,
        group_panels: dict[str, pd.DataFrame],
        columns: list[str],
        forward_return: pd.DataFrame,
    ) -> pd.DataFrame:
        """Reduce one category's bottom factors into a category factor."""
        group_ic = pd.DataFrame(
            {
                column: _cross_sectional_ic(group_panels[column], forward_return)
                for column in columns
            }
        ).reindex(forward_return.index)
        output = pd.DataFrame(np.nan, index=forward_return.index, columns=forward_return.columns)

        for position in range(self.ic_window, len(forward_return.index)):
            today = forward_return.index[position]
            history = group_ic.iloc[position - self.ic_window:position]
            weights = self._group_weights(history).reindex(columns).fillna(0.0)
            today_factors = pd.DataFrame(
                {column: group_panels[column].loc[today] for column in columns}
            )
            output.loc[today] = today_factors.mul(weights, axis=1).sum(axis=1, min_count=1)
        return _zscore(output)

    def _reduce_groups(
        self,
        panels: dict[str, dict[str, pd.DataFrame]],
        groups: OrderedDict[str, list[str]],
        forward_return: pd.DataFrame,
    ) -> pd.DataFrame:
        """Reduce bottom factors into category factors."""
        if self.use_orthogonal_ic:
            return self._reduce_groups_with_orthogonalization(
                panels,
                groups,
                forward_return,
            )

        group_outputs = {}
        for group_name, columns in groups.items():
            group_outputs[group_name] = self._reduce_one_group(
                panels[group_name],
                columns,
                forward_return,
            )

        return pd.concat(group_outputs, axis=1)

    def _reduce_groups_with_orthogonalization(
        self,
        panels: dict[str, dict[str, pd.DataFrame]],
        groups: OrderedDict[str, list[str]],
        forward_return: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sequentially reduce groups after orthogonalizing to prior reduced groups."""
        group_outputs = {}
        previous_reduced_groups: list[pd.DataFrame] = []
        for group_name, columns in groups.items():
            group_panels = panels[group_name]
            if previous_reduced_groups:
                group_panels = {
                    column: _residualize_panel(group_panels[column], previous_reduced_groups)
                    for column in columns
                }
            output = self._reduce_one_group(group_panels, columns, forward_return)
            group_outputs[group_name] = output
            previous_reduced_groups.append(output)

        return pd.concat(group_outputs, axis=1)

    def _combine_categories(
        self,
        reduced_groups: pd.DataFrame,
        forward_return: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine reduced category factors by Max-IC style weights."""
        group_names = list(reduced_groups.columns.get_level_values(0).unique())
        category_factors = {
            group_name: reduced_groups[group_name].reindex(forward_return.index)
            for group_name in group_names
        }
        category_ic = pd.DataFrame(
            {
                group_name: _cross_sectional_ic(category_factors[group_name], forward_return)
                for group_name in group_names
            }
        ).reindex(forward_return.index)
        result = pd.DataFrame(np.nan, index=forward_return.index, columns=forward_return.columns)

        for position in range(self.ic_window, len(forward_return.index)):
            today = forward_return.index[position]
            history = slice(position - self.ic_window, position)
            ic_mean = category_ic.iloc[history].mean(axis=0, skipna=True)
            today_factors = pd.DataFrame(
                {group_name: category_factors[group_name].loc[today] for group_name in group_names}
            )
            if ic_mean.notna().sum() == 0:
                continue

            corr = today_factors.corr(min_periods=3).reindex(
                index=group_names,
                columns=group_names,
            )
            sigma = corr.fillna(0.0).to_numpy(dtype=float)
            sigma = (sigma + sigma.T) / 2
            np.fill_diagonal(sigma, 1.0)
            ic_vector = ic_mean.reindex(group_names).fillna(0.0).to_numpy(dtype=float)
            try:
                weights = np.linalg.solve(sigma, ic_vector)
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(sigma) @ ic_vector

            finite_values = today_factors.replace([np.inf, -np.inf], np.nan)
            result.loc[today] = -finite_values.mul(weights, axis=1).sum(axis=1, min_count=1)

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
        if "adj_close" not in data.columns:
            raise ValueError("字段 adj_close 缺失")

        groups = self._factor_groups(data)
        close = data["adj_close"].unstack("symbol").astype(float).sort_index()
        close = close.replace([np.inf, -np.inf], np.nan)
        forward_return = close.shift(-2) / close.shift(-1) - 1

        panels = self._prepare_factor_panels(data, close, groups)
        reduced_groups = self._reduce_groups(panels, groups, forward_return)
        result = self._combine_categories(reduced_groups, forward_return)
        return neutralize_market_cap_sector(result, data)


@register_factor
class HaitongBestIC(HaitongDimensionReduction):
    name = "haitong_best_ic_24m"
    description = "海通因子库治理29：类别内 IC 最高降维"
    method = ReductionMethod.BEST_IC


@register_factor
class HaitongBestICOrth(HaitongDimensionReduction):
    name = "haitong_best_ic_orth_24m"
    description = "海通因子库治理29：类别内正交 IC 最高降维"
    method = ReductionMethod.BEST_IC_ORTH


@register_factor
class HaitongICWeight(HaitongDimensionReduction):
    name = "haitong_ic_weight_24m"
    description = "海通因子库治理29：类别内 IC 加权降维"
    method = ReductionMethod.IC_WEIGHT


@register_factor
class HaitongICWeightOrth(HaitongDimensionReduction):
    name = "haitong_ic_weight_orth_24m"
    description = "海通因子库治理29：类别内正交 IC 加权降维"
    method = ReductionMethod.IC_WEIGHT_ORTH


@register_factor
class HaitongPCAIC(HaitongDimensionReduction):
    name = "haitong_pca_ic_24m"
    description = "海通因子库治理29：类别内 IC 序列 PCA 降维"
    method = ReductionMethod.PCA_IC


@register_factor
class HaitongPCAICOrth(HaitongDimensionReduction):
    name = "haitong_pca_ic_orth_24m"
    description = "海通因子库治理29：类别内正交 IC 序列 PCA 降维"
    method = ReductionMethod.PCA_IC_ORTH
