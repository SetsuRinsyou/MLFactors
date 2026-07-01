from factors.base import BaseFactor
from factors.registry import register_factor
import numpy as np
import pandas as pd


@register_factor
class MaxICWeight(BaseFactor):
    """
    最大IC权重因子：基于海通选股因子系列研究的多因子组合优化策略。
    该因子通过计算各个单因子的IC值，并根据IC值进行加权，得到一个综合的多因子信号。
    """

    name = "max_ic_weight_10d"
    description = "最大IC权重因子：10日IC均值加权"

    def __init__(self, ic_window: int = 10) -> None:
        self.ic_window = ic_window
        self.name = f"max_ic_weight_{self.ic_window}d"

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        factor_columns = [
            "beta_252d",
            "beta_mn_252d",
            "beta_mp_252d",
            "beta_n_252d",
            "beta_p_252d",
            "dip_rebound_10d",
            "high_r_std_84d",
            "hml_r_std_84d",
            "industry_momentum_1d",
            # "max_daily_return_5d",
            # "multi_horizon_llt_50d",
            # "multi_horizon_ma_50d",
            "open_high_surge_10d",
            "pair_reversal_60d",
            "r_std_84d",
            "reversal_1m",
            "reversal_3m",
            "reversal_6m",
            "size_factor_1d",
            "size_squared_1d",
            "slp_reversal_20d",
            "smallcap_growth_1d",
            "style_category_momentum_1d",
            "trading_amount_20d",
            "turnover_cv_20d",
            "vff3_63d",
            "vff3_up_63d",
            "vff3_upd_63d",
            "volume_3m",
            "volume_price_corr_15d",
        ]
        if self.ic_window <= 0:
            raise ValueError("ic_window 必须为正整数")
        if "close" not in data.columns:
            raise ValueError("字段 close 缺失")
        if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
            raise ValueError("因子输入必须使用 (date, symbol) MultiIndex")

        missing = [column for column in factor_columns if column not in data.columns]
        if missing:
            raise ValueError(f"factor_result 字段缺失: {', '.join(missing)}")

        close = data["close"].unstack("symbol").astype(float).sort_index()
        close = close.replace([np.inf, -np.inf], np.nan)
        raw_factors = {
            column: data[column]
            .unstack("symbol")
            .astype(float)
            .reindex(index=close.index, columns=close.columns)
            .replace([np.inf, -np.inf], np.nan)
            for column in factor_columns
        }

        available_dates = pd.DataFrame(
            {column: raw_factors[column].notna().any(axis=1) for column in factor_columns}
        ).all(axis=1)
        if not available_dates.any():
            raise ValueError("没有找到所有 factor_result 同时具备的日期")
        start_date = available_dates[available_dates].index[0]

        close = close.loc[start_date:]
        raw_factors = {
            column: factor.loc[start_date:]
            for column, factor in raw_factors.items()
        }
        forward_return = close.shift(-2) / close.shift(-1) - 1
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        normalized_factors = {}
        for column, factor in raw_factors.items():
            mean = factor.mean(axis=1, skipna=True)
            std = factor.std(axis=1, skipna=True, ddof=0).replace(0, np.nan)
            normalized_factors[column] = factor.sub(mean, axis=0).div(std, axis=0)

        factor_ic = pd.DataFrame(index=close.index, columns=factor_columns, dtype=float)
        corr_matrices = []
        for date in close.index:
            factor_frame = pd.DataFrame(
                {column: normalized_factors[column].loc[date] for column in factor_columns}
            )
            factor_ic.loc[date] = factor_frame.rank(axis=0).corrwith(
                forward_return.loc[date].rank(),
                axis=0,
            )
            corr_matrices.append(factor_frame.corr(min_periods=3).to_numpy(dtype=float))

        corr_matrices = np.asarray(corr_matrices, dtype=float)
        for position in range(self.ic_window + 1, len(close)):
            window_start = position - self.ic_window - 1
            window_end = position - 1
            ic_mean = factor_ic.iloc[window_start:window_end].mean(axis=0, skipna=True)
            if ic_mean.notna().sum() == 0:
                continue

            with np.errstate(invalid="ignore"):
                sigma = np.nanmean(corr_matrices[window_start:window_end], axis=0)
            sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
            sigma = (sigma + sigma.T) / 2
            np.fill_diagonal(sigma, 1.0)

            ic_vector = ic_mean.fillna(0.0).to_numpy(dtype=float)
            try:
                weights = np.linalg.solve(sigma, ic_vector)
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(sigma) @ ic_vector

            today = close.index[position]
            today_factors = pd.DataFrame(
                {column: normalized_factors[column].loc[today] for column in factor_columns}
            )
            finite_values = today_factors.replace([np.inf, -np.inf], np.nan)
            result.loc[today] = -finite_values.mul(weights, axis=1).sum(axis=1, min_count=1)

        result.index.name = "date"
        result.columns.name = "symbol"
        if not constituents:
            return result

        mask = pd.DataFrame(False, index=result.index, columns=result.columns)
        for date in result.index:
            allowed = constituents.get(str(pd.Timestamp(date).date()), set())
            if allowed:
                present = result.columns.intersection(allowed)
                mask.loc[date, present] = True
        return result.where(mask)
