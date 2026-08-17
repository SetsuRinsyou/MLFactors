"""计算 Runner 的评估报告数据并组织输出。"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factors_eval import FactorEvalResult, calc_forward_returns, eval as evaluate_factor
from settings import SETTINGS
POSITIVE_OPERATIONS = {
    "BETA", "CNTD", "CNTN", "CNTP", "HIGH0", "IMAX", "IMIN", "IMXD",
    "KLOW", "KLOW2", "KMID", "KMID2", "KSFT", "KSFT2", "LOW0", "MA",
    "MAX", "MIN", "OPEN0", "QTLD", "QTLU", "RANK", "ROC", "RSV",
    "SUMD", "SUMN", "SUMP", "VWAP0",
}
NEGATIVE_OPERATIONS = {"KLEN", "KUP", "KUP2", "RESI", "STD", "WVMA"}
POSITIVE_FACTORS = {
    "cfp_1d", "current_ratio_1d", "eps_growth_63d", "gross_margin_1d",
    "industry_momentum_1d", "inventory_turnover_1d",
    "net_income_cash_ratio_1d", "net_income_growth_63d",
    "net_profit_margin_1d", "quick_ratio_1d", "revenue_growth_63d",
    "reversal_1m", "reversal_3m", "roe_growth_63d", "slp_reversal_20d",
    "smallcap_growth_1d", "total_asset_turnover_1d",
}
NEGATIVE_FACTORS = {
    "current_liability_ratio_1d", "debt_per_share_1d", "debt_to_asset_1d",
    "high_r_std_84d", "long_term_debt_ratio_1d", "max_daily_return_5d",
    "operating_expense_ratio_1d", "r_std_84d", "size_factor_1d", "sp_1d",
    "total_asset_growth_63d", "turnover_cv_20d", "vff3_63d",
    "vff3_up_63d", "vff3_upd_63d",
}
def factor_direction(factor_name: str, config: dict[str, Any]) -> str:
    """按经济逻辑返回 +、- 或 ~；全部派生因子固定为 ~。"""
    params = config.get("params", {})
    if params.get("expansion_kind"):
        return "~"
    operation = str(params.get("operation", "")).upper()
    if operation in POSITIVE_OPERATIONS or factor_name in POSITIVE_FACTORS:
        return "+"
    if operation in NEGATIVE_OPERATIONS or factor_name in NEGATIVE_FACTORS:
        return "-"
    return "~"


def weekly_ic_statistics(
    ic_series: pd.Series,
    direction: str,
    full_period_ic_mean: float,
) -> dict[str, Any]:
    """以自然周全部有效交易日IC均值计算胜率、自相关和最长失效。"""
    values = pd.to_numeric(ic_series, errors="coerce").dropna().sort_index()
    if not isinstance(values.index, pd.DatetimeIndex):
        values.index = pd.to_datetime(values.index)
    weekly = values.groupby(
        values.index.to_period(SETTINGS.report.weekly_ic_frequency)
    ).mean()
    if direction == "+":
        reference = 1
    elif direction == "-":
        reference = -1
    elif pd.notna(full_period_ic_mean):
        reference = int(np.sign(full_period_ic_mean))
    else:
        reference = 0
    if weekly.empty or reference == 0:
        return {
            "basis": "~", "sign": "~", "week_count": len(weekly),
            "win_rate": np.nan, "autocorrelation": np.nan,
            "max_failure_streak": np.nan, "failure_interval": "~",
        }
    oriented = weekly * reference
    failures = oriented <= 0
    group_ids = failures.ne(failures.shift()).cumsum()
    failure_groups = [
        part for _, part in failures.groupby(group_ids) if bool(part.iloc[0])
    ]
    longest = (
        max(failure_groups, key=len) if failure_groups else pd.Series(dtype=bool)
    )
    interval = (
        f"{longest.index[0]} 至 {longest.index[-1]}"
        if not longest.empty else "无"
    )
    return {
        "basis": "经济方向" if direction in {"+", "-"} else "全时段IC均值符号",
        "sign": "+" if reference > 0 else "-",
        "week_count": int(len(weekly)),
        "win_rate": float((oriented > 0).mean()),
        "autocorrelation": (
            float(weekly.autocorr(lag=1)) if len(weekly) >= 3 else np.nan
        ),
        "max_failure_streak": int(len(longest)),
        "failure_interval": interval,
    }


@dataclass(frozen=True)
class BacktestWindow:
    """自然年分段回测窗口。"""

    year: int
    start: pd.Timestamp
    end: pd.Timestamp


class RunnerReportsMixin:
    """提供 Runner 使用的报告评估与输出方法。"""

    def _iter_yearly_windows(self) -> list[BacktestWindow]:
        """生成覆盖已加载数据范围的自然年窗口。"""
        if self.data.empty:
            raise ValueError("尚未加载行情数据")
        data_dates = pd.DatetimeIndex(
            self.data.index.get_level_values("date").unique()
        ).sort_values()
        first_date, last_date = data_dates.min(), data_dates.max()
        windows = []
        for year in range(first_date.year, last_date.year + 1):
            start = max(pd.Timestamp(year=year, month=1, day=1), first_date)
            end = min(pd.Timestamp(year=year, month=12, day=31), last_date)
            if start <= end:
                windows.append(BacktestWindow(year=year, start=start, end=end))
        return windows

    def save_daily_metrics(
        self,
        evaluations: dict[str, dict[int, FactorEvalResult]],
    ) -> Path:
        """保存带信号日和下一真实交易日的双版本日频截面指标。"""
        version_frames = []
        for version in SETTINGS.report.factor_versions:
            evaluation = evaluations[version].get(SETTINGS.report.daily_metrics_period)
            if evaluation is None:
                raise ValueError(
                    f"{version} 日频指标要求 forward_periods 包含 "
                    f"{SETTINGS.report.daily_metrics_period}"
                )
            version_frames.append(
                self._suffix_metrics(evaluation.daily_metrics, version)
            )
        daily_metrics = pd.concat(version_frames, axis=1).sort_index()
        daily_metrics.index = pd.to_datetime(daily_metrics.index)
        daily_metrics.index.name = "signal_date"
        trading_calendar = self.data_loader.trading_calendar
        if trading_calendar.empty:
            trading_calendar = pd.DatetimeIndex(
                self.data.index.get_level_values("date").unique()
            ).sort_values()
        available_date_map = pd.Series(
            trading_calendar[1:].to_numpy(),
            index=trading_calendar[:-1],
        )
        daily_metrics.insert(
            0,
            "available_date",
            daily_metrics.index.map(available_date_map),
        )
        self.daily_metrics = daily_metrics.reset_index()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "daily_factor_metrics.csv"
        self.daily_metrics.to_csv(
            output_path,
            index=False,
            na_rep="",
            date_format="%Y-%m-%d",
        )
        return output_path

    def _load_market_state_segments(
        self,
    ) -> dict[str, dict[int, list[pd.DatetimeIndex]]]:
        """按2014—2019、2020—2026拆分状态并保留有效连续片段。"""
        if self.market_state_path is None:
            return {}
        if not self.market_state_path.exists():
            raise FileNotFoundError(
                f"市场状态文件不存在: {self.market_state_path}"
            )
        states = pd.read_csv(self.market_state_path, encoding="utf-8-sig")
        states = states.rename(
            columns={
                "signal date": "signal_date",
                "available date": "available_date",
            }
        )
        required = {"signal_date", "state"}
        missing = required.difference(states.columns)
        if missing:
            raise ValueError(
                f"市场状态文件缺少字段: {', '.join(sorted(missing))}"
            )
        states["signal_date"] = pd.to_datetime(
            states["signal_date"],
            errors="raise",
        )
        states["state"] = pd.to_numeric(states["state"], errors="raise").astype(int)
        states = states[["signal_date", "state"]].sort_values("signal_date")
        if states["signal_date"].duplicated().any():
            raise ValueError("市场状态文件存在重复 signal_date")
        unexpected_states = set(states["state"]).difference(
            SETTINGS.report.market_state_names
        )
        if unexpected_states:
            raise ValueError(f"市场状态取值非法: {sorted(unexpected_states)}")

        calendar = self.data_loader.trading_calendar
        if calendar.empty:
            return {}
        states = states.loc[
            states["signal_date"].between(calendar.min(), calendar.max())
        ].copy()
        if states.empty:
            return {}
        expected_dates = calendar[
            (calendar >= states["signal_date"].min())
            & (calendar <= states["signal_date"].max())
        ]
        actual_dates = pd.DatetimeIndex(states["signal_date"])
        if not actual_dates.equals(expected_dates):
            missing_dates = expected_dates.difference(actual_dates)
            extra_dates = actual_dates.difference(expected_dates)
            raise ValueError(
                "市场状态日期与真实交易日不一致；"
                f"缺少={missing_dates[:5].date.tolist()}，"
                f"额外={extra_dates[:5].date.tolist()}"
            )

        eras: dict[str, dict[int, list[pd.DatetimeIndex]]] = {}
        for era, (start, end) in SETTINGS.report.market_state_eras.items():
            era_states = states.loc[states["signal_date"].between(start, end)].copy()
            era_states["segment_id"] = era_states["state"].ne(
                era_states["state"].shift()
            ).cumsum()
            segments = {state: [] for state in SETTINGS.report.market_state_names}
            for _, segment in era_states.groupby("segment_id", sort=True):
                if len(segment) < SETTINGS.report.minimum_market_state_segment_days:
                    continue
                state = int(segment["state"].iloc[0])
                segments[state].append(
                    pd.DatetimeIndex(segment["signal_date"]).sort_values()
                )
            eras[era] = segments
        return eras

    def save_market_state_summary(
        self,
        signal_versions: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """分两个时代、按连续市场状态片段计算双版本回测。"""
        segments_by_era = self._load_market_state_segments()
        if not segments_by_era:
            self.market_state_summary = pd.DataFrame()
            return self.market_state_summary

        rows = []
        data_dates = self.data.index.get_level_values("date")
        benchmark_column = (
            SETTINGS.report.preferred_benchmark_column
            if SETTINGS.report.preferred_benchmark_column in self.benchmark.columns
            else (self.benchmark.columns[0] if not self.benchmark.empty else None)
        )
        for era, segments_by_state in segments_by_era.items():
            for state in SETTINGS.report.market_state_names:
                segments = segments_by_state[state]
                if not segments:
                    continue
                state_dates = pd.DatetimeIndex(
                    sorted({date for segment in segments for date in segment})
                )
                state_data = self.data.loc[data_dates.isin(state_dates)]
                segment_returns = []
                benchmark_returns = []
                sampling_dates = []
                for segment_dates in segments:
                    segment_data = self.data.loc[data_dates.isin(segment_dates)]
                    segment_returns.append(
                        calc_forward_returns(
                            segment_data,
                            SETTINGS.report.daily_metrics_period,
                            price_col=self.eval_price_col,
                        )
                    )
                    if benchmark_column is not None:
                        prices = self.benchmark[benchmark_column].reindex(segment_dates)
                        benchmark_returns.append(
                            (
                                prices.shift(
                                    -(1 + SETTINGS.report.daily_metrics_period)
                                )
                                / prices.shift(-1)
                                - 1
                            )
                            .dropna()
                        )
                    sampling_dates.extend(
                        segment_dates[::SETTINGS.report.daily_metrics_period]
                    )
                forward_returns = pd.concat(segment_returns).sort_index()
                state_benchmark_returns = (
                    pd.concat(benchmark_returns).sort_index()
                    if benchmark_returns else None
                )

                summaries = []
                for version in SETTINGS.report.factor_versions:
                    signals = signal_versions[version].reindex(state_dates)
                    signals.index.name = "date"
                    signals.columns.name = "symbol"
                    evaluation = evaluate_factor(
                        signals,
                        state_data,
                        forward_period=SETTINGS.report.daily_metrics_period,
                        n_groups=self.n_groups,
                        ic_method=self.ic_method,
                        price_col=self.eval_price_col,
                        forward_returns=forward_returns,
                        sampling_dates=pd.DatetimeIndex(sampling_dates),
                        benchmark_returns=state_benchmark_returns,
                    )
                    version_summary = evaluation.summary.copy()
                    full_ic_mean = float(
                        self.evaluations[version][SETTINGS.report.daily_metrics_period]
                        .summary.iloc[0]["IC_mean"]
                    )
                    weekly = weekly_ic_statistics(
                        evaluation.full_ic_series,
                        factor_direction(self.factor_name, self.factor_config),
                        full_ic_mean,
                    )
                    version_summary["weekly_IC_win_rate"] = weekly["win_rate"]
                    version_summary["weekly_IC_week_count"] = weekly["week_count"]
                    version_summary["weekly_IC_reference_sign"] = weekly["sign"]
                    summaries.append(self._suffix_metrics(version_summary, version))
                row = pd.concat(summaries, axis=1).reset_index()
                row.insert(0, "state", state)
                row.insert(0, "era", era)
                row.insert(2, "segment_count", len(segments))
                row.insert(3, "state_trading_days", len(state_dates))
                rows.append(row)

        self.market_state_summary = (
            pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        )
        return self.market_state_summary

    @staticmethod
    def _grouped_rank_ic(sample: pd.DataFrame) -> pd.DataFrame:
        """向量化计算每个signal_date×industry截面的Spearman RankIC。"""
        valid = sample.dropna(subset=["factor", "forward_return"]).copy()
        if valid.empty:
            return pd.DataFrame(
                columns=["signal_date", SETTINGS.data.industry_column, "IC"]
            )
        keys = [valid["signal_date"], valid[SETTINGS.data.industry_column]]
        valid["factor_rank"] = valid.groupby(
            keys,
            sort=False,
        )["factor"].rank(method="average")
        valid["return_rank"] = valid.groupby(
            keys,
            sort=False,
        )["forward_return"].rank(method="average")
        valid["rank_product"] = valid["factor_rank"] * valid["return_rank"]
        valid["factor_square"] = valid["factor_rank"] ** 2
        valid["return_square"] = valid["return_rank"] ** 2
        grouped = valid.groupby(
            ["signal_date", SETTINGS.data.industry_column],
            sort=True,
        ).agg(
            valid_stock_count=("factor", "size"),
            factor_sum=("factor_rank", "sum"),
            return_sum=("return_rank", "sum"),
            factor_square_sum=("factor_square", "sum"),
            return_square_sum=("return_square", "sum"),
            product_sum=("rank_product", "sum"),
        )
        n = grouped["valid_stock_count"].astype(float)
        numerator = n * grouped["product_sum"] - (
            grouped["factor_sum"] * grouped["return_sum"]
        )
        denominator = np.sqrt(
            (
                n * grouped["factor_square_sum"]
                - grouped["factor_sum"] ** 2
            )
            * (
                n * grouped["return_square_sum"]
                - grouped["return_sum"] ** 2
            )
        )
        grouped["IC"] = numerator.divide(denominator.where(denominator > 0))
        grouped.loc[grouped["valid_stock_count"] < 3, "IC"] = np.nan
        return grouped.reset_index()[
            [
                "signal_date",
                SETTINGS.data.industry_column,
                "valid_stock_count",
                "IC",
            ]
        ]

    def save_industry_daily_ic(
        self,
        signal_versions: dict[str, pd.DataFrame],
        save: bool = True,
    ) -> Path | pd.DataFrame:
        """计算行业日频Rank IC；正式MD流水线可只保留内存结果。"""
        forward_returns = calc_forward_returns(
            self.data,
            SETTINGS.report.daily_metrics_period,
            price_col=self.eval_price_col,
        ).rename("forward_return")
        industry = self.data[SETTINGS.data.industry_column].astype("string")
        industry = industry[industry.notna() & industry.str.strip().ne("")]
        membership = (
            industry.rename(SETTINGS.data.industry_column)
            .reset_index()
            .groupby(["date", SETTINGS.data.industry_column], sort=True)
            .size()
            .rename("industry_stock_count")
            .reset_index()
            .rename(columns={"date": "signal_date"})
        )
        base = pd.concat([industry, forward_returns], axis=1).reset_index()
        base = base.rename(columns={"date": "signal_date"})
        version_results = []
        for version in SETTINGS.report.factor_versions:
            factor = signal_versions[version].stack().rename("factor")
            factor.index.names = ["signal_date", "symbol"]
            sample = base.merge(
                factor.reset_index(),
                on=["signal_date", "symbol"],
                how="left",
            )
            result = self._grouped_rank_ic(sample).rename(
                columns={
                    "valid_stock_count": f"valid_stock_count_{version}",
                    "IC": f"IC_{version}",
                }
            )
            version_results.append(result)

        industry_ic = membership
        for result in version_results:
            industry_ic = industry_ic.merge(
                result,
                on=["signal_date", SETTINGS.data.industry_column],
                how="left",
            )
        calendar = self.data_loader.trading_calendar
        available_date_map = pd.Series(
            calendar[1:].to_numpy(),
            index=calendar[:-1],
        )
        industry_ic.insert(
            1,
            "available_date",
            industry_ic["signal_date"].map(available_date_map),
        )
        self.industry_daily_ic = industry_ic
        if not save:
            return industry_ic
        output_path = self.output_dir / "industry_daily_ic.csv"
        industry_ic.to_csv(
            output_path,
            index=False,
            na_rep="",
            date_format="%Y-%m-%d",
        )
        return output_path

    def _evaluate_yearly_window(
        self,
        signal_versions: dict[str, pd.DataFrame],
        window: BacktestWindow,
    ) -> pd.DataFrame | None:
        """在自然年边界内重算报告周期指标并返回内存汇总。"""
        if self.data.empty:
            return None
        data_dates = self.data.index.get_level_values("date")
        window_data = self.data.loc[
            (data_dates >= window.start) & (data_dates <= window.end)
        ]
        window_signal_versions = {}
        for version in SETTINGS.report.factor_versions:
            signals = signal_versions[version]
            signal_dates = pd.DatetimeIndex(signals.index)
            window_signal_versions[version] = signals.loc[
                (signal_dates >= window.start) & (signal_dates <= window.end)
            ]
        if window_data.empty or all(
            signals.dropna(how="all").empty
            for signals in window_signal_versions.values()
        ):
            return None

        evaluations, summary = self.evaluate_versions(
            window_data,
            window_signal_versions,
            forward_periods=(SETTINGS.report.daily_metrics_period,),
        )
        direction = factor_direction(self.factor_name, self.factor_config)
        for version in SETTINGS.report.factor_versions:
            full_ic_mean = float(
                self.evaluations[version][SETTINGS.report.daily_metrics_period]
                .summary.iloc[0]["IC_mean"]
            )
            weekly = weekly_ic_statistics(
                evaluations[version][SETTINGS.report.daily_metrics_period].full_ic_series,
                direction,
                full_ic_mean,
            )
            period = SETTINGS.report.daily_metrics_period
            summary.loc[period, f"weekly_IC_win_rate_{version}"] = weekly["win_rate"]
            summary.loc[period, f"weekly_IC_week_count_{version}"] = weekly["week_count"]
            summary.loc[period, f"weekly_IC_reference_sign_{version}"] = weekly["sign"]
        yearly_summary = summary.reset_index()
        yearly_summary.insert(0, "year", window.year)
        return yearly_summary

    def save_yearly_reports(
        self,
        signal_versions: dict[str, pd.DataFrame],
    ) -> None:
        """计算自然年报告周期汇总并保存年度趋势图。"""
        from plot import save_yearly_summary_trends

        windows = self._iter_yearly_windows()
        yearly_dir = self.output_dir / "yearly"
        if yearly_dir.is_dir():
            shutil.rmtree(yearly_dir)

        summary_frames: list[pd.DataFrame] = []
        for window in windows:
            yearly_summary = self._evaluate_yearly_window(
                signal_versions,
                window,
            )
            if yearly_summary is None:
                continue
            summary_frames.append(yearly_summary)

        if not summary_frames:
            self.yearly_summary = pd.DataFrame()
            return

        self.yearly_summary = pd.concat(summary_frames, ignore_index=True)
        save_yearly_summary_trends(
            self.yearly_summary,
            self.output_dir / "yearly_factor_summary_trends.png",
            factor_name=self.factor_name,
        )

    def build_reports(
        self,
        signal_versions: dict[str, pd.DataFrame],
        save_periodic_reports: bool = True,
    ) -> dict[str, dict[int, FactorEvalResult]]:
        """生成普通 Runner 流程所需的全部报告。"""
        from report_render import build_factor_report

        self.evaluations, self.summary = self.evaluate_versions(
            self.data,
            signal_versions,
        )
        self.save_daily_metrics(self.evaluations)
        self.save_market_state_summary(signal_versions)
        self.save_industry_daily_ic(signal_versions)
        if save_periodic_reports:
            self.save_yearly_reports(signal_versions)
        build_factor_report(self, signal_versions, self.factor_config)
        return self.evaluations

    def build_markdown_report(
        self,
        signal_versions: dict[str, pd.DataFrame],
    ) -> None:
        """从已保存因子值生成正式 Markdown 报告。"""
        from report_render import build_factor_report

        self.raw_result = signal_versions["raw"]
        self.neutralized_result = signal_versions["neutralization"]
        self.evaluations, self.summary = self.evaluate_versions(
            self.data,
            signal_versions,
        )
        self.save_market_state_summary(signal_versions)
        self.save_industry_daily_ic(signal_versions, save=False)
        yearly = []
        for window in self._iter_yearly_windows():
            summary = self._evaluate_yearly_window(signal_versions, window)
            if summary is not None:
                yearly.append(summary)
        self.yearly_summary = (
            pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame()
        )
        build_factor_report(self, signal_versions, self.factor_config)


def remove_legacy_report_outputs(output_dir: Path) -> None:
    """删除正式 Markdown 流水线不再保留的旧报告附件。"""
    for filename in (
        "factor_report.md",
        "report.md",
        "report.html",
        "daily_factor_metrics.csv",
        "factor_summary.csv",
        "industry_daily_ic.csv",
        "market_state_summary.csv",
        "yearly_summary.csv",
        "yearly_factor_summary_trends.png",
    ):
        path = output_dir / filename
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("*.png"):
        path.unlink()
    yearly_dir = output_dir / "yearly"
    if yearly_dir.is_dir():
        shutil.rmtree(yearly_dir)
