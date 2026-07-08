"""加载 CSV 数据并计算注册因子。"""

from dataclasses import dataclass
import importlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from dataloader import DataLoader
from factors_eval import FactorEvalResult, eval as evaluate_factor
from factors.registry import FactorRegistry
from plot import FactorPlotter


DEFAULT_SECURITY_STATUS_PATH = (
    Path(__file__).resolve().parent
    / "cache"
    / "tushare_security_status_missing_tables_20100104_20260623_20260706.csv"
)


@dataclass(frozen=True)
class BacktestWindow:
    """分段回测窗口。"""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    title: str


class Runner:
    """管理因子的数据加载、计算和结果查看。"""

    def __init__(
        self,
        factor_name: str,
        factor_params: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        data_columns: list[str] | None = None,
        data_dir: str | Path = Path(__file__).resolve().parent / "cache" / "csv",
        factor_dir: str | Path | None = None,
        constituents_path: str | Path | None = None,
        constituents: str | None = None,
        forward_periods: tuple[int, ...] = (1, 5, 10, 21),
        n_groups: int = 5,
        ic_method: str = "rank",
        max_lag: int = 20,
        output_dir: str | Path | None = None,
        eval_price_col: str = "adj_close",
        security_status_path: str | Path | None = DEFAULT_SECURITY_STATUS_PATH,
        yearly_reports: bool = True,
        monthly_reports_start: str | date | None = "2024-01-01",
    ) -> None:
        self.factor_name = factor_name
        self.forward_periods = forward_periods
        self.n_groups = n_groups
        self.ic_method = ic_method
        self.max_lag = max_lag
        self.eval_price_col = eval_price_col
        self.security_status_path = Path(security_status_path) if security_status_path is not None else None
        self.output_dir = Path(output_dir or Path("outputs") / factor_name)
        self.yearly_reports = yearly_reports
        self.monthly_reports_start = (
            pd.Timestamp(monthly_reports_start)
            if monthly_reports_start is not None
            else None
        )
        self.factor = FactorRegistry.get(factor_name)(**(factor_params or {}))
        self.data = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.result = pd.DataFrame()
        self.evaluations: dict[int, FactorEvalResult] = {}
        self.summary = pd.DataFrame()
        self.periodic_summary: dict[str, pd.DataFrame] = {}
        loader_columns = (
            None
            if data_columns is None
            else list(dict.fromkeys([*data_columns, eval_price_col]))
        )
        self.data_loader = DataLoader(
            data_dir=data_dir,
            symbols=symbols,
            start=start,
            end=end,
            columns=loader_columns,
            factor_dir=factor_dir,
            constituents_path=constituents_path,
            constituent_index=constituents,
            security_status_path=security_status_path,
        )

    def calculate(self,
                  combined_data: pd.DataFrame | None = None,
                  save: bool = False) -> pd.DataFrame:
        """计算因子；save=True 时按股票保存完整回测期因子值。"""
        signals = self.factor.generate_signals(combined_data, None)
        self.result = signals
        if save:
            factor_dir = self.output_dir / "factors"
            factor_dir.mkdir(parents=True, exist_ok=True)
            symbols = combined_data.index.get_level_values("symbol").unique()
            factor_result = signals.reindex(columns=symbols)
            for symbol in symbols:
                dates = combined_data.xs(symbol, level="symbol").index
                factor_data = factor_result[symbol].reindex(dates).rename(self.factor_name)
                factor_data.index = factor_data.index.strftime("%Y-%m-%d")
                factor_data.index.name = "date"
                factor_data.to_csv(
                    factor_dir / f"{symbol}.csv",
                    na_rep="",
                )
        return signals

    def evaluate(self, combined_data, signals) -> dict[int, FactorEvalResult]:
        """计算 1、5、10、21 日等指定周期的因子评估结果。"""
        evaluations = {
            period: evaluate_factor(
                signals,
                combined_data,
                forward_period=period,
                n_groups=self.n_groups,
                ic_method=self.ic_method,
                max_lag=self.max_lag,
                price_col=self.eval_price_col,
            )
            for period in self.forward_periods
        }
        summary = pd.concat(
            [evaluation.summary for evaluation in evaluations.values()]
        ).sort_index()
        return evaluations, summary

    @staticmethod
    def _slice_market_data(
        data: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """按日期切片 ``(date, symbol)`` MultiIndex 行情数据。"""
        if data.empty:
            return data
        dates = data.index.get_level_values("date")
        return data.loc[(dates >= start) & (dates <= end)]

    @staticmethod
    def _slice_signals(
        signals: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """按日期切片 date × symbol 因子宽表。"""
        if signals.empty:
            return signals
        signal_dates = pd.DatetimeIndex(signals.index)
        return signals.loc[(signal_dates >= start) & (signal_dates <= end)]

    def _data_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """返回已加载行情数据的起止日期。"""
        if self.data.empty:
            raise ValueError("尚未加载行情数据")
        dates = pd.DatetimeIndex(self.data.index.get_level_values("date").unique())
        return dates.min(), dates.max()

    def _iter_yearly_windows(self) -> list[BacktestWindow]:
        """生成覆盖当前数据区间的年度窗口。"""
        start_date, end_date = self._data_date_range()
        windows = []
        for year in range(start_date.year, end_date.year + 1):
            window_start = max(pd.Timestamp(year=year, month=1, day=1), start_date)
            window_end = min(pd.Timestamp(year=year, month=12, day=31), end_date)
            if window_start <= window_end:
                windows.append(
                    BacktestWindow(
                        name=str(year),
                        start=window_start,
                        end=window_end,
                        title=f"{year} 年",
                    )
                )
        return windows

    def _iter_monthly_windows(self) -> list[BacktestWindow]:
        """生成从 monthly_reports_start 开始的逐月窗口。"""
        if self.monthly_reports_start is None:
            return []

        data_start, data_end = self._data_date_range()
        first_date = max(self.monthly_reports_start, data_start)
        month_start = pd.Timestamp(
            year=first_date.year,
            month=first_date.month,
            day=1,
        )
        windows = []
        while month_start <= data_end:
            month_end = month_start + pd.offsets.MonthEnd(1)
            window_start = max(month_start, first_date)
            window_end = min(month_end, data_end)
            if window_start <= window_end:
                name = month_start.strftime("%Y-%m")
                windows.append(
                    BacktestWindow(
                        name=name,
                        start=window_start,
                        end=window_end,
                        title=f"{name} 月",
                    )
                )
            month_start = month_start + pd.offsets.MonthBegin(1)
        return windows

    def _benchmark_cumulative(self, evaluation: FactorEvalResult, period: int) -> pd.DataFrame | None:
        """计算与分层收益日期对齐的基准累计收益。"""
        if self.benchmark.empty:
            return None
        benchmark_returns = (
            self.benchmark.shift(-(1 + period))
            / self.benchmark.shift(-1)
            - 1
        )
        benchmark_returns = benchmark_returns.reindex(
            evaluation.layered.group_returns.index
        )
        return (1 + benchmark_returns).cumprod() - 1

    def save_reports(
        self,
        evaluations: dict[int, FactorEvalResult],
        summary: pd.DataFrame,
        output_dir: str | Path | None = None,
        report_title: str | None = None,
        data: pd.DataFrame | None = None,
        periodic_links: dict[str, list[Path]] | None = None,
    ) -> Path:
        """保存 CSV、综合评估图和包含表格与图片的 Markdown 报告。"""
        output_dir = Path(output_dir or self.output_dir)
        report_data = data if data is not None else self.data
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "factor_summary.csv")

        image_files: list[tuple[int, Path]] = []
        for period, evaluation in evaluations.items():
            output_path = output_dir / f"{self.factor_name}_{period}d.png"
            benchmark_cumulative = self._benchmark_cumulative(evaluation, period)
            FactorPlotter(
                evaluation,
                factor_name=self.factor_name,
                benchmark_cumulative=benchmark_cumulative,
            ).save(output_path)
            image_files.append((period, output_path))

        data_dates = report_data.index.get_level_values("date")

        report_lines = [
            f"# {report_title or f'{self.factor_name} 因子评估报告'}",
            "",
            "## 运行配置",
            "",
            f"- 数据区间：{data_dates.min().date()} 至 {data_dates.max().date()}",
            f"- 股票数量：{report_data.index.get_level_values('symbol').nunique()}",
            f"- 前向收益周期：{', '.join(f'{period} 日' for period in self.forward_periods)}",
            f"- 分层数量：{self.n_groups}",
            f"- IC 方法：{self.ic_method}",
            f"- 状态过滤：{'启用' if self.security_status_path is not None else '未启用'}",
            "",
            "## 多周期评估汇总",
            "",
            summary.reset_index().to_markdown(index=False),
            "",
            "## 评估图表",
            "",
        ]
        for period, image_file in image_files:
            report_lines.extend([
                f"### {period} 日前向收益",
                "",
                f"![{self.factor_name} {period} 日评估图]({image_file.name})",
                "",
            ])

        if periodic_links:
            report_lines.extend(["## 分段报告", ""])
            if periodic_links.get("yearly"):
                report_lines.extend(["### 年度回测", ""])
                for report_file in periodic_links["yearly"]:
                    report_lines.append(f"- [{report_file.parent.name}]({report_file.as_posix()})")
                report_lines.append("")
            if periodic_links.get("monthly"):
                report_lines.extend(["### 月度回测", ""])
                for report_file in periodic_links["monthly"]:
                    report_lines.append(f"- [{report_file.parent.name}]({report_file.as_posix()})")
                report_lines.append("")

        report_path = output_dir / "report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        return report_path

    def _summary_for_window(self, summary: pd.DataFrame, scope: str, window: BacktestWindow) -> pd.DataFrame:
        """给分段汇总表补充窗口信息。"""
        result = summary.reset_index()
        result.insert(0, "scope", scope)
        result.insert(1, "window", window.name)
        result.insert(2, "start", window.start.date().isoformat())
        result.insert(3, "end", window.end.date().isoformat())
        return result

    def _evaluate_and_save_window(
        self,
        signals: pd.DataFrame,
        scope: str,
        window: BacktestWindow,
    ) -> tuple[Path, pd.DataFrame] | None:
        """评估并保存单个年度或月度窗口。"""
        window_data = self._slice_market_data(self.data, window.start, window.end)
        window_signals = self._slice_signals(signals, window.start, window.end)
        if window_data.empty or window_signals.dropna(how="all").empty:
            return None

        evaluations, summary = self.evaluate(self.data, window_signals)
        report_dir = self.output_dir / scope / window.name
        report_path = self.save_reports(
            evaluations,
            summary,
            output_dir=report_dir,
            report_title=f"{self.factor_name} 因子评估报告 - {window.title}",
            data=window_data,
        )
        return report_path.relative_to(self.output_dir), self._summary_for_window(
            summary,
            scope,
            window,
        )

    def save_periodic_reports(self, signals: pd.DataFrame) -> dict[str, list[Path]]:
        """额外保存年度回测图表，以及 2024 年 1 月起的月度回测图表。"""
        report_links: dict[str, list[Path]] = {"yearly": [], "monthly": []}
        summary_frames: dict[str, list[pd.DataFrame]] = {"yearly": [], "monthly": []}

        if self.yearly_reports:
            for window in self._iter_yearly_windows():
                saved = self._evaluate_and_save_window(signals, "yearly", window)
                if saved is None:
                    continue
                report_path, summary = saved
                report_links["yearly"].append(report_path)
                summary_frames["yearly"].append(summary)

        for window in self._iter_monthly_windows():
            saved = self._evaluate_and_save_window(signals, "monthly", window)
            if saved is None:
                continue
            report_path, summary = saved
            report_links["monthly"].append(report_path)
            summary_frames["monthly"].append(summary)

        self.periodic_summary = {}
        for scope, frames in summary_frames.items():
            if frames:
                scope_summary = pd.concat(frames, ignore_index=True)
                scope_summary.to_csv(self.output_dir / f"{scope}_summary.csv", index=False)
                self.periodic_summary[scope] = scope_summary

        return {scope: links for scope, links in report_links.items() if links}

    def latest(self) -> pd.Series:
        """返回最近一个有因子结果的交易日。"""
        latest_result = self.result.dropna(how="all")
        if latest_result.empty:
            return pd.Series(dtype=float, name=self.factor_name)
        result = latest_result.iloc[-1].dropna().sort_values()
        result.name = latest_result.index[-1]
        return result

    def run(
        self,
        save_factor: bool = False,
        save_periodic_reports: bool = True,
    ) -> dict[int, FactorEvalResult]:
        """依次加载数据、计算因子、执行多周期评估并保存结果。"""
        self.data = self.data_loader.load_all()
        self.benchmark = self.data_loader.benchmark
        signals = self.calculate(combined_data=self.data, save=save_factor)
        self.evaluations, self.summary = self.evaluate(self.data, signals)
        periodic_links = (
            self.save_periodic_reports(signals)
            if save_periodic_reports
            else None
        )
        self.save_reports(
            self.evaluations,
            self.summary,
            periodic_links=periodic_links,
        )
        return self.evaluations


if __name__ == "__main__":
    factor_config_path = "config/factor_configs_onlysize.json"
    with open(factor_config_path, "r", encoding="utf-8") as f:
        factor_configs = json.load(f)

    index_runs = [
        {
            "name": "hs300",
            "data_dir": "cache/hs300_csv",
            "constituents": "000300.SH",
        },
        {
            "name": "zz500",
            "data_dir": "cache/zz500_csv",
            "constituents": "000905.SH",
        },
        {
            "name": "zz1000",
            "data_dir": "cache/zz1000_csv",
            "constituents": "000852.SH",
        },
    ]

    for factor_name, config in factor_configs.items():
        importlib.import_module(config["module"])
        for index_config in index_runs:
            data_dir = Path(index_config["data_dir"])
            runner = Runner(
                factor_name=factor_name,
                factor_params=config["params"],
                symbols=None,
                start="2010-01-01",
                n_groups=5,
                output_dir=Path("outputs") / index_config["name"] / factor_name,
                data_columns=config["columns"],
                data_dir=data_dir,
                constituents_path=data_dir / "index_members_rebalance.csv",
                constituents=index_config["constituents"],
            )
            runner.run(save_factor=True)
            print(f"{factor_name} 已保存到: {runner.output_dir.resolve()}")

    # 组合因子回测
    # scheme_configs = {
    #     "max_ic_weight_fundamental_10d": {"params": {"ic_window": 10},
    #                                       "columns": ["adj_close"]},
    #     "max_ic_weight_price_10d": {"params": {"ic_window": 10},
    #                                 "columns": ["adj_close"]},
    #     "max_ic_weight_risk_10d": {"params": {"ic_window": 10},
    #                                "columns": ["adj_close"]},
    #     "max_ic_weight_all_10d": {"params": {"ic_window": 10},
    #                              "columns": ["adj_close"]},
    # }

    # for scheme_name, config in scheme_configs.items():
    #     runner = Runner(
    #         factor_name=scheme_name,
    #         factor_params=config["params"],
    #         symbols=None,
    #         start="2010-01-01",
    #         n_groups=5,
    #         output_dir=f"scheme_outputs/hs300/{scheme_name}",
    #         data_columns=config["columns"],
    #         data_dir="cache/hs300_csv",
    #         factor_dir="factor_results/hs300_adjusted_factor_wide",
    #     )
    #     runner.run(save_factor=True)
    #     print(f"{scheme_name} 已保存到: {runner.output_dir.resolve()}")
    #     runner = Runner(
    #         factor_name=scheme_name,
    #         factor_params=config["params"],
    #         symbols=None,
    #         start="2010-01-01",
    #         n_groups=5,
    #         output_dir=f"scheme_outputs/zz500/{scheme_name}",
    #         data_columns=config["columns"],
    #         data_dir="cache/zz500_csv",
    #         factor_dir="factor_results/zz500_adjusted_factor_wide",
    #     )
    #     runner.run(save_factor=True)
    #     print(f"{scheme_name} 已保存到: {runner.output_dir.resolve()}")

    #     runner = Runner(
    #         factor_name=scheme_name,
    #         factor_params=config["params"],
    #         symbols=None,
    #         start="2010-01-01",
    #         n_groups=5,
    #         output_dir=f"scheme_outputs/zz1000/{scheme_name}",
    #         data_columns=config["columns"],
    #         data_dir="cache/zz1000_csv",
    #         factor_dir="factor_results/zz1000_adjusted_factor_wide",
    #     )
    #     runner.run(save_factor=True)
    #     print(f"{scheme_name} 已保存到: {runner.output_dir.resolve()}")
