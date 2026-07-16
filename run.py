"""加载 CSV 数据并计算注册因子。"""

from dataclasses import dataclass
import importlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from dataloader import DataLoader
from factors_eval import FactorEvalResult, eval as evaluate_factor
from factors.registry import FactorRegistry
from plot import FactorPlotter, save_yearly_summary_trends


DEFAULT_SECURITY_STATUS_PATH = (
    Path(__file__).resolve().parent
    / "cache"
    / "tushare_security_status_missing_tables_20100104_20260623_20260706.csv"
)


DAILY_METRICS_PERIOD = 5


@dataclass(frozen=True)
class BacktestWindow:
    """自然年分段回测窗口。"""

    year: int
    start: pd.Timestamp
    end: pd.Timestamp


class Runner:
    """管理因子的数据加载、计算和结果查看。"""

    def __init__(
        self,
        factor_name: str,
        relay_class: str,
        factor_params: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        data_columns: list[str] | None = None,
        data_dir: str | Path = Path(__file__).resolve().parent / "cache" / "csv",
        factor_dir: str | Path | None = None,
        factor_columns: list[str] | None = None,
        constituents_path: str | Path | None = None,
        constituents: str | None = None,
        forward_periods: tuple[int, ...] = (DAILY_METRICS_PERIOD,),
        n_groups: int = 5,
        ic_method: str = "rank",
        output_dir: str | Path | None = None,
        eval_price_col: str = "adj_close",
        security_status_path: str | Path | None = DEFAULT_SECURITY_STATUS_PATH,
    ) -> None:
        self.factor_name = factor_name
        self.forward_periods = forward_periods
        self.n_groups = n_groups
        self.ic_method = ic_method
        self.eval_price_col = eval_price_col
        self.security_status_path = Path(security_status_path) if security_status_path is not None else None
        self.output_dir = Path(output_dir or Path("outputs") / factor_name)
        self.factor = FactorRegistry.get(relay_class)(**(factor_params or {}))
        self.data = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.result = pd.DataFrame()
        self.evaluations: dict[int, FactorEvalResult] = {}
        self.summary = pd.DataFrame()
        self.yearly_summary = pd.DataFrame()
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
            factor_columns=factor_columns,
            constituents_path=constituents_path,
            constituent_index=constituents,
            security_status_path=security_status_path,
        )

    def calculate(self,
                  combined_data: pd.DataFrame | None = None,
                  save: bool = False) -> pd.DataFrame:
        """计算因子；save=True 时按股票保存完整回测期因子值。"""
        data = self.data if combined_data is None else combined_data
        if data is None or data.empty:
            raise ValueError("没有可用于计算因子的 combined_data")

        signals = self.factor.generate_signals(data, None)
        self.result = signals
        if save:
            factor_dir = self.output_dir / "factors"
            factor_dir.mkdir(parents=True, exist_ok=True)
            symbols = data.index.get_level_values("symbol").unique()
            factor_result = signals.reindex(columns=symbols)
            for symbol in symbols:
                dates = data.xs(symbol, level="symbol").index
                factor_data = factor_result[symbol].reindex(dates).rename(self.factor_name)
                factor_data.index = factor_data.index.strftime("%Y-%m-%d")
                factor_data.index.name = "date"
                factor_data.to_csv(
                    factor_dir / f"{symbol}.csv",
                    na_rep="",
                )
        return signals

    def evaluate(
        self,
        combined_data: pd.DataFrame,
        signals: pd.DataFrame,
        forward_periods: tuple[int, ...] | None = None,
    ) -> tuple[dict[int, FactorEvalResult], pd.DataFrame]:
        """计算指定周期的因子评估结果。"""
        periods = forward_periods or self.forward_periods
        evaluations = {
            period: evaluate_factor(
                signals,
                combined_data,
                forward_period=period,
                n_groups=self.n_groups,
                ic_method=self.ic_method,
                price_col=self.eval_price_col,
            )
            for period in periods
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
        dates = pd.DatetimeIndex(signals.index)
        return signals.loc[(dates >= start) & (dates <= end)]

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
        yearly_report_links: list[Path] | None = None,
        yearly_trend_image: Path | None = None,
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
            f"- 前向收益周期：{', '.join(f'{period} 日' for period in evaluations)}",
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

        if yearly_report_links:
            report_lines.extend(["## 年度回测", ""])
            for report_file in yearly_report_links:
                report_lines.append(
                    f"- [{report_file.parent.name}]({report_file.as_posix()})"
                )
            report_lines.append("")

        if yearly_trend_image is not None:
            report_lines.extend([
                "## 年度指标变化",
                "",
                "下图展示年度 `factor_summary.csv` 的原始指标值，保留正负方向。",
                "",
                f"![{self.factor_name} 年度指标变化]({yearly_trend_image.as_posix()})",
                "",
            ])

        report_path = output_dir / "report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        return report_path

    def save_daily_metrics(
        self,
        evaluations: dict[int, FactorEvalResult],
    ) -> Path:
        """保存 5 日前向收益口径的日频截面因子指标。"""
        evaluation = evaluations.get(DAILY_METRICS_PERIOD)
        if evaluation is None:
            raise ValueError(
                f"日频指标要求 forward_periods 包含 {DAILY_METRICS_PERIOD}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "daily_factor_metrics.csv"
        evaluation.daily_metrics.to_csv(output_path, na_rep="")
        return output_path

    def _evaluate_and_save_yearly_window(
        self,
        signals: pd.DataFrame,
        window: BacktestWindow,
    ) -> tuple[Path, pd.DataFrame] | None:
        """在自然年边界内重新计算并保存 5 日评估报告。"""
        window_data = self._slice_market_data(self.data, window.start, window.end)
        window_signals = self._slice_signals(signals, window.start, window.end)
        if window_data.empty or window_signals.dropna(how="all").empty:
            return None

        evaluations, summary = self.evaluate(
            window_data,
            window_signals,
            forward_periods=(DAILY_METRICS_PERIOD,),
        )
        report_dir = self.output_dir / "yearly" / str(window.year)
        report_path = self.save_reports(
            evaluations,
            summary,
            output_dir=report_dir,
            report_title=f"{self.factor_name} 因子评估报告 - {window.year} 年",
            data=window_data,
        )
        yearly_summary = summary.reset_index()
        yearly_summary.insert(0, "year", window.year)
        return report_path.relative_to(self.output_dir), yearly_summary

    def save_yearly_reports(
        self,
        signals: pd.DataFrame,
    ) -> tuple[list[Path], Path | None]:
        """保存每自然年的 5 日报告、汇总 CSV 和年度指标趋势图。"""
        report_links: list[Path] = []
        summary_frames: list[pd.DataFrame] = []
        for window in self._iter_yearly_windows():
            saved = self._evaluate_and_save_yearly_window(signals, window)
            if saved is None:
                continue
            report_path, yearly_summary = saved
            report_links.append(report_path)
            summary_frames.append(yearly_summary)

        if not summary_frames:
            self.yearly_summary = pd.DataFrame()
            return report_links, None

        self.yearly_summary = pd.concat(summary_frames, ignore_index=True)
        self.yearly_summary.to_csv(
            self.output_dir / "yearly_summary.csv",
            index=False,
        )
        trend_path = save_yearly_summary_trends(
            self.yearly_summary,
            self.output_dir / "yearly_factor_summary_trends.png",
            factor_name=self.factor_name,
        )
        return report_links, trend_path

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
        """加载数据、计算因子，并保存总时段和可选年度回测结果。"""
        self.data = self.data_loader.load_all()
        self.benchmark = self.data_loader.benchmark
        signals = self.calculate(combined_data=self.data, save=save_factor)
        self.evaluations, self.summary = self.evaluate(self.data, signals)
        self.save_daily_metrics(self.evaluations)
        yearly_links, yearly_trend_image = (
            self.save_yearly_reports(signals)
            if save_periodic_reports
            else ([], None)
        )
        self.save_reports(
            self.evaluations,
            self.summary,
            yearly_report_links=yearly_links,
            yearly_trend_image=(
                yearly_trend_image.relative_to(self.output_dir)
                if yearly_trend_image is not None
                else None
            ),
        )
        return self.evaluations


if __name__ == "__main__":
    factor_config_path = "config/factor_configs.json"
    with open(factor_config_path, "r", encoding="utf-8") as f:
        factor_configs = json.load(f)

    data_dir = Path("cache/zz500_csv")
    for factor_name, config in tqdm(
        factor_configs.items(),
        total=len(factor_configs),
        desc="自主因子回测",
        unit="factor",
    ):
        importlib.import_module(config["module"])
        runner = Runner(
            factor_name=factor_name,
            relay_class=config["relay_class"],
            factor_params=config["params"],
            start="2010-01-01",
            data_columns=[column for column in config["columns"]],
            data_dir=data_dir,
            factor_dir=None,
            constituents_path=data_dir.parent / "zz500_index_members_rebalance.csv",
            constituents="000905.SH",
            output_dir=Path("outputs/zz500/review") / factor_name,
        )
        runner.run()
