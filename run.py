"""加载 CSV 数据并计算注册因子。"""

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
        self.factor = FactorRegistry.get(factor_name)(**(factor_params or {}))
        self.data = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.result = pd.DataFrame()
        self.evaluations: dict[int, FactorEvalResult] = {}
        self.summary = pd.DataFrame()
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

    def evaluate(self, combined_data, signals) -> dict[int, FactorEvalResult]:
        """计算 1、5、10、21 日等指定周期的因子评估结果。"""
        evaluations = {
            period: evaluate_factor(
                signals,
                combined_data,
                forward_period=period,
                n_groups=self.n_groups,
                ic_method=self.ic_method,
                price_col=self.eval_price_col,
            )
            for period in self.forward_periods
        }
        summary = pd.concat(
            [evaluation.summary for evaluation in evaluations.values()]
        ).sort_index()
        return evaluations, summary

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
    ) -> Path:
        """保存 CSV、综合评估图和包含表格与图片的 Markdown 报告。"""
        output_dir = self.output_dir
        report_data = self.data
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "factor_summary.csv")
        full_ic_series = pd.concat(
            {
                f"{period}d": evaluation.full_ic_series
                for period, evaluation in evaluations.items()
            },
            axis=1,
        ).sort_index()
        full_ic_series.index.name = "date"
        full_ic_series.to_csv(output_dir / "full_ic_series.csv")

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
            f"# {self.factor_name} 因子评估报告",
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

        report_path = output_dir / "report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        return report_path

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
    ) -> dict[int, FactorEvalResult]:
        """依次加载数据、计算因子、执行多周期评估并保存结果。"""
        self.data = self.data_loader.load_all()
        self.benchmark = self.data_loader.benchmark
        signals = self.calculate(combined_data=self.data, save=save_factor)
        self.evaluations, self.summary = self.evaluate(self.data, signals)
        self.save_reports(
            self.evaluations,
            self.summary,
        )
        return self.evaluations


if __name__ == "__main__":
    factor_config_path = "config/factor_configs.json"
    with open(factor_config_path, "r", encoding="utf-8") as f:
        factor_configs = json.load(f)

    index_runs = [
        # {
        #     "name": "hs300",
        #     "data_dir": "cache/hs300_csv",
        #     "constituents": "000300.SH",
        # },
        {
            "name": "zz500",
            "data_dir": "cache/zz500_csv",
            "constituents": "000905.SH",
        },
        # {
        #     "name": "zz1000",
        #     "data_dir": "cache/zz1000_csv",
        #     "constituents": "000852.SH",
        # },
    ]

    for factor_name, config in factor_configs.items():
        importlib.import_module(config["module"])
        for index_config in index_runs:
            index_name = index_config["name"]
            data_dir = Path(index_config["data_dir"])
            runner = Runner(
                factor_name=factor_name,
                factor_params=config["params"],
                symbols=None,
                start="2010-01-01",
                n_groups=5,
                output_dir=Path("outputs") / index_name / factor_name,
                data_columns=config["columns"],
                data_dir=data_dir,
                constituents_path=data_dir.parent / f"{index_name}_index_members_rebalance.csv",
                constituents=index_config["constituents"],
            )
            runner.run(save_factor=True)
            print(f"{factor_name} 已保存到: {runner.output_dir.resolve()}")
