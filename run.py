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
    ) -> None:
        self.factor_name = factor_name
        self.forward_periods = forward_periods
        self.n_groups = n_groups
        self.ic_method = ic_method
        self.max_lag = max_lag
        self.output_dir = Path(output_dir or Path("outputs") / factor_name)
        self.factor = FactorRegistry.get(factor_name)(**(factor_params or {}))
        self.data = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.result = pd.DataFrame()
        self.evaluations: dict[int, FactorEvalResult] = {}
        self.summary = pd.DataFrame()
        self.data_loader = DataLoader(
            data_dir=data_dir,
            symbols=symbols,
            start=start,
            end=end,
            columns=data_columns,
            factor_dir=factor_dir,
            constituents_path=constituents_path,
            constituent_index=constituents,
        )

    def calculate(self,
                  combined_data: pd.DataFrame | None = None,
                  save: bool = False) -> pd.DataFrame:
        """计算因子；save=True 时按股票保存完整回测期因子值。"""
        signals = self.factor.generate_signals(combined_data, None)
        self.result = signals
        if save:
            factor_dir = self.output_dir / "factor"
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
            )
            for period in self.forward_periods
        }
        summary = pd.concat(
            [evaluation.summary for evaluation in evaluations.values()]
        ).sort_index()
        return evaluations, summary

    def save_reports(self, evaluations, summary) -> Path:
        """保存 CSV、综合评估图和包含表格与图片的 Markdown 报告。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(self.output_dir / "factor_summary.csv")

        image_files = []
        for period, evaluation in evaluations.items():
            output_path = self.output_dir / f"{self.factor_name}_{period}d.png"
            benchmark_returns = (
                self.benchmark.shift(-(1 + period))
                / self.benchmark.shift(-1)
                - 1
            )
            benchmark_returns = benchmark_returns.reindex(
                evaluation.layered.group_returns.index
            )
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            FactorPlotter(
                evaluation,
                factor_name=self.factor_name,
                benchmark_cumulative=benchmark_cumulative,
            ).save(output_path)
            image_files.append(output_path)

        report_lines = [
            f"# {self.factor_name} 因子评估报告",
            "",
            "## 运行配置",
            "",
            f"- 数据区间：{self.data.index.get_level_values('date').min().date()} 至 "
            f"{self.data.index.get_level_values('date').max().date()}",
            f"- 股票数量：{self.data.index.get_level_values('symbol').nunique()}",
            f"- 前向收益周期：{', '.join(f'{period} 日' for period in self.forward_periods)}",
            f"- 分层数量：{self.n_groups}",
            f"- IC 方法：{self.ic_method}",
            "",
            "## 多周期评估汇总",
            "",
            summary.reset_index().to_markdown(index=False),
            "",
            "## 评估图表",
            "",
        ]
        for period, image_file in zip(evaluations, image_files):
            report_lines.extend([
                f"### {period} 日前向收益",
                "",
                f"![{self.factor_name} {period} 日评估图]({image_file.name})",
                "",
            ])

        report_path = self.output_dir / "report.md"
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
        self.save_reports(self.evaluations, self.summary)
        return self.evaluations


if __name__ == "__main__":
    constituents_path = "cache/tushare_index_weight_20100104_20260623_20260701083808.csv"

    factor_config_path = "config/factor_configs_onlysize.json"
    with open(factor_config_path, "r", encoding="utf-8") as f:
        factor_configs = json.load(f)

    for factor_name, config in factor_configs.items():
        importlib.import_module(config["module"])
        runner = Runner(
            factor_name=factor_name,
            factor_params=config["params"],
            symbols=None,
            start="2010-01-01",
            n_groups=5,
            output_dir=f"outputs/hs300/{factor_name}",
            data_columns=config["columns"],
            data_dir="cache/hs300_csv",
            constituents_path=constituents_path,
            constituents="000300.SH",
        )
        runner.run(save_factor=True)
        print(f"{factor_name} 已保存到: {runner.output_dir.resolve()}")

        runner = Runner(
            factor_name=factor_name,
            factor_params=config["params"],
            symbols=None,
            start="2010-01-01",
            n_groups=5,
            output_dir=f"outputs/zz500/{factor_name}",
            data_columns=config["columns"],
            data_dir="cache/zz500_csv",
            constituents_path=constituents_path,
            constituents="000905.SH",
        )
        runner.run(save_factor=True)
        print(f"{factor_name} 已保存到: {runner.output_dir.resolve()}")

        runner = Runner(
            factor_name=factor_name,
            factor_params=config["params"],
            symbols=None,
            start="2010-01-01",
            n_groups=5,
            output_dir=f"outputs/zz1000/{factor_name}",
            data_columns=config["columns"],
            data_dir="cache/zz1000_csv",
            constituents_path=constituents_path,
            constituents="000852.SH",
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
