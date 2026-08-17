"""加载 CSV 数据并计算注册因子。"""

import argparse
import importlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from dataloader import DataLoader
from factor_pipeline import financial_factor_names, run_full_pipeline
from factors_eval import (
    FactorEvalResult,
    eval as evaluate_factor,
)
from factors.registry import FactorRegistry
from neutralization import neutralize_factor_values
from report_calculate import RunnerReportsMixin
from settings import SETTINGS, configure_settings


_USE_SETTINGS = object()


class Runner(RunnerReportsMixin):
    """管理因子的数据加载、计算和结果查看。"""

    def __init__(
        self,
        factor_name: str,
        relay_class: str,
        factor_params: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        start: str | date | None | object = _USE_SETTINGS,
        end: str | date | None = None,
        data_columns: list[str] | None = None,
        data_dir: str | Path | object = _USE_SETTINGS,
        factor_dir: str | Path | None = None,
        factor_columns: list[str] | None = None,
        constituents_path: str | Path | None = None,
        constituents: str | None = None,
        forward_periods: tuple[int, ...] | object = _USE_SETTINGS,
        n_groups: int | object = _USE_SETTINGS,
        ic_method: str | object = _USE_SETTINGS,
        output_dir: str | Path | None = None,
        eval_price_col: str | object = _USE_SETTINGS,
        security_status_path: str | Path | None | object = _USE_SETTINGS,
        market_state_path: str | Path | None | object = _USE_SETTINGS,
        is_financial_factor: bool = False,
        factor_config: dict[str, Any] | None = None,
    ) -> None:
        if start is _USE_SETTINGS:
            start = SETTINGS.runtime.start
        if data_dir is _USE_SETTINGS:
            data_dir = SETTINGS.paths.data_dir
        if forward_periods is _USE_SETTINGS:
            forward_periods = SETTINGS.evaluation.ic_decay_periods
        if n_groups is _USE_SETTINGS:
            n_groups = SETTINGS.evaluation.n_groups
        if ic_method is _USE_SETTINGS:
            ic_method = SETTINGS.evaluation.ic_method
        if eval_price_col is _USE_SETTINGS:
            eval_price_col = SETTINGS.data.evaluation_price_column
        if security_status_path is _USE_SETTINGS:
            security_status_path = SETTINGS.paths.security_status_path
        if market_state_path is _USE_SETTINGS:
            market_state_path = SETTINGS.paths.market_state_path
        if output_dir is None:
            output_dir = SETTINGS.paths.output_root / factor_name

        self.factor_name = factor_name
        self.factor_config = factor_config or {
            "params": factor_params or {},
            "columns": data_columns or [],
            "relay_class": relay_class,
        }
        self.is_financial_factor = is_financial_factor
        self.forward_periods = forward_periods
        self.n_groups = n_groups
        self.ic_method = ic_method
        self.eval_price_col = eval_price_col
        self.security_status_path = (
            Path(security_status_path)
            if security_status_path is not None
            else None
        )
        self.market_state_path = (
            Path(market_state_path) if market_state_path is not None else None
        )
        self.output_dir = Path(output_dir)
        self.factor = FactorRegistry.get(relay_class)(**(factor_params or {}))
        self.data = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.result = pd.DataFrame()
        self.raw_result = pd.DataFrame()
        self.neutralized_result = pd.DataFrame()
        self.evaluations: dict[str, dict[int, FactorEvalResult]] = {}
        self.summary = pd.DataFrame()
        self.daily_metrics = pd.DataFrame()
        self.yearly_summary = pd.DataFrame()
        self.market_state_summary = pd.DataFrame()
        self.industry_daily_ic = pd.DataFrame()
        loader_columns = (
            None
            if data_columns is None
            else list(
                dict.fromkeys(
                    [
                        *data_columns,
                        eval_price_col,
                        SETTINGS.data.market_cap_column,
                        SETTINGS.data.industry_column,
                    ]
                )
            )
        )
        if loader_columns is not None and self.is_financial_factor:
            loader_columns = list(
                dict.fromkeys(
                    [*loader_columns, SETTINGS.data.report_publish_date_column]
                )
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
        neutralized_signals = neutralize_factor_values(signals, data)
        self.result = signals
        self.raw_result = signals
        self.neutralized_result = neutralized_signals
        if save:
            self.save_signals(
                signals,
                neutralized_signals=neutralized_signals,
                combined_data=data,
            )
        return signals

    def save_signals(
        self,
        signals: pd.DataFrame,
        combined_data: pd.DataFrame | None = None,
        neutralized_signals: pd.DataFrame | None = None,
    ) -> Path:
        """按股票保存已有的 ``date × symbol`` 因子宽表。

        该方法也可用于复用已落盘的因子值生成报告，避免重新调用
        ``factor.generate_signals``。普通因子输出 signal_date、available_date、
        原始因子值和中性化因子值；财务类因子额外输出距最近财报发布日期的
        自然日天数。
        """
        data = self.data if combined_data is None else combined_data
        if data is None or data.empty:
            raise ValueError("没有可用于保存因子值的 combined_data")
        if (
            self.is_financial_factor
            and SETTINGS.data.report_publish_date_column not in data.columns
        ):
            raise ValueError(
                f"财务类因子 {self.factor_name} 缺少字段: "
                f"{SETTINGS.data.report_publish_date_column}"
            )
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)
        if neutralized_signals is None:
            neutralized_signals = neutralize_factor_values(signals, data)
        elif not isinstance(neutralized_signals.index, pd.DatetimeIndex):
            neutralized_signals = neutralized_signals.copy()
            neutralized_signals.index = pd.to_datetime(neutralized_signals.index)

        factor_dir = self.output_dir / "factors"
        factor_dir.mkdir(parents=True, exist_ok=True)
        symbols = data.index.get_level_values("symbol").unique()
        expected_symbols = {str(symbol) for symbol in symbols}
        for csv_path in factor_dir.glob("*.csv"):
            if csv_path.stem not in expected_symbols:
                csv_path.unlink()
        raw_result = signals.reindex(columns=symbols)
        neutralized_result = neutralized_signals.reindex(columns=symbols)
        trading_calendar = self.data_loader.trading_calendar
        if trading_calendar.empty:
            trading_calendar = pd.DatetimeIndex(
                data.index.get_level_values("date").unique()
            ).sort_values()
        available_date_map = pd.Series(
            trading_calendar[1:].to_numpy(),
            index=trading_calendar[:-1],
        )

        for symbol in symbols:
            symbol_data = data.xs(symbol, level="symbol").sort_index()
            dates = pd.DatetimeIndex(symbol_data.index)
            factor_data = pd.DataFrame(
                {
                    "signal_date": dates,
                    "available_date": dates.map(available_date_map),
                    f"{self.factor_name}_raw": (
                        raw_result[symbol].reindex(dates).to_numpy()
                    ),
                    f"{self.factor_name}_neutralization": (
                        neutralized_result[symbol].reindex(dates).to_numpy()
                    ),
                }
            )
            if self.is_financial_factor:
                publish_dates = pd.to_datetime(
                    symbol_data[SETTINGS.data.report_publish_date_column],
                    errors="coerce",
                )
                publish_dates = publish_dates.where(
                    publish_dates.to_numpy() <= dates.to_numpy()
                )
                latest_publish_dates = publish_dates.ffill().cummax()
                report_age = (
                    pd.Series(dates, index=symbol_data.index)
                    - latest_publish_dates
                ).dt.days
                factor_data[SETTINGS.data.days_since_report_column] = (
                    report_age.reindex(factor_data["signal_date"]).to_numpy()
                )

            factor_data.to_csv(
                factor_dir / f"{symbol}.csv",
                index=False,
                na_rep="",
                date_format="%Y-%m-%d",
            )
        return factor_dir

    def evaluate(
        self,
        combined_data: pd.DataFrame,
        signals: pd.DataFrame,
        forward_periods: tuple[int, ...] | None = None,
    ) -> tuple[dict[int, FactorEvalResult], pd.DataFrame]:
        """计算指定周期的因子评估结果。"""
        periods = forward_periods or self.forward_periods
        benchmark_prices = None
        if not self.benchmark.empty:
            preferred_benchmark = SETTINGS.report.preferred_benchmark_column
            benchmark_column = (
                preferred_benchmark
                if preferred_benchmark in self.benchmark.columns
                else self.benchmark.columns[0]
            )
            data_dates = pd.DatetimeIndex(
                combined_data.index.get_level_values("date").unique()
            ).sort_values()
            benchmark_prices = self.benchmark[benchmark_column].reindex(data_dates)
        evaluations = {
            period: evaluate_factor(
                signals,
                combined_data,
                forward_period=period,
                n_groups=self.n_groups,
                ic_method=self.ic_method,
                price_col=self.eval_price_col,
                benchmark_prices=benchmark_prices,
                ic_only=period != SETTINGS.report.daily_metrics_period,
            )
            for period in periods
        }
        summary = pd.concat(
            [evaluation.summary for evaluation in evaluations.values()]
        ).sort_index()
        return evaluations, summary

    @staticmethod
    def _suffix_metrics(frame: pd.DataFrame, version: str) -> pd.DataFrame:
        """Append a factor-value version suffix to every metric column."""
        return frame.rename(
            columns={column: f"{column}_{version}" for column in frame.columns}
        )

    def evaluate_versions(
        self,
        combined_data: pd.DataFrame,
        signal_versions: dict[str, pd.DataFrame],
        forward_periods: tuple[int, ...] | None = None,
    ) -> tuple[dict[str, dict[int, FactorEvalResult]], pd.DataFrame]:
        """Evaluate raw and neutralized factor values with identical settings."""
        missing_versions = set(SETTINGS.report.factor_versions).difference(
            signal_versions
        )
        if missing_versions:
            raise ValueError(
                "因子值缺少版本: " + ", ".join(sorted(missing_versions))
            )

        evaluations: dict[str, dict[int, FactorEvalResult]] = {}
        summaries = []
        for version in SETTINGS.report.factor_versions:
            version_evaluations, version_summary = self.evaluate(
                combined_data,
                signal_versions[version],
                forward_periods=forward_periods,
            )
            unexpected = set(version_summary.columns).difference(
                FactorEvalResult.SUMMARY_METRIC_COLUMNS
            )
            missing = set(FactorEvalResult.SUMMARY_METRIC_COLUMNS).difference(
                version_summary.columns
            )
            if unexpected or missing:
                raise ValueError(
                    f"{version} 回测汇总字段不符合约定；"
                    f"缺少={sorted(missing)}，额外={sorted(unexpected)}"
                )
            evaluations[version] = version_evaluations
            summaries.append(self._suffix_metrics(version_summary, version))

        summary = pd.concat(summaries, axis=1).sort_index()
        return evaluations, summary

    def run(
        self,
        save_factor: bool = False,
        save_periodic_reports: bool = True,
        build_report: bool = True,
    ) -> dict[str, dict[int, FactorEvalResult]]:
        """加载数据并计算因子；可延迟到统一回填后再生成报告。"""
        self.data = self.data_loader.load_all()
        self.benchmark = self.data_loader.benchmark
        raw_signals = self.calculate(combined_data=self.data, save=save_factor)
        if not build_report:
            return {}
        signal_versions = {
            "raw": raw_signals,
            "neutralization": self.neutralized_result,
        }
        return self.build_reports(
            signal_versions,
            save_periodic_reports=save_periodic_reports,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按完整配置生成并评估因子。")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="完整运行配置JSON；所有本次执行参数均从该文件读取。",
    )
    args = parser.parse_args()

    try:
        configure_settings(args.config)
    except ValueError as error:
        parser.error(str(error))

    if SETTINGS.runtime.mode == "pipeline":
        run_full_pipeline(runner_class=Runner)
        raise SystemExit(0)

    factor_config_path = SETTINGS.paths.factor_config
    with factor_config_path.open("r", encoding="utf-8") as f:
        factor_configs = json.load(f)
    if SETTINGS.runtime.factor is not None:
        if SETTINGS.runtime.factor not in factor_configs:
            parser.error(
                f"因子配置 {factor_config_path} 中不存在因子: "
                f"{SETTINGS.runtime.factor}"
            )
        factor_configs = {
            SETTINGS.runtime.factor: factor_configs[SETTINGS.runtime.factor]
        }
    financial_factors = financial_factor_names(factor_configs)

    data_dir = SETTINGS.paths.data_dir
    for factor_name, config in tqdm(
        factor_configs.items(),
        total=len(factor_configs),
        desc="因子生成",
        unit="factor",
    ):
        if config.get("params", {}).get("expansion_kind"):
            parser.error(
                "派生因子必须使用 runtime.mode=pipeline 的完整配置运行。"
            )
        importlib.import_module(config["module"])
        runner = Runner(
            factor_name=factor_name,
            relay_class=config["relay_class"],
            factor_params=config["params"],
            start=SETTINGS.runtime.start,
            data_columns=[column for column in config["columns"]],
            data_dir=data_dir,
            factor_dir=None,
            constituents_path=SETTINGS.paths.constituents_path,
            constituents=SETTINGS.data.constituent_index,
            output_dir=SETTINGS.paths.output_root / factor_name,
            is_financial_factor=factor_name in financial_factors,
            factor_config=config,
        )
        runner.run(save_factor=SETTINGS.runtime.save_factor)
