"""加载 CSV 数据并计算注册因子。"""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import importlib
import json
import multiprocessing as mp
import os
import re
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import traceback
from typing import Any
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import DataLoader
from factors_eval import (
    FactorEvalResult,
    SUMMARY_METRIC_COLUMNS,
    calc_forward_returns,
    eval as evaluate_factor,
)
from factors.registry import FactorRegistry
from factor_report import build_factor_report, factor_direction, weekly_ic_statistics
from neutralization import (
    DEFAULT_INDUSTRY_COLUMN,
    DEFAULT_MARKET_CAP_COLUMN,
    neutralize_factor_values,
)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SECURITY_STATUS_PATH = (
    PROJECT_ROOT
    / "cache"
    / "tushare_security_status_missing_tables_20100104_20260623_20260706.csv"
)


DAILY_METRICS_PERIOD = 5
IC_DECAY_PERIODS = (5, 10, 15, 20, 40, 60)
DEFAULT_START_DATE = "2014-01-02"
MARKET_STATE_ERAS = {
    "2014-2019": (pd.Timestamp("2014-01-02"), pd.Timestamp("2019-12-31")),
    "2020-2026": (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-12-31")),
}
MIN_MARKET_STATE_SEGMENT_DAYS = 7
MARKET_STATE_NAMES = {
    0: "全面牛市",
    1: "结构性牛市",
    2: "非趋势",
    3: "熊市",
}
DEFAULT_MARKET_STATE_PATH = (
    PROJECT_ROOT
    / "zz500_lightgbm_prediction_4class_signals.csv"
)
REPORT_PUBLISH_DATE_COLUMN = "publish_date"
DAYS_SINCE_REPORT_COLUMN = "days_since_latest_report_publish_date"
FACTOR_VERSIONS = ("raw", "neutralization")
DEFAULT_FINANCIAL_FACTOR_CONFIG = (
    PROJECT_ROOT / "config" / "financial_factors_in_199.json"
)
DEFAULT_BASE_FACTOR_CONFIG = PROJECT_ROOT / "config" / "factor_configs_199.json"
DEFAULT_FULL_FACTOR_CONFIG = PROJECT_ROOT / "config" / "factor_configs_5947.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "cache" / "zz500_csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "zz500"
RUNTIME_ROOT = PROJECT_ROOT / ".runtime"
WINDOW_PARAM_KEYS = ("window", "lookback", "period")
HARDCODED_WINDOW_FACTORS = {
    "eps_growth_63d", "equity_growth_63d", "net_income_growth_63d",
    "revenue_growth_63d", "reversal_1m", "reversal_3m",
    "roe_growth_63d", "total_asset_growth_63d",
    "trading_amount_20d", "volume_3m",
}


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
        start: str | date | None = DEFAULT_START_DATE,
        end: str | date | None = None,
        data_columns: list[str] | None = None,
        data_dir: str | Path = Path(__file__).resolve().parent / "cache" / "csv",
        factor_dir: str | Path | None = None,
        factor_columns: list[str] | None = None,
        constituents_path: str | Path | None = None,
        constituents: str | None = None,
        forward_periods: tuple[int, ...] = IC_DECAY_PERIODS,
        n_groups: int = 5,
        ic_method: str = "rank",
        output_dir: str | Path | None = None,
        eval_price_col: str = "adj_close",
        security_status_path: str | Path | None = DEFAULT_SECURITY_STATUS_PATH,
        market_state_path: str | Path | None = DEFAULT_MARKET_STATE_PATH,
        is_financial_factor: bool = False,
        factor_config: dict[str, Any] | None = None,
    ) -> None:
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
        self.output_dir = Path(output_dir or Path("outputs") / factor_name)
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
                        DEFAULT_MARKET_CAP_COLUMN,
                        DEFAULT_INDUSTRY_COLUMN,
                    ]
                )
            )
        )
        if loader_columns is not None and self.is_financial_factor:
            loader_columns = list(
                dict.fromkeys([*loader_columns, REPORT_PUBLISH_DATE_COLUMN])
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
        if self.is_financial_factor and REPORT_PUBLISH_DATE_COLUMN not in data.columns:
            raise ValueError(
                f"财务类因子 {self.factor_name} 缺少字段: "
                f"{REPORT_PUBLISH_DATE_COLUMN}"
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
                    symbol_data[REPORT_PUBLISH_DATE_COLUMN],
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
                factor_data[DAYS_SINCE_REPORT_COLUMN] = (
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
            benchmark_column = (
                "ZZ500" if "ZZ500" in self.benchmark.columns
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
                ic_only=period != DAILY_METRICS_PERIOD,
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
        missing_versions = set(FACTOR_VERSIONS).difference(signal_versions)
        if missing_versions:
            raise ValueError(
                "因子值缺少版本: " + ", ".join(sorted(missing_versions))
            )

        evaluations: dict[str, dict[int, FactorEvalResult]] = {}
        summaries = []
        for version in FACTOR_VERSIONS:
            version_evaluations, version_summary = self.evaluate(
                combined_data,
                signal_versions[version],
                forward_periods=forward_periods,
            )
            unexpected = set(version_summary.columns).difference(
                SUMMARY_METRIC_COLUMNS
            )
            missing = set(SUMMARY_METRIC_COLUMNS).difference(
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
        evaluations: dict[str, dict[int, FactorEvalResult]],
        summary: pd.DataFrame,
        output_dir: str | Path | None = None,
        report_title: str | None = None,
        data: pd.DataFrame | None = None,
        yearly_trend_image: Path | None = None,
    ) -> Path:
        """保存全时段图片和可分页检索的HTML报告。"""
        from html import escape
        from plot import FactorPlotter

        output_dir = Path(output_dir or self.output_dir)
        report_data = data if data is not None else self.data
        output_dir.mkdir(parents=True, exist_ok=True)
        for legacy_name in (
            "report.md",
            "factor_summary.csv",
            "market_state_summary.csv",
            "yearly_summary.csv",
        ):
            legacy_path = output_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

        image_files: list[tuple[str, int, Path]] = []
        for version in FACTOR_VERSIONS:
            for period, evaluation in evaluations[version].items():
                if period != DAILY_METRICS_PERIOD:
                    continue
                output_path = (
                    output_dir
                    / f"{self.factor_name}_{version}_{period}d.png"
                )
                benchmark_cumulative = self._benchmark_cumulative(
                    evaluation,
                    period,
                )
                FactorPlotter(
                    evaluation,
                    factor_name=f"{self.factor_name}_{version}",
                    benchmark_cumulative=benchmark_cumulative,
                ).save(output_path)
                image_files.append((version, period, output_path))

        data_dates = report_data.index.get_level_values("date")
        market_state = self.market_state_summary.copy()
        if not market_state.empty:
            market_state.insert(
                1,
                "state_name",
                market_state["state"].map(MARKET_STATE_NAMES),
            )

        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            values = frame.copy()
            for column in values.columns:
                if pd.api.types.is_datetime64_any_dtype(values[column]):
                    values[column] = values[column].dt.strftime("%Y-%m-%d")
            return json.loads(values.to_json(orient="records"))

        datasets = {
            "full": records(summary.reset_index()),
            "daily": records(self.daily_metrics),
            "yearly": records(self.yearly_summary),
            "market": records(market_state),
        }
        charts = []
        for version, period, image_file in image_files:
            charts.append(
                "<figure><figcaption>"
                f"{escape(version)} · {period}日前向收益"
                "</figcaption><img loading=\"lazy\" src=\""
                f"{escape(image_file.relative_to(output_dir).as_posix())}"
                "\"></figure>"
            )
        if yearly_trend_image is not None:
            charts.append(
                "<figure><figcaption>年度指标趋势</figcaption>"
                f"<img loading=\"lazy\" src=\"{escape(yearly_trend_image.as_posix())}\">"
                "</figure>"
            )

        title = report_title or f"{self.factor_name} 因子评估报告"
        html_template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<script>
window.addEventListener('DOMContentLoaded',()=>{
  const safe=v=>String(v==null?'':v).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));
  window.oneRow=row=>{
    if(!row)return '<div class="context">当前窗口没有有效数据</div>';
    const keys=Object.keys(row);
    const common=keys.filter(k=>!k.endsWith('_raw')&&!k.endsWith('_neutralization'));
    const metrics=[...new Set(keys.filter(k=>k.endsWith('_raw')||k.endsWith('_neutralization')).map(k=>k.replace(/_(raw|neutralization)$/,'')))];
    const cols=[...common,...metrics];
    const versionRow=(version,label)=>`<tr><td style="font-weight:800;text-align:left">${label}</td>${cols.map(c=>`<td>${safe(common.includes(c)?row[c]:row[`${c}_${version}`])}</td>`).join('')}</tr>`;
    return `<table><thead><tr><th>版本</th>${cols.map(c=>`<th>${safe(c)}</th>`).join('')}</tr></thead><tbody>${versionRow('raw','Raw')}${versionRow('neutralization','Neutralization')}</tbody></table>`;
  };
  for(const [cardId,inputId] of [['dailyCard','dailyPicker'],['periodCard','periodPicker'],['stateCard','statePicker']]){
    const card=document.getElementById(cardId),input=document.getElementById(inputId);
    if(card&&input)card.addEventListener('click',()=>input.dispatchEvent(new Event('change')));
  }
  const active=document.querySelector('.selector.active');
  const activeInput=active&&active.querySelector('input,select');
  if(activeInput)activeInput.dispatchEvent(new Event('change'));
});
</script>
<style>
:root{--navy:#102a43;--navy2:#174765;--paper:#fff;--bg:#edf2f7;--ink:#172b4d;--muted:#6b7c93;--line:#dce4ec;--shadow:0 12px 32px rgba(16,42,67,.10)}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#e9f0f5,#f8fafc 50%,#edf3f7);color:var(--ink);font:14px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}.hero{padding:30px 34px 66px;background:linear-gradient(120deg,var(--navy),var(--navy2));color:#fff}.hero-inner,.workspace{max-width:1700px;margin:auto}.eyebrow{color:#a9c9d9;font-size:12px;letter-spacing:.14em;text-transform:uppercase}.hero h1{margin:6px 0 8px;font-size:30px}.hero p{margin:0;color:#d7e5ed}.workspace{margin-top:-40px;padding:0 24px 38px}.meta{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:10px;margin-bottom:16px}.meta div,.selector,.card,figure{background:rgba(255,255,255,.98);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}.meta div{padding:13px}.meta b,.selector span{display:block;margin-bottom:5px;color:var(--muted);font-size:11px;letter-spacing:.04em}.selectors{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:14px;margin-bottom:16px}.selector{padding:16px;transition:.18s}.selector.active{border-color:#5d8da6;box-shadow:0 12px 34px rgba(47,111,145,.18)}.selector input,.selector select{width:100%;height:42px;padding:0 11px;border:1px solid #cdd8e2;border-radius:9px;background:#f9fbfc;color:var(--ink);font-weight:600;outline:none}.selector input:focus,.selector select:focus{border-color:#5285a0;box-shadow:0 0 0 3px rgba(82,133,160,.12)}.card{padding:17px;margin-bottom:16px}.card-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.card h2{margin:0;font-size:18px}.context{color:var(--muted)}.table-wrap{overflow:auto;max-height:68vh;border:1px solid var(--line);border-radius:10px}table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th{position:sticky;top:0;background:#eaf0f4;color:#496276;font-size:12px}th:first-child,td:first-child{text-align:left}.charts{display:grid;grid-template-columns:repeat(2,minmax(440px,1fr));gap:16px}figure{margin:0;padding:13px}figcaption{margin-bottom:8px;font-weight:700}img{display:block;width:100%;height:auto}details summary{cursor:pointer;font-weight:700}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}.toolbar input,.toolbar select,.toolbar button{padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:#fff}.toolbar input{min-width:260px}
@media(max-width:1000px){.meta{grid-template-columns:repeat(2,1fr)}.selectors,.charts{grid-template-columns:1fr}.hero{padding-inline:22px}.workspace{padding-inline:14px}}
</style></head><body><header class="hero"><div class="hero-inner"><div class="eyebrow">MLFactors · Factor Research</div><h1>__TITLE__</h1><p>Raw / Neutralization · Rank IC · T+1至T+6收益</p></div></header><main class="workspace"><section class="meta">__META__</section><section class="selectors"><label class="selector" id="dailyCard"><span>01 · 选择具体交易日</span><input id="dailyPicker" type="date"></label><label class="selector active" id="periodCard"><span>02 · 选择自然年或全时段</span><select id="periodPicker"></select></label><label class="selector" id="stateCard"><span>03 · 选择大盘状态</span><select id="statePicker"></select></label></section><section class="card"><div class="card-head"><h2>所选窗口指标</h2><div id="context" class="context">全时段</div></div><div id="selectedTable" class="table-wrap"></div></section><section class="card"><div class="card-head"><h2>评估图表</h2><div class="context">图片与HTML同目录，页面内直接展示</div></div><div class="charts">__CHARTS__</div></section><details class="card"><summary>展开完整日频指标表</summary><div class="table-app" data-key="daily"></div></details></main>
<script>const DATA=__DATA__,states={};function esc(v){return String(v==null?'':v).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));}function oneRow(row){if(!row)return '<div class="context">当前窗口没有有效数据</div>';const cols=Object.keys(row);return `<table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody><tr>${cols.map(c=>`<td>${esc(row[c])}</td>`).join('')}</tr></tbody></table>`;}const cards=[document.getElementById('dailyCard'),document.getElementById('periodCard'),document.getElementById('stateCard')];function show(row,label,card){cards.forEach(x=>x.classList.toggle('active',x===card));document.getElementById('selectedTable').innerHTML=oneRow(row);document.getElementById('context').textContent=label;}const dailyPicker=document.getElementById('dailyPicker'),periodPicker=document.getElementById('periodPicker'),statePicker=document.getElementById('statePicker');const dates=DATA.daily.map(x=>x.signal_date);dailyPicker.min=dates[0];dailyPicker.max=dates[dates.length-1];dailyPicker.value=dates[dates.length-1];dailyPicker.onchange=()=>show(DATA.daily.find(x=>x.signal_date===dailyPicker.value),`交易日 ${dailyPicker.value}`,cards[0]);periodPicker.innerHTML='<option value="full">全时段</option>'+DATA.yearly.map(x=>`<option value="${x.year}">${x.year}年</option>`).join('');periodPicker.onchange=()=>periodPicker.value==='full'?show(DATA.full[0],'全时段',cards[1]):show(DATA.yearly.find(x=>String(x.year)===periodPicker.value),`${periodPicker.value}年`,cards[1]);const stateKey=x=>`${x.era||''}|${x.state}`;statePicker.innerHTML=DATA.market.map(x=>`<option value="${stateKey(x)}">${x.era||'全时段'} · ${x.state} · ${x.state_name}</option>`).join('');statePicker.onchange=()=>{const row=DATA.market.find(x=>stateKey(x)===statePicker.value);show(row,`${row.era||'全时段'} · ${row.state} · ${row.state_name}`,cards[2]);};function render(app){const key=app.dataset.key,data=DATA[key]||[],state=states[key]||(states[key]={page:0,size:100,q:''});const filtered=state.q?data.filter(r=>JSON.stringify(r).toLowerCase().includes(state.q)):data,pages=Math.max(1,Math.ceil(filtered.length/state.size));state.page=Math.min(state.page,pages-1);const rows=filtered.slice(state.page*state.size,(state.page+1)*state.size),cols=data.length?Object.keys(data[0]):[];app.innerHTML=`<div class="toolbar"><input placeholder="筛选日频表..." value="${esc(state.q)}"><select><option>50</option><option selected>100</option><option>250</option></select><button class="prev">上一页</button><button class="next">下一页</button><span>${filtered.length}行 · ${state.page+1}/${pages}页</span></div><div class="table-wrap"><table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${esc(r[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;app.querySelector('input').oninput=e=>{state.q=e.target.value.toLowerCase();state.page=0;render(app)};app.querySelector('select').value=String(state.size);app.querySelector('select').onchange=e=>{state.size=+e.target.value;state.page=0;render(app)};app.querySelector('.prev').onclick=()=>{state.page=Math.max(0,state.page-1);render(app)};app.querySelector('.next').onclick=()=>{state.page=Math.min(pages-1,state.page+1);render(app)};}document.querySelectorAll('.table-app').forEach(render);show(DATA.full[0],'全时段',cards[1]);</script></body></html>"""
        meta = "".join(
            f"<div><b>{escape(label)}</b>{escape(str(value))}</div>"
            for label, value in (
                ("数据区间", f"{data_dates.min().date()} 至 {data_dates.max().date()}"),
                ("股票数量", report_data.index.get_level_values("symbol").nunique()),
                ("前向收益", "T+1 至 T+6"),
                ("IC方法", self.ic_method),
                ("因子版本", "raw / neutralization"),
                ("中性化", "log市值 + 申万一级行业OLS残差"),
            )
        )
        html_content = (
            html_template.replace("__TITLE__", escape(title))
            .replace("__META__", meta)
            .replace("__CHARTS__", "".join(charts))
            .replace("__DATA__", json.dumps(datasets, ensure_ascii=False))
        )
        report_path = output_dir / "report.html"
        report_path.write_text(html_content, encoding="utf-8")
        return report_path

    def save_daily_metrics(
        self,
        evaluations: dict[str, dict[int, FactorEvalResult]],
    ) -> Path:
        """保存带信号日和下一真实交易日的双版本日频截面指标。"""
        version_frames = []
        for version in FACTOR_VERSIONS:
            evaluation = evaluations[version].get(DAILY_METRICS_PERIOD)
            if evaluation is None:
                raise ValueError(
                    f"{version} 日频指标要求 forward_periods 包含 "
                    f"{DAILY_METRICS_PERIOD}"
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
        unexpected_states = set(states["state"]).difference(MARKET_STATE_NAMES)
        if unexpected_states:
            raise ValueError(f"市场状态取值非法: {sorted(unexpected_states)}")

        calendar = self.data_loader.trading_calendar
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
        for era, (start, end) in MARKET_STATE_ERAS.items():
            era_states = states.loc[states["signal_date"].between(start, end)].copy()
            era_states["segment_id"] = era_states["state"].ne(
                era_states["state"].shift()
            ).cumsum()
            segments = {state: [] for state in MARKET_STATE_NAMES}
            for _, segment in era_states.groupby("segment_id", sort=True):
                if len(segment) < MIN_MARKET_STATE_SEGMENT_DAYS:
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
            "ZZ500" if "ZZ500" in self.benchmark.columns
            else (self.benchmark.columns[0] if not self.benchmark.empty else None)
        )
        for era, segments_by_state in segments_by_era.items():
            for state in MARKET_STATE_NAMES:
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
                            DAILY_METRICS_PERIOD,
                            price_col=self.eval_price_col,
                        )
                    )
                    if benchmark_column is not None:
                        prices = self.benchmark[benchmark_column].reindex(segment_dates)
                        benchmark_returns.append(
                            (prices.shift(-(1 + DAILY_METRICS_PERIOD)) / prices.shift(-1) - 1)
                            .dropna()
                        )
                    sampling_dates.extend(segment_dates[::DAILY_METRICS_PERIOD])
                forward_returns = pd.concat(segment_returns).sort_index()
                state_benchmark_returns = (
                    pd.concat(benchmark_returns).sort_index()
                    if benchmark_returns else None
                )

                summaries = []
                for version in FACTOR_VERSIONS:
                    signals = signal_versions[version].reindex(state_dates)
                    signals.index.name = "date"
                    signals.columns.name = "symbol"
                    evaluation = evaluate_factor(
                        signals,
                        state_data,
                        forward_period=DAILY_METRICS_PERIOD,
                        n_groups=self.n_groups,
                        ic_method=self.ic_method,
                        price_col=self.eval_price_col,
                        forward_returns=forward_returns,
                        sampling_dates=pd.DatetimeIndex(sampling_dates),
                        benchmark_returns=state_benchmark_returns,
                    )
                    version_summary = evaluation.summary.copy()
                    full_ic_mean = float(
                        self.evaluations[version][DAILY_METRICS_PERIOD]
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
            return pd.DataFrame(columns=["signal_date", "industry_SW_1", "IC"])
        keys = [valid["signal_date"], valid["industry_SW_1"]]
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
            ["signal_date", "industry_SW_1"],
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
            ["signal_date", "industry_SW_1", "valid_stock_count", "IC"]
        ]

    def save_industry_daily_ic(
        self,
        signal_versions: dict[str, pd.DataFrame],
        save: bool = True,
    ) -> Path | pd.DataFrame:
        """计算行业日频Rank IC；正式MD流水线可只保留内存结果。"""
        forward_returns = calc_forward_returns(
            self.data,
            DAILY_METRICS_PERIOD,
            price_col=self.eval_price_col,
        ).rename("forward_return")
        industry = self.data[DEFAULT_INDUSTRY_COLUMN].astype("string")
        industry = industry[industry.notna() & industry.str.strip().ne("")]
        membership = (
            industry.rename("industry_SW_1")
            .reset_index()
            .groupby(["date", "industry_SW_1"], sort=True)
            .size()
            .rename("industry_stock_count")
            .reset_index()
            .rename(columns={"date": "signal_date"})
        )
        base = pd.concat([industry, forward_returns], axis=1).reset_index()
        base = base.rename(columns={"date": "signal_date"})
        version_results = []
        for version in FACTOR_VERSIONS:
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
                on=["signal_date", "industry_SW_1"],
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

    def _evaluate_and_save_yearly_window(
        self,
        signal_versions: dict[str, pd.DataFrame],
        window: BacktestWindow,
    ) -> pd.DataFrame | None:
        """在自然年边界内重算5日指标并返回内存汇总。"""
        window_data = self._slice_market_data(self.data, window.start, window.end)
        window_signal_versions = {
            version: self._slice_signals(
                signal_versions[version],
                window.start,
                window.end,
            )
            for version in FACTOR_VERSIONS
        }
        if window_data.empty or all(
            signals.dropna(how="all").empty
            for signals in window_signal_versions.values()
        ):
            return None

        evaluations, summary = self.evaluate_versions(
            window_data,
            window_signal_versions,
            forward_periods=(DAILY_METRICS_PERIOD,),
        )
        direction = factor_direction(self.factor_name, self.factor_config)
        for version in FACTOR_VERSIONS:
            full_ic_mean = float(
                self.evaluations[version][DAILY_METRICS_PERIOD]
                .summary.iloc[0]["IC_mean"]
            )
            weekly = weekly_ic_statistics(
                evaluations[version][DAILY_METRICS_PERIOD].full_ic_series,
                direction,
                full_ic_mean,
            )
            summary.loc[DAILY_METRICS_PERIOD, f"weekly_IC_win_rate_{version}"] = weekly["win_rate"]
            summary.loc[DAILY_METRICS_PERIOD, f"weekly_IC_week_count_{version}"] = weekly["week_count"]
            summary.loc[DAILY_METRICS_PERIOD, f"weekly_IC_reference_sign_{version}"] = weekly["sign"]
        yearly_summary = summary.reset_index()
        yearly_summary.insert(0, "year", window.year)
        return yearly_summary

    def save_yearly_reports(
        self,
        signal_versions: dict[str, pd.DataFrame],
    ) -> tuple[list[Path], Path | None]:
        """计算自然年5日汇总并保存年度趋势图。"""
        from plot import save_yearly_summary_trends

        windows = self._iter_yearly_windows()
        yearly_dir = self.output_dir / "yearly"
        if yearly_dir.is_dir():
            shutil.rmtree(yearly_dir)

        summary_frames: list[pd.DataFrame] = []
        for window in windows:
            yearly_summary = self._evaluate_and_save_yearly_window(
                signal_versions,
                window,
            )
            if yearly_summary is None:
                continue
            summary_frames.append(yearly_summary)

        if not summary_frames:
            self.yearly_summary = pd.DataFrame()
            return [], None

        self.yearly_summary = pd.concat(summary_frames, ignore_index=True)
        trend_path = save_yearly_summary_trends(
            self.yearly_summary,
            self.output_dir / "yearly_factor_summary_trends.png",
            factor_name=self.factor_name,
        )
        return [], trend_path

    def latest(self, version: str = "raw") -> pd.Series:
        """返回指定版本最近一个有因子结果的交易日。"""
        if version not in FACTOR_VERSIONS:
            raise ValueError(f"未知因子版本: {version}")
        source = (
            self.raw_result if version == "raw" else self.neutralized_result
        )
        latest_result = source.dropna(how="all")
        if latest_result.empty:
            return pd.Series(dtype=float, name=f"{self.factor_name}_{version}")
        result = latest_result.iloc[-1].dropna().sort_values()
        result.name = latest_result.index[-1]
        return result

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
        self.evaluations, self.summary = self.evaluate_versions(
            self.data,
            signal_versions,
        )
        self.save_daily_metrics(self.evaluations)
        self.save_market_state_summary(signal_versions)
        self.save_industry_daily_ic(signal_versions)
        _, yearly_trend_image = (
            self.save_yearly_reports(signal_versions)
            if save_periodic_reports
            else ([], None)
        )
        build_factor_report(
            self,
            signal_versions,
            self.factor_config,
        )
        return self.evaluations


# ---------------------------------------------------------------------------
# Full 5,947-factor pipeline
# ---------------------------------------------------------------------------

_PIPELINE_CONFIGS: dict[str, dict[str, Any]] = {}
_PIPELINE_BASE_CONFIGS: dict[str, dict[str, Any]] = {}
_PIPELINE_MARKET_DATA: pd.DataFrame | None = None
_PIPELINE_BENCHMARK = pd.DataFrame()
_PIPELINE_CALENDAR = pd.DatetimeIndex([])
_PIPELINE_DATES = pd.DatetimeIndex([])
_PIPELINE_SYMBOLS: list[str] = []
_PIPELINE_WORK_DIR: Path | None = None
_PIPELINE_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
_PIPELINE_FINANCIAL_FACTORS: set[str] = set()


def _pipeline_result(name: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"factor": name, "status": status, **extra}


def _run_parallel(
    function: Any,
    items: list[Any],
    workers: int,
    description: str,
) -> None:
    failures = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(function, item): item for item in items}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=description, unit="factor"
        ):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
                tqdm.write(f"FAILED {result['factor']}\n{result['traceback']}")
    if failures:
        raise RuntimeError(f"{description}失败: {len(failures)}")


def _factor_values_ready(output_root: Path, name: str) -> bool:
    factor_dir = output_root / name / "factors"
    files = list(factor_dir.glob("*.csv")) if factor_dir.is_dir() else []
    if not files:
        return False
    try:
        columns = set(pd.read_csv(files[0], nrows=0).columns)
    except Exception:
        return False
    return {f"{name}_raw", f"{name}_neutralization"}.issubset(columns)


def _calculate_base_values(task: tuple[str, dict[str, Any], str, str]) -> dict[str, Any]:
    name, config, start, output_root_text = task
    try:
        importlib.import_module(config["module"])
        output_root = Path(output_root_text)
        runner = Runner(
            factor_name=name,
            relay_class=config["relay_class"],
            factor_params=config.get("params", {}),
            start=start,
            data_columns=config.get("columns", []),
            data_dir=DEFAULT_DATA_DIR,
            constituents_path=DEFAULT_DATA_DIR / "constituents_daily.csv",
            constituents="000905.SH",
            output_dir=output_root / name,
            is_financial_factor=name in _PIPELINE_FINANCIAL_FACTORS,
            factor_config=config,
        )
        runner.run(save_factor=True, build_report=False)
        return _pipeline_result(name, "completed")
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _robust_z(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        median = np.nanmedian(values, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(values - median), axis=1, keepdims=True)
        scale = 1.4826 * mad
        fallback = np.nanstd(values, axis=1, keepdims=True)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback)
        result = (values - median) / np.where(scale > 1e-12, scale, np.nan)
    return np.clip(result, -5.0, 5.0).astype("float32")


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        np.save(file, values, allow_pickle=False)
    temporary.replace(path)


def _matrix_path(name: str, version: str, standardized: bool = False) -> Path:
    assert _PIPELINE_WORK_DIR is not None
    prefix = "z_" if standardized else ""
    return _PIPELINE_WORK_DIR / "base_matrices" / f"{name}__{prefix}{version}.npy"


def _load_saved_versions(name: str) -> dict[str, pd.DataFrame]:
    date_positions = {
        date.strftime("%Y-%m-%d"): position
        for position, date in enumerate(_PIPELINE_DATES)
    }
    arrays = {
        version: np.full(
            (len(_PIPELINE_DATES), len(_PIPELINE_SYMBOLS)), np.nan, dtype="float32"
        )
        for version in FACTOR_VERSIONS
    }
    value_columns = {
        "raw": f"{name}_raw",
        "neutralization": f"{name}_neutralization",
    }
    for symbol_position, symbol in enumerate(_PIPELINE_SYMBOLS):
        path = _PIPELINE_OUTPUT_ROOT / name / "factors" / f"{symbol}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            usecols=["signal_date", *value_columns.values()],
            dtype={column: "float32" for column in value_columns.values()},
        )
        positions = np.fromiter(
            (date_positions.get(value, -1) for value in frame["signal_date"]),
            dtype="int32",
            count=len(frame),
        )
        valid = positions >= 0
        for version, column in value_columns.items():
            arrays[version][positions[valid], symbol_position] = frame.loc[
                valid, column
            ].to_numpy(dtype="float32")
    return {
        version: pd.DataFrame(values, index=_PIPELINE_DATES, columns=_PIPELINE_SYMBOLS)
        .rename_axis(index="date", columns="symbol")
        for version, values in arrays.items()
    }


def _build_base_matrix(name: str) -> dict[str, Any]:
    try:
        versions = _load_saved_versions(name)
        for version, frame in versions.items():
            values = frame.to_numpy(dtype="float32")
            _atomic_save_npy(_matrix_path(name, version), values)
            _atomic_save_npy(_matrix_path(name, version, standardized=True), _robust_z(values))
        return _pipeline_result(name, "completed")
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _eval_formula_ast(node: ast.AST, variables: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(node, ast.Expression):
        return _eval_formula_ast(node.body, variables)
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return np.asarray(float(node.value), dtype="float32")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_formula_ast(node.operand, variables)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_formula_ast(node.left, variables)
        right = _eval_formula_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"派生公式包含不允许的语法: {ast.dump(node)}")


def _load_base_matrix(name: str, role: str, version: str) -> np.ndarray:
    standardized = role in {"Z", "S"}
    return np.load(_matrix_path(name, version, standardized), mmap_mode="r")


def _evaluate_combination(params: dict[str, Any], version: str) -> np.ndarray:
    directions = dict(zip(params["input_factors"], params["input_directions"]))
    variables: dict[str, np.ndarray] = {}
    counter = 0

    def replace_token(match: re.Match[str]) -> str:
        nonlocal counter
        role, factor_name = match.group(1), match.group(2)
        if factor_name not in directions:
            raise ValueError(f"公式引用未声明的基础因子: {factor_name}")
        variable = f"v{counter}"
        counter += 1
        values = _load_base_matrix(factor_name, role, version)
        variables[variable] = directions[factor_name] * values if role == "S" else values
        return variable

    expression = params["formula"].replace("×", "*")
    expression = re.sub(r"([0-9.]+)\[", r"\1*[", expression)
    expression = re.sub(r"([XZS])\(([^)]+)\)", replace_token, expression)
    expression = expression.replace("[", "(").replace("]", ")")
    expression = re.sub(r"(?<=[0-9.)])(?=v\d+)", "*", expression)
    with np.errstate(all="ignore"):
        return np.asarray(
            _eval_formula_ast(ast.parse(expression, mode="eval"), variables),
            dtype="float32",
        )


def _evaluate_transform(params: dict[str, Any], version: str) -> np.ndarray:
    values = np.asarray(
        _load_base_matrix(params["input_factors"][0], "Z", version), dtype="float32"
    )
    code = params["transformation"]
    with np.errstate(all="ignore"):
        if code == "T01":
            result = np.where(values == 0, np.nan, 1.0 / (values + 1e-6 * np.sign(values)))
        elif code == "T02": result = np.abs(values)
        elif code == "T03": result = np.maximum(values, 0)
        elif code == "T04": result = np.maximum(-values, 0)
        elif code == "T05": result = -np.abs(values - 1)
        elif code == "T06": result = -np.abs(values + 1)
        elif code == "T07": result = values / (1 + values**2)
        elif code == "T08": result = values * (np.abs(values) >= 1)
        elif code == "T09": result = values * (np.abs(values) < 1)
        elif code == "T10": result = np.abs(values) * (1 + 0.5 * (values > 0))
        elif code == "T11": result = np.sign(values) * (np.abs(values) >= 1)
        elif code == "T12": result = -np.abs(values**2 - 1)
        else: raise ValueError(f"未知单因子变换: {code}")
    result = np.asarray(result, dtype="float32")
    result[~np.isfinite(values)] = np.nan
    return result


def _make_memory_runner(name: str, config: dict[str, Any]) -> Runner:
    assert _PIPELINE_MARKET_DATA is not None
    runner = Runner.__new__(Runner)
    runner.factor_name = name
    runner.factor_config = config
    runner.is_financial_factor = any(
        value in _PIPELINE_FINANCIAL_FACTORS
        for value in config.get("params", {}).get("input_factors", [name])
    )
    runner.forward_periods = IC_DECAY_PERIODS
    runner.n_groups = 5
    runner.ic_method = "rank"
    runner.eval_price_col = "adj_close"
    runner.security_status_path = DEFAULT_SECURITY_STATUS_PATH
    runner.market_state_path = DEFAULT_MARKET_STATE_PATH
    runner.output_dir = _PIPELINE_OUTPUT_ROOT / name
    runner.data = _PIPELINE_MARKET_DATA
    runner.benchmark = _PIPELINE_BENCHMARK
    runner.result = pd.DataFrame()
    runner.raw_result = pd.DataFrame()
    runner.neutralized_result = pd.DataFrame()
    runner.evaluations = {}
    runner.summary = pd.DataFrame()
    runner.daily_metrics = pd.DataFrame()
    runner.yearly_summary = pd.DataFrame()
    runner.market_state_summary = pd.DataFrame()
    runner.industry_daily_ic = pd.DataFrame()
    runner.data_loader = SimpleNamespace(trading_calendar=_PIPELINE_CALENDAR)
    return runner


def _calculate_derived_values(name: str) -> dict[str, Any]:
    try:
        config = _PIPELINE_CONFIGS[name]
        params = config["params"]
        arrays = {}
        for version in FACTOR_VERSIONS:
            values = (
                _evaluate_transform(params, version)
                if params["expansion_kind"] == "single_transform"
                else _evaluate_combination(params, version)
            )
            arrays[version] = pd.DataFrame(
                values, index=_PIPELINE_DATES, columns=_PIPELINE_SYMBOLS, dtype="float32"
            ).rename_axis(index="date", columns="symbol")
        arrays["neutralization"] = neutralize_factor_values(
            arrays["neutralization"], _PIPELINE_MARKET_DATA
        ).astype("float32")
        runner = _make_memory_runner(name, config)
        runner.save_signals(
            arrays["raw"],
            combined_data=_PIPELINE_MARKET_DATA,
            neutralized_signals=arrays["neutralization"],
        )
        return _pipeline_result(name, "completed")
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _is_window_factor(name: str, config: dict[str, Any]) -> bool:
    params = config.get("params", {})
    return name in HARDCODED_WINDOW_FACTORS or any(key in params for key in WINDOW_PARAM_KEYS)


def _fill_factor_values(name: str) -> dict[str, Any]:
    try:
        factor_dir = _PIPELINE_OUTPUT_ROOT / name / "factors"
        if not factor_dir.is_dir():
            raise FileNotFoundError(f"因子值目录不存在: {factor_dir}")
        files_changed = cells_filled = 0
        for path in factor_dir.glob("*.csv"):
            frame = pd.read_csv(path)
            columns = [
                column for column in (f"{name}_raw", f"{name}_neutralization")
                if column in frame.columns
            ]
            if not columns:
                raise ValueError(f"{path} 缺少因子值列")
            changed = False
            for column in columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                missing = values.isna()
                filled = values.ffill().bfill()
                fillable = missing & filled.notna()
                if fillable.any():
                    frame[column] = filled
                    cells_filled += int(fillable.sum())
                    changed = True
            if changed:
                temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(path)
                files_changed += 1
        return _pipeline_result(
            name, "completed", files_changed=files_changed, cells_filled=cells_filled
        )
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _remove_legacy_report_outputs(output_dir: Path) -> None:
    """删除旧流水线生成、但正式Markdown流水线不再保留的报告附件。"""
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


def _build_markdown_report(name: str) -> dict[str, Any]:
    try:
        config = _PIPELINE_CONFIGS[name]
        _remove_legacy_report_outputs(_PIPELINE_OUTPUT_ROOT / name)
        signals = _load_saved_versions(name)
        runner = _make_memory_runner(name, config)
        runner.raw_result = signals["raw"]
        runner.neutralized_result = signals["neutralization"]
        runner.evaluations, runner.summary = runner.evaluate_versions(
            _PIPELINE_MARKET_DATA, signals
        )
        runner.save_market_state_summary(signals)
        runner.save_industry_daily_ic(signals, save=False)
        yearly = []
        for window in runner._iter_yearly_windows():
            summary = runner._evaluate_and_save_yearly_window(signals, window)
            if summary is not None:
                yearly.append(summary)
        runner.yearly_summary = pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame()
        build_factor_report(runner, signals, config)
        return _pipeline_result(name, "completed")
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _validate_full_config(
    full_configs: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """校验唯一全量配置，并从中拆出基础与派生因子。"""
    base_configs = {
        name: config
        for name, config in full_configs.items()
        if not config.get("params", {}).get("expansion_kind")
    }
    if len(base_configs) != 199 or len(full_configs) != 5947:
        raise ValueError(
            f"配置数量错误: base={len(base_configs)} full={len(full_configs)}"
        )
    derived = []
    for name, config in full_configs.items():
        if name in base_configs:
            continue
        params = config.get("params", {})
        kind = params.get("expansion_kind")
        required = {"formula", "input_factors", "input_directions"}
        if kind == "single_transform": required.add("transformation")
        elif kind == "factor_combination": required.add("combination")
        else: raise ValueError(f"{name} 缺少有效 expansion_kind")
        missing = sorted(required.difference(params))
        if missing: raise ValueError(f"{name} 派生配置不完整: {missing}")
        if not set(params["input_factors"]).issubset(base_configs):
            raise ValueError(f"{name} 引用了非基础输入")
        if len(params["input_factors"]) != len(params["input_directions"]):
            raise ValueError(f"{name} 输入方向数量不一致")
        derived.append(name)
    if len(derived) != 5748:
        raise ValueError(f"派生因子数错误: {len(derived)}")
    return base_configs, sorted(derived)


def run_full_pipeline(
    full_config_path: Path,
    output_root: Path,
    start: str,
    workers: int,
    resume: bool,
) -> None:
    """从零生成因子值，统一回填后，再生成全部Markdown报告。"""
    global _PIPELINE_CONFIGS, _PIPELINE_BASE_CONFIGS, _PIPELINE_MARKET_DATA
    global _PIPELINE_BENCHMARK, _PIPELINE_CALENDAR, _PIPELINE_DATES
    global _PIPELINE_SYMBOLS, _PIPELINE_WORK_DIR, _PIPELINE_OUTPUT_ROOT
    global _PIPELINE_FINANCIAL_FACTORS

    workers = max(1, workers)
    _PIPELINE_CONFIGS = json.loads(full_config_path.read_text(encoding="utf-8"))
    _PIPELINE_BASE_CONFIGS, derived_names = _validate_full_config(_PIPELINE_CONFIGS)
    financial = json.loads(DEFAULT_FINANCIAL_FACTOR_CONFIG.read_text(encoding="utf-8"))
    _PIPELINE_FINANCIAL_FACTORS = set(financial["factor_names"])
    _PIPELINE_OUTPUT_ROOT = output_root.resolve()
    _PIPELINE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    _PIPELINE_WORK_DIR = RUNTIME_ROOT / f"factor_pipeline_{uuid.uuid4().hex}"
    (_PIPELINE_WORK_DIR / "base_matrices").mkdir(parents=True)

    try:
        base_names = sorted(_PIPELINE_BASE_CONFIGS)
        base_tasks = [
            (name, _PIPELINE_BASE_CONFIGS[name], start, str(_PIPELINE_OUTPUT_ROOT))
            for name in base_names
            if not (resume and _factor_values_ready(_PIPELINE_OUTPUT_ROOT, name))
        ]
        print(f"BASE_VALUES_START factors={len(base_tasks)} workers={workers}", flush=True)
        if base_tasks:
            _run_parallel(_calculate_base_values, base_tasks, workers, "基础因子值")
        print("BASE_VALUES_DONE factors=199 reports=0", flush=True)

        loader = DataLoader(
            data_dir=DEFAULT_DATA_DIR,
            start=start,
            columns=[
                "adj_close", "publish_date",
                DEFAULT_MARKET_CAP_COLUMN, DEFAULT_INDUSTRY_COLUMN,
            ],
            constituents_path=DEFAULT_DATA_DIR / "constituents_daily.csv",
            constituent_index="000905.SH",
            security_status_path=DEFAULT_SECURITY_STATUS_PATH,
        )
        _PIPELINE_MARKET_DATA = loader.load_all().sort_index()
        _PIPELINE_BENCHMARK = loader.benchmark
        _PIPELINE_CALENDAR = loader.trading_calendar
        _PIPELINE_DATES = pd.DatetimeIndex(
            _PIPELINE_MARKET_DATA.index.get_level_values("date").unique()
        ).sort_values()
        _PIPELINE_SYMBOLS = sorted(
            str(value) for value in _PIPELINE_MARKET_DATA.index.get_level_values("symbol").unique()
        )

        print(f"BASE_MATRIX_START factors=199 workers={min(workers, 32)}", flush=True)
        _run_parallel(_build_base_matrix, base_names, min(workers, 32), "基础因子矩阵")
        print("BASE_MATRIX_DONE", flush=True)

        derived_tasks = [
            name for name in derived_names
            if not (resume and _factor_values_ready(_PIPELINE_OUTPUT_ROOT, name))
        ]
        print(f"DERIVED_VALUES_START factors={len(derived_tasks)} workers={workers}", flush=True)
        if derived_tasks:
            _run_parallel(_calculate_derived_values, derived_tasks, workers, "派生因子值")
        print("DERIVED_VALUES_DONE factors=5748 reports=0", flush=True)
        shutil.rmtree(_PIPELINE_WORK_DIR / "base_matrices", ignore_errors=True)
        print("BASE_MATRIX_CLEANED", flush=True)

        window_bases = {
            name for name, config in _PIPELINE_BASE_CONFIGS.items()
            if _is_window_factor(name, config)
        }
        fill_names = set(window_bases)
        for name, config in _PIPELINE_CONFIGS.items():
            if any(
                input_name in window_bases
                for input_name in config.get("params", {}).get("input_factors", [])
            ):
                fill_names.add(name)
        print(f"FILL_START factors={len(fill_names)} workers={workers}", flush=True)
        _run_parallel(_fill_factor_values, sorted(fill_names), workers, "缺失值回填")
        print(f"FILL_DONE factors={len(fill_names)}", flush=True)

        report_names = sorted(_PIPELINE_CONFIGS)
        print(f"REPORT_START factors={len(report_names)} workers={workers}", flush=True)
        _run_parallel(_build_markdown_report, report_names, workers, "Markdown报告")
        print(f"REPORT_DONE completed={len(report_names)} failed=0", flush=True)
    finally:
        if _PIPELINE_WORK_DIR is not None:
            shutil.rmtree(_PIPELINE_WORK_DIR, ignore_errors=True)
        if RUNTIME_ROOT.is_dir() and not any(RUNTIME_ROOT.iterdir()):
            RUNTIME_ROOT.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量计算并评估配置文件中的因子。")
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="运行正式全流程：基础值→派生值→统一回填→全部Markdown报告。",
    )
    parser.add_argument(
        "--factor-config",
        type=Path,
        default=Path("config/factor_configs_199.json"),
        help=(
            "单独运行基础因子时使用的配置；正式 --pipeline 不读取该文件。"
        ),
    )
    parser.add_argument(
        "--save-factor",
        action="store_true",
        help="按股票保存逐日 raw/neutralization 因子值；未指定时只保存评估结果。",
    )
    parser.add_argument(
        "--factor",
        default=None,
        help="只运行配置中的指定因子；未指定时运行配置内全部因子。",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help=f"因子计算和回测起始日，默认使用 {DEFAULT_START_DATE}。",
    )
    parser.add_argument(
        "--financial-factor-config",
        type=Path,
        default=DEFAULT_FINANCIAL_FACTOR_CONFIG,
        help=(
            "财务类因子名单配置；默认使用 "
            "config/financial_factors_in_199.json。"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/zz500"),
        help="单因子输出根目录，默认使用 outputs/zz500。",
    )
    parser.add_argument(
        "--full-factor-config",
        type=Path,
        default=DEFAULT_FULL_FACTOR_CONFIG,
        help="全流程使用的5947因子配置。",
    )
    parser.add_argument("--workers", type=int, default=40, help="全流程并行进程数。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="全流程中跳过已经存在有效因子值列的因子。",
    )
    args = parser.parse_args()

    if args.pipeline:
        run_full_pipeline(
            full_config_path=args.full_factor_config,
            output_root=args.output_root,
            start=args.start,
            workers=args.workers,
            resume=args.resume,
        )
        raise SystemExit(0)

    factor_config_path = args.factor_config
    with factor_config_path.open("r", encoding="utf-8") as f:
        factor_configs = json.load(f)
    if args.factor is not None:
        if args.factor not in factor_configs:
            parser.error(
                f"因子配置 {factor_config_path} 中不存在因子: {args.factor}"
            )
        factor_configs = {args.factor: factor_configs[args.factor]}
    with args.financial_factor_config.open("r", encoding="utf-8") as f:
        financial_factor_config = json.load(f)
    financial_factor_names = set(financial_factor_config["factor_names"])

    data_dir = Path("cache/zz500_csv")
    for factor_name, config in tqdm(
        factor_configs.items(),
        total=len(factor_configs),
        desc="因子生成",
        unit="factor",
    ):
        if config.get("params", {}).get("expansion_kind"):
            parser.error("派生因子必须通过 --pipeline 运行，不能按普通注册因子执行。")
        importlib.import_module(config["module"])
        runner = Runner(
            factor_name=factor_name,
            relay_class=config["relay_class"],
            factor_params=config["params"],
            start=args.start,
            data_columns=[column for column in config["columns"]],
            data_dir=data_dir,
            factor_dir=None,
            constituents_path=data_dir / "constituents_daily.csv",
            constituents="000905.SH",
            output_dir=args.output_root / factor_name,
            is_financial_factor=factor_name in financial_factor_names,
            factor_config=config,
        )
        runner.run(save_factor=args.save_factor)
