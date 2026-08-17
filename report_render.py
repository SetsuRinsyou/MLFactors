"""渲染人工可审阅的单因子统一 Markdown 报告。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factors_eval import (
    FactorEvalResult,
    calc_forward_returns,
    calc_tail_group_returns,
    eval as evaluate_factor,
)
from report_calculate import factor_direction, weekly_ic_statistics
from settings import SETTINGS


def decay_statistics(evaluations: dict[int, FactorEvalResult]) -> dict[str, Any]:
    """按传入持有期的IC强度计算峰值和持久跌破50%的半衰期。"""
    rows = []
    for horizon, evaluation in sorted(evaluations.items()):
        summary = evaluation.summary.iloc[0]
        rows.append({
            "horizon": int(horizon),
            "IC_mean": float(summary["IC_mean"]),
            "IC_std": float(summary["IC_std"]),
            "ICIR": float(summary["ICIR"]),
            "t_stat": float(summary["t_stat"]),
            "p_value": float(summary["p_value"]),
            "strength": abs(float(summary["IC_mean"])),
        })
    frame = pd.DataFrame(rows)
    valid = frame[np.isfinite(frame["strength"])]
    if valid.empty:
        return {"frame": frame, "peak_horizon": "~", "peak_strength": np.nan, "half_life": "~"}
    peak_index = valid["strength"].idxmax()
    peak_horizon = int(frame.loc[peak_index, "horizon"])
    peak_strength = float(frame.loc[peak_index, "strength"])
    after = frame.loc[frame["horizon"] > peak_horizon].reset_index(drop=True)
    half_life: str | int = f">{int(frame['horizon'].max())}"
    threshold = peak_strength / 2.0
    for position, row in after.iterrows():
        later = after.loc[position:, "strength"]
        if np.isfinite(row["strength"]) and row["strength"] <= threshold and (later <= threshold).all():
            half_life = int(row["horizon"] - peak_horizon)
            break
    return {
        "frame": frame,
        "peak_horizon": peak_horizon,
        "peak_strength": peak_strength,
        "half_life": half_life,
    }


def _escape(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(_escape(value) for value in headers) + " |"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(_escape(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def number(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "~" if value is None else str(value)
    return "~" if not np.isfinite(value) else f"{value:.{digits}f}"


def percent(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "~"
    return "~" if not np.isfinite(value) else f"{value:.2%}"


def integer(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "~"
    return "~" if not np.isfinite(value) else str(int(value))


def interval_return(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    return float((1.0 + values).prod() - 1.0) if not values.empty else np.nan


def relative_returns(returns: pd.Series, benchmark: pd.Series) -> pd.Series:
    aligned = pd.concat([returns.rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return (1.0 + aligned["portfolio"]) / (1.0 + aligned["benchmark"]) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    nav = (1.0 + values).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def _benchmark_prices(runner: Any, dates: pd.DatetimeIndex | None = None) -> pd.Series | None:
    if runner.benchmark.empty:
        return None
    column = (
        SETTINGS.report.preferred_benchmark_column
        if SETTINGS.report.preferred_benchmark_column in runner.benchmark.columns
        else runner.benchmark.columns[0]
    )
    series = runner.benchmark[column]
    return series.reindex(dates) if dates is not None else series


def _category(factor_name: str, config: dict[str, Any]) -> str:
    params = config.get("params", {})
    rules_path = SETTINGS.paths.factor_category_config
    rules = json.loads(rules_path.read_text(encoding="utf-8"))["categories"]
    operation = str(params.get("operation", "")).upper()
    matches = [
        rule["name"] for rule in rules
        if factor_name in rule.get("factors", []) or operation in rule.get("operations", [])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"因子 {factor_name} 必须且只能匹配一个因子类别，实际={matches}；"
            "请先重新生成 config/factors/factor_categories_5947.json"
        )
    return matches[0]


def _missing_rate_text(runner: Any, signal_versions: dict[str, pd.DataFrame]) -> str:
    eligible_index = runner.data.index
    values = signal_versions["raw"].stack(future_stack=True).reindex(eligible_index)
    valid = int(np.isfinite(pd.to_numeric(values, errors="coerce")).sum())
    eligible = int(len(eligible_index))
    missing_rate = (eligible - valid) / eligible if eligible else np.nan
    return percent(missing_rate)


def _core_rows(
    evaluations: dict[str, dict[int, FactorEvalResult]], direction: str,
) -> tuple[list[list[str]], dict[str, dict[str, Any]]]:
    rows = []
    weekly_stats = {}
    for version in SETTINGS.report.factor_versions:
        evaluation = evaluations[version][SETTINGS.report.daily_metrics_period]
        summary = evaluation.summary.iloc[0]
        stats = weekly_ic_statistics(evaluation.full_ic_series, direction, float(summary["IC_mean"]))
        weekly_stats[version] = stats
        skewness = pd.to_numeric(evaluation.full_ic_series, errors="coerce").dropna().skew()
        rows.append([
            SETTINGS.report.version_names[version], number(summary["IC_mean"]), number(summary["IC_std"]),
            number(summary["ICIR"]), number(summary["t_stat"]), number(summary["p_value"], 6),
            stats["sign"], stats["week_count"], percent(stats["win_rate"]),
            number(skewness), number(stats["autocorrelation"]),
            stats["max_failure_streak"], stats["failure_interval"],
            percent(evaluation.turnover.mean()),
            percent(summary[f"top_{SETTINGS.report.core_tail_percentage}%_cumulative_return"]),
            percent(summary[f"bottom_{SETTINGS.report.core_tail_percentage}%_cumulative_return"]),
            percent(summary[f"top_{SETTINGS.report.core_tail_percentage}%_max_drawdown"]),
            percent(summary[f"bottom_{SETTINGS.report.core_tail_percentage}%_max_drawdown"]),
        ])
    return rows, weekly_stats


def _decay_rows(evaluations: dict[str, dict[int, FactorEvalResult]]) -> tuple[list[list[str]], list[list[str]]]:
    decay = {version: decay_statistics(evaluations[version]) for version in SETTINGS.report.factor_versions}
    horizons = sorted(set(decay["raw"]["frame"]["horizon"]))
    rows = []
    for horizon in horizons:
        raw = decay["raw"]["frame"].set_index("horizon").loc[horizon]
        neutral = decay["neutralization"]["frame"].set_index("horizon").loc[horizon]
        rows.append([
            f"{horizon}日", number(raw["IC_mean"]), number(raw["ICIR"]), number(raw["p_value"], 6),
            number(neutral["IC_mean"]), number(neutral["ICIR"]), number(neutral["p_value"], 6),
        ])
    conclusions = [[
        SETTINGS.report.version_names[version], f"{decay[version]['peak_horizon']}日",
        number(decay[version]["peak_strength"]), decay[version]["half_life"],
    ] for version in SETTINGS.report.factor_versions]
    return rows, conclusions


def _group_rows(
    evaluations: dict[str, dict[int, FactorEvalResult]],
    version: str,
) -> list[list[str]]:
    rows = []
    evaluation = evaluations[version][SETTINGS.report.daily_metrics_period]
    benchmark = evaluation.sampled_benchmark_returns
    for group in evaluation.layered.group_returns.columns:
        returns = evaluation.layered.group_returns[group]
        relative = relative_returns(returns, benchmark)
        rows.append([
            f"Group {group}", percent(interval_return(returns)),
            percent(interval_return(relative)), percent(max_drawdown(returns)),
        ])
    return rows


def _tail_rows(
    evaluations: dict[str, dict[int, FactorEvalResult]],
    version: str,
) -> list[list[str]]:
    rows = []
    summary = evaluations[version][SETTINGS.report.daily_metrics_period].summary.iloc[0]
    for pct in SETTINGS.evaluation.tail_percentages:
        for side in ("top", "bottom"):
            key = f"{side}_{pct}%"
            rows.append([
                f"{side.title()} {pct}%",
                percent(summary[f"{key}_cumulative_return"]),
                percent(summary[f"{key}_excess_return"]),
                percent(summary[f"{key}_max_drawdown"]),
            ])
    return rows


def _lag_rows(
    runner: Any,
    signal_versions: dict[str, pd.DataFrame],
) -> dict[str, list[list[str]]]:
    rows = {version: [] for version in SETTINGS.report.factor_versions}
    prices = _benchmark_prices(runner)
    baseline = {}
    for lag in SETTINGS.report.report_lags:
        for version in SETTINGS.report.factor_versions:
            if lag == 0:
                evaluation = runner.evaluations[version][SETTINGS.report.daily_metrics_period]
            else:
                signals = signal_versions[version].shift(lag)
                evaluation = evaluate_factor(
                    signals, runner.data, forward_period=SETTINGS.report.daily_metrics_period,
                    n_groups=runner.n_groups,
                    ic_method=runner.ic_method, price_col=runner.eval_price_col,
                    benchmark_prices=prices, portfolio_only=True,
                )
            summary = evaluation.summary.iloc[0]
            strength = abs(float(summary["IC_mean"]))
            if lag == 0:
                baseline[version] = strength
            retention = strength / baseline[version] if baseline.get(version, 0) else np.nan
            rows[version].append([
                lag, number(summary["IC_mean"]), percent(retention),
                percent(summary[f"top_{SETTINGS.report.core_tail_percentage}%_cumulative_return"]),
                percent(summary[f"top_{SETTINGS.report.core_tail_percentage}%_excess_return"]),
                percent(summary[f"bottom_{SETTINGS.report.core_tail_percentage}%_cumulative_return"]),
                percent(summary[f"bottom_{SETTINGS.report.core_tail_percentage}%_excess_return"]),
            ])
    return rows


def _size_rows(
    runner: Any,
    signal_versions: dict[str, pd.DataFrame],
) -> dict[str, list[list[str]]]:
    if SETTINGS.data.market_cap_column not in runner.data.columns:
        return {
            version: [["~", "~", "~", "~", "~", "缺少market_cap"]]
            for version in SETTINGS.report.factor_versions
        }
    cap = (
        runner.data[SETTINGS.data.market_cap_column]
        .unstack()
        .reindex_like(signal_versions["raw"])
    )
    ranks = cap.rank(axis=1, pct=True, method="first")
    lower_quantile, upper_quantile = SETTINGS.report.size_bucket_quantiles
    masks = {
        "中证500内部小盘组": ranks <= lower_quantile,
        "中证500内部中盘组": (
            (ranks > lower_quantile) & (ranks <= upper_quantile)
        ),
        "中证500内部大盘组": ranks > upper_quantile,
    }
    rows = {version: [] for version in SETTINGS.report.factor_versions}
    prices = _benchmark_prices(runner)
    for bucket, mask in masks.items():
        for version in SETTINGS.report.factor_versions:
            evaluation = evaluate_factor(
                signal_versions[version].where(mask), runner.data,
                forward_period=SETTINGS.report.daily_metrics_period,
                n_groups=runner.n_groups, ic_method=runner.ic_method,
                price_col=runner.eval_price_col, benchmark_prices=prices,
                portfolio_only=True,
            )
            summary = evaluation.summary.iloc[0]
            rows[version].append([
                bucket, number(summary["IC_mean"]),
                percent(summary[f"top_{SETTINGS.report.core_tail_percentage}%_cumulative_return"]),
                percent(summary[f"top_{SETTINGS.report.core_tail_percentage}%_excess_return"]),
                percent(summary[f"bottom_{SETTINGS.report.core_tail_percentage}%_cumulative_return"]),
                percent(summary[f"bottom_{SETTINGS.report.core_tail_percentage}%_excess_return"]),
            ])
    return rows


def _industry_rows(
    runner: Any,
    signal_versions: dict[str, pd.DataFrame],
    direction: str,
) -> dict[str, list[list[str]]]:
    frame = runner.industry_daily_ic.copy()
    if frame.empty:
        return {version: [] for version in SETTINGS.report.factor_versions}
    forward_returns = calc_forward_returns(
        runner.data,
        SETTINGS.report.daily_metrics_period,
        price_col=runner.eval_price_col,
    )
    industry = runner.data[SETTINGS.data.industry_column].astype("string")
    trading_dates = pd.DatetimeIndex(runner.data.index.get_level_values("date").unique()).sort_values()
    sampled_dates = trading_dates[::SETTINGS.report.daily_metrics_period]
    benchmark = runner.evaluations["raw"][
        SETTINGS.report.daily_metrics_period
    ].sampled_benchmark_returns
    rows = {version: [] for version in SETTINGS.report.factor_versions}
    global_sign = {
        version: float(
            runner.evaluations[version][SETTINGS.report.daily_metrics_period]
            .summary.iloc[0]["IC_mean"]
        )
        for version in SETTINGS.report.factor_versions
    }
    factors = {}
    for version in SETTINGS.report.factor_versions:
        factor = signal_versions[version].stack().rename("factor")
        factor.index.names = ["date", "symbol"]
        factors[version] = factor
    for industry_name in sorted(
        frame[SETTINGS.data.industry_column].dropna().unique()
    ):
        membership = industry.eq(industry_name)
        selected_index = membership[membership].index
        for version in SETTINGS.report.factor_versions:
            daily_ic = frame.loc[
                frame[SETTINGS.data.industry_column].eq(industry_name)
            ].set_index("signal_date")[f"IC_{version}"]
            daily_ic.index = pd.to_datetime(daily_ic.index)
            stats = weekly_ic_statistics(daily_ic, direction, global_sign[version])
            factor = factors[version].reindex(selected_index).dropna()
            factor = factor[factor.index.get_level_values("date").isin(sampled_dates)]
            returns = forward_returns.reindex(factor.index)
            tails = calc_tail_group_returns(
                factor,
                returns,
                fractions=(SETTINGS.report.core_tail_fraction,),
            )
            top = tails.get(
                f"top_{SETTINGS.report.core_tail_percentage}%_return", pd.Series(dtype=float)
            )
            bottom = tails.get(
                f"bottom_{SETTINGS.report.core_tail_percentage}%_return", pd.Series(dtype=float)
            )
            top_relative = relative_returns(top, benchmark)
            bottom_relative = relative_returns(bottom, benchmark)
            rows[version].append([
                industry_name, number(daily_ic.mean()),
                percent(stats["win_rate"]), percent(interval_return(top)),
                percent(interval_return(top_relative)),
                percent(interval_return(bottom)),
                percent(interval_return(bottom_relative)),
            ])
    return rows


def _period_ic_rows(
    frame: pd.DataFrame,
    leading: list[str],
    version: str,
    market_state: bool = False,
) -> list[list[str]]:
    if frame.empty:
        return []
    rows = []
    for record in frame.to_dict(orient="records"):
        row = []
        for column in leading:
            value = record.get(column, "~")
            if market_state and column == "state":
                value = f"{value} · {SETTINGS.report.market_state_names.get(value, '~')}"
            row.append(value)
        row.extend([
            number(record.get(f"IC_mean_{version}", np.nan)),
            number(record.get(f"IC_std_{version}", np.nan)),
            number(record.get(f"ICIR_{version}", np.nan)),
            number(record.get(f"t_stat_{version}", np.nan)),
            number(record.get(f"p_value_{version}", np.nan), 6),
            record.get(f"weekly_IC_reference_sign_{version}", "~"),
            integer(record.get(f"weekly_IC_week_count_{version}", np.nan)),
            percent(record.get(f"weekly_IC_win_rate_{version}", np.nan)),
        ])
        rows.append(row)
    return rows


def build_factor_report(
    runner: Any,
    signal_versions: dict[str, pd.DataFrame],
    factor_config: dict[str, Any],
) -> Path:
    """把全时段、市场状态和年度结果合并为唯一 factor_report.md。"""
    direction = factor_direction(runner.factor_name, factor_config)
    data_dates = pd.DatetimeIndex(runner.data.index.get_level_values("date").unique()).sort_values()
    performance_dates = runner.evaluations["raw"][
        SETTINGS.report.daily_metrics_period
    ].full_ic_series.index
    core_rows, _ = _core_rows(runner.evaluations, direction)
    decay_rows, decay_conclusions = _decay_rows(runner.evaluations)
    lag_rows = _lag_rows(runner, signal_versions)
    size_rows = _size_rows(runner, signal_versions)
    industry_rows = _industry_rows(runner, signal_versions, direction)
    params = factor_config.get("params", {})
    metadata_rows = [
        ["因子名称", runner.factor_name],
        ["因子类别", _category(runner.factor_name, factor_config)],
        ["经济方向", direction],
        ["参数", f"`{json.dumps(params, ensure_ascii=False)}`"],
        ["股票池", "中证500历史成分股"],
        ["因子数据区间", f"{data_dates.min().date()} 至 {data_dates.max().date()}"],
        [
            f"{SETTINGS.report.daily_metrics_period}日收益有效区间",
            f"{performance_dates.min().date()} 至 {performance_dates.max().date()}",
        ],
        ["调仓频率", f"{SETTINGS.report.daily_metrics_period}个交易日"],
        [
            "收益窗口",
            f"T+1买入、T+{SETTINGS.report.daily_metrics_period + 1}卖出",
        ],
        ["样本过滤", "历史成分股、ST、停牌"],
        ["中性化", "对数市值＋申万一级行业OLS残差"],
        ["空缺率", _missing_rate_text(runner, signal_versions)],
    ]

    sections = [
        f"# {runner.factor_name} 单因子验收与评估报告",
        "\n> 本文件是该因子的唯一汇总报告。`原始因子值`对应内部 `_raw`，`中性化因子值`对应内部 `_neutralization`。",
        "\n## 一、基本信息\n",
        markdown_table(["项目", "内容"], metadata_rows),
        "\n## 二、核心指标\n",
        markdown_table(
            ["因子版本", "IC均值", "IC标准差", "ICIR", "t-stat", "p-value", "胜率基准符号", "自然周数", "自然周IC胜率", "IC偏度", "IC滞后1期自相关系数", "最大连续失效周数", "最长失效区间", f"{SETTINGS.report.daily_metrics_period}日单边换手率", f"Top{SETTINGS.report.core_tail_percentage}区间绝对收益", f"Bottom{SETTINGS.report.core_tail_percentage}区间绝对收益", f"Top{SETTINGS.report.core_tail_percentage}绝对最大回撤", f"Bottom{SETTINGS.report.core_tail_percentage}绝对最大回撤"],
            core_rows,
        ),
        "\n## 三、细节表现\n",
        "\n### IC衰减\n",
        markdown_table(["持有期", "原始IC均值", "原始ICIR", "原始p值", "中性化IC均值", "中性化ICIR", "中性化p值"], decay_rows),
        "\n#### 衰减结论\n",
        markdown_table(["因子版本", "IC峰值持有期", "峰值IC强度", "IC半衰期（交易日）"], decay_conclusions),
    ]

    sections.append("\n### 五分组表现\n")
    for version in SETTINGS.report.factor_versions:
        sections.extend([
            f"\n#### {SETTINGS.report.version_names[version]}\n",
            markdown_table(
                ["分组", "区间绝对收益", "区间超额收益", "绝对最大回撤"],
                _group_rows(runner.evaluations, version),
            ),
        ])

    sections.append("\n### Top/Bottom阈值表现\n")
    for version in SETTINGS.report.factor_versions:
        sections.extend([
            f"\n#### {SETTINGS.report.version_names[version]}\n",
            markdown_table(
                ["组合", "区间绝对收益", "区间超额收益", "绝对最大回撤"],
                _tail_rows(runner.evaluations, version),
            ),
        ])

    sections.append("\n### 滞后检验\n")
    for version in SETTINGS.report.factor_versions:
        sections.extend([
            f"\n#### {SETTINGS.report.version_names[version]}\n",
            markdown_table(
                ["滞后交易日", "IC均值", "IC强度保留率", "Top20区间绝对收益", "Top20区间超额收益", "Bottom20区间绝对收益", "Bottom20区间超额收益"],
                lag_rows[version],
            ),
        ])

    sections.append("\n## 四、市值表现\n")
    for version in SETTINGS.report.factor_versions:
        sections.extend([
            f"\n### {SETTINGS.report.version_names[version]}\n",
            markdown_table(
                ["市值层", "IC均值", "Top20区间绝对收益", "Top20区间超额收益", "Bottom20区间绝对收益", "Bottom20区间超额收益"],
                size_rows[version],
            ),
        ])

    sections.append("\n## 五、行业表现\n")
    for version in SETTINGS.report.factor_versions:
        sections.extend([
            f"\n### {SETTINGS.report.version_names[version]}\n",
            markdown_table(
                ["行业", "IC均值", "自然周IC胜率", "Top20区间绝对收益", "Top20区间超额收益", "Bottom20区间绝对收益", "Bottom20区间超额收益"],
                industry_rows[version],
            ),
        ])

    sections.append("\n## 六、市场状态表现\n")
    for version in SETTINGS.report.factor_versions:
        for era in SETTINGS.report.market_state_eras:
            era_frame = runner.market_state_summary.loc[
                runner.market_state_summary["era"].eq(era)
            ]
            sections.extend([
                f"\n### {era} · {SETTINGS.report.version_names[version]}\n",
                markdown_table(
                    ["状态", "连续片段数", "状态交易日", "IC均值", "IC标准差", "ICIR", "t-stat", "p-value", "胜率基准符号", "自然周数", "自然周IC胜率"],
                    _period_ic_rows(
                        era_frame,
                        ["state", "segment_count", "state_trading_days"],
                        version,
                        market_state=True,
                    ),
                ),
            ])

    sections.append("\n## 七、年度表现\n")
    for version in SETTINGS.report.factor_versions:
        sections.extend([
            f"\n### {SETTINGS.report.version_names[version]}\n",
            markdown_table(
                ["年份", "IC均值", "IC标准差", "ICIR", "t-stat", "p-value", "胜率基准符号", "自然周数", "自然周IC胜率"],
                _period_ic_rows(runner.yearly_summary, ["year"], version),
            ),
        ])
    runner.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = runner.output_dir / "factor_report.md"
    temporary = output_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return output_path
