"""生成人工可审阅的单因子统一 Markdown 报告。"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from factors_eval import (
    FactorEvalResult,
    calc_forward_returns,
    calc_tail_group_returns,
    eval as evaluate_factor,
)


FACTOR_VERSIONS = ("raw", "neutralization")
VERSION_NAMES = {"raw": "原始因子值", "neutralization": "中性化因子值"}
TAIL_PCTS = (10, 20, 30)
MARKET_STATE_NAMES = {
    0: "全面牛市",
    1: "结构性牛市",
    2: "非趋势",
    3: "熊市",
}

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

CUSTOM_FACTOR_FORMULAS = {
    "cfp_1d": r"\frac{\mathrm{OCF}_t}{\mathrm{MarketCap}_t}",
    "current_liability_ratio_1d": r"\frac{\mathrm{CurrentLiabilities}_t}{\mathrm{TotalLiabilities}_t}",
    "current_ratio_1d": r"\frac{\mathrm{CurrentAssets}_t}{\mathrm{CurrentLiabilities}_t}",
    "debt_per_share_1d": r"\frac{\mathrm{TotalDebt}_t}{\mathrm{BasicShares}_t}",
    "debt_to_asset_1d": r"\frac{\mathrm{TotalLiabilities}_t}{\mathrm{TotalAssets}_t}",
    "dip_rebound_10d": r"\frac{1}{10}\sum_{k=0}^{9}\ln\!\left(\frac{C_{t-k}}{L_{t-k}}\right)",
    "eps_growth_63d": r"\frac{\mathrm{EPS}_t-\mathrm{EPS}_{t-63}}{|\mathrm{EPS}_{t-63}|}",
    "equity_growth_63d": r"\frac{\mathrm{Equity}_t-\mathrm{Equity}_{t-63}}{|\mathrm{Equity}_{t-63}|}",
    "fixed_ratio_1d": r"\frac{\mathrm{NetPPE}_t}{\mathrm{TotalAssets}_t}",
    "gross_margin_1d": r"\frac{\mathrm{GrossProfit}_t}{\mathrm{Revenue}_t}",
    "high_r_std_84d": r"\operatorname{Std}_{84}\!\left(\frac{H_t}{C_{t-1}}-1\right)",
    "hml_r_std_84d": r"\operatorname{Std}_{84}\!\left(\frac{H_t}{C_{t-1}}-1\right)-\operatorname{Std}_{84}\!\left(\frac{L_t}{C_{t-1}}-1\right)",
    "industry_momentum_1d": r"\frac{\sum_{j\in\mathcal I(i),\,j\ne i}r_{j,t}}{N_{\mathcal I(i),t}-1}",
    "inventory_turnover_1d": r"\frac{\mathrm{CostOfRevenue}_t}{\mathrm{Inventory}_t}",
    "long_term_debt_ratio_1d": r"\frac{\mathrm{NonCurrentDebt}_t}{\mathrm{TotalAssets}_t}",
    "max_daily_return_5d": r"\frac{1}{5}\sum_{r\in\operatorname{Top5}\{r_{t-20},\ldots,r_t\}}r",
    "net_income_cash_ratio_1d": r"\frac{\mathrm{OCF}_t}{\mathrm{NetIncome}_t}",
    "net_income_growth_63d": r"\frac{\mathrm{NetIncome}_t-\mathrm{NetIncome}_{t-63}}{|\mathrm{NetIncome}_{t-63}|}",
    "net_profit_margin_1d": r"\frac{\mathrm{NetIncome}_t}{\mathrm{Revenue}_t}",
    "open_high_surge_10d": r"\frac{1}{10}\sum_{k=0}^{9}\ln\!\left(\frac{H_{t-k}}{O_{t-k}}\right)",
    "operating_expense_ratio_1d": r"\frac{\mathrm{OperatingExpense}_t}{\mathrm{Revenue}_t}",
    "quick_ratio_1d": r"\frac{\mathrm{CurrentAssets}_t-\mathrm{Inventory}_t}{\mathrm{CurrentLiabilities}_t}",
    "r_std_84d": r"\operatorname{Std}_{84}(r_t),\qquad r_t=\frac{C_t}{C_{t-1}}-1",
    "revenue_growth_63d": r"\frac{\mathrm{Revenue}_t-\mathrm{Revenue}_{t-63}}{|\mathrm{Revenue}_{t-63}|}",
    "reversal_1m": r"-\left(\frac{C_t}{C_{t-20}}-1\right)",
    "reversal_3m": r"-\left(\frac{C_t}{C_{t-60}}-1\right)",
    "roe_growth_63d": r"\frac{\mathrm{ROE}_t-\mathrm{ROE}_{t-63}}{|\mathrm{ROE}_{t-63}|}",
    "size_factor_1d": r"\ln(\mathrm{MarketCap}_t)",
    "size_squared_1d": r"\left[\ln(\mathrm{MarketCap}_t)\right]^2",
    "slp_reversal_20d": r"-\frac{C_t-C_k}{C_k(t-k)},\qquad k=\text{20日窗口内较近的最高或最低价位置}",
    "smallcap_growth_1d": r"Z_{\mathrm{CS}}\!\left[\operatorname{Winsor}_{1\%,99\%}\!\left(-\ln\mathrm{MarketCap}_t\right)\right]",
    "sp_1d": r"\frac{\mathrm{MarketCap}_t}{\mathrm{Revenue}_t}",
    "total_asset_growth_63d": r"\frac{\mathrm{TotalAssets}_t-\mathrm{TotalAssets}_{t-63}}{|\mathrm{TotalAssets}_{t-63}|}",
    "total_asset_turnover_1d": r"\frac{\mathrm{Revenue}_t}{\mathrm{TotalAssets}_t}",
    "trading_amount_20d": r"\frac{1}{20}\sum_{k=0}^{19}C_{t-k}V_{t-k}",
    "turnover_cv_20d": r"\frac{\operatorname{Std}_{20}(\mathrm{Turnover}_t)}{\operatorname{Mean}_{20}(\mathrm{Turnover}_t)},\quad \mathrm{Turnover}_t=\frac{V_tC_t}{\mathrm{MarketCap}_t}",
    "vff3_63d": r"\operatorname{Std}_{63}(\varepsilon_t),\quad r_t=\alpha+\beta_M\mathrm{MKT}_t+\beta_S\mathrm{SMB}_t+\beta_H\mathrm{HML}_t+\varepsilon_t",
    "vff3_up_63d": r"\operatorname{Std}_{63}(\varepsilon_t\mid\varepsilon_t>0)",
    "vff3_upd_63d": r"\operatorname{Std}_{63}(\varepsilon_t\mid\varepsilon_t>0)+\operatorname{Std}_{63}(\varepsilon_t\mid\varepsilon_t<0)",
    "volume_3m": r"\frac{1}{60}\sum_{k=0}^{59}V_{t-k}",
    "volume_price_corr_15d": r"\operatorname{Corr}_{10}\!\left(C_t,\frac{V_tC_t}{\mathrm{MarketCap}_t}\right)",
}


def factor_formula(factor_name: str, config: dict[str, Any]) -> str:
    """Return the report formula without depending on aggregation code."""
    params = config.get("params", {})
    expansion_kind = params.get("expansion_kind")
    if expansion_kind == "single_transform":
        base = params["input_factors"][0].replace("_", r"\_")
        return params["formula"] + rf"\qquad u=Z_v\!\left(\mathrm{{{base}}}\right)"
    if expansion_kind == "factor_combination":
        formula = params["formula"].replace("×", r"\times")
        def replace_factor_token(match: re.Match[str]) -> str:
            escaped_name = match.group(2).replace("_", r"\_")
            return (
                rf"\operatorname{{{match.group(1)}}}\!\left("
                rf"\mathrm{{{escaped_name}}}\right)"
            )
        return re.sub(
            r"\b([XZS])\(([A-Za-z0-9_]+)\)",
            replace_factor_token,
            formula,
        )
    if factor_name in CUSTOM_FACTOR_FORMULAS:
        return CUSTOM_FACTOR_FORMULAS[factor_name]
    operation = str(params.get("operation", "")).upper()
    window = params.get("window", 1)
    formulas = {
        "HIGH0": r"\frac{H_t}{C_t}", "LOW0": r"\frac{L_t}{C_t}",
        "OPEN0": r"\frac{O_t}{C_t}", "VWAP0": r"\frac{\mathrm{VWAP}_t}{C_t}",
        "KMID": r"\frac{C_t-O_t}{O_t}", "KMID2": r"\frac{C_t-O_t}{H_t-L_t}",
        "KUP": r"\frac{H_t-\max(O_t,C_t)}{O_t}", "KUP2": r"\frac{H_t-\max(O_t,C_t)}{H_t-L_t}",
        "KLOW": r"\frac{\min(O_t,C_t)-L_t}{O_t}", "KLOW2": r"\frac{\min(O_t,C_t)-L_t}{H_t-L_t}",
        "KSFT": r"\frac{2C_t-H_t-L_t}{O_t}", "KSFT2": r"\frac{2C_t-H_t-L_t}{H_t-L_t}",
        "KLEN": r"\frac{H_t-L_t}{O_t}",
        "BETA": rf"\frac{{\operatorname{{Slope}}(C_{{t-{window}+1:t}}\sim\mathrm{{time}})}}{{C_t}}",
        "RESI": rf"\frac{{\operatorname{{Residual}}_t(C_{{t-{window}+1:t}}\sim\mathrm{{time}})}}{{C_t}}",
        "RSQR": rf"R^2(C_{{t-{window}+1:t}}\sim\mathrm{{time}})",
        "ROC": rf"\frac{{C_{{t-{window}}}}}{{C_t}}", "MA": rf"\frac{{\operatorname{{Mean}}_{{{window}}}(C_t)}}{{C_t}}",
        "MAX": rf"\frac{{\operatorname{{Max}}_{{{window}}}(H_t)}}{{C_t}}", "MIN": rf"\frac{{\operatorname{{Min}}_{{{window}}}(L_t)}}{{C_t}}",
        "QTLU": rf"\frac{{Q^{{80\%}}_{{{window}}}(C_t)}}{{C_t}}", "QTLD": rf"\frac{{Q^{{20\%}}_{{{window}}}(C_t)}}{{C_t}}",
        "RANK": rf"\operatorname{{PercentileRank}}\!\left(C_t;C_{{t-{window}+1:t}}\right)",
        "IMAX": rf"\frac{{\operatorname{{argmax}}_{{{window}}}(H)+1}}{{{window}}}", "IMIN": rf"\frac{{\operatorname{{argmin}}_{{{window}}}(L)+1}}{{{window}}}",
        "IMXD": rf"\frac{{\operatorname{{argmax}}_{{{window}}}(H)-\operatorname{{argmin}}_{{{window}}}(L)}}{{{window}}}",
        "RSV": rf"\frac{{C_t-\operatorname{{Min}}_{{{window}}}(L)}}{{\operatorname{{Max}}_{{{window}}}(H)-\operatorname{{Min}}_{{{window}}}(L)}}",
        "CNTP": rf"\operatorname{{Mean}}_{{{window}}}\!\left(\mathbf 1[C_t>C_{{t-1}}]\right)",
        "CNTN": rf"\operatorname{{Mean}}_{{{window}}}\!\left(\mathbf 1[C_t<C_{{t-1}}]\right)",
        "CNTD": rf"\operatorname{{Mean}}_{{{window}}}(\mathbf 1[C_t>C_{{t-1}}])-\operatorname{{Mean}}_{{{window}}}(\mathbf 1[C_t<C_{{t-1}}])",
        "SUMP": rf"\frac{{\sum_{{{window}}}\max(\Delta C_t,0)}}{{\sum_{{{window}}}|\Delta C_t|}}", "SUMN": rf"\frac{{\sum_{{{window}}}\max(-\Delta C_t,0)}}{{\sum_{{{window}}}|\Delta C_t|}}",
        "SUMD": rf"\frac{{\sum_{{{window}}}\max(\Delta C_t,0)-\sum_{{{window}}}\max(-\Delta C_t,0)}}{{\sum_{{{window}}}|\Delta C_t|}}",
        "VMA": rf"\frac{{\operatorname{{Mean}}_{{{window}}}(V_t)}}{{V_t}}", "VSUMP": rf"\frac{{\sum_{{{window}}}\max(\Delta V_t,0)}}{{\sum_{{{window}}}|\Delta V_t|}}",
        "VSUMN": rf"\frac{{\sum_{{{window}}}\max(-\Delta V_t,0)}}{{\sum_{{{window}}}|\Delta V_t|}}", "VSUMD": rf"\frac{{\sum_{{{window}}}\max(\Delta V_t,0)-\sum_{{{window}}}\max(-\Delta V_t,0)}}{{\sum_{{{window}}}|\Delta V_t|}}",
        "CORR": rf"\operatorname{{Corr}}_{{{window}}}\!\left(C_t,\ln(V_t+1)\right)",
        "CORD": rf"\operatorname{{Corr}}_{{{window}}}\!\left(\frac{{C_t}}{{C_{{t-1}}}},\ln\!\left(\frac{{V_t}}{{V_{{t-1}}}}+1\right)\right)",
        "STD": rf"\frac{{\operatorname{{Std}}_{{{window}}}(C_t)}}{{C_t}}", "VSTD": rf"\frac{{\operatorname{{Std}}_{{{window}}}(V_t)}}{{V_t}}",
        "WVMA": rf"\frac{{\operatorname{{Std}}_{{{window}}}(|r_t|V_t)}}{{\operatorname{{Mean}}_{{{window}}}(|r_t|V_t)}}",
    }
    if operation not in formulas:
        raise ValueError(f"因子 {factor_name} 缺少可核验的数学公式")
    return formulas[operation]


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
    weekly = values.groupby(values.index.to_period("W-FRI")).mean()
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
    failure_groups = [part for _, part in failures.groupby(group_ids) if bool(part.iloc[0])]
    longest = max(failure_groups, key=len) if failure_groups else pd.Series(dtype=bool)
    interval = (
        f"{longest.index[0]} 至 {longest.index[-1]}" if not longest.empty else "无"
    )
    return {
        "basis": "经济方向" if direction in {"+", "-"} else "全时段IC均值符号",
        "sign": "+" if reference > 0 else "-",
        "week_count": int(len(weekly)),
        "win_rate": float((oriented > 0).mean()),
        "autocorrelation": float(weekly.autocorr(lag=1)) if len(weekly) >= 3 else np.nan,
        "max_failure_streak": int(len(longest)),
        "failure_interval": interval,
    }


def decay_statistics(evaluations: dict[int, FactorEvalResult]) -> dict[str, Any]:
    """按5/10/15/20/40/60日IC强度计算峰值和持久跌破50%的半衰期。"""
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
    half_life: str | int = ">60"
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
    column = "ZZ500" if "ZZ500" in runner.benchmark.columns else runner.benchmark.columns[0]
    series = runner.benchmark[column]
    return series.reindex(dates) if dates is not None else series


def _category(factor_name: str, config: dict[str, Any]) -> str:
    params = config.get("params", {})
    rules_path = Path(__file__).resolve().parent / "config/factor_categories_5947.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8"))["categories"]
    operation = str(params.get("operation", "")).upper()
    matches = [
        rule["name"] for rule in rules
        if factor_name in rule.get("factors", []) or operation in rule.get("operations", [])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"因子 {factor_name} 必须且只能匹配一个因子类别，实际={matches}；"
            "请先重新生成 config/factor_categories_5947.json"
        )
    return matches[0]


def _formula(factor_name: str, config: dict[str, Any]) -> str:
    return factor_formula(factor_name, config)


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
    for version in FACTOR_VERSIONS:
        evaluation = evaluations[version][5]
        summary = evaluation.summary.iloc[0]
        stats = weekly_ic_statistics(evaluation.full_ic_series, direction, float(summary["IC_mean"]))
        weekly_stats[version] = stats
        skewness = pd.to_numeric(evaluation.full_ic_series, errors="coerce").dropna().skew()
        rows.append([
            VERSION_NAMES[version], number(summary["IC_mean"]), number(summary["IC_std"]),
            number(summary["ICIR"]), number(summary["t_stat"]), number(summary["p_value"], 6),
            stats["sign"], stats["week_count"], percent(stats["win_rate"]),
            number(skewness), number(stats["autocorrelation"]),
            stats["max_failure_streak"], stats["failure_interval"],
            percent(evaluation.turnover.mean()),
            percent(summary["top_20%_cumulative_return"]),
            percent(summary["bottom_20%_cumulative_return"]),
            percent(summary["top_20%_max_drawdown"]),
            percent(summary["bottom_20%_max_drawdown"]),
        ])
    return rows, weekly_stats


def _decay_rows(evaluations: dict[str, dict[int, FactorEvalResult]]) -> tuple[list[list[str]], list[list[str]]]:
    decay = {version: decay_statistics(evaluations[version]) for version in FACTOR_VERSIONS}
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
        VERSION_NAMES[version], f"{decay[version]['peak_horizon']}日",
        number(decay[version]["peak_strength"]), decay[version]["half_life"],
    ] for version in FACTOR_VERSIONS]
    return rows, conclusions


def _group_rows(
    evaluations: dict[str, dict[int, FactorEvalResult]],
    version: str,
) -> list[list[str]]:
    rows = []
    evaluation = evaluations[version][5]
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
    summary = evaluations[version][5].summary.iloc[0]
    for pct in TAIL_PCTS:
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
    rows = {version: [] for version in FACTOR_VERSIONS}
    prices = _benchmark_prices(runner)
    baseline = {}
    for lag in range(4):
        for version in FACTOR_VERSIONS:
            if lag == 0:
                evaluation = runner.evaluations[version][5]
            else:
                signals = signal_versions[version].shift(lag)
                evaluation = evaluate_factor(
                    signals, runner.data, forward_period=5, n_groups=runner.n_groups,
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
                percent(summary["top_20%_cumulative_return"]),
                percent(summary["top_20%_excess_return"]),
                percent(summary["bottom_20%_cumulative_return"]),
                percent(summary["bottom_20%_excess_return"]),
            ])
    return rows


def _size_rows(
    runner: Any,
    signal_versions: dict[str, pd.DataFrame],
) -> dict[str, list[list[str]]]:
    if "market_cap" not in runner.data.columns:
        return {
            version: [["~", "~", "~", "~", "~", "缺少market_cap"]]
            for version in FACTOR_VERSIONS
        }
    cap = runner.data["market_cap"].unstack().reindex_like(signal_versions["raw"])
    ranks = cap.rank(axis=1, pct=True, method="first")
    masks = {
        "中证500内部小盘组": ranks <= 1 / 3,
        "中证500内部中盘组": (ranks > 1 / 3) & (ranks <= 2 / 3),
        "中证500内部大盘组": ranks > 2 / 3,
    }
    rows = {version: [] for version in FACTOR_VERSIONS}
    prices = _benchmark_prices(runner)
    for bucket, mask in masks.items():
        for version in FACTOR_VERSIONS:
            evaluation = evaluate_factor(
                signal_versions[version].where(mask), runner.data,
                forward_period=5, n_groups=runner.n_groups, ic_method=runner.ic_method,
                price_col=runner.eval_price_col, benchmark_prices=prices,
                portfolio_only=True,
            )
            summary = evaluation.summary.iloc[0]
            rows[version].append([
                bucket, number(summary["IC_mean"]),
                percent(summary["top_20%_cumulative_return"]),
                percent(summary["top_20%_excess_return"]),
                percent(summary["bottom_20%_cumulative_return"]),
                percent(summary["bottom_20%_excess_return"]),
            ])
    return rows


def _industry_rows(
    runner: Any,
    signal_versions: dict[str, pd.DataFrame],
    direction: str,
) -> dict[str, list[list[str]]]:
    frame = runner.industry_daily_ic.copy()
    if frame.empty:
        return {version: [] for version in FACTOR_VERSIONS}
    forward_returns = calc_forward_returns(runner.data, 5, price_col=runner.eval_price_col)
    industry = runner.data["industry_SW_1"].astype("string")
    trading_dates = pd.DatetimeIndex(runner.data.index.get_level_values("date").unique()).sort_values()
    sampled_dates = trading_dates[::5]
    benchmark = runner.evaluations["raw"][5].sampled_benchmark_returns
    rows = {version: [] for version in FACTOR_VERSIONS}
    global_sign = {
        version: float(runner.evaluations[version][5].summary.iloc[0]["IC_mean"])
        for version in FACTOR_VERSIONS
    }
    factors = {}
    for version in FACTOR_VERSIONS:
        factor = signal_versions[version].stack().rename("factor")
        factor.index.names = ["date", "symbol"]
        factors[version] = factor
    for industry_name in sorted(frame["industry_SW_1"].dropna().unique()):
        membership = industry.eq(industry_name)
        selected_index = membership[membership].index
        for version in FACTOR_VERSIONS:
            daily_ic = frame.loc[frame["industry_SW_1"].eq(industry_name)].set_index("signal_date")[f"IC_{version}"]
            daily_ic.index = pd.to_datetime(daily_ic.index)
            stats = weekly_ic_statistics(daily_ic, direction, global_sign[version])
            factor = factors[version].reindex(selected_index).dropna()
            factor = factor[factor.index.get_level_values("date").isin(sampled_dates)]
            returns = forward_returns.reindex(factor.index)
            tails = calc_tail_group_returns(factor, returns, fractions=(0.20,))
            top = tails.get("top_20%_return", pd.Series(dtype=float))
            bottom = tails.get("bottom_20%_return", pd.Series(dtype=float))
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
                value = f"{value} · {MARKET_STATE_NAMES.get(value, '~')}"
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
    performance_dates = runner.evaluations["raw"][5].full_ic_series.index
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
        ["计算公式", f"`${_formula(runner.factor_name, factor_config)}$`"],
        ["参数", f"`{json.dumps(params, ensure_ascii=False)}`"],
        ["股票池", "中证500历史成分股"],
        ["因子数据区间", f"{data_dates.min().date()} 至 {data_dates.max().date()}"],
        ["5日收益有效区间", f"{performance_dates.min().date()} 至 {performance_dates.max().date()}"],
        ["调仓频率", "5个交易日"],
        ["收益窗口", "T+1买入、T+6卖出"],
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
            ["因子版本", "IC均值", "IC标准差", "ICIR", "t-stat", "p-value", "胜率基准符号", "自然周数", "自然周IC胜率", "IC偏度", "IC滞后1期自相关系数", "最大连续失效周数", "最长失效区间", "5日单边换手率", "Top20区间绝对收益", "Bottom20区间绝对收益", "Top20绝对最大回撤", "Bottom20绝对最大回撤"],
            core_rows,
        ),
        "\n## 三、细节表现\n",
        "\n### IC衰减\n",
        markdown_table(["持有期", "原始IC均值", "原始ICIR", "原始p值", "中性化IC均值", "中性化ICIR", "中性化p值"], decay_rows),
        "\n#### 衰减结论\n",
        markdown_table(["因子版本", "IC峰值持有期", "峰值IC强度", "IC半衰期（交易日）"], decay_conclusions),
    ]

    sections.append("\n### 五分组表现\n")
    for version in FACTOR_VERSIONS:
        sections.extend([
            f"\n#### {VERSION_NAMES[version]}\n",
            markdown_table(
                ["分组", "区间绝对收益", "区间超额收益", "绝对最大回撤"],
                _group_rows(runner.evaluations, version),
            ),
        ])

    sections.append("\n### Top/Bottom阈值表现\n")
    for version in FACTOR_VERSIONS:
        sections.extend([
            f"\n#### {VERSION_NAMES[version]}\n",
            markdown_table(
                ["组合", "区间绝对收益", "区间超额收益", "绝对最大回撤"],
                _tail_rows(runner.evaluations, version),
            ),
        ])

    sections.append("\n### 滞后检验\n")
    for version in FACTOR_VERSIONS:
        sections.extend([
            f"\n#### {VERSION_NAMES[version]}\n",
            markdown_table(
                ["滞后交易日", "IC均值", "IC强度保留率", "Top20区间绝对收益", "Top20区间超额收益", "Bottom20区间绝对收益", "Bottom20区间超额收益"],
                lag_rows[version],
            ),
        ])

    sections.append("\n## 四、市值表现\n")
    for version in FACTOR_VERSIONS:
        sections.extend([
            f"\n### {VERSION_NAMES[version]}\n",
            markdown_table(
                ["市值层", "IC均值", "Top20区间绝对收益", "Top20区间超额收益", "Bottom20区间绝对收益", "Bottom20区间超额收益"],
                size_rows[version],
            ),
        ])

    sections.append("\n## 五、行业表现\n")
    for version in FACTOR_VERSIONS:
        sections.extend([
            f"\n### {VERSION_NAMES[version]}\n",
            markdown_table(
                ["行业", "IC均值", "自然周IC胜率", "Top20区间绝对收益", "Top20区间超额收益", "Bottom20区间绝对收益", "Bottom20区间超额收益"],
                industry_rows[version],
            ),
        ])

    sections.append("\n## 六、市场状态表现\n")
    for version in FACTOR_VERSIONS:
        for era in ("2014-2019", "2020-2026"):
            era_frame = runner.market_state_summary.loc[
                runner.market_state_summary["era"].eq(era)
            ]
            sections.extend([
                f"\n### {era} · {VERSION_NAMES[version]}\n",
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
    for version in FACTOR_VERSIONS:
        sections.extend([
            f"\n### {VERSION_NAMES[version]}\n",
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
