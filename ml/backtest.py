"""按现有中证500口径回测按股票保存的机器学习组合因子。"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import DataLoader, load_symbol_aliases
from factors_eval import calc_forward_returns, eval as evaluate_factor


FORWARD_PERIOD = 5
TOP_FRACTION = 0.20
INDEX_CODE = "000905.SH"
SELECTION_EFFICIENCY = "top_20%_selection_rank_efficiency"
CAPTURE_EFFICIENCY = "top_20%_return_capture_rank_efficiency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", help="方法名，同时作为默认目录名和因子列名")
    parser.add_argument("--factor-column", default=None)
    parser.add_argument("--factor-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None, help="默认取因子值最后日期")
    return parser.parse_args()


def default_security_status_path() -> Path:
    files = sorted(PROJECT_ROOT.glob("cache/tushare_security_status*.csv"))
    if not files:
        raise FileNotFoundError("cache下未找到证券停牌/ST状态文件")
    return files[-1]


def load_factor_values(
    factor_dir: Path,
    factor_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> pd.Series:
    files = sorted(factor_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到ML因子文件: {factor_dir}")
    aliases = load_symbol_aliases(PROJECT_ROOT / "cache" / "zz500_csv")
    frames: list[pd.DataFrame] = []
    for number, path in enumerate(files, start=1):
        header = pd.read_csv(path, nrows=0).columns.tolist()
        required = {"signal_date", "available_date", factor_column}
        if not required.issubset(header):
            missing = ", ".join(sorted(required.difference(header)))
            raise ValueError(f"{path} 缺少列: {missing}")
        frame = pd.read_csv(
            path,
            usecols=["signal_date", factor_column],
            parse_dates=["signal_date"],
        ).rename(columns={"signal_date": "date", factor_column: "factor"})
        frame["symbol"] = aliases.get(path.stem, path.stem)
        frames.append(frame)
        if number == 1 or number % 200 == 0 or number == len(files):
            print(f"[factor] {number}/{len(files)} files", flush=True)

    values = pd.concat(frames, ignore_index=True)
    values["factor"] = pd.to_numeric(values["factor"], errors="coerce")
    values = values[
        (values["date"] >= start)
        & (end is None or values["date"].le(end))
    ].dropna(subset=["date", "symbol", "factor"])
    if values.duplicated(["date", "symbol"]).any():
        duplicates = values.loc[
            values.duplicated(["date", "symbol"], keep=False),
            ["date", "symbol"],
        ].head()
        raise ValueError(f"因子值存在重复(date, symbol):\n{duplicates}")
    if values.empty:
        raise ValueError("指定时段内没有有效ML因子值")
    return (
        values.set_index(["date", "symbol"])["factor"]
        .sort_index()
        .rename(factor_column)
    )


def load_market_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data_dir = PROJECT_ROOT / "cache" / "zz500_csv"
    loader = DataLoader(
        data_dir=data_dir,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        columns=["adj_close"],
        constituents_path=data_dir / "constituents_daily.csv",
        constituent_index=INDEX_CODE,
        security_status_path=default_security_status_path(),
    )
    market = loader.load_all()
    if market.empty:
        raise ValueError("中证500回测行情为空")
    return market[["adj_close"]].sort_index()


def calc_rank_efficiencies(
    factor: pd.Series,
    forward_returns: pd.Series,
    top_fraction: float = TOP_FRACTION,
) -> pd.DataFrame:
    """计算因子选股→收益排名、收益最优→因子排名的双向Top排名效率。"""
    combined = pd.concat(
        [factor.rename("factor"), forward_returns.rename("returns")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    records: list[tuple[pd.Timestamp, float, float]] = []
    for current_date, cross_section in combined.groupby(level="date", sort=True):
        cross_section = cross_section.droplevel("date").sort_index()
        count = len(cross_section)
        if count < 2:
            continue
        selected_count = max(1, math.ceil(count * top_fraction))
        ideal_rank_sum = selected_count * (selected_count + 1) / 2.0
        factor_rank = cross_section["factor"].rank(
            method="first", ascending=False
        )
        return_rank = cross_section["returns"].rank(
            method="first", ascending=False
        )
        factor_top = factor_rank.nsmallest(selected_count).index
        return_top = return_rank.nsmallest(selected_count).index
        selection_efficiency = ideal_rank_sum / return_rank.loc[factor_top].sum()
        capture_efficiency = ideal_rank_sum / factor_rank.loc[return_top].sum()
        records.append(
            (current_date, selection_efficiency, capture_efficiency)
        )
    return pd.DataFrame.from_records(
        records,
        columns=["date", SELECTION_EFFICIENCY, CAPTURE_EFFICIENCY],
    ).set_index("date")


def evaluate_window(
    factor: pd.Series,
    market: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    market_dates = market.index.get_level_values("date")
    factor_dates = factor.index.get_level_values("date")
    window_market = market.loc[(market_dates >= start) & (market_dates <= end)]
    window_factor = factor.loc[(factor_dates >= start) & (factor_dates <= end)]
    if window_market.empty or window_factor.empty:
        raise ValueError(f"{start.date()}至{end.date()}没有可回测数据")

    result = evaluate_factor(
        window_factor,
        window_market,
        forward_period=FORWARD_PERIOD,
        n_groups=5,
        ic_method="rank",
        price_col="adj_close",
    )
    forward_returns = calc_forward_returns(
        window_market, FORWARD_PERIOD, price_col="adj_close"
    )
    efficiencies = calc_rank_efficiencies(window_factor, forward_returns)
    result.daily_metrics = (
        result.daily_metrics.join(efficiencies, how="outer").sort_index()
    )
    result.summary[SELECTION_EFFICIENCY] = round(
        efficiencies[SELECTION_EFFICIENCY].mean(), 4
    )
    result.summary[CAPTURE_EFFICIENCY] = round(
        efficiencies[CAPTURE_EFFICIENCY].mean(), 4
    )
    return result


def atomic_to_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, **kwargs)
    os.replace(temporary, path)


def run_backtest(
    method: str,
    factor_column: str,
    factor_dir: Path,
    output_dir: Path,
    start: pd.Timestamp,
    configured_end: pd.Timestamp | None,
) -> None:
    factor = load_factor_values(factor_dir, factor_column, start, configured_end)
    signal_end = pd.Timestamp(factor.index.get_level_values("date").max())
    end = min(signal_end, configured_end) if configured_end is not None else signal_end
    market = load_market_data(start, end)
    common_index = factor.index.intersection(market.index)
    factor = factor.loc[common_index].sort_index()
    if factor.empty:
        raise ValueError("ML因子与中证500有效行情没有交集")
    print(
        f"[data] method={method}, factor_rows={len(factor):,}, "
        f"stocks={factor.index.get_level_values('symbol').nunique()}, "
        f"range={start.date()}..{end.date()}",
        flush=True,
    )

    overall = evaluate_window(factor, market, start, end)
    atomic_to_csv(
        overall.summary,
        output_dir / "factor_summary.csv",
    )
    atomic_to_csv(
        overall.daily_metrics,
        output_dir / "daily_factor_metrics.csv",
        na_rep="",
    )

    yearly_rows: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        try:
            result = evaluate_window(factor, market, year_start, year_end)
        except ValueError:
            continue
        row = result.summary.reset_index()
        row.insert(0, "year", year)
        yearly_rows.append(row)
        print(
            f"[year] {year} "
            f"IC={row.loc[0, 'IC_mean']:.4f} "
            f"selection_eff={row.loc[0, SELECTION_EFFICIENCY]:.4f} "
            f"capture_eff={row.loc[0, CAPTURE_EFFICIENCY]:.4f}",
            flush=True,
        )
    if not yearly_rows:
        raise ValueError("没有生成任何年度回测结果")
    yearly_summary = pd.concat(yearly_rows, ignore_index=True)
    atomic_to_csv(
        yearly_summary,
        output_dir / "yearly_summary.csv",
        index=False,
    )
    print(
        f"[done] overall={output_dir / 'factor_summary.csv'}, "
        f"yearly={output_dir / 'yearly_summary.csv'}, "
        f"daily={output_dir / 'daily_factor_metrics.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    args = parse_args()
    method_dir = ML_DIR / args.method
    run_backtest(
        method=args.method,
        factor_column=args.factor_column or args.method,
        factor_dir=(args.factor_dir or method_dir / "output" / "factors").resolve(),
        output_dir=(args.output_dir or method_dir / "output" / "backtest").resolve(),
        start=pd.Timestamp(args.start),
        configured_end=pd.Timestamp(args.end) if args.end else None,
    )
