"""统计 period=5 回测中 Top 20% 组合的实际选股及科技行业占比。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dataloader import DataLoader
from run_single_factors_parallel import (
    DEFAULT_SECURITY_STATUS_PATH,
    DEFAULT_INDEX_ORDER,
    INDEX_RUNS,
)


DEFAULT_START = "2025-04-01"
DEFAULT_END = "2026-06-30"
DEFAULT_WIDE_ROOT = Path("factor_results")
DEFAULT_CACHE_ROOT = Path("cache")
DEFAULT_OUTPUT = Path("outputs") / "top20_selection_202504_202606.csv"
REBALANCE_PERIOD = 5
QUANTILES = 5


def load_backtest_universe(
    index_name: str,
    cache_root: Path,
    end: str,
    security_status_path: Path | None,
) -> pd.DataFrame:
    """加载与原回测完全相同的成分股、ST 和停牌过滤后的股票池。

    返回 ``date, symbol, stock_name, industry_1``。交易日序列也来自该股票池，因而
    ``[::5]`` 的调仓日期与 ``factors_eval.eval`` 保持一致。
    """
    if index_name not in INDEX_RUNS:
        raise ValueError(f"不支持的指数: {index_name}")
    index_config = INDEX_RUNS[index_name]
    data_dir = cache_root / f"{index_name}_csv"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"未找到缓存目录: {data_dir}")

    loader = DataLoader(
        data_dir=data_dir,
        start=index_config["start"],
        end=end,
        columns=["stock_name", "industry_1"],
        constituents_path=data_dir / "index_members_rebalance.csv",
        constituent_index=index_config["constituents"],
        security_status_path=security_status_path,
    )
    data = loader.load_all()
    if data.empty:
        raise ValueError(f"{index_name} 回测股票池为空")

    universe = data.reset_index()[["date", "symbol", "stock_name", "industry_1"]]
    universe["date"] = pd.to_datetime(universe["date"])
    # 必须保留指数回测起点以来的全部交易日。原回测先在全时段执行 [::5]，
    # 再进入任意观察窗口；若此处从观察起点重新计数，会使调仓相位发生偏移。
    return universe


def rebalance_dates(universe: pd.DataFrame, start: str, end: str) -> pd.DatetimeIndex:
    """按全样本交易日每 5 日取一次调仓日，再截取目标观察区间。"""
    all_dates = pd.DatetimeIndex(universe["date"].unique()).sort_values()
    dates = all_dates[::REBALANCE_PERIOD]
    return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def load_factor_values_at_rebalance_dates(
    wide_dir: Path,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[str]]:
    """从每股票宽表读取调仓日期上的全部因子值。"""
    csv_paths = sorted(wide_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"未找到因子宽表: {wide_dir}")

    first_columns = pd.read_csv(csv_paths[0], nrows=0).columns.tolist()
    factor_names = [column for column in first_columns if column != "date"]
    if not factor_names:
        raise ValueError(f"{csv_paths[0]} 未包含因子列")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, parse_dates=["date"])
        if "date" not in frame.columns:
            raise ValueError(f"{csv_path} 缺少 date 列")
        missing = set(factor_names).difference(frame.columns)
        if missing:
            raise ValueError(
                f"{csv_path} 的因子列与 {csv_paths[0].name} 不一致，缺少: "
                f"{', '.join(sorted(missing))}"
            )
        frame = frame.loc[frame["date"].isin(dates), ["date", *factor_names]]
        if frame.empty:
            continue
        frame["symbol"] = csv_path.stem
        frames.append(frame)

    if not frames:
        raise ValueError(f"{wide_dir} 在指定调仓日没有因子值")
    return pd.concat(frames, ignore_index=True), factor_names


def select_top_quintile(
    candidates: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    """按回测的 rank(method='first') + qcut 口径选取每期最高 20%。"""
    selected: list[pd.DataFrame] = []
    for _, cross_section in candidates.groupby("date", sort=True):
        values = cross_section.dropna(subset=[factor_name]).copy()
        if len(values) < QUANTILES:
            continue
        ranks = values[factor_name].rank(method="first")
        groups = pd.qcut(
            ranks,
            QUANTILES,
            labels=False,
            duplicates="drop",
        )
        selected.append(values.loc[groups == groups.max()])

    if not selected:
        return pd.DataFrame(columns=candidates.columns)
    return pd.concat(selected, ignore_index=True)


def summarize_selections(
    selections: pd.DataFrame,
    technology_industries: set[str],
) -> pd.DataFrame:
    """按指数、因子、股票汇总重复入选次数及科技行业选中次数。"""
    selections = selections.copy()
    selections["is_technology"] = selections["industry_1"].isin(technology_industries)
    selections["date"] = pd.to_datetime(selections["date"])

    stock_rows = []
    for (index_name, factor_name, symbol), rows in selections.groupby(
        ["index_name", "factor_name", "symbol"],
        sort=True,
    ):
        rows = rows.sort_values("date")
        latest_name = rows["stock_name"].dropna()
        latest_industry = rows["industry_1"].dropna()
        latest_industry_value = (
            latest_industry.iloc[-1] if not latest_industry.empty else pd.NA
        )
        stock_rows.append({
            "index_name": index_name,
            "factor_name": factor_name,
            "symbol": symbol,
            "selection_count": int(len(rows)),
            "technology_selection_count": int(rows["is_technology"].sum()),
            "technology_selection_ratio_for_stock": float(rows["is_technology"].mean()),
            "latest_stock_name": latest_name.iloc[-1] if not latest_name.empty else pd.NA,
            "latest_industry_1": latest_industry_value,
            "is_technology_at_latest_selection": latest_industry_value in technology_industries,
            "selected_dates": "|".join(rows["date"].dt.strftime("%Y-%m-%d")),
        })
    details = pd.DataFrame(stock_rows)

    factor_totals = (
        selections.groupby(["index_name", "factor_name"], as_index=False)
        .agg(
            factor_total_selection_count=("symbol", "size"),
            factor_technology_selection_count=("is_technology", "sum"),
        )
    )
    factor_totals["factor_technology_selection_ratio"] = (
        factor_totals["factor_technology_selection_count"]
        / factor_totals["factor_total_selection_count"]
    )
    details = details.merge(factor_totals, on=["index_name", "factor_name"], how="left")

    index_totals = (
        selections.groupby("index_name", as_index=False)
        .agg(
            index_total_selection_count=("symbol", "size"),
            index_technology_selection_count=("is_technology", "sum"),
        )
    )
    index_totals["index_technology_selection_ratio"] = (
        index_totals["index_technology_selection_count"]
        / index_totals["index_total_selection_count"]
    )
    details = details.merge(index_totals, on="index_name", how="left")

    overall_total = int(len(selections))
    overall_technology = int(selections["is_technology"].sum())
    details["overall_total_selection_count"] = overall_total
    details["overall_technology_selection_count"] = overall_technology
    details["overall_technology_selection_ratio"] = (
        overall_technology / overall_total if overall_total else pd.NA
    )

    return details.sort_values(
        ["index_name", "factor_name", "selection_count", "symbol"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def analyze_top20_selections(
    wide_root: Path,
    cache_root: Path,
    indices: list[str],
    start: str,
    end: str,
    technology_industries: set[str],
    output_path: Path,
    security_status_path: Path | None,
) -> pd.DataFrame:
    """生成 period=5 的 Top 20% 选股统计，并保存为一个 CSV 文件。"""
    selected_frames: list[pd.DataFrame] = []
    for index_name in indices:
        universe = load_backtest_universe(
            index_name=index_name,
            cache_root=cache_root,
            end=end,
            security_status_path=security_status_path,
        )
        dates = rebalance_dates(universe, start=start, end=end)
        if dates.empty:
            raise ValueError(f"{index_name} 在 {start} 至 {end} 没有调仓日")

        wide_dir = wide_root / f"{index_name}_factor_wide"
        factor_values, factor_names = load_factor_values_at_rebalance_dates(
            wide_dir,
            dates,
        )
        candidates = universe[universe["date"].isin(dates)].merge(
            factor_values,
            on=["date", "symbol"],
            how="left",
            validate="one_to_one",
        )
        for factor_name in factor_names:
            selected = select_top_quintile(candidates, factor_name)
            if selected.empty:
                continue
            selected_frames.append(
                selected[["date", "symbol", "stock_name", "industry_1"]].assign(
                    index_name=index_name,
                    factor_name=factor_name,
                )
            )

    if not selected_frames:
        raise ValueError("没有生成任何 Top 20% 选股记录")
    details = summarize_selections(
        pd.concat(selected_frames, ignore_index=True),
        technology_industries=technology_industries,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_path, index=False, na_rep="")
    return details


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="按 period=5 回测口径统计 Top 20% 实际选股和科技行业占比"
    )
    parser.add_argument(
        "--wide-root",
        default=str(DEFAULT_WIDE_ROOT),
        help="按指数保存每股票因子宽表的根目录，默认 factor_results",
    )
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="指数缓存目录的根目录，默认 cache",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=DEFAULT_INDEX_ORDER,
        default=list(DEFAULT_INDEX_ORDER),
        help="需要统计的指数，默认 zz500 zz1000 hs300",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="观察起始日")
    parser.add_argument("--end", default=DEFAULT_END, help="观察结束日")
    parser.add_argument(
        "--technology-industries",
        nargs="+",
        required=True,
        help="科技行业在 cache 的 industry_1 中对应的精确行业名称，必须显式给出",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="选股明细输出 CSV，默认 outputs/top20_selection_202504_202606.csv",
    )
    parser.add_argument(
        "--security-status-path",
        default=str(DEFAULT_SECURITY_STATUS_PATH),
        help="ST/停牌状态文件；传空字符串可关闭状态过滤",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    security_status_path = (
        Path(args.security_status_path) if args.security_status_path else None
    )
    result = analyze_top20_selections(
        wide_root=Path(args.wide_root),
        cache_root=Path(args.cache_root),
        indices=args.indices,
        start=args.start,
        end=args.end,
        technology_industries=set(args.technology_industries),
        output_path=Path(args.output),
        security_status_path=security_status_path,
    )
    overall_ratio = result["overall_technology_selection_ratio"].iloc[0]
    print(
        f"已保存 {len(result)} 条按股票汇总的 Top 20% 选股记录: "
        f"{Path(args.output).resolve()}"
    )
    print(f"科技行业选中次数占比: {overall_ratio:.4%}")
