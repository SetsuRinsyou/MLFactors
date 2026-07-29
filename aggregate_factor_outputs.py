"""Aggregate saved factor values into one wide CSV per stock."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUTS_DIR = Path("outputs/zz500/2026-07-22_wo_eval")
WIDE_OUTPUT_DIR = OUTPUTS_DIR / "factor_wide"
ADD_MARKET = True
MARKET_DIR = Path("cache/zz500_csv")


def iter_factor_dirs(outputs_dir: Path):
    for factor_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        factor_values_dir = factor_dir / "factors"
        if not factor_values_dir.is_dir():
            factor_values_dir = factor_dir / "factor"
        if not factor_values_dir.is_dir():
            continue
        yield factor_dir.name, factor_values_dir


def iter_factor_dirs(outputs_dir: Path):
    for factor_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        factor_values_dir = factor_dir / "factors"
        if not factor_values_dir.is_dir():
            factor_values_dir = factor_dir / "factor"
        if factor_values_dir.is_dir():
            yield factor_dir.name, factor_values_dir


def discover_symbols(factor_dirs: list[tuple[str, Path]]) -> list[str]:
    symbols = set()
    for _, factor_values_dir in factor_dirs:
        symbols.update(csv_path.stem for csv_path in factor_values_dir.glob("*.csv"))
    return sorted(symbols)


def read_factor_series(factor_name: str, csv_path: Path) -> pd.Series:
    data = pd.read_csv(csv_path)
    date_column = "signal_date" if "signal_date" in data.columns else "date"
    if date_column not in data.columns:
        raise ValueError(f"{csv_path} 缺少 signal_date/date 列")

    metadata_columns = {
        "date",
        "signal_date",
        "available_date",
        "days_since_latest_report_publish_date",
    }
    value_columns = [
        column for column in data.columns if column not in metadata_columns
    ]
    if len(value_columns) != 1:
        raise ValueError(f"{csv_path} 必须只有一个因子值列")

    series = data.set_index(date_column)[value_columns[0]]
    series.index.name = "date"
    series.name = factor_name
    return series


def read_market_series(csv_path: Path) -> pd.Series:
    columns = pd.read_csv(csv_path, nrows=0).columns
    date_column = "date" if "date" in columns else "trade_date"
    if date_column not in columns:
        raise ValueError(f"{csv_path} 缺少 date 或 trade_date 列")
    if "adj_close" not in columns:
        raise ValueError(f"{csv_path} 缺少 adj_close 列")

    data = pd.read_csv(
        csv_path,
        usecols=[date_column, "adj_close"],
        dtype={date_column: str},
    )
    series = data.set_index(date_column)["adj_close"]
    series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
    return series


def aggregate_factor_outputs(
    outputs_dir: Path = OUTPUTS_DIR,
    output_dir: Path = WIDE_OUTPUT_DIR,
    symbols: list[str] | None = None,
) -> None:
    factor_dirs = list(iter_factor_dirs(outputs_dir))
    if not factor_dirs:
        raise FileNotFoundError(f"{outputs_dir} 下没有找到 */factors/*.csv 或 */factor/*.csv")

    target_symbols = sorted(symbols) if symbols is not None else discover_symbols(factor_dirs)
    if not target_symbols:
        raise FileNotFoundError(f"{outputs_dir} 下没有找到可聚合的股票 CSV")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for symbol in target_symbols:
        series_list = []
        for factor_name, factor_values_dir in factor_dirs:
            csv_path = factor_values_dir / f"{symbol}.csv"
            if csv_path.exists():
                series_list.append(read_factor_series(factor_name, csv_path))
        if not series_list:
            continue
        wide = pd.concat(series_list, axis=1)
        output_path = output_dir / f"{symbol}.csv"
        if output_path.exists():
            existing = pd.read_csv(output_path)
            if "date" not in existing.columns:
                raise ValueError(f"{output_path} 缺少 date 列，无法增量聚合")
            existing = existing.set_index("date")

            # Preserve previously aggregated factors, replacing only columns
            # produced by the current input directory.
            existing = existing.drop(columns=wide.columns, errors="ignore")
            wide = pd.concat([existing, wide], axis=1)

        wide = wide.sort_index()
        wide.index.name = "date"
        wide.to_csv(output_path, na_rep="")
        saved_count += 1

    print(f"已聚合 {saved_count} 只股票到: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="聚合单因子输出为按股票拆分的宽表")
    parser.add_argument(
        "--outputs-dir",
        default=str(OUTPUTS_DIR),
        help="单因子输出目录，例如 outputs_new/zz500",
    )
    parser.add_argument(
        "--output-dir",
        default=str(WIDE_OUTPUT_DIR),
        help="宽表输出目录，例如 factor_results/zz500_factor_wide",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="只聚合指定股票代码；默认聚合全部股票",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate_factor_outputs(
        outputs_dir=Path(args.outputs_dir),
        output_dir=Path(args.output_dir),
        symbols=args.symbols,
    )
