"""Aggregate saved factor values into one wide CSV per stock."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


OUTPUTS_DIR = Path("outputs")
WIDE_OUTPUT_DIR = OUTPUTS_DIR / "factor_wide"


def iter_factor_files(outputs_dir: Path):
    for factor_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        factor_values_dir = factor_dir / "factor"
        if not factor_values_dir.is_dir():
            continue
        for csv_path in sorted(factor_values_dir.glob("*.csv")):
            yield factor_dir.name, csv_path.stem, csv_path


def read_factor_series(factor_name: str, csv_path: Path) -> pd.Series:
    data = pd.read_csv(csv_path)
    if "date" not in data.columns:
        raise ValueError(f"{csv_path} 缺少 date 列")

    value_columns = [column for column in data.columns if column != "date"]
    if len(value_columns) != 1:
        raise ValueError(f"{csv_path} 必须只有一个因子值列")

    series = data.set_index("date")[value_columns[0]]
    series.name = factor_name
    return series


def aggregate_factor_outputs(
    outputs_dir: Path = OUTPUTS_DIR,
    output_dir: Path = WIDE_OUTPUT_DIR,
) -> None:
    symbol_series = defaultdict(list)
    for factor_name, symbol, csv_path in iter_factor_files(outputs_dir):
        symbol_series[symbol].append(read_factor_series(factor_name, csv_path))

    if not symbol_series:
        raise FileNotFoundError(f"{outputs_dir} 下没有找到 */factor/*.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    for symbol, series_list in sorted(symbol_series.items()):
        wide = pd.concat(series_list, axis=1).sort_index()
        wide.index.name = "date"
        wide.to_csv(output_dir / f"{symbol}.csv", na_rep="")

    print(f"已聚合 {len(symbol_series)} 只股票到: {output_dir.resolve()}")


if __name__ == "__main__":
    aggregate_factor_outputs()
