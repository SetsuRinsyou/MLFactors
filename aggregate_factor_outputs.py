"""将单因子目录中的因子值聚合为按股票拆分的宽表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUTS_DIR = Path("outputs/zz500")
DEFAULT_WIDE_OUTPUT_DIR = Path("outputs/factor_wide")


def iter_factor_dirs(outputs_dir: Path):
    """遍历包含标准 ``factors`` 子目录的单因子输出。"""
    for factor_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        values_dir = factor_dir / "factors"
        if values_dir.is_dir():
            yield factor_dir.name, values_dir


def discover_symbols(factor_dirs: list[tuple[str, Path]]) -> list[str]:
    symbols = set()
    for _, values_dir in factor_dirs:
        symbols.update(path.stem for path in values_dir.glob("*.csv"))
    return sorted(symbols)


def read_factor_values(factor_name: str, csv_path: Path) -> pd.DataFrame:
    """读取一个股票的原始值和中性化值。"""
    frame = pd.read_csv(csv_path)
    date_column = "signal_date" if "signal_date" in frame.columns else "date"
    expected = [f"{factor_name}_raw", f"{factor_name}_neutralization"]
    missing = [column for column in [date_column, *expected] if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少标准列: {missing}")
    values = frame.set_index(date_column)[expected]
    values.index.name = "date"
    return values


def aggregate_factor_outputs(
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    output_dir: Path = DEFAULT_WIDE_OUTPUT_DIR,
    symbols: list[str] | None = None,
) -> None:
    """把全部单因子CSV横向合并为每只股票一张宽表。"""
    factor_dirs = list(iter_factor_dirs(outputs_dir))
    if not factor_dirs:
        raise FileNotFoundError(f"{outputs_dir} 下没有找到 */factors/*.csv")
    target_symbols = sorted(symbols) if symbols else discover_symbols(factor_dirs)
    if not target_symbols:
        raise FileNotFoundError(f"{outputs_dir} 下没有可聚合的股票CSV")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for symbol in target_symbols:
        frames = []
        for factor_name, values_dir in factor_dirs:
            path = values_dir / f"{symbol}.csv"
            if path.exists():
                frames.append(read_factor_values(factor_name, path))
        if not frames:
            continue
        wide = pd.concat(frames, axis=1).sort_index()
        wide.index.name = "date"
        wide.to_csv(output_dir / f"{symbol}.csv", na_rep="")
        saved += 1
    print(f"已聚合 {saved} 只股票到: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WIDE_OUTPUT_DIR)
    parser.add_argument("--symbols", nargs="+", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate_factor_outputs(args.outputs_dir, args.output_dir, args.symbols)
