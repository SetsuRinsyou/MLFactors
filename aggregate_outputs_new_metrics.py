"""Aggregate factor metrics from outputs_new into per-index period CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INDICES = ("zz500", "zz1000", "hs300")
PERIODS = (1, 5, 10, 21)
METRICS = ("IC_mean", "ICIR", "t_stat", "p_value", "timing_IC_mean")
METRIC_LABELS = {
    "IC_mean": "IC",
    "ICIR": "ICIR",
    "t_stat": "t_stat",
    "p_value": "p_value",
    "timing_IC_mean": "timing_IC_mean",
}
YEAR_WINDOWS = tuple(str(year) for year in range(2021, 2027))
MONTH_WINDOWS = tuple(
    f"{year}-{month:02d}"
    for year in range(2024, 2027)
    for month in range(1, 13)
    if (year < 2026 or month <= 6)
)
SCOPE_WEIGHTS = {
    "全时段": 0.6,
    "2021-2026加和": 0.3,
    "2024-01到2026-06加和": 0.1,
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file, returning an empty frame when it does not exist."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _metric_columns(prefix: str) -> list[str]:
    """Build output column names for one scope."""
    return [f"{prefix}{METRIC_LABELS[metric]}" for metric in METRICS]


def _empty_metric_values() -> dict[str, float]:
    """Return NaN metric values."""
    return {metric: float("nan") for metric in METRICS}


def _full_period_values(factor_dir: Path, period: int) -> dict[str, float]:
    """Load absolute full-period metrics for one factor and one period."""
    summary = _read_csv(factor_dir / "factor_summary.csv")
    if summary.empty or "period" not in summary.columns:
        return _empty_metric_values()

    rows = summary[summary["period"].eq(period)]
    if rows.empty:
        return _empty_metric_values()

    row = rows.iloc[0]
    values = {}
    for metric in METRICS:
        values[metric] = abs(pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0])
    return values


def _window_sum_values(
    factor_dir: Path,
    filename: str,
    period: int,
    windows: tuple[str, ...],
) -> dict[str, float]:
    """Sum absolute metrics across selected yearly or monthly windows."""
    summary = _read_csv(factor_dir / filename)
    if summary.empty or not {"period", "window"}.issubset(summary.columns):
        return _empty_metric_values()

    rows = summary[
        summary["period"].eq(period)
        & summary["window"].astype(str).isin(windows)
    ]
    if rows.empty:
        return _empty_metric_values()

    values = {}
    for metric in METRICS:
        if metric not in rows.columns:
            values[metric] = float("nan")
            continue
        values[metric] = (
            pd.to_numeric(rows[metric], errors="coerce")
            .abs()
            .sum(min_count=1)
        )
    return values


def _weighted_values(
    full_values: dict[str, float],
    yearly_values: dict[str, float],
    monthly_values: dict[str, float],
) -> dict[str, float]:
    """Calculate weighted metrics across the three requested scopes."""
    scopes = {
        "全时段": full_values,
        "2021-2026加和": yearly_values,
        "2024-01到2026-06加和": monthly_values,
    }
    result = {}
    for metric in METRICS:
        weighted_parts = [
            scopes[scope][metric] * weight
            for scope, weight in SCOPE_WEIGHTS.items()
        ]
        result[metric] = sum(weighted_parts)
    return result


def _append_metric_group(
    row: dict[str, object],
    prefix: str,
    values: dict[str, float],
) -> None:
    """Append one metric group to an output row."""
    for metric in METRICS:
        row[f"{prefix}{METRIC_LABELS[metric]}"] = values[metric]


def aggregate_index_period(output_root: Path, index_name: str, period: int) -> pd.DataFrame:
    """Aggregate all factor rows for one index and one period."""
    index_dir = output_root / index_name
    rows: list[dict[str, object]] = []
    if not index_dir.is_dir():
        return pd.DataFrame(columns=["因子名"])

    for factor_dir in sorted(path for path in index_dir.iterdir() if path.is_dir()):
        full_values = _full_period_values(factor_dir, period)
        yearly_values = _window_sum_values(
            factor_dir,
            "yearly_summary.csv",
            period,
            YEAR_WINDOWS,
        )
        monthly_values = _window_sum_values(
            factor_dir,
            "monthly_summary.csv",
            period,
            MONTH_WINDOWS,
        )
        weighted_values = _weighted_values(full_values, yearly_values, monthly_values)

        row: dict[str, object] = {"因子名": factor_dir.name}
        _append_metric_group(row, "全时段的", full_values)
        _append_metric_group(row, "2021-2026加和的", yearly_values)
        _append_metric_group(row, "2024-01到2026-06加和的", monthly_values)
        _append_metric_group(row, "加权后的", weighted_values)
        rows.append(row)

    columns = (
        ["因子名"]
        + _metric_columns("全时段的")
        + _metric_columns("2021-2026加和的")
        + _metric_columns("2024-01到2026-06加和的")
        + _metric_columns("加权后的")
    )
    return pd.DataFrame(rows, columns=columns)


def aggregate_all(output_root: Path, output_dir: Path) -> list[Path]:
    """Write all index-period aggregate CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for index_name in INDICES:
        for period in PERIODS:
            summary = aggregate_index_period(output_root, index_name, period)
            output_path = output_dir / f"{index_name}_period_{period}_metrics_summary.csv"
            summary.to_csv(output_path, index=False, encoding="utf-8-sig")
            written_paths.append(output_path)
    return written_paths


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="汇总 outputs_new 下各指数、各因子的核心指标",
    )
    parser.add_argument(
        "--output-root",
        default="outputs_new",
        help="待汇总的输出根目录，默认 outputs_new",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="总 CSV 输出目录，默认 <output-root>/aggregate_csv",
    )
    return parser.parse_args()


def main() -> None:
    """Script entrypoint."""
    args = parse_args()
    output_root = Path(args.output_root)
    output_dir = Path(args.summary_dir) if args.summary_dir else output_root / "aggregate_csv"
    written_paths = aggregate_all(output_root, output_dir)
    print(f"已输出 {len(written_paths)} 个总 CSV 到: {output_dir.resolve()}")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
