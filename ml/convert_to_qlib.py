"""将按股票拆分的232因子宽表转换为Qlib可直接加载的Parquet数据集。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import qlib
from qlib.data.dataset.loader import StaticDataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "factor_results_232" / "zz500_factor_wide"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "zz500_csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "zz500" / "dataset.parquet"
HORIZON = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coverage-threshold", type=float, default=0.30)
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def load_market_data(path: Path) -> pd.DataFrame:
    market = pd.read_csv(
        path,
        usecols=["trade_date", "adj_close", "zz500_close"],
        encoding="utf-8-sig",
    ).rename(columns={"trade_date": "datetime"})
    market["datetime"] = pd.to_datetime(market["datetime"], errors="raise")
    market = market.drop_duplicates("datetime", keep="last").sort_values("datetime")
    for column in ("adj_close", "zz500_close"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
        market[column] = market[column].replace([np.inf, -np.inf], np.nan)
        market[f"{column}_fwd_{HORIZON}d"] = (
            market[column].shift(-(HORIZON + 1)) / market[column].shift(-1) - 1.0
        )
    return market


def atomic_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def convert(
    source_dir: Path,
    cache_dir: Path,
    output: Path,
    coverage_threshold: float,
) -> dict:
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到因子宽表: {source_dir}")

    header = read_header(files[0])
    if not header or header[0] != "date":
        raise ValueError(f"{files[0]} 首列必须为 date")
    factor_names = header[1:]
    if len(factor_names) != 232:
        raise ValueError(f"预期232个因子，实际发现{len(factor_names)}个")

    output.parent.mkdir(parents=True, exist_ok=True)
    flat_tmp = output.parent / ".dataset_flat.tmp.parquet"
    grouped_tmp = output.parent / ".dataset.tmp.parquet"
    for temporary in (flat_tmp, grouped_tmp):
        temporary.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    missing_cache: list[str] = []
    try:
        for number, factor_path in enumerate(files, start=1):
            if read_header(factor_path) != header:
                raise ValueError(f"因子列与首个文件不一致: {factor_path}")
            cache_path = cache_dir / factor_path.name
            if not cache_path.exists():
                missing_cache.append(factor_path.stem)
                continue

            factors = pd.read_csv(factor_path, encoding="utf-8-sig")
            factors = factors.rename(columns={"date": "datetime"})
            factors["datetime"] = pd.to_datetime(factors["datetime"], errors="raise")
            if factors["datetime"].duplicated().any():
                raise ValueError(f"存在重复日期: {factor_path}")
            factors[factor_names] = factors[factor_names].apply(
                pd.to_numeric, errors="coerce"
            ).astype("float32")
            factors[factor_names] = factors[factor_names].replace(
                [np.inf, -np.inf], np.nan
            )

            market = load_market_data(cache_path)
            frame = factors.merge(market, on="datetime", how="left", validate="one_to_one")
            frame.insert(1, "instrument", factor_path.stem)
            frame["fwd_return_5d"] = frame.pop(f"adj_close_fwd_{HORIZON}d")
            benchmark = frame.pop(f"zz500_close_fwd_{HORIZON}d")
            frame["fwd_excess_return_5d"] = frame["fwd_return_5d"] - benchmark
            numeric_tail = [
                "adj_close",
                "zz500_close",
                "fwd_return_5d",
                "fwd_excess_return_5d",
            ]
            frame[numeric_tail] = frame[numeric_tail].astype("float32")
            frame = frame[
                ["datetime", "instrument", *factor_names, *numeric_tail]
            ].sort_values("datetime")

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(flat_tmp, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(frame)
            if number == 1 or number % 100 == 0 or number == len(files):
                print(
                    f"[convert] {number}/{len(files)} files, {total_rows:,} rows",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    if missing_cache:
        flat_tmp.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"{len(missing_cache)}只股票缺少行情缓存，例如: {missing_cache[:10]}"
        )
    if total_rows == 0:
        raise ValueError("转换结果为空")

    flat = pd.read_parquet(flat_tmp)
    flat = flat.sort_values(["datetime", "instrument"], kind="mergesort")
    if flat.duplicated(["datetime", "instrument"]).any():
        raise ValueError("转换结果存在重复的(datetime, instrument)")

    nonnull_cells = flat[factor_names].notna().sum(axis=1)
    coverage = (
        pd.DataFrame(
            {
                "datetime": flat["datetime"],
                "nonnull_cells": nonnull_cells,
            }
        )
        .groupby("datetime", sort=True)
        .agg(nonnull_cells=("nonnull_cells", "sum"), stocks=("nonnull_cells", "size"))
    )
    coverage["coverage"] = coverage["nonnull_cells"] / (
        coverage["stocks"] * len(factor_names)
    )
    eligible_dates = coverage.index[coverage["coverage"] > coverage_threshold]
    if eligible_dates.empty:
        raise ValueError(f"没有日期的因子非空率超过{coverage_threshold:.0%}")
    first_eligible_date = pd.Timestamp(eligible_dates[0])

    grouped = pd.concat(
        {
            "feature": flat[factor_names],
            "label": flat[["fwd_return_5d", "fwd_excess_return_5d"]],
            "meta": flat[["adj_close", "zz500_close"]],
        },
        axis=1,
    )
    grouped.index = pd.MultiIndex.from_frame(
        flat[["datetime", "instrument"]], names=["datetime", "instrument"]
    )
    grouped.columns.names = ["group", "field"]
    grouped.to_parquet(grouped_tmp, compression="zstd", index=True)
    atomic_replace(grouped_tmp, output)
    flat_tmp.unlink(missing_ok=True)

    date_min = pd.Timestamp(flat["datetime"].min())
    date_max = pd.Timestamp(flat["datetime"].max())
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "qlib_version": qlib.__version__,
        "source_dir": str(source_dir.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "dataset": str(output.resolve()),
        "file_count": len(files),
        "row_count": int(total_rows),
        "factor_count": len(factor_names),
        "factor_names": factor_names,
        "date_min": date_min.date().isoformat(),
        "date_max": date_max.date().isoformat(),
        "coverage_threshold": coverage_threshold,
        "first_eligible_date": first_eligible_date.date().isoformat(),
        "first_eligible_coverage": float(coverage.loc[first_eligible_date, "coverage"]),
        "label": "adj_close[t+6] / adj_close[t+1] - 1",
        "raw_feature_nan_preserved": True,
    }
    manifest_path = output.with_name("manifest.json")
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    atomic_replace(manifest_tmp, manifest_path)

    del flat, grouped, nonnull_cells
    gc.collect()
    loaded = StaticDataLoader(str(output)).load(
        start_time=first_eligible_date,
        end_time=first_eligible_date,
    )
    if loaded.empty or list(loaded["feature"].columns) != factor_names:
        raise RuntimeError("Qlib StaticDataLoader验证失败")
    print(
        f"[done] {output} | rows={total_rows:,} | factors={len(factor_names)} | "
        f"first_train_date={first_eligible_date.date()}",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    args = parse_args()
    convert(args.source_dir, args.cache_dir, args.output, args.coverage_threshold)
