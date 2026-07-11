"""Run configurable Haitong multi-factor scheme factors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import os
from pathlib import Path
from typing import Any

from aggregate_factor_outputs import aggregate_factor_outputs
from run import Runner


INDEX_RUNS = {
    "hs300": {
        "data_dir": Path("cache/hs300_csv"),
        "constituents": "000300.SH",
    },
    "zz500": {
        "data_dir": Path("cache/zz500_csv"),
        "constituents": "000905.SH",
    },
    "zz1000": {
        "data_dir": Path("cache/zz1000_csv"),
        "constituents": "000852.SH",
    },
}


def load_factor_configs(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configured multi-factor scheme factors on index universes."
    )
    parser.add_argument(
        "--config",
        default="config/factor_configs_haitong_dim_reduction.json",
        help="Scheme factor config JSON.",
    )
    parser.add_argument(
        "--single-factor-output-root",
        default="outputs",
        help="Root directory of saved single-factor outputs.",
    )
    parser.add_argument(
        "--factor-wide-root",
        default="factor_results",
        help="Root directory for aggregated wide bottom-factor files.",
    )
    parser.add_argument(
        "--output-root",
        default="scheme_outputs",
        help="Root directory for scheme-factor reports.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes. Defaults to min(6, CPU count, task count). Use 1 for serial.",
    )
    parser.add_argument("--start", default="2010-01-01", help="Backtest start date.")
    parser.add_argument("--end", default=None, help="Backtest end date.")
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=sorted(INDEX_RUNS),
        default=sorted(INDEX_RUNS),
        help="Index universes to run.",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="Scheme factor names to run. Defaults to all factors in config.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional stock symbols for a small sampled run.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Reuse existing wide bottom-factor files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scheme factors whose factor_summary.csv already exists.",
    )
    parser.add_argument(
        "--no-save-factor",
        action="store_true",
        help="Do not save per-symbol scheme factor values.",
    )
    parser.add_argument(
        "--no-periodic-reports",
        action="store_true",
        help="Do not save yearly and monthly reports.",
    )
    return parser.parse_args()


def run_scheme_task(task: dict[str, Any]) -> str:
    """Run one index and one scheme factor."""
    index_name = task["index_name"]
    index_config = INDEX_RUNS[index_name]
    factor_name = task["factor_name"]
    config = task["config"]

    importlib.import_module(config["module"])
    factor_params = dict(config.get("params", {}))
    factor_params.update(config.get("index_params", {}).get(index_name, {}))
    runner = Runner(
        factor_name=factor_name,
        factor_params=factor_params,
        symbols=task["symbols"],
        start=task["start"],
        end=task["end"],
        n_groups=5,
        output_dir=Path(task["output_dir"]),
        data_columns=config.get("columns"),
        data_dir=index_config["data_dir"],
        factor_dir=Path(task["factor_wide_dir"]),
        constituents_path=index_config["data_dir"] / "index_members_rebalance.csv",
        constituents=index_config["constituents"],
    )
    runner.run(
        save_factor=not task["no_save_factor"],
        save_periodic_reports=not task["no_periodic_reports"],
    )
    return f"Saved {factor_name} for {index_name}: {runner.output_dir.resolve()}"


def resolve_workers(requested_workers: int | None, task_count: int) -> int:
    """Resolve parallel worker count."""
    if task_count <= 0:
        return 0
    if requested_workers is not None:
        if requested_workers <= 0:
            raise ValueError("--workers 必须为正整数")
        return min(requested_workers, task_count)
    return min(6, os.cpu_count() or 1, task_count)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    factor_configs = load_factor_configs(config_path)
    selected_factors = args.factors or list(factor_configs)

    missing = [factor for factor in selected_factors if factor not in factor_configs]
    if missing:
        raise ValueError(f"Unknown scheme factors: {', '.join(missing)}")

    single_factor_output_root = Path(args.single_factor_output_root)
    factor_wide_root = Path(args.factor_wide_root)
    output_root = Path(args.output_root)

    tasks: list[dict[str, Any]] = []
    for index_name in args.indices:
        factor_wide_dir = factor_wide_root / f"{index_name}_factor_wide"

        if not args.skip_aggregate:
            aggregate_factor_outputs(
                outputs_dir=single_factor_output_root / index_name,
                output_dir=factor_wide_dir,
                symbols=args.symbols,
            )

        for factor_name in selected_factors:
            config = factor_configs[factor_name]
            output_dir = output_root / index_name / factor_name
            if args.skip_existing and (output_dir / "factor_summary.csv").exists():
                print(f"Skip existing: {output_dir}")
                continue

            tasks.append(
                {
                    "index_name": index_name,
                    "factor_name": factor_name,
                    "config": config,
                    "factor_wide_dir": str(factor_wide_dir),
                    "output_dir": str(output_dir),
                    "symbols": args.symbols,
                    "start": args.start,
                    "end": args.end,
                    "no_save_factor": args.no_save_factor,
                    "no_periodic_reports": args.no_periodic_reports,
                }
            )

    workers = resolve_workers(args.workers, len(tasks))
    if workers == 0:
        print("No scheme factor tasks to run.")
        return

    print(f"Running {len(tasks)} scheme factor tasks with {workers} worker(s).", flush=True)
    if workers == 1:
        for task in tasks:
            print(run_scheme_task(task), flush=True)
        return

    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_scheme_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append((task["index_name"], task["factor_name"], exc))
                print(
                    f"Failed {task['factor_name']} for {task['index_name']}: {exc}",
                    flush=True,
                )

    if failures:
        failed_names = ", ".join(
            f"{index_name}/{factor_name}" for index_name, factor_name, _ in failures
        )
        raise RuntimeError(f"{len(failures)} scheme factor task(s) failed: {failed_names}")


if __name__ == "__main__":
    main()
