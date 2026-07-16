"""并行批量回测全部单因子。

默认在中证500、中证1000、沪深300依次回测 74 个单因子；
每个指数内部按 32 进程并行。多周期 LLT 和 MA 因子固定排除。
完成记录会实时追加到 ``<output_root>/<index>_completed_factors.csv``。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

for thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[thread_env] = "1"

DEFAULT_CONFIG_PATHS = (
    "config/factor_configs_fundamental.json",
    "config/factor_configs_price.json",
    "config/factor_configs_risk.json",
    "config/factor_configs_sector.json",
)
EXCLUDED_FACTOR_MODULES = frozenset({
    "factors.price.multi_horizon_llt_daily",
    "factors.price.multi_horizon_ma_daily",
})
EXPECTED_FACTOR_COUNT = 74
DEFAULT_SECURITY_STATUS_PATH = (
    Path(__file__).resolve().parent
    / "cache"
    / "tushare_security_status_missing_tables_20100104_20260623_20260706.csv"
)

INDEX_RUNS = {
    "zz500": {
        "label": "中证500",
        "data_dir": "cache/zz500_csv",
        "constituents": "000905.SH",
        "start": "2010-01-12",
    },
    "zz1000": {
        "label": "中证1000",
        "data_dir": "cache/zz1000_csv",
        "constituents": "000852.SH",
        "start": "2015-06-30",
    },
    "hs300": {
        "label": "沪深300",
        "data_dir": "cache/hs300_csv",
        "constituents": "000300.SH",
        "start": "2016-01-29",
    },
}

DEFAULT_INDEX_ORDER = ("zz500", "zz1000", "hs300")
PROGRESS_COLUMNS = [
    "factor_name",
    "status",
    "duration_seconds",
    "started_at",
    "finished_at",
    "output_dir",
    "error",
]


def _now() -> str:
    """返回秒级时间戳。"""
    return datetime.now().isoformat(timespec="seconds")


def _load_factor_configs(config_paths: list[Path]) -> dict[str, dict[str, Any]]:
    """合并各类别单因子配置，并拒绝重名因子。"""
    factor_configs: dict[str, dict[str, Any]] = {}
    for config_path in config_paths:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        duplicate_names = sorted(set(config).intersection(factor_configs))
        if duplicate_names:
            raise ValueError(
                f"配置文件 {config_path} 与此前配置存在重名因子: "
                f"{', '.join(duplicate_names)}"
            )
        factor_configs.update(config)
    return factor_configs


def _is_default_excluded(factor_config: dict[str, Any]) -> bool:
    """判断是否为固定排除的多周期 LLT/MA 因子。"""
    return factor_config.get("module") in EXCLUDED_FACTOR_MODULES


def _factor_outputs_exist(output_root: Path, index_name: str, factor_name: str) -> bool:
    """检查某个因子是否已经保存了按股票拆分的因子值 CSV。"""
    factor_dir = output_root / index_name / factor_name / "factors"
    return factor_dir.is_dir() and any(factor_dir.glob("*.csv"))


def _completed_successes(
    progress_path: Path,
    output_root: Path,
    index_name: str,
    require_factor_outputs: bool,
) -> set[str]:
    """读取已成功完成的因子名，用于断点续跑。"""
    if not progress_path.exists() or progress_path.stat().st_size == 0:
        return set()
    with progress_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "factor_name" not in reader.fieldnames:
            return set()
        return {
            row["factor_name"]
            for row in reader
            if row.get("status") == "success"
            and (
                not require_factor_outputs
                or _factor_outputs_exist(output_root, index_name, row["factor_name"])
            )
        }


def _ensure_progress_file(progress_path: Path, overwrite: bool = False) -> None:
    """确保进度 CSV 存在并包含表头。"""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not progress_path.exists() or progress_path.stat().st_size == 0:
        with progress_path.open("w", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=PROGRESS_COLUMNS).writeheader()


def _append_progress(progress_path: Path, row: dict[str, Any]) -> None:
    """追加一条完成记录并立即落盘。"""
    with progress_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PROGRESS_COLUMNS)
        writer.writerow({column: row.get(column, "") for column in PROGRESS_COLUMNS})
        file.flush()
        os.fsync(file.fileno())


def run_one_factor(
    output_root: Path,
    index_name: str,
    index_config: dict[str, str],
    factor_name: str,
    factor_config: dict[str, Any],
    save_factor: bool,
    save_periodic_reports: bool,
    security_status_path: str | None,
) -> dict[str, Any]:
    """子进程执行单个因子。"""
    started_at = _now()
    start_time = time.perf_counter()
    output_dir = output_root / index_name / factor_name

    try:
        from run import Runner

        importlib.import_module(factor_config["module"])
        data_dir = Path(index_config["data_dir"])
        runner = Runner(
            factor_name=factor_name,
            factor_params=factor_config.get("params", {}),
            symbols=None,
            start=index_config["start"],
            n_groups=5,
            output_dir=output_dir,
            data_columns=factor_config.get("columns"),
            data_dir=data_dir,
            constituents_path=data_dir / "index_members_rebalance.csv",
            constituents=index_config["constituents"],
            security_status_path=security_status_path,
        )
        runner.run(
            save_factor=save_factor,
            save_periodic_reports=save_periodic_reports,
        )
        status = "success"
        error = ""
    except Exception:
        status = "failed"
        error = traceback.format_exc()

    return {
        "factor_name": factor_name,
        "status": status,
        "duration_seconds": f"{time.perf_counter() - start_time:.2f}",
        "started_at": started_at,
        "finished_at": _now(),
        "output_dir": str(output_dir),
        "error": error,
    }


def run_index(
    output_root: Path,
    index_name: str,
    factor_configs: dict[str, dict[str, Any]],
    workers: int,
    save_factor: bool,
    save_periodic_reports: bool,
    resume: bool,
    force: bool,
    security_status_path: str | None,
    dry_run: bool,
    heartbeat_seconds: int,
) -> None:
    """按指定并行度跑完一个指数上的全部因子。"""
    index_config = INDEX_RUNS[index_name]
    progress_path = output_root / f"{index_name}_completed_factors.csv"
    _ensure_progress_file(progress_path, overwrite=force)

    completed = (
        _completed_successes(
            progress_path,
            output_root=output_root,
            index_name=index_name,
            require_factor_outputs=save_factor,
        )
        if resume and not force
        else set()
    )
    tasks = [
        (factor_name, config)
        for factor_name, config in factor_configs.items()
        if factor_name not in completed
    ]

    print(
        f"[{_now()}] 开始 {index_name}({index_config['label']}): "
        f"start={index_config['start']}, total={len(factor_configs)}, "
        f"skip_success={len(completed)}, pending={len(tasks)}, workers={workers}, "
        f"save_factor={save_factor}, periodic_reports={save_periodic_reports}",
        flush=True,
    )
    if dry_run:
        print(
            f"[{_now()}] dry-run: {index_name} 不执行，进度文件: {progress_path}",
            flush=True,
        )
        return
    if not tasks:
        return

    finished = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                run_one_factor,
                output_root,
                index_name,
                index_config,
                factor_name,
                factor_config,
                save_factor,
                save_periodic_reports,
                security_status_path,
            ): factor_name
            for factor_name, factor_config in tasks
        }
        pending = set(future_map)
        while pending:
            done, pending = wait(
                pending,
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                print(
                    f"[{_now()}] {index_name} heartbeat: "
                    f"finished={finished}/{len(tasks)}, "
                    f"running_or_pending={len(pending)}, "
                    f"progress_file={progress_path}",
                    flush=True,
                )
                continue

            for future in done:
                result = future.result()
                _append_progress(progress_path, result)
                finished += 1
                print(
                    f"[{_now()}] {index_name} {finished}/{len(tasks)} "
                    f"{result['factor_name']} {result['status']} "
                    f"{result['duration_seconds']}s",
                    flush=True,
                )

                if finished % 5 == 0:
                    print(
                        f"[{_now()}] {index_name} 已完成 {finished} 个，"
                        f"进度文件: {progress_path}",
                        flush=True,
                    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="并行批量回测全部单因子")
    parser.add_argument(
        "--config",
        nargs="+",
        default=list(DEFAULT_CONFIG_PATHS),
        help="单因子配置文件，可传多个；默认合并基本面、价格、风险、行业配置",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="输出根目录，默认 outputs；例如 outputs_new",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=tuple(INDEX_RUNS),
        default=list(DEFAULT_INDEX_ORDER),
        help="默认顺序: zz500 zz1000 hs300",
    )
    parser.add_argument(
        "--save-factor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存每只股票的每日因子值，默认启用；可用 --no-save-factor 关闭",
    )
    parser.add_argument(
        "--no-periodic-reports",
        action="store_true",
        help="只保存总区间报告，不生成年度报告",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不跳过进度文件中已有 success 的因子",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖对应指数的进度文件并重新记录",
    )
    parser.add_argument(
        "--security-status-path",
        default=str(DEFAULT_SECURITY_STATUS_PATH),
        help="传空字符串可关闭 ST/停牌/退市过滤",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行回测")
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="没有因子完成时，主进程输出心跳日志的间隔秒数",
    )
    parser.add_argument(
        "--exclude-factors",
        nargs="+",
        default=[],
        help="在固定排除多周期 LLT/MA 的基础上，额外跳过指定因子名",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    args.heartbeat_seconds = max(1, args.heartbeat_seconds)
    output_root = Path(args.output_root)
    config_paths = [Path(config_path) for config_path in args.config]
    factor_configs = _load_factor_configs(config_paths)
    default_excluded = sorted(
        factor_name
        for factor_name, factor_config in factor_configs.items()
        if _is_default_excluded(factor_config)
    )
    factor_configs = {
        factor_name: factor_config
        for factor_name, factor_config in factor_configs.items()
        if not _is_default_excluded(factor_config)
    }
    extra_excluded = set(args.exclude_factors)
    if extra_excluded:
        unknown_excluded = sorted(extra_excluded - set(factor_configs))
        if unknown_excluded:
            print(
                f"[{_now()}] 注意: 以下排除因子不在配置文件中: "
                f"{', '.join(unknown_excluded)}",
                flush=True,
            )
        factor_configs = {
            factor_name: factor_config
            for factor_name, factor_config in factor_configs.items()
            if factor_name not in extra_excluded
        }
        print(
            f"[{_now()}] 已额外排除 {len(extra_excluded) - len(unknown_excluded)} 个因子: "
            f"{', '.join(sorted(extra_excluded - set(unknown_excluded)))}",
            flush=True,
        )
    security_status_path = args.security_status_path or None

    print(
        f"[{_now()}] CPU={os.cpu_count()} workers={args.workers} "
        f"factor_count={len(factor_configs)} config={', '.join(map(str, config_paths))} "
        f"output_root={output_root} save_factor={args.save_factor} "
        f"periodic_reports={not args.no_periodic_reports}",
        flush=True,
    )
    print(
        f"[{_now()}] 固定排除 {len(default_excluded)} 个多周期因子: "
        f"{', '.join(default_excluded)}",
        flush=True,
    )
    if not extra_excluded and len(factor_configs) != EXPECTED_FACTOR_COUNT:
        raise ValueError(
            f"固定排除后应有 {EXPECTED_FACTOR_COUNT} 个单因子，"
            f"实际为 {len(factor_configs)}。请检查配置文件。"
        )

    for index_name in args.indices:
        run_index(
            output_root=output_root,
            index_name=index_name,
            factor_configs=factor_configs,
            workers=args.workers,
            save_factor=args.save_factor,
            save_periodic_reports=not args.no_periodic_reports,
            resume=not args.no_resume,
            force=args.force,
            security_status_path=security_status_path,
            dry_run=args.dry_run,
            heartbeat_seconds=args.heartbeat_seconds,
        )

    print(f"[{_now()}] 全部指定指数执行完成", flush=True)


if __name__ == "__main__":
    main()
