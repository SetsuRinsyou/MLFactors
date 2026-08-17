"""执行完整的 5,947 因子生成与报告流水线。"""

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import traceback
from typing import Any
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import DataLoader
from neutralization import neutralize_factor_values
from report_calculate import remove_legacy_report_outputs
from settings import SETTINGS


# ---------------------------------------------------------------------------
# Full 5,947-factor pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    runner_class: type[Any]
    configs: dict[str, dict[str, Any]]
    base_configs: dict[str, dict[str, Any]]
    derived_names: list[str]
    financial_factors: set[str]
    work_dir: Path
    market_data: pd.DataFrame | None = None
    benchmark: pd.DataFrame = field(default_factory=pd.DataFrame)
    calendar: pd.DatetimeIndex = field(
        default_factory=lambda: pd.DatetimeIndex([])
    )
    dates: pd.DatetimeIndex = field(
        default_factory=lambda: pd.DatetimeIndex([])
    )
    symbols: list[str] = field(default_factory=list)

    def attach_market_data(
        self,
        market_data: pd.DataFrame,
        benchmark: pd.DataFrame,
        calendar: pd.DatetimeIndex,
    ) -> None:
        self.market_data = market_data.sort_index()
        self.benchmark = benchmark
        self.calendar = calendar
        self.dates = pd.DatetimeIndex(
            self.market_data.index.get_level_values("date").unique()
        ).sort_values()
        self.symbols = sorted(
            str(value)
            for value in self.market_data.index.get_level_values("symbol").unique()
        )

    def require_market_data(self) -> pd.DataFrame:
        if self.market_data is None:
            raise RuntimeError("流水线上下文尚未加载行情数据")
        return self.market_data


_WORKER_CONTEXT: PipelineContext | None = None


def _set_worker_context(context: PipelineContext) -> None:
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is not None:
        raise RuntimeError("流水线上下文已经初始化")
    _WORKER_CONTEXT = context


def _get_worker_context() -> PipelineContext:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("流水线上下文尚未初始化")
    return _WORKER_CONTEXT


def _clear_worker_context() -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = None


def _pipeline_result(name: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"factor": name, "status": status, **extra}


def financial_factor_names(
    configs: dict[str, dict[str, Any]],
) -> set[str]:
    """从基础因子所属模块识别直接依赖财报字段的因子。"""
    return {
        name
        for name, config in configs.items()
        if config.get("module", "").startswith("factors.fundamental.")
    }


def _run_parallel(
    function: Any,
    items: list[Any],
    workers: int,
    description: str,
) -> None:
    failures = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(function, item): item for item in items}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=description, unit="factor"
        ):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
                tqdm.write(f"FAILED {result['factor']}\n{result['traceback']}")
    if failures:
        raise RuntimeError(f"{description}失败: {len(failures)}")


def _factor_values_ready(output_root: Path, name: str) -> bool:
    factor_dir = output_root / name / "factors"
    files = list(factor_dir.glob("*.csv")) if factor_dir.is_dir() else []
    if not files:
        return False
    try:
        columns = set(pd.read_csv(files[0], nrows=0).columns)
    except Exception:
        return False
    return {f"{name}_raw", f"{name}_neutralization"}.issubset(columns)


def _calculate_base_values(name: str) -> dict[str, Any]:
    try:
        context = _get_worker_context()
        config = context.base_configs[name]
        importlib.import_module(config["module"])
        runner = context.runner_class(
            factor_name=name,
            relay_class=config["relay_class"],
            factor_params=config.get("params", {}),
            start=SETTINGS.runtime.start,
            data_columns=config.get("columns", []),
            data_dir=SETTINGS.paths.data_dir,
            constituents_path=SETTINGS.paths.constituents_path,
            constituents=SETTINGS.data.constituent_index,
            output_dir=SETTINGS.paths.output_root / name,
            is_financial_factor=name in context.financial_factors,
            factor_config=config,
        )
        runner.run(save_factor=True, build_report=False)
        return _pipeline_result(name, "completed")
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _robust_z(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        median = np.nanmedian(values, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(values - median), axis=1, keepdims=True)
        scale = 1.4826 * mad
        fallback = np.nanstd(values, axis=1, keepdims=True)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback)
        result = (values - median) / np.where(scale > 1e-12, scale, np.nan)
    return np.clip(result, -5.0, 5.0).astype("float32")


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        np.save(file, values, allow_pickle=False)
    temporary.replace(path)


def _matrix_path(name: str, version: str, standardized: bool = False) -> Path:
    context = _get_worker_context()
    prefix = "z_" if standardized else ""
    return context.work_dir / "base_matrices" / f"{name}__{prefix}{version}.npy"


def _load_saved_versions(name: str) -> dict[str, pd.DataFrame]:
    context = _get_worker_context()
    date_positions = {
        date.strftime("%Y-%m-%d"): position
        for position, date in enumerate(context.dates)
    }
    arrays = {
        version: np.full(
            (len(context.dates), len(context.symbols)), np.nan, dtype="float32"
        )
        for version in SETTINGS.report.factor_versions
    }
    value_columns = {
        "raw": f"{name}_raw",
        "neutralization": f"{name}_neutralization",
    }
    for symbol_position, symbol in enumerate(context.symbols):
        path = SETTINGS.paths.output_root / name / "factors" / f"{symbol}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            usecols=["signal_date", *value_columns.values()],
            dtype={column: "float32" for column in value_columns.values()},
        )
        positions = np.fromiter(
            (date_positions.get(value, -1) for value in frame["signal_date"]),
            dtype="int32",
            count=len(frame),
        )
        valid = positions >= 0
        for version, column in value_columns.items():
            arrays[version][positions[valid], symbol_position] = frame.loc[
                valid, column
            ].to_numpy(dtype="float32")
    return {
        version: pd.DataFrame(values, index=context.dates, columns=context.symbols)
        .rename_axis(index="date", columns="symbol")
        for version, values in arrays.items()
    }


def _build_base_matrix(name: str) -> dict[str, Any]:
    try:
        versions = _load_saved_versions(name)
        for version, frame in versions.items():
            values = frame.to_numpy(dtype="float32")
            _atomic_save_npy(_matrix_path(name, version), values)
            _atomic_save_npy(_matrix_path(name, version, standardized=True), _robust_z(values))
        return _pipeline_result(name, "completed")
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _eval_formula_ast(node: ast.AST, variables: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(node, ast.Expression):
        return _eval_formula_ast(node.body, variables)
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return np.asarray(float(node.value), dtype="float32")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_formula_ast(node.operand, variables)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_formula_ast(node.left, variables)
        right = _eval_formula_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"派生公式包含不允许的语法: {ast.dump(node)}")


def _load_base_matrix(name: str, role: str, version: str) -> np.ndarray:
    standardized = role in {"Z", "S"}
    return np.load(_matrix_path(name, version, standardized), mmap_mode="r")


def _evaluate_combination(params: dict[str, Any], version: str) -> np.ndarray:
    directions = dict(zip(params["input_factors"], params["input_directions"]))
    variables: dict[str, np.ndarray] = {}
    counter = 0

    def replace_token(match: re.Match[str]) -> str:
        nonlocal counter
        role, factor_name = match.group(1), match.group(2)
        if factor_name not in directions:
            raise ValueError(f"公式引用未声明的基础因子: {factor_name}")
        variable = f"v{counter}"
        counter += 1
        values = _load_base_matrix(factor_name, role, version)
        variables[variable] = directions[factor_name] * values if role == "S" else values
        return variable

    expression = params["formula"].replace("×", "*")
    expression = re.sub(r"([0-9.]+)\[", r"\1*[", expression)
    expression = re.sub(r"([XZS])\(([^)]+)\)", replace_token, expression)
    expression = expression.replace("[", "(").replace("]", ")")
    expression = re.sub(r"(?<=[0-9.)])(?=v\d+)", "*", expression)
    with np.errstate(all="ignore"):
        return np.asarray(
            _eval_formula_ast(ast.parse(expression, mode="eval"), variables),
            dtype="float32",
        )


def _evaluate_transform(params: dict[str, Any], version: str) -> np.ndarray:
    values = np.asarray(
        _load_base_matrix(params["input_factors"][0], "Z", version), dtype="float32"
    )
    code = params["transformation"]
    with np.errstate(all="ignore"):
        if code == "T01":
            result = np.where(values == 0, np.nan, 1.0 / (values + 1e-6 * np.sign(values)))
        elif code == "T02": result = np.abs(values)
        elif code == "T03": result = np.maximum(values, 0)
        elif code == "T04": result = np.maximum(-values, 0)
        elif code == "T05": result = -np.abs(values - 1)
        elif code == "T06": result = -np.abs(values + 1)
        elif code == "T07": result = values / (1 + values**2)
        elif code == "T08": result = values * (np.abs(values) >= 1)
        elif code == "T09": result = values * (np.abs(values) < 1)
        elif code == "T10": result = np.abs(values) * (1 + 0.5 * (values > 0))
        elif code == "T11": result = np.sign(values) * (np.abs(values) >= 1)
        elif code == "T12": result = -np.abs(values**2 - 1)
        else: raise ValueError(f"未知单因子变换: {code}")
    result = np.asarray(result, dtype="float32")
    result[~np.isfinite(values)] = np.nan
    return result


def _make_memory_runner(name: str, config: dict[str, Any]) -> Any:
    context = _get_worker_context()
    market_data = context.require_market_data()
    runner = context.runner_class.__new__(context.runner_class)
    runner.factor_name = name
    runner.factor_config = config
    runner.is_financial_factor = any(
        value in context.financial_factors
        for value in config.get("params", {}).get("input_factors", [name])
    )
    runner.forward_periods = SETTINGS.evaluation.ic_decay_periods
    runner.n_groups = SETTINGS.evaluation.n_groups
    runner.ic_method = SETTINGS.evaluation.ic_method
    runner.eval_price_col = SETTINGS.data.evaluation_price_column
    runner.security_status_path = SETTINGS.paths.security_status_path
    runner.market_state_path = SETTINGS.paths.market_state_path
    runner.output_dir = SETTINGS.paths.output_root / name
    runner.data = market_data
    runner.benchmark = context.benchmark
    runner.result = pd.DataFrame()
    runner.raw_result = pd.DataFrame()
    runner.neutralized_result = pd.DataFrame()
    runner.evaluations = {}
    runner.summary = pd.DataFrame()
    runner.daily_metrics = pd.DataFrame()
    runner.yearly_summary = pd.DataFrame()
    runner.market_state_summary = pd.DataFrame()
    runner.industry_daily_ic = pd.DataFrame()
    runner.data_loader = SimpleNamespace(trading_calendar=context.calendar)
    return runner


def _calculate_derived_values(name: str) -> dict[str, Any]:
    try:
        context = _get_worker_context()
        market_data = context.require_market_data()
        config = context.configs[name]
        params = config["params"]
        arrays = {}
        for version in SETTINGS.report.factor_versions:
            values = (
                _evaluate_transform(params, version)
                if params["expansion_kind"] == "single_transform"
                else _evaluate_combination(params, version)
            )
            arrays[version] = pd.DataFrame(
                values, index=context.dates, columns=context.symbols, dtype="float32"
            ).rename_axis(index="date", columns="symbol")
        arrays["neutralization"] = neutralize_factor_values(
            arrays["neutralization"], market_data
        ).astype("float32")
        runner = _make_memory_runner(name, config)
        runner.save_signals(
            arrays["raw"],
            combined_data=market_data,
            neutralized_signals=arrays["neutralization"],
        )
        return _pipeline_result(name, "completed")
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _fill_factor_values(name: str) -> dict[str, Any]:
    try:
        context = _get_worker_context()
        factor_dir = SETTINGS.paths.output_root / name / "factors"
        if not factor_dir.is_dir():
            raise FileNotFoundError(f"因子值目录不存在: {factor_dir}")
        files_changed = cells_filled = 0
        for path in factor_dir.glob("*.csv"):
            frame = pd.read_csv(path)
            columns = [
                column for column in (f"{name}_raw", f"{name}_neutralization")
                if column in frame.columns
            ]
            if not columns:
                raise ValueError(f"{path} 缺少因子值列")
            changed = False
            for column in columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                missing = values.isna()
                filled = values.ffill().bfill()
                fillable = missing & filled.notna()
                if fillable.any():
                    frame[column] = filled
                    cells_filled += int(fillable.sum())
                    changed = True
            if changed:
                temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(path)
                files_changed += 1
        return _pipeline_result(
            name, "completed", files_changed=files_changed, cells_filled=cells_filled
        )
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _build_markdown_report(name: str) -> dict[str, Any]:
    try:
        context = _get_worker_context()
        config = context.configs[name]
        remove_legacy_report_outputs(SETTINGS.paths.output_root / name)
        signals = _load_saved_versions(name)
        runner = _make_memory_runner(name, config)
        runner.build_markdown_report(signals)
        return _pipeline_result(name, "completed")
    except Exception:
        return _pipeline_result(name, "failed", traceback=traceback.format_exc())


def _validate_full_config(
    full_configs: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """校验唯一全量配置，并从中拆出基础与派生因子。"""
    base_configs = {
        name: config
        for name, config in full_configs.items()
        if not config.get("params", {}).get("expansion_kind")
    }
    if len(base_configs) != 199 or len(full_configs) != 5947:
        raise ValueError(
            f"配置数量错误: base={len(base_configs)} full={len(full_configs)}"
        )
    derived = []
    for name, config in full_configs.items():
        if name in base_configs:
            continue
        params = config.get("params", {})
        kind = params.get("expansion_kind")
        required = {"formula", "input_factors", "input_directions"}
        if kind == "single_transform": required.add("transformation")
        elif kind == "factor_combination": required.add("combination")
        else: raise ValueError(f"{name} 缺少有效 expansion_kind")
        missing = sorted(required.difference(params))
        if missing: raise ValueError(f"{name} 派生配置不完整: {missing}")
        if not set(params["input_factors"]).issubset(base_configs):
            raise ValueError(f"{name} 引用了非基础输入")
        if len(params["input_factors"]) != len(params["input_directions"]):
            raise ValueError(f"{name} 输入方向数量不一致")
        derived.append(name)
    if len(derived) != 5748:
        raise ValueError(f"派生因子数错误: {len(derived)}")
    return base_configs, sorted(derived)


def run_full_pipeline(runner_class: type[Any]) -> None:
    """从零生成因子值，统一回填后，再生成全部Markdown报告。"""
    _clear_worker_context()
    configs = json.loads(SETTINGS.paths.factor_config.read_text(encoding="utf-8"))
    base_configs, derived_names = _validate_full_config(configs)
    SETTINGS.paths.output_root.mkdir(parents=True, exist_ok=True)
    SETTINGS.paths.runtime_root.mkdir(parents=True, exist_ok=True)
    work_dir = SETTINGS.paths.runtime_root / f"factor_pipeline_{uuid.uuid4().hex}"
    (work_dir / "base_matrices").mkdir(parents=True)
    context = PipelineContext(
        runner_class=runner_class,
        configs=configs,
        base_configs=base_configs,
        derived_names=derived_names,
        financial_factors=financial_factor_names(base_configs),
        work_dir=work_dir,
    )
    _set_worker_context(context)

    try:
        workers = SETTINGS.runtime.workers
        base_names = sorted(context.base_configs)
        base_tasks = [
            name
            for name in base_names
            if not (
                SETTINGS.runtime.resume
                and _factor_values_ready(SETTINGS.paths.output_root, name)
            )
        ]
        print(f"BASE_VALUES_START factors={len(base_tasks)} workers={workers}", flush=True)
        if base_tasks:
            _run_parallel(_calculate_base_values, base_tasks, workers, "基础因子值")
        print("BASE_VALUES_DONE factors=199 reports=0", flush=True)

        loader = DataLoader(
            data_dir=SETTINGS.paths.data_dir,
            start=SETTINGS.runtime.start,
            columns=[
                SETTINGS.data.evaluation_price_column,
                SETTINGS.data.report_publish_date_column,
                SETTINGS.data.market_cap_column,
                SETTINGS.data.industry_column,
            ],
            constituents_path=SETTINGS.paths.constituents_path,
            constituent_index=SETTINGS.data.constituent_index,
            security_status_path=SETTINGS.paths.security_status_path,
        )
        context.attach_market_data(
            loader.load_all(),
            loader.benchmark,
            loader.trading_calendar,
        )

        matrix_workers = min(
            workers,
            SETTINGS.pipeline.base_matrix_worker_limit,
        )
        print(
            f"BASE_MATRIX_START factors=199 workers={matrix_workers}",
            flush=True,
        )
        _run_parallel(_build_base_matrix, base_names, matrix_workers, "基础因子矩阵")
        print("BASE_MATRIX_DONE", flush=True)

        derived_tasks = [
            name
            for name in context.derived_names
            if not (
                SETTINGS.runtime.resume
                and _factor_values_ready(SETTINGS.paths.output_root, name)
            )
        ]
        print(f"DERIVED_VALUES_START factors={len(derived_tasks)} workers={workers}", flush=True)
        if derived_tasks:
            _run_parallel(_calculate_derived_values, derived_tasks, workers, "派生因子值")
        print("DERIVED_VALUES_DONE factors=5748 reports=0", flush=True)
        shutil.rmtree(context.work_dir / "base_matrices", ignore_errors=True)
        print("BASE_MATRIX_CLEANED", flush=True)

        window_bases = {
            name for name, config in context.base_configs.items()
            if name in SETTINGS.pipeline.hardcoded_window_factors
            or any(
                key in config.get("params", {})
                for key in SETTINGS.pipeline.window_parameter_keys
            )
        }
        fill_names = set(window_bases)
        for name, config in context.configs.items():
            if any(
                input_name in window_bases
                for input_name in config.get("params", {}).get("input_factors", [])
            ):
                fill_names.add(name)
        print(f"FILL_START factors={len(fill_names)} workers={workers}", flush=True)
        _run_parallel(_fill_factor_values, sorted(fill_names), workers, "缺失值回填")
        print(f"FILL_DONE factors={len(fill_names)}", flush=True)

        report_names = sorted(context.configs)
        print(f"REPORT_START factors={len(report_names)} workers={workers}", flush=True)
        _run_parallel(_build_markdown_report, report_names, workers, "Markdown报告")
        print(f"REPORT_DONE completed={len(report_names)} failed=0", flush=True)
    finally:
        shutil.rmtree(context.work_dir, ignore_errors=True)
        _clear_worker_context()
        if (
            SETTINGS.paths.runtime_root.is_dir()
            and not any(SETTINGS.paths.runtime_root.iterdir())
        ):
            SETTINGS.paths.runtime_root.rmdir()
