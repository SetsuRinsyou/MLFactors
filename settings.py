"""项目范围内唯一的共享配置加载与访问入口。"""

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class PathSettings:
    project_root: Path
    data_dir: Path
    output_root: Path
    runtime_root: Path
    factor_config: Path
    factor_category_config: Path
    constituents_path: Path
    security_status_path: Path
    market_state_path: Path


@dataclass(frozen=True)
class DataSettings:
    report_publish_date_column: str
    days_since_report_column: str
    market_cap_column: str
    industry_column: str
    constituent_index: str
    evaluation_price_column: str


@dataclass(frozen=True)
class EvaluationSettings:
    ic_decay_periods: tuple[int, ...]
    n_groups: int
    ic_method: str
    tail_percentages: tuple[int, ...]
    timing_win_horizon_weights: Mapping[int, float]

    @property
    def tail_fractions(self) -> tuple[float, ...]:
        return tuple(value / 100 for value in self.tail_percentages)


@dataclass(frozen=True)
class ReportSettings:
    daily_metrics_period: int
    factor_versions: tuple[str, ...]
    version_names: Mapping[str, str]
    core_tail_percentage: int
    report_lags: tuple[int, ...]
    size_bucket_quantiles: tuple[float, float]
    weekly_ic_frequency: str
    preferred_benchmark_column: str
    market_state_eras: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]]
    minimum_market_state_segment_days: int
    market_state_names: Mapping[int, str]

    @property
    def core_tail_fraction(self) -> float:
        return self.core_tail_percentage / 100


@dataclass(frozen=True)
class PipelineSettings:
    window_parameter_keys: tuple[str, ...]
    hardcoded_window_factors: frozenset[str]
    base_matrix_worker_limit: int


@dataclass(frozen=True)
class RuntimeSettings:
    mode: str
    start: str
    workers: int
    resume: bool
    save_factor: bool
    factor: str | None


@dataclass(frozen=True)
class ProjectSettings:
    source_path: Path
    paths: PathSettings
    data: DataSettings
    evaluation: EvaluationSettings
    report: ReportSettings
    pipeline: PipelineSettings
    runtime: RuntimeSettings


_PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _load_settings(path: str | Path) -> ProjectSettings:
    """读取一套完整JSON配置并转换为只读、带类型的项目设置。"""
    source_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        paths = raw["paths"]
        data = raw["data"]
        evaluation = raw["evaluation"]
        report = raw["report"]
        pipeline = raw["pipeline"]
        runtime = raw["runtime"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"无法读取完整运行配置 {source_path}: {error}") from error

    mode = str(runtime["mode"])
    if mode not in {"factor", "pipeline"}:
        raise ValueError("runtime.mode 必须是 factor 或 pipeline")
    workers = int(runtime["workers"])
    if workers <= 0:
        raise ValueError("runtime.workers 必须为正整数")
    if not isinstance(runtime["resume"], bool):
        raise ValueError("runtime.resume 必须是布尔值")
    if not isinstance(runtime["save_factor"], bool):
        raise ValueError("runtime.save_factor 必须是布尔值")
    factor = runtime.get("factor")
    if factor is not None and not isinstance(factor, str):
        raise ValueError("runtime.factor 必须是字符串或 null")

    tail_percentages = tuple(int(value) for value in evaluation["tail_percentages"])
    core_tail_percentage = int(report["core_tail_percentage"])
    if core_tail_percentage not in tail_percentages:
        raise ValueError("report.core_tail_percentage 必须包含在尾部比例中")
    if int(evaluation["n_groups"]) <= 0:
        raise ValueError("evaluation.n_groups 必须为正整数")
    if int(pipeline["base_matrix_worker_limit"]) <= 0:
        raise ValueError("pipeline.base_matrix_worker_limit 必须为正整数")
    try:
        pd.Timestamp(runtime["start"])
    except (TypeError, ValueError) as error:
        raise ValueError("runtime.start 必须是有效日期") from error

    settings = ProjectSettings(
        source_path=source_path,
        paths=PathSettings(
            project_root=_PROJECT_ROOT,
            data_dir=_project_path(paths["data_dir"]),
            output_root=_project_path(paths["output_root"]),
            runtime_root=_project_path(paths["runtime_root"]),
            factor_config=_project_path(paths["factor_config"]),
            factor_category_config=_project_path(paths["factor_category_config"]),
            constituents_path=_project_path(paths["constituents_path"]),
            security_status_path=_project_path(paths["security_status_path"]),
            market_state_path=_project_path(paths["market_state_path"]),
        ),
        data=DataSettings(
            report_publish_date_column=str(data["report_publish_date_column"]),
            days_since_report_column=str(data["days_since_report_column"]),
            market_cap_column=str(data["market_cap_column"]),
            industry_column=str(data["industry_column"]),
            constituent_index=str(data["constituent_index"]),
            evaluation_price_column=str(data["evaluation_price_column"]),
        ),
        evaluation=EvaluationSettings(
            ic_decay_periods=tuple(
                int(value) for value in evaluation["ic_decay_periods"]
            ),
            n_groups=int(evaluation["n_groups"]),
            ic_method=str(evaluation["ic_method"]),
            tail_percentages=tail_percentages,
            timing_win_horizon_weights=MappingProxyType(
                {
                    int(key): float(value)
                    for key, value in evaluation[
                        "timing_win_horizon_weights"
                    ].items()
                }
            ),
        ),
        report=ReportSettings(
            daily_metrics_period=int(report["daily_metrics_period"]),
            factor_versions=tuple(str(value) for value in report["factor_versions"]),
            version_names=MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in report["version_names"].items()
                }
            ),
            core_tail_percentage=core_tail_percentage,
            report_lags=tuple(int(value) for value in report["report_lags"]),
            size_bucket_quantiles=tuple(
                float(value) for value in report["size_bucket_quantiles"]
            ),
            weekly_ic_frequency=str(report["weekly_ic_frequency"]),
            preferred_benchmark_column=str(
                report["preferred_benchmark_column"]
            ),
            market_state_eras=MappingProxyType(
                {
                    str(name): (
                        pd.Timestamp(bounds[0]),
                        pd.Timestamp(bounds[1]),
                    )
                    for name, bounds in report["market_state_eras"].items()
                }
            ),
            minimum_market_state_segment_days=int(
                report["minimum_market_state_segment_days"]
            ),
            market_state_names=MappingProxyType(
                {
                    int(key): str(value)
                    for key, value in report["market_state_names"].items()
                }
            ),
        ),
        pipeline=PipelineSettings(
            window_parameter_keys=tuple(
                str(value) for value in pipeline["window_parameter_keys"]
            ),
            hardcoded_window_factors=frozenset(
                str(value) for value in pipeline["hardcoded_window_factors"]
            ),
            base_matrix_worker_limit=int(
                pipeline["base_matrix_worker_limit"]
            ),
        ),
        runtime=RuntimeSettings(
            mode=mode,
            start=str(runtime["start"]),
            workers=workers,
            resume=runtime["resume"],
            save_factor=runtime["save_factor"],
            factor=factor,
        ),
    )
    if len(settings.report.size_bucket_quantiles) != 2:
        raise ValueError("report.size_bucket_quantiles 必须恰好包含两个值")
    return settings


def load_settings(path: str | Path) -> ProjectSettings:
    """读取并完整校验一份运行配置。"""
    try:
        return _load_settings(path)
    except ValueError:
        raise
    except (KeyError, TypeError) as error:
        source_path = Path(path).expanduser().resolve()
        raise ValueError(f"运行配置字段无效 {source_path}: {error}") from error


class _SettingsProxy:
    def __init__(self) -> None:
        self._settings: ProjectSettings | None = None

    def _configure(self, settings: ProjectSettings) -> None:
        self._settings = settings

    @property
    def current(self) -> ProjectSettings:
        if self._settings is None:
            raise RuntimeError("尚未通过 --config 加载项目设置")
        return self._settings

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current, name)


SETTINGS = _SettingsProxy()


def configure_settings(path: str | Path) -> ProjectSettings:
    """加载并切换唯一SETTINGS入口当前使用的完整配置。"""
    settings = load_settings(path)
    SETTINGS._configure(settings)
    return settings
