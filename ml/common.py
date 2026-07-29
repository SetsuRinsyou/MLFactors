"""232因子滚动机器学习共用的数据、评估、日志和输出逻辑。"""

from __future__ import annotations

import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.data.dataset.loader import StaticDataLoader
from scipy.stats import rankdata


METRIC_NAMES = (
    "loss",
    "IC",
    "ICIR",
    "top_20pct_period_return",
    "top_20pct_sharpe",
)
SELECTION_METRIC = "test_top_20pct_period_return"


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def preprocess_features(features: pd.DataFrame, mad_scale: float = 5.0) -> np.ndarray:
    """逐日做MAD去极值和截面Z-score，最后仅在内存中以0填充NaN。"""
    values = features.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    dates = features.index.get_level_values("datetime").to_numpy()
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(dates)]

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = values[start:end]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(block, axis=0)
            mad = np.nanmedian(np.abs(block - median), axis=0)
        clip_mask = np.isfinite(mad) & (mad > 1e-12)
        if clip_mask.any():
            block[:, clip_mask] = np.clip(
                block[:, clip_mask],
                median[clip_mask] - mad_scale * mad[clip_mask],
                median[clip_mask] + mad_scale * mad[clip_mask],
            )

        finite = np.isfinite(block)
        count = finite.sum(axis=0)
        total = np.where(finite, block, 0.0).sum(axis=0, dtype=np.float64)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        centered = np.where(finite, block - mean, 0.0)
        variance = np.divide(
            np.square(centered, dtype=np.float64).sum(axis=0),
            count,
            out=np.zeros_like(total),
            where=count > 0,
        )
        std = np.sqrt(variance)
        std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
        block = (block - mean) / std
        block[~np.isfinite(block)] = 0.0
        values[start:end] = block.astype(np.float32, copy=False)
    return np.ascontiguousarray(values)


def make_extreme_labels(
    returns: pd.Series, tail_fraction: float = 0.30
) -> np.ndarray:
    """每日未来超额收益前后tail_fraction分别标记为+1/-1，中间为0。"""
    ranks = returns.groupby(level="datetime", sort=False).rank(
        method="first", pct=True
    )
    labels = np.zeros(len(returns), dtype=np.int8)
    valid = returns.notna().to_numpy()
    rank_values = ranks.to_numpy()
    labels[valid & (rank_values <= tail_fraction)] = -1
    labels[valid & (rank_values > 1.0 - tail_fraction)] = 1
    return labels


@dataclass(frozen=True)
class RollingWindow:
    number: int
    total: int
    output_month: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    predict_start: pd.Timestamp
    predict_end: pd.Timestamp


@dataclass(frozen=True)
class DataSlice:
    index: pd.MultiIndex
    features: np.ndarray
    returns: np.ndarray
    labels: np.ndarray
    configured_start: pd.Timestamp
    configured_end: pd.Timestamp
    effective_end: pd.Timestamp
    purged_dates: int


@dataclass
class RollingData:
    index: pd.MultiIndex
    features: np.ndarray
    returns: np.ndarray
    labels: np.ndarray
    row_dates: np.ndarray
    calendar: pd.DatetimeIndex

    def slice(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        purge_tail_dates: int = 0,
    ) -> DataSlice:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        segment_dates = self.calendar[
            (self.calendar >= start) & (self.calendar <= end)
        ]
        if len(segment_dates) <= purge_tail_dates:
            raise ValueError(f"{start.date()}至{end.date()}交易日不足")
        effective_end = pd.Timestamp(segment_dates[-(purge_tail_dates + 1)])
        left = int(np.searchsorted(self.row_dates, start.to_datetime64(), side="left"))
        right = int(
            np.searchsorted(
                self.row_dates, effective_end.to_datetime64(), side="right"
            )
        )
        if left >= right:
            raise ValueError(f"{start.date()}至{effective_end.date()}没有样本")
        return DataSlice(
            index=self.index[left:right],
            features=self.features[left:right],
            returns=self.returns[left:right],
            labels=self.labels[left:right],
            configured_start=start,
            configured_end=end,
            effective_end=effective_end,
            purged_dates=purge_tail_dates,
        )


def load_rolling_data(
    dataset_path: Path,
    factor_names: list[str],
    start: str,
    end: str,
    mad_scale: float,
    tail_fraction: float,
) -> RollingData:
    """通过Qlib StaticDataLoader加载一次全区间数据并完成逐日预处理。"""
    frame = StaticDataLoader(str(dataset_path)).load(
        start_time=start,
        end_time=end,
    )
    if frame.empty:
        raise ValueError("Qlib数据集为空")
    if list(frame["feature"].columns) != factor_names:
        raise ValueError("Qlib数据集因子列与manifest不一致")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()

    index = frame.index
    row_dates = index.get_level_values("datetime").to_numpy()
    calendar = pd.DatetimeIndex(row_dates).unique().sort_values()
    returns = frame["label"]["fwd_return_5d"].to_numpy(dtype=np.float64)
    labels = make_extreme_labels(
        frame["label"]["fwd_excess_return_5d"], tail_fraction
    )
    print(
        "[preprocess] MAD winsorize -> cross-sectional zscore -> runtime fillna(0)",
        flush=True,
    )
    features = preprocess_features(frame["feature"], mad_scale)
    return RollingData(index, features, returns, labels, row_dates, calendar)


def make_rolling_windows(
    train_start: str,
    output_start: str,
    output_end: str,
) -> list[RollingWindow]:
    """生成扩展训练集、12个月测试集、随后1个月预测集。"""
    first = pd.Period(output_start, freq="M")
    last = pd.Period(output_end, freq="M")
    periods = list(pd.period_range(first, last, freq="M"))
    windows: list[RollingWindow] = []
    for number, output_period in enumerate(periods, start=1):
        train_end = (output_period - 13).end_time.normalize()
        test_period = output_period - 12
        test_end_period = output_period - 1
        windows.append(
            RollingWindow(
                number=number,
                total=len(periods),
                output_month=str(output_period),
                train_start=pd.Timestamp(train_start),
                train_end=train_end,
                test_start=test_period.start_time.normalize(),
                test_end=test_end_period.end_time.normalize(),
                predict_start=output_period.start_time.normalize(),
                predict_end=output_period.end_time.normalize(),
            )
        )
    return windows


class FactorEvaluator:
    """计算IC、ICIR及五组错位5日Top 20%组合的阶段表现。"""

    def __init__(
        self,
        index: pd.MultiIndex,
        returns: np.ndarray,
        horizon: int,
        top_fraction: float,
    ) -> None:
        self.horizon = int(horizon)
        self.top_fraction = float(top_fraction)
        values = np.asarray(returns, dtype=np.float64)
        dates = index.get_level_values("datetime").to_numpy()
        boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(dates)]
        self.groups: list[tuple[np.ndarray, np.ndarray, float]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            idx = np.arange(start, end)
            idx = idx[np.isfinite(values[idx])]
            if len(idx) < 3:
                continue
            return_rank = rankdata(values[idx], method="average")
            return_rank -= return_rank.mean()
            denom = float(np.sqrt(np.square(return_rank).sum()))
            if denom > 0:
                self.groups.append((idx, return_rank, denom))
        self.returns = values

    def evaluate(self, scores: np.ndarray) -> dict[str, float]:
        scores = np.asarray(scores, dtype=np.float64)
        ic_values: list[float] = []
        daily_top_returns: list[float] = []
        for idx, return_rank, return_denom in self.groups:
            score_rank = rankdata(scores[idx], method="average")
            score_rank -= score_rank.mean()
            score_denom = float(np.sqrt(np.square(score_rank).sum()))
            ic_values.append(
                float(np.dot(score_rank, return_rank) / (score_denom * return_denom))
                if score_denom > 0
                else np.nan
            )
            count = max(1, math.ceil(len(idx) * self.top_fraction))
            local = np.argpartition(scores[idx], -count)[-count:]
            daily_top_returns.append(float(np.mean(self.returns[idx[local]])))

        ic = np.asarray(ic_values, dtype=np.float64)
        top_returns = np.asarray(daily_top_returns, dtype=np.float64)
        ic_means: list[float] = []
        icirs: list[float] = []
        period_returns: list[float] = []
        sharpes: list[float] = []
        periods_per_year = 252.0 / self.horizon
        for offset in range(self.horizon):
            ic_sample = ic[offset:: self.horizon]
            ic_sample = ic_sample[np.isfinite(ic_sample)]
            if len(ic_sample):
                ic_means.append(float(ic_sample.mean()))
            if len(ic_sample) >= 2 and ic_sample.std(ddof=1) > 0:
                icirs.append(float(ic_sample.mean() / ic_sample.std(ddof=1)))

            return_sample = top_returns[offset:: self.horizon]
            return_sample = return_sample[np.isfinite(return_sample)]
            if len(return_sample) and np.all(return_sample > -1.0):
                period_returns.append(float(np.prod(1.0 + return_sample) - 1.0))
            if len(return_sample) >= 2 and return_sample.std(ddof=1) > 0:
                sharpes.append(
                    float(
                        return_sample.mean()
                        / return_sample.std(ddof=1)
                        * np.sqrt(periods_per_year)
                    )
                )
        return {
            "IC": float(np.mean(ic_means)) if ic_means else np.nan,
            "ICIR": float(np.mean(icirs)) if icirs else np.nan,
            "top_20pct_period_return": (
                float(np.mean(period_returns)) if period_returns else np.nan
            ),
            "top_20pct_sharpe": float(np.mean(sharpes)) if sharpes else np.nan,
        }


def binary_logloss(scores: np.ndarray, labels: np.ndarray) -> float:
    selected = labels != 0
    if not selected.any():
        return np.nan
    return float(np.logaddexp(0.0, -labels[selected] * scores[selected]).mean())


def selection_improved(
    record: dict[str, float | int],
    best_record: dict[str, float | int] | None,
) -> bool:
    current = float(record[SELECTION_METRIC])
    if not np.isfinite(current):
        return False
    if best_record is None:
        return True
    previous = float(best_record[SELECTION_METRIC])
    if current > previous + 1e-12:
        return True
    return (
        abs(current - previous) <= 1e-12
        and float(record["test_loss"]) < float(best_record["test_loss"])
    )


def format_epoch_log(
    window: RollingWindow,
    epoch: int,
    epochs: int,
    seconds: float,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    learning_rate: float | None = None,
) -> str:
    lr = "" if learning_rate is None else f" lr={learning_rate:.8f}"
    return (
        f"roll={window.number:02d}/{window.total:02d} output={window.output_month} "
        f"epoch={epoch:02d}/{epochs:02d} seconds={seconds:.3f}{lr} "
        f"train(loss={train_metrics['loss']:.6f}, IC={train_metrics['IC']:.5f}, "
        f"ICIR={train_metrics['ICIR']:.5f}, "
        f"top20_period_ret={train_metrics['top_20pct_period_return']:.4%}, "
        f"top20_sharpe={train_metrics['top_20pct_sharpe']:.4f}) "
        f"test(loss={test_metrics['loss']:.6f}, IC={test_metrics['IC']:.5f}, "
        f"ICIR={test_metrics['ICIR']:.5f}, "
        f"top20_period_ret={test_metrics['top_20pct_period_return']:.4%}, "
        f"top20_sharpe={test_metrics['top_20pct_sharpe']:.4f})"
    )


def reset_artifacts(script_dir: Path, model_name: str) -> tuple[Path, Path]:
    """一次新运行开始时清理旧训练产物，保留代码和配置。"""
    model_dir = script_dir / "model"
    log_dir = script_dir / "log"
    output_dir = script_dir / "output"
    for directory in (model_dir, log_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for path in model_dir.iterdir():
        if path.is_file():
            path.unlink()
    for path in log_dir.iterdir():
        if path.is_file():
            path.unlink()
    factors_dir = output_dir / "factors"
    if factors_dir.exists():
        shutil.rmtree(factors_dir)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
    factors_dir.mkdir(parents=True)
    return model_dir / model_name, log_dir / "epochs.log"


def build_available_date_map(
    calendar: pd.DatetimeIndex,
    last_available_date: str | None,
) -> dict[pd.Timestamp, pd.Timestamp]:
    mapping = {
        pd.Timestamp(signal): pd.Timestamp(available)
        for signal, available in zip(calendar[:-1], calendar[1:])
    }
    if last_available_date is not None:
        candidate = pd.Timestamp(last_available_date)
        if candidate <= calendar[-1]:
            raise ValueError("last_available_date必须晚于数据集最后交易日")
        mapping[pd.Timestamp(calendar[-1])] = candidate
    return mapping


def append_factor_scores(
    factors_dir: Path,
    index: pd.MultiIndex,
    scores: np.ndarray,
    factor_name: str,
    available_dates: dict[pd.Timestamp, pd.Timestamp],
) -> int:
    """按股票将一个预测月的连续分数追加到三列CSV。"""
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                index.get_level_values("datetime")
            ).normalize(),
            "instrument": index.get_level_values("instrument").astype(str),
            factor_name: np.asarray(scores, dtype=np.float64),
        }
    )
    frame["available_date"] = frame["signal_date"].map(available_dates)
    if frame["available_date"].isna().any():
        missing = frame.loc[frame["available_date"].isna(), "signal_date"].max()
        raise ValueError(
            f"无法确定{missing.date()}的下一真实交易日，请更新last_available_date"
        )
    frame = frame.sort_values(["instrument", "signal_date"])
    columns = ["signal_date", "available_date", factor_name]
    for instrument, block in frame.groupby("instrument", sort=False):
        path = factors_dir / f"{instrument}.csv"
        block[columns].to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False,
            date_format="%Y-%m-%d",
        )
    return len(frame)
