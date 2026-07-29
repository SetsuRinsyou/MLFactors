"""按月滚动训练232因子线性SVM并输出下一月组合因子。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from common import (
    FactorEvaluator,
    METRIC_NAMES,
    append_factor_scores,
    build_available_date_map,
    format_epoch_log,
    load_rolling_data,
    make_rolling_windows,
    reset_artifacts,
    resolve_path,
    selection_improved,
)
from model import QlibLinearSVM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    return parser.parse_args()


def classification_loss(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    weights: np.ndarray,
) -> float:
    selected = labels != 0
    if not selected.any():
        return np.nan
    hinge = np.maximum(0.0, 1.0 - labels[selected] * scores[selected]).mean()
    return float(hinge + 0.5 * alpha * np.dot(weights, weights))


def atomic_dump(model: QlibLinearSVM, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, path)


def main(config_path: Path) -> None:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        resolve_path(config["manifest"], config_path.parent).read_text(
            encoding="utf-8"
        )
    )
    factor_names = manifest["factor_names"]
    if len(factor_names) != 232:
        raise ValueError(f"预期232个因子，manifest中为{len(factor_names)}个")

    windows = make_rolling_windows(
        manifest["first_eligible_date"],
        config["output_start"],
        manifest["date_max"],
    )
    data = load_rolling_data(
        resolve_path(config["dataset"], config_path.parent),
        factor_names,
        manifest["first_eligible_date"],
        manifest["date_max"],
        config["mad_scale"],
        config["tail_fraction"],
    )
    available_dates = build_available_date_map(
        data.calendar, config.get("last_available_date")
    )
    model_path, log_path = reset_artifacts(SCRIPT_DIR, "latest.joblib")
    factors_dir = SCRIPT_DIR / "output" / "factors"
    epochs = int(config["epochs_per_roll"])
    purge_dates = int(config["horizon"]) + 1

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        for window in windows:
            train = data.slice(
                window.train_start, window.train_end, purge_tail_dates=purge_dates
            )
            test = data.slice(
                window.test_start, window.test_end, purge_tail_dates=purge_dates
            )
            predict = data.slice(window.predict_start, window.predict_end)
            fit_mask = train.labels != 0
            x_fit = np.ascontiguousarray(train.features[fit_mask])
            y_fit = train.labels[fit_mask]
            train_evaluator = FactorEvaluator(
                train.index, train.returns, config["horizon"], config["top_fraction"]
            )
            test_evaluator = FactorEvaluator(
                test.index, test.returns, config["horizon"], config["top_fraction"]
            )
            model = QlibLinearSVM(
                factor_names=factor_names,
                epochs=epochs,
                alpha=config["alpha"],
                tail_fraction=config["tail_fraction"],
                mad_scale=config["mad_scale"],
                random_state=config["random_state"],
            )
            header = (
                f"[roll] {window.number:02d}/{window.total:02d} "
                f"output={window.output_month} "
                f"train={window.train_start.date()}..{train.effective_end.date()} "
                f"test={window.test_start.date()}..{test.effective_end.date()} "
                f"predict={window.predict_start.date()}..{window.predict_end.date()} "
                f"rows(train/test/predict/fit)="
                f"{len(train.index):,}/{len(test.index):,}/"
                f"{len(predict.index):,}/{len(x_fit):,}"
            )
            print(header, flush=True)
            log.write(header + "\n")

            best_record: dict[str, float | int] | None = None
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                model.fit_epoch(x_fit, y_fit)
                train_scores = model.decision_function(train.features)
                test_scores = model.decision_function(test.features)
                weights = model.estimator.coef_.ravel().astype(np.float64)
                train_metrics = train_evaluator.evaluate(train_scores)
                test_metrics = test_evaluator.evaluate(test_scores)
                train_metrics["loss"] = classification_loss(
                    train_scores, train.labels, config["alpha"], weights
                )
                test_metrics["loss"] = classification_loss(
                    test_scores, test.labels, config["alpha"], weights
                )
                record = {
                    "roll": window.number,
                    "output_month": window.output_month,
                    "epoch": epoch,
                    **{
                        f"train_{name}": train_metrics[name]
                        for name in METRIC_NAMES
                    },
                    **{f"test_{name}": test_metrics[name] for name in METRIC_NAMES},
                }
                line = format_epoch_log(
                    window,
                    epoch,
                    epochs,
                    time.perf_counter() - started,
                    train_metrics,
                    test_metrics,
                )
                print(line, flush=True)
                log.write(line + "\n")
                if selection_improved(record, best_record):
                    best_record = record.copy()
                    model.rolling_window = window
                    model.best_record = best_record
                    atomic_dump(model, model_path)

            selected_model: QlibLinearSVM = joblib.load(model_path)
            scores = selected_model.decision_function(predict.features)
            written = append_factor_scores(
                factors_dir, predict.index, scores, "SVM", available_dates
            )
            selected_line = (
                f"[selected] output={window.output_month} "
                f"epoch={best_record['epoch']} "
                f"test_top20_period_ret="
                f"{best_record['test_top_20pct_period_return']:.4%} "
                f"written={written:,} model={model_path}"
            )
            print(selected_line, flush=True)
            log.write(selected_line + "\n")
            del (
                train,
                test,
                predict,
                x_fit,
                y_fit,
                train_evaluator,
                test_evaluator,
                model,
                selected_model,
                scores,
            )
            gc.collect()

    print(f"[done] model={model_path}, factors={factors_dir}, log={log_path}", flush=True)


if __name__ == "__main__":
    main(parse_args().config)
