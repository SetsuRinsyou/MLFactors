"""按月滚动训练232因子LightGBM GBDT并输出下一月组合因子。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
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
    binary_logloss,
    build_available_date_map,
    format_epoch_log,
    load_rolling_data,
    make_rolling_windows,
    reset_artifacts,
    resolve_path,
    selection_improved,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    return parser.parse_args()


def atomic_save_model(booster: lgb.Booster, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    booster.save_model(str(temporary), num_iteration=booster.current_iteration())
    os.replace(temporary, path)


def make_params(config: dict) -> dict:
    return {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": config["learning_rate"],
        "num_leaves": config["num_leaves"],
        "max_depth": config["max_depth"],
        "min_data_in_leaf": config["min_data_in_leaf"],
        "feature_fraction": config["feature_fraction"],
        "bagging_fraction": config["bagging_fraction"],
        "bagging_freq": config["bagging_freq"],
        "lambda_l1": config["lambda_l1"],
        "lambda_l2": config["lambda_l2"],
        "min_gain_to_split": config["min_gain_to_split"],
        "num_threads": config["num_threads"],
        "seed": config["random_state"],
        "feature_fraction_seed": config["random_state"],
        "bagging_seed": config["random_state"],
        "data_random_seed": config["random_state"],
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


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
    model_path, log_path = reset_artifacts(SCRIPT_DIR, "latest.txt")
    factors_dir = SCRIPT_DIR / "output" / "factors"
    epochs = int(config["epochs_per_roll"])
    purge_dates = int(config["horizon"]) + 1
    params = make_params(config)

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
            y_fit = (train.labels[fit_mask] == 1).astype(np.int8)
            train_set = lgb.Dataset(
                x_fit,
                label=y_fit,
                feature_name=factor_names,
                free_raw_data=True,
            )
            train_set.construct()
            del x_fit, y_fit
            booster = lgb.Booster(params=params, train_set=train_set)
            train_evaluator = FactorEvaluator(
                train.index, train.returns, config["horizon"], config["top_fraction"]
            )
            test_evaluator = FactorEvaluator(
                test.index, test.returns, config["horizon"], config["top_fraction"]
            )
            header = (
                f"[roll] {window.number:02d}/{window.total:02d} "
                f"output={window.output_month} "
                f"train={window.train_start.date()}..{train.effective_end.date()} "
                f"test={window.test_start.date()}..{test.effective_end.date()} "
                f"predict={window.predict_start.date()}..{window.predict_end.date()} "
                f"rows(train/test/predict/fit)="
                f"{len(train.index):,}/{len(test.index):,}/"
                f"{len(predict.index):,}/{int(fit_mask.sum()):,}"
            )
            print(header, flush=True)
            log.write(header + "\n")

            best_record: dict[str, float | int] | None = None
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                booster.update()
                train_scores = booster.predict(train.features, raw_score=True)
                test_scores = booster.predict(test.features, raw_score=True)
                train_metrics = train_evaluator.evaluate(train_scores)
                test_metrics = test_evaluator.evaluate(test_scores)
                train_metrics["loss"] = binary_logloss(train_scores, train.labels)
                test_metrics["loss"] = binary_logloss(test_scores, test.labels)
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
                    atomic_save_model(booster, model_path)

            selected_model = lgb.Booster(model_file=str(model_path))
            scores = selected_model.predict(predict.features, raw_score=True)
            written = append_factor_scores(
                factors_dir, predict.index, scores, "LightGBM", available_dates
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
                fit_mask,
                train_set,
                booster,
                train_evaluator,
                test_evaluator,
                selected_model,
                scores,
            )
            gc.collect()

    print(f"[done] model={model_path}, factors={factors_dir}, log={log_path}", flush=True)


if __name__ == "__main__":
    main(parse_args().config)
