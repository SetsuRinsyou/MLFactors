"""按月滚动训练GPU版232因子MLP并输出下一月组合因子。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

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


class FactorMLP(nn.Module):
    """232→512→256→128→64→1，隐藏块为Linear→LayerNorm→GELU→Dropout。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: list[float],
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 4 or len(dropout) != 4:
            raise ValueError("MLP必须配置4个隐藏层和4个dropout值")
        layers: list[nn.Module] = []
        previous = input_dim
        for width, probability in zip(hidden_dims, dropout):
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(probability),
                ]
            )
            previous = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(previous, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(features)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_gpu(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；MLP按要求禁止回退到CPU")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"GPU编号{gpu_id}无效，可用范围为0至{torch.cuda.device_count() - 1}"
        )
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def scheduled_learning_rate(epoch: int, config: dict) -> float:
    base = float(config["learning_rate"])
    minimum = float(config["minimum_learning_rate"])
    warmup = int(config["warmup_epochs"])
    total = int(config["epochs_per_roll"])
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def make_train_loader(
    features: np.ndarray,
    labels: np.ndarray,
    config: dict,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def train_epoch(
    model: FactorMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> None:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    smoothing = float(config["label_smoothing"])
    clip_norm = float(config["gradient_clip_norm"])
    use_scaler = scaler.is_enabled()
    for features, hard_targets in loader:
        features = features.to(device, non_blocking=True)
        hard_targets = hard_targets.to(device, non_blocking=True)
        targets = hard_targets * (1.0 - smoothing) + 0.5 * smoothing
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
        ):
            loss = criterion(model(features), targets)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()


@torch.inference_mode()
def predict_logits(
    model: FactorMLP,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    model.eval()
    scores = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        batch = torch.from_numpy(features[start:end]).to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
        ):
            scores[start:end] = model(batch).float().cpu().numpy()
    return scores.astype(np.float64)


def atomic_save_checkpoint(
    model: FactorMLP,
    path: Path,
    record: dict[str, float | int],
    factor_names: list[str],
    config: dict,
    window,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "record": record,
        "rolling_window": window,
        "factor_names": factor_names,
        "input_dim": len(factor_names),
        "hidden_dims": config["hidden_dims"],
        "dropout": config["dropout"],
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> FactorMLP:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = FactorMLP(
        payload["input_dim"], payload["hidden_dims"], payload["dropout"]
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def main(config_path: Path, cli_gpu_id: int | None) -> None:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gpu_id = int(config["gpu_id"] if cli_gpu_id is None else cli_gpu_id)
    device = require_gpu(gpu_id)
    torch.set_float32_matmul_precision("high")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"[device] gpu_id={gpu_id}, name={gpu_name}", flush=True)

    manifest = json.loads(
        resolve_path(config["manifest"], config_path.parent).read_text(
            encoding="utf-8"
        )
    )
    factor_names = manifest["factor_names"]
    if len(factor_names) != 232:
        raise ValueError(f"预期232个因子，manifest中为{len(factor_names)}个")
    if config["hidden_dims"][0] < 400:
        raise ValueError("第一隐藏层维度必须不小于400")

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
    model_path, log_path = reset_artifacts(SCRIPT_DIR, "latest.pt")
    factors_dir = SCRIPT_DIR / "output" / "factors"
    epochs = int(config["epochs_per_roll"])
    purge_dates = int(config["horizon"]) + 1
    amp_enabled = bool(config["mixed_precision"])
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        for window in windows:
            seed = int(config["random_state"]) + window.number
            set_random_seed(seed)
            train = data.slice(
                window.train_start, window.train_end, purge_tail_dates=purge_dates
            )
            test = data.slice(
                window.test_start, window.test_end, purge_tail_dates=purge_dates
            )
            predict = data.slice(window.predict_start, window.predict_end)
            fit_mask = train.labels != 0
            x_fit = np.ascontiguousarray(train.features[fit_mask])
            y_fit = (train.labels[fit_mask] == 1).astype(np.float32)
            train_loader = make_train_loader(x_fit, y_fit, config, seed)
            model = FactorMLP(
                len(factor_names), config["hidden_dims"], config["dropout"]
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            scaler = torch.amp.GradScaler(
                "cuda", enabled=amp_enabled and amp_dtype == torch.float16
            )
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
                f"{len(predict.index):,}/{len(x_fit):,}"
            )
            print(header, flush=True)
            log.write(header + "\n")

            best_record: dict[str, float | int] | None = None
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                learning_rate = scheduled_learning_rate(epoch, config)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    config,
                    amp_enabled,
                    amp_dtype,
                )
                train_scores = predict_logits(
                    model,
                    train.features,
                    device,
                    int(config["inference_batch_size"]),
                    amp_enabled,
                    amp_dtype,
                )
                test_scores = predict_logits(
                    model,
                    test.features,
                    device,
                    int(config["inference_batch_size"]),
                    amp_enabled,
                    amp_dtype,
                )
                if not np.isfinite(train_scores).all() or not np.isfinite(
                    test_scores
                ).all():
                    raise FloatingPointError(
                        f"{window.output_month} epoch {epoch}产生NaN或Inf预测"
                    )
                train_metrics = train_evaluator.evaluate(train_scores)
                test_metrics = test_evaluator.evaluate(test_scores)
                train_metrics["loss"] = binary_logloss(train_scores, train.labels)
                test_metrics["loss"] = binary_logloss(test_scores, test.labels)
                record = {
                    "roll": window.number,
                    "output_month": window.output_month,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
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
                    learning_rate,
                )
                print(line, flush=True)
                log.write(line + "\n")
                if selection_improved(record, best_record):
                    best_record = record.copy()
                    atomic_save_checkpoint(
                        model, model_path, record, factor_names, config, window
                    )

            selected_model = load_checkpoint(model_path, device)
            scores = predict_logits(
                selected_model,
                predict.features,
                device,
                int(config["inference_batch_size"]),
                amp_enabled,
                amp_dtype,
            )
            written = append_factor_scores(
                factors_dir, predict.index, scores, "MLP", available_dates
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
                x_fit,
                y_fit,
                train_loader,
                model,
                optimizer,
                scaler,
                train_evaluator,
                test_evaluator,
                selected_model,
                scores,
            )
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[done] model={model_path}, factors={factors_dir}, log={log_path}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.gpu_id)
