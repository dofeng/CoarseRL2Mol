"""Shared file discovery, training reproducibility and checkpoint output."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch


def collect_pt_files(paths: Sequence[str | Path]) -> list[str]:
    files: list[str] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(str(candidate) for candidate in sorted(path.glob("*.pt")))
        elif path.is_file() and path.suffix == ".pt":
            files.append(str(path))
        else:
            raise FileNotFoundError(f"No .pt files found at '{path}'")
    if not files:
        raise FileNotFoundError(f"No .pt files found in: {paths}")
    return files


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    residual_sum = float(np.sum((y_true - y_pred) ** 2))
    total_sum = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total_sum <= 1e-12:
        return float("nan")
    return 1.0 - residual_sum / total_sum


def save_checkpoint(path: str | Path,
                    model: torch.nn.Module,
                    config,
                    epoch: int,
                    metrics: dict,
                    optimizer_state: dict | None = None,
                    scheduler_state: dict | None = None,
                    extra: dict | None = None):
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def metrics_to_dataframe(metrics_by_split: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for split, metrics in metrics_by_split.items():
        row = {"split": split}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
