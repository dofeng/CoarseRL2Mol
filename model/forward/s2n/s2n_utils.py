"""Utilities for S2N training, evaluation and prediction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import torch

from model.forward.checkpoints import load_s2n_checkpoint
from model.forward.common import (
    metrics_to_dataframe,
    r2_score_np,
    save_checkpoint as save_s2n_checkpoint,
)

try:
    from .s2n_model import (
        S2NModel,
        SU_GROUPS,
        apply_element_feasibility_mask,
    )
    from ...shared.inverse_common import E_SU, SU_DEFS
    from ...shared.coarse_graph import _node_color
except ImportError:
    from model.forward.s2n.s2n_model import (
        S2NModel,
        SU_GROUPS,
        apply_element_feasibility_mask,
    )
    from model.shared.inverse_common import E_SU, SU_DEFS
    from model.shared.coarse_graph import _node_color


def save_json(path: str | Path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def predict_loader(model: S2NModel,
                   loader,
                   device: torch.device,
                   apply_mask: bool = False) -> dict[str, torch.Tensor | list[str]]:
    preds, trues, elements, paths = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            spectrum = batch["spectrum"].to(device)
            elem = batch["elements"].to(device)
            pred = model.infer_su_hist(spectrum, elem, apply_mask=False)
            if apply_mask:
                pred = apply_element_feasibility_mask(pred, elem)
            preds.append(pred.cpu())
            trues.append(batch["n_su_true"].cpu().float())
            elements.append(batch["elements"].cpu().float())
            if "path" in batch:
                paths.extend(list(batch["path"]))
    return {
        "pred": torch.cat(preds, dim=0),
        "true": torch.cat(trues, dim=0),
        "elements": torch.cat(elements, dim=0),
        "paths": paths,
    }


def compute_s2n_metrics(pred: torch.Tensor,
                        true: torch.Tensor,
                        elements: torch.Tensor) -> dict:
    pred = pred.float().cpu()
    true = true.float().cpu()
    elements = elements.float().cpu()
    carbon = elements[:, 0:1].clamp(min=1.0)

    diff = torch.abs(pred - true)
    elem_pred = pred @ E_SU.float()
    elem_diff = torch.abs(elem_pred - elements)

    metrics = {
        "flatten_mae": float(diff.mean().item()),
        "flatten_r2": float(r2_score_np(true.numpy(), pred.numpy())),
        "per_sample_mae_mean": float(diff.mean(dim=1).mean().item()),
        "per_sample_mae_median": float(diff.mean(dim=1).median().item()),
        "mae_per_100C": float((diff.sum(dim=1) / carbon.squeeze(1) * 100.0).mean().item()),
        "total_su_mae": float(torch.abs(pred.sum(dim=1) - true.sum(dim=1)).mean().item()),
        "element_mae_raw": float(elem_diff.mean().item()),
        "element_mae_per_100C": float((elem_diff.sum(dim=1) / carbon.squeeze(1) * 100.0).mean().item()),
        "max_true_count": float(true.max().item()),
        "max_pred_count": float(pred.max().item()),
        "pred_true_max_ratio": float((pred.max() / true.max().clamp(min=1.0)).item()),
    }

    for threshold in [25.0, 50.0, 100.0]:
        mask = true > threshold
        if mask.any():
            high_err = torch.abs(pred[mask] - true[mask])
            high_bias = pred[mask] - true[mask]
            metrics[f"true_gt_{int(threshold)}_n"] = int(mask.sum().item())
            metrics[f"true_gt_{int(threshold)}_mae"] = float(high_err.mean().item())
            metrics[f"true_gt_{int(threshold)}_bias_mean"] = float(high_bias.mean().item())
            metrics[f"true_gt_{int(threshold)}_pred_true_ratio_mean"] = float((pred[mask] / true[mask].clamp(min=1.0)).mean().item())
        else:
            metrics[f"true_gt_{int(threshold)}_n"] = 0
            metrics[f"true_gt_{int(threshold)}_mae"] = float("nan")
            metrics[f"true_gt_{int(threshold)}_bias_mean"] = float("nan")
            metrics[f"true_gt_{int(threshold)}_pred_true_ratio_mean"] = float("nan")

    group_metrics = {}
    report_groups = ["carbonyl", "aromatic", "unsaturated", "aliphatic", "hetero_terminal"]
    for name in report_groups:
        ids = SU_GROUPS[name]
        gp = pred[:, ids].sum(dim=1)
        gt = true[:, ids].sum(dim=1)
        group_metrics[f"{name}_mae"] = float(torch.abs(gp - gt).mean().item())
        group_metrics[f"{name}_mae_per_100C"] = float((torch.abs(gp - gt) / carbon.squeeze(1) * 100.0).mean().item())
    metrics.update(group_metrics)

    absent = {}
    element_col = {"O": 2, "N": 3, "S": 4, "X": 5}
    required = {
        "O": SU_GROUPS["oxygen_related"],
        "N": SU_GROUPS["nitrogen_related"],
        "S": SU_GROUPS["sulfur_related"],
        "X": SU_GROUPS["halogen_related"],
    }
    for symbol, ids in required.items():
        mask = elements[:, element_col[symbol]] <= 0.5
        if mask.any():
            false_mass = pred[mask][:, ids].sum(dim=1)
            absent[f"absent_{symbol}_false_count_mean"] = float(false_mass.mean().item())
            absent[f"absent_{symbol}_false_positive_rate"] = float((false_mass > 0.5).float().mean().item())
        else:
            absent[f"absent_{symbol}_false_count_mean"] = float("nan")
            absent[f"absent_{symbol}_false_positive_rate"] = float("nan")
    metrics.update(absent)
    return metrics


def save_prediction_table(pred: torch.Tensor,
                          true: torch.Tensor | None,
                          output_csv: str | Path):
    rows = []
    pred = pred.detach().cpu().float()
    true = true.detach().cpu().float() if true is not None else None
    for idx, (name, _) in enumerate(SU_DEFS):
        row = {
            "SU_Index": idx,
            "SU_Name": name,
            "Predicted_Count": float(pred[idx].item()),
            "Predicted_Rounded": int(round(float(pred[idx].item()))),
        }
        if true is not None:
            row["True_Count"] = float(true[idx].item())
            row["Abs_Error"] = abs(float(pred[idx].item()) - float(true[idx].item()))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def save_s2n_plot_data(result: dict,
                       output_dir: str | Path,
                       prefix: str = "eval"):
    """Save raw tables used by S2N plots and error summaries.

    The generated CSV files are intentionally redundant: each one maps cleanly
    to a common figure type so figures can be redrawn later in Origin, Python
    or Illustrator without rerunning model inference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = result["pred"].detach().cpu().float()
    true = result["true"].detach().cpu().float()
    elements = result["elements"].detach().cpu().float()
    paths = list(result.get("paths", []))
    if len(paths) < pred.shape[0]:
        paths.extend([f"sample_{i}" for i in range(len(paths), pred.shape[0])])

    carbon = elements[:, 0].clamp(min=1.0)
    diff = torch.abs(pred - true)

    # 1. Flattened regression/scatter source data.
    rows = []
    for sample_idx in range(pred.shape[0]):
        for su_idx, (su_name, _) in enumerate(SU_DEFS):
            rows.append({
                "sample_index": sample_idx,
                "path": paths[sample_idx],
                "su_index": su_idx,
                "su_name": su_name,
                "true_count": float(true[sample_idx, su_idx].item()),
                "pred_count": float(pred[sample_idx, su_idx].item()),
                "abs_error": float(diff[sample_idx, su_idx].item()),
            })
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_regression_points.csv", index=False)

    # 2. Per-sample error summary.
    sample_rows = []
    for sample_idx in range(pred.shape[0]):
        sum_abs = float(diff[sample_idx].sum().item())
        sample_rows.append({
            "sample_index": sample_idx,
            "path": paths[sample_idx],
            "C": float(elements[sample_idx, 0].item()),
            "H": float(elements[sample_idx, 1].item()),
            "O": float(elements[sample_idx, 2].item()),
            "N": float(elements[sample_idx, 3].item()),
            "S": float(elements[sample_idx, 4].item()),
            "X": float(elements[sample_idx, 5].item()),
            "mae": float(diff[sample_idx].mean().item()),
            "sum_abs_error": sum_abs,
            "mae_per_100C": float(sum_abs / float(carbon[sample_idx].item()) * 100.0),
            "total_true_su": float(true[sample_idx].sum().item()),
            "total_pred_su": float(pred[sample_idx].sum().item()),
            "total_abs_error": float(abs(pred[sample_idx].sum().item() - true[sample_idx].sum().item())),
        })
    pd.DataFrame(sample_rows).to_csv(output_dir / f"{prefix}_per_sample_errors.csv", index=False)

    # 3. Per-SU aggregate error summary.
    su_rows = []
    for su_idx, (su_name, _) in enumerate(SU_DEFS):
        e = diff[:, su_idx]
        su_rows.append({
            "su_index": su_idx,
            "su_name": su_name,
            "support_true_sum": float(true[:, su_idx].sum().item()),
            "pred_sum": float(pred[:, su_idx].sum().item()),
            "bias_mean": float((pred[:, su_idx] - true[:, su_idx]).mean().item()),
            "mae": float(e.mean().item()),
            "rmse": float(torch.sqrt(torch.mean((pred[:, su_idx] - true[:, su_idx]) ** 2)).item()),
            "false_positive_rate_when_true_zero": float(((true[:, su_idx] <= 0.5) & (pred[:, su_idx] > 0.5)).float().mean().item()),
        })
    pd.DataFrame(su_rows).to_csv(output_dir / f"{prefix}_per_su_errors.csv", index=False)

    # 3b. High-count point diagnostics.
    high_rows = []
    for threshold in [25.0, 50.0, 100.0]:
        mask = true > threshold
        if not mask.any():
            continue
        idxs = torch.nonzero(mask, as_tuple=False)
        for sample_idx, su_idx in idxs.tolist():
            true_v = float(true[sample_idx, su_idx].item())
            pred_v = float(pred[sample_idx, su_idx].item())
            high_rows.append({
                "threshold": threshold,
                "sample_index": sample_idx,
                "path": paths[sample_idx],
                "su_index": su_idx,
                "su_name": SU_DEFS[su_idx][0],
                "true_count": true_v,
                "pred_count": pred_v,
                "abs_error": abs(pred_v - true_v),
                "signed_error": pred_v - true_v,
                "pred_true_ratio": pred_v / max(true_v, 1.0),
            })
    pd.DataFrame(high_rows).to_csv(output_dir / f"{prefix}_high_count_points.csv", index=False)

    # 4. Group-level source data and aggregate errors.
    report_groups = ["carbonyl", "aromatic", "unsaturated", "aliphatic", "hetero_terminal"]
    group_rows = []
    group_summary_rows = []
    for group_name in report_groups:
        ids = SU_GROUPS[group_name]
        g_pred = pred[:, ids].sum(dim=1)
        g_true = true[:, ids].sum(dim=1)
        g_err = torch.abs(g_pred - g_true)
        for sample_idx in range(pred.shape[0]):
            group_rows.append({
                "sample_index": sample_idx,
                "path": paths[sample_idx],
                "group": group_name,
                "true_count": float(g_true[sample_idx].item()),
                "pred_count": float(g_pred[sample_idx].item()),
                "abs_error": float(g_err[sample_idx].item()),
                "abs_error_per_100C": float(g_err[sample_idx].item() / float(carbon[sample_idx].item()) * 100.0),
            })
        group_summary_rows.append({
            "group": group_name,
            "mae": float(g_err.mean().item()),
            "mae_per_100C": float((g_err / carbon * 100.0).mean().item()),
            "bias_mean": float((g_pred - g_true).mean().item()),
            "true_sum": float(g_true.sum().item()),
            "pred_sum": float(g_pred.sum().item()),
        })
    pd.DataFrame(group_rows).to_csv(output_dir / f"{prefix}_group_errors_by_sample.csv", index=False)
    pd.DataFrame(group_summary_rows).to_csv(output_dir / f"{prefix}_group_error_summary.csv", index=False)

    # 5. Element reconstruction source data.
    elem_pred = pred @ E_SU.float()
    elem_symbols = ["C", "H", "O", "N", "S", "X"]
    elem_rows = []
    for sample_idx in range(pred.shape[0]):
        for elem_idx, symbol in enumerate(elem_symbols):
            elem_rows.append({
                "sample_index": sample_idx,
                "path": paths[sample_idx],
                "element": symbol,
                "true_count": float(elements[sample_idx, elem_idx].item()),
                "pred_from_su_count": float(elem_pred[sample_idx, elem_idx].item()),
                "abs_error": float(abs(elem_pred[sample_idx, elem_idx].item() - elements[sample_idx, elem_idx].item())),
            })
    pd.DataFrame(elem_rows).to_csv(output_dir / f"{prefix}_element_reconstruction.csv", index=False)


def save_scale_stratified_s2n_eval(result: dict,
                                   output_dir: str | Path,
                                   prefix: str = "eval",
                                   value_key: str = "total_true_su",
                                   bins: Sequence[float] | None = None):
    """Save S2N metrics stratified by molecular scale.

    ``value_key`` can be ``total_true_su`` or ``carbon``. This avoids judging
    S2N only from large merged graphs and makes small/medium/large behavior
    visible in one evaluation run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = result["pred"].detach().cpu().float()
    true = result["true"].detach().cpu().float()
    elements = result["elements"].detach().cpu().float()
    paths = list(result.get("paths", []))
    if len(paths) < pred.shape[0]:
        paths.extend([f"sample_{i}" for i in range(len(paths), pred.shape[0])])

    if value_key == "carbon":
        values = elements[:, 0]
        default_bins = [0, 80, 160, 260, 380, 10**9]
        label_prefix = "C"
    else:
        values = true.sum(dim=1)
        default_bins = [0, 80, 160, 260, 380, 10**9]
        label_prefix = "total_su"
    edges = list(bins or default_bins)
    if len(edges) < 2:
        raise ValueError("Scale stratification requires at least two bin edges.")

    summary_rows = []
    point_rows = []
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        is_last = idx == len(edges) - 2
        mask = (values >= lo) & ((values <= hi) if is_last else (values < hi))
        label_hi = "inf" if hi >= 10**8 else f"{hi:g}"
        label = f"{label_prefix}_{lo:g}_{label_hi}"
        if not mask.any():
            summary_rows.append({"bin": label, "lo": lo, "hi": hi, "n": 0})
            continue

        sub_pred = pred[mask]
        sub_true = true[mask]
        sub_elements = elements[mask]
        metrics = compute_s2n_metrics(sub_pred, sub_true, sub_elements)
        row = {"bin": label, "lo": lo, "hi": hi, "n": int(mask.sum().item())}
        row.update(metrics)
        summary_rows.append(row)

        sample_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        for sample_idx in sample_indices:
            point_rows.append({
                "bin": label,
                "sample_index": int(sample_idx),
                "path": paths[sample_idx],
                "scale_value": float(values[sample_idx].item()),
                "carbon": float(elements[sample_idx, 0].item()),
                "total_true_su": float(true[sample_idx].sum().item()),
                "total_pred_su": float(pred[sample_idx].sum().item()),
                "mae": float(torch.abs(pred[sample_idx] - true[sample_idx]).mean().item()),
                "mae_per_100C": float(torch.abs(pred[sample_idx] - true[sample_idx]).sum().item() / max(float(elements[sample_idx, 0].item()), 1.0) * 100.0),
            })

    pd.DataFrame(summary_rows).to_csv(output_dir / f"{prefix}_{value_key}_scale_summary.csv", index=False)
    pd.DataFrame(point_rows).to_csv(output_dir / f"{prefix}_{value_key}_scale_points.csv", index=False)


def plot_total_su_regression(pred: torch.Tensor,
                             true: torch.Tensor,
                             output_png: str | Path,
                             title: str = "S2N total SU count"):
    import matplotlib.pyplot as plt

    y_true = true.detach().cpu().float().sum(dim=1).numpy()
    y_pred = pred.detach().cpu().float().sum(dim=1).numpy()
    r2 = r2_score_np(y_true, y_pred)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    if math.isclose(lo, hi):
        hi = lo + 1.0

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.45, color="#2878B5", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
    ax.set_xlabel("True total SU count", fontsize=12)
    ax.set_ylabel("Predicted total SU count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.tick_params(direction="in", width=1.1)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_su_bar(pred: torch.Tensor,
                output_png: str | Path,
                true: torch.Tensor | None = None,
                title: str = "Structural Unit Distribution"):
    import matplotlib.pyplot as plt

    pred_np = pred.detach().cpu().numpy().astype(float)
    x = np.arange(len(SU_DEFS))
    colors = [_node_color(int(idx)) for idx in x]

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(18, 5.5))
    if true is None:
        bars = ax.bar(x, pred_np, color=colors, edgecolor="white", linewidth=0.5)
        for bar, value in zip(bars, pred_np):
            if value >= 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(int(round(value))), ha="center", va="bottom", fontsize=9)
    else:
        true_np = true.detach().cpu().numpy().astype(float)
        width = 0.38
        ax.bar(x - width / 2, true_np, width, label="True", color="#4C72B0")
        ax.bar(x + width / 2, pred_np, width, label="Predicted", color="#DD8452")
        ax.legend(frameon=False, fontsize=11)
    ax.set_xlabel("SU ID", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=8)
    ax.tick_params(direction="in", width=1.1)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regression(pred: torch.Tensor,
                    true: torch.Tensor,
                    output_png: str | Path,
                    title: str = "S2N predicted vs true SU counts"):
    import matplotlib.pyplot as plt

    y_true = true.detach().cpu().numpy().reshape(-1)
    y_pred = pred.detach().cpu().numpy().reshape(-1)
    r2 = r2_score_np(y_true, y_pred)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    if math.isclose(lo, hi):
        hi = lo + 1.0

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=9, alpha=0.35, color="#2878B5", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
    ax.set_xlabel("True SU count", fontsize=12)
    ax.set_ylabel("Predicted SU count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.tick_params(direction="in", width=1.1)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
