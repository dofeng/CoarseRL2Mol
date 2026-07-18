"""Utilities for shift-only G2S training, evaluation and prediction."""

from __future__ import annotations

import json
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch

try:
    from .g2s_model import G2SConfig, NMR_VAE, PPM_REGIONS, SU_GROUPS
    from ...shared.coarse_graph import NUM_SU_TYPES, SU_DEFS, _node_color
except ImportError:
    from model.forward.g2s.g2s_model import G2SConfig, NMR_VAE, PPM_REGIONS, SU_GROUPS
    from model.shared.coarse_graph import NUM_SU_TYPES, SU_DEFS, _node_color


def save_json(path: str | Path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_g2s_checkpoint(model_path: str | Path,
                        device: torch.device | str = "cpu") -> tuple[NMR_VAE, G2SConfig, dict]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        config = G2SConfig.from_dict(ckpt.get("config"))
        state_dict = ckpt["model_state_dict"]
        metadata = dict(ckpt)
    else:
        config = G2SConfig(predict_intensity=True)
        state_dict = ckpt
        metadata = {"legacy_state_dict_only": True}
    model = NMR_VAE.from_config(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, metadata


def save_g2s_checkpoint(path: str | Path,
                        model: NMR_VAE,
                        config: G2SConfig,
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


@torch.no_grad()
def predict_loader(model: NMR_VAE, loader, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    preds, trues, center_sus, mol_ids, original_centers, source_paths = [], [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        (mu_pred, _), _ = model(batch)
        preds.append(mu_pred.detach().cpu().view(-1))
        trues.append(batch.real_ppm.detach().cpu().view(-1).float())
        if hasattr(batch, "center_su"):
            center_sus.append(batch.center_su.detach().cpu().view(-1).long())
        else:
            su = batch.x[batch.center_id][:, :NUM_SU_TYPES].argmax(dim=-1).detach().cpu()
            center_sus.append(su)
        mol_ids.append(batch.mol_id.detach().cpu().view(-1).long())
        original_centers.append(batch.original_center_id.detach().cpu().view(-1).long())
        if hasattr(batch, "source_path"):
            source_paths.extend(list(batch.source_path))
        else:
            source_paths.extend(["" for _ in range(int(mu_pred.numel()))])

    return {
        "pred": torch.cat(preds, dim=0),
        "true": torch.cat(trues, dim=0),
        "center_su": torch.cat(center_sus, dim=0),
        "mol_id": torch.cat(mol_ids, dim=0),
        "original_center_id": torch.cat(original_centers, dim=0),
        "source_path": source_paths,
    }


def compute_g2s_metrics(pred: torch.Tensor,
                        true: torch.Tensor,
                        center_su: torch.Tensor | None = None) -> dict:
    pred_np = pred.detach().cpu().numpy().reshape(-1).astype(float)
    true_np = true.detach().cpu().numpy().reshape(-1).astype(float)
    err = np.abs(pred_np - true_np)
    metrics = {
        "n": int(true_np.size),
        "mae": float(np.mean(err)) if err.size else float("nan"),
        "median_ae": float(np.median(err)) if err.size else float("nan"),
        "rmse": rmse_np(true_np, pred_np),
        "r2": r2_score_np(true_np, pred_np),
        "p90_ae": float(np.percentile(err, 90)) if err.size else float("nan"),
        "p95_ae": float(np.percentile(err, 95)) if err.size else float("nan"),
    }

    for name, (lo, hi) in PPM_REGIONS.items():
        mask = (true_np >= float(lo)) & (true_np < float(hi))
        if mask.any():
            e = np.abs(pred_np[mask] - true_np[mask])
            metrics[f"region_{name}_n"] = int(mask.sum())
            metrics[f"region_{name}_mae"] = float(np.mean(e))
            metrics[f"region_{name}_rmse"] = rmse_np(true_np[mask], pred_np[mask])
        else:
            metrics[f"region_{name}_n"] = 0
            metrics[f"region_{name}_mae"] = float("nan")
            metrics[f"region_{name}_rmse"] = float("nan")

    if center_su is not None:
        su_np = center_su.detach().cpu().numpy().reshape(-1).astype(int)
        for name, ids in SU_GROUPS.items():
            mask = np.isin(su_np, np.asarray(ids, dtype=int))
            if mask.any():
                e = np.abs(pred_np[mask] - true_np[mask])
                metrics[f"group_{name}_n"] = int(mask.sum())
                metrics[f"group_{name}_mae"] = float(np.mean(e))
                metrics[f"group_{name}_rmse"] = rmse_np(true_np[mask], pred_np[mask])
            else:
                metrics[f"group_{name}_n"] = 0
                metrics[f"group_{name}_mae"] = float("nan")
                metrics[f"group_{name}_rmse"] = float("nan")

        for su_id, (su_name, _) in enumerate(SU_DEFS):
            mask = su_np == int(su_id)
            if mask.any():
                e = np.abs(pred_np[mask] - true_np[mask])
                metrics[f"su_{su_id:02d}_{su_name}_n"] = int(mask.sum())
                metrics[f"su_{su_id:02d}_{su_name}_mae"] = float(np.mean(e))
    return metrics


def metrics_to_dataframe(metrics_by_split: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for split, metrics in metrics_by_split.items():
        row = {"split": split}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def save_prediction_points(data: dict[str, torch.Tensor], output_csv: str | Path):
    pred = data["pred"].detach().cpu().numpy()
    true = data["true"].detach().cpu().numpy()
    su = data["center_su"].detach().cpu().numpy().astype(int)
    source_paths = data.get("source_path", [])
    rows = []
    for i in range(len(pred)):
        su_id = int(su[i])
        rows.append({
            "index": i,
            "source_path": source_paths[i] if i < len(source_paths) else "",
            "mol_id": int(data["mol_id"][i].item()),
            "original_center_id": int(data["original_center_id"][i].item()),
            "center_su": su_id,
            "center_su_name": SU_DEFS[su_id][0] if 0 <= su_id < len(SU_DEFS) else "",
            "true_ppm": float(true[i]),
            "pred_ppm": float(pred[i]),
            "abs_error_ppm": float(abs(pred[i] - true[i])),
        })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def save_g2s_plot_data(data: dict[str, torch.Tensor],
                       output_dir: str | Path,
                       prefix: str = "eval"):
    """Save raw G2S prediction/error tables for later figure redrawing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = data["pred"].detach().cpu().float()
    true = data["true"].detach().cpu().float()
    center_su = data["center_su"].detach().cpu().long()
    mol_id = data["mol_id"].detach().cpu().long()
    original_center_id = data["original_center_id"].detach().cpu().long()
    source_paths = list(data.get("source_path", []))
    err = torch.abs(pred - true)

    # 1. Raw regression/scatter source table.
    rows = []
    for idx in range(int(pred.numel())):
        su_id = int(center_su[idx].item())
        rows.append({
            "point_index": idx,
            "source_path": source_paths[idx] if idx < len(source_paths) else "",
            "mol_id": int(mol_id[idx].item()),
            "original_center_id": int(original_center_id[idx].item()),
            "center_su": su_id,
            "center_su_name": SU_DEFS[su_id][0] if 0 <= su_id < len(SU_DEFS) else "",
            "true_ppm": float(true[idx].item()),
            "pred_ppm": float(pred[idx].item()),
            "abs_error_ppm": float(err[idx].item()),
            "signed_error_ppm": float((pred[idx] - true[idx]).item()),
        })
    points_df = pd.DataFrame(rows)
    points_df.to_csv(output_dir / f"{prefix}_regression_points.csv", index=False)

    # 2. Per-molecule summary.
    mol_rows = []
    for mid in sorted(points_df["mol_id"].unique().tolist()):
        sub = points_df[points_df["mol_id"] == mid]
        mol_rows.append({
            "mol_id": int(mid),
            "n_centers": int(len(sub)),
            "mae": float(sub["abs_error_ppm"].mean()),
            "median_ae": float(sub["abs_error_ppm"].median()),
            "rmse": float(np.sqrt(np.mean(sub["signed_error_ppm"].to_numpy(dtype=float) ** 2))),
            "p95_ae": float(np.percentile(sub["abs_error_ppm"].to_numpy(dtype=float), 95)),
        })
    pd.DataFrame(mol_rows).to_csv(output_dir / f"{prefix}_per_molecule_errors.csv", index=False)

    # 3. Per-SU summary.
    su_rows = []
    for su_id, (su_name, _) in enumerate(SU_DEFS):
        mask = points_df["center_su"] == int(su_id)
        if not bool(mask.any()):
            continue
        sub = points_df[mask]
        signed = sub["signed_error_ppm"].to_numpy(dtype=float)
        ae = sub["abs_error_ppm"].to_numpy(dtype=float)
        su_rows.append({
            "center_su": su_id,
            "center_su_name": su_name,
            "n": int(len(sub)),
            "mae": float(np.mean(ae)),
            "median_ae": float(np.median(ae)),
            "rmse": float(np.sqrt(np.mean(signed ** 2))),
            "bias_mean": float(np.mean(signed)),
            "p95_ae": float(np.percentile(ae, 95)),
        })
    pd.DataFrame(su_rows).to_csv(output_dir / f"{prefix}_per_su_errors.csv", index=False)

    # 4. SU group summary.
    group_rows = []
    for group_name, ids in SU_GROUPS.items():
        mask = points_df["center_su"].isin(ids)
        if not bool(mask.any()):
            continue
        sub = points_df[mask]
        signed = sub["signed_error_ppm"].to_numpy(dtype=float)
        ae = sub["abs_error_ppm"].to_numpy(dtype=float)
        group_rows.append({
            "group": group_name,
            "n": int(len(sub)),
            "mae": float(np.mean(ae)),
            "median_ae": float(np.median(ae)),
            "rmse": float(np.sqrt(np.mean(signed ** 2))),
            "bias_mean": float(np.mean(signed)),
            "p95_ae": float(np.percentile(ae, 95)),
        })
    pd.DataFrame(group_rows).to_csv(output_dir / f"{prefix}_group_error_summary.csv", index=False)

    # 5. PPM region summary.
    region_rows = []
    for region_name, (lo, hi) in PPM_REGIONS.items():
        mask = (points_df["true_ppm"] >= float(lo)) & (points_df["true_ppm"] < float(hi))
        if not bool(mask.any()):
            continue
        sub = points_df[mask]
        signed = sub["signed_error_ppm"].to_numpy(dtype=float)
        ae = sub["abs_error_ppm"].to_numpy(dtype=float)
        region_rows.append({
            "region": region_name,
            "ppm_lo": float(lo),
            "ppm_hi": float(hi),
            "n": int(len(sub)),
            "mae": float(np.mean(ae)),
            "median_ae": float(np.median(ae)),
            "rmse": float(np.sqrt(np.mean(signed ** 2))),
            "bias_mean": float(np.mean(signed)),
            "p95_ae": float(np.percentile(ae, 95)),
        })
    pd.DataFrame(region_rows).to_csv(output_dir / f"{prefix}_region_error_summary.csv", index=False)


def plot_shift_regression(pred: torch.Tensor,
                          true: torch.Tensor,
                          output_png: str | Path,
                          title: str = "G2S predicted vs true chemical shift"):
    import matplotlib.pyplot as plt

    y_pred = pred.detach().cpu().numpy().reshape(-1).astype(float)
    y_true = true.detach().cpu().numpy().reshape(-1).astype(float)
    mae = float(np.mean(np.abs(y_pred - y_true))) if y_true.size else float("nan")
    r2 = r2_score_np(y_true, y_pred)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(y_true, y_pred, s=10, alpha=0.35, color="#2878B5", edgecolors="none")
    ax.plot([0, 240], [0, 240], "k--", linewidth=1.0)
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 240)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True chemical shift (ppm)", fontsize=12)
    ax.set_ylabel("Predicted chemical shift (ppm)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f} ppm",
            transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.tick_params(direction="in", width=1.1)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(pred: torch.Tensor,
                         true: torch.Tensor,
                         output_png: str | Path,
                         title: str = "G2S absolute error distribution"):
    import matplotlib.pyplot as plt

    err = np.abs(pred.detach().cpu().numpy().reshape(-1) - true.detach().cpu().numpy().reshape(-1))
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(err, bins=50, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Absolute error (ppm)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(direction="in", width=1.1)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_mae(metrics: dict,
                   output_png: str | Path,
                   prefix: str = "group_",
                   title: str = "G2S MAE by SU group"):
    import matplotlib.pyplot as plt

    names, values = [], []
    for name in SU_GROUPS:
        key = f"{prefix}{name}_mae"
        if key in metrics and not (isinstance(metrics[key], float) and math.isnan(metrics[key])):
            names.append(name.replace("_", "\n"))
            values.append(float(metrics[key]))
    if not values:
        return
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.2
    colors = ["#B55A4C", "#2878B5", "#2A9D8F", "#E69F00", "#7E57C2"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(np.arange(len(values)), values, color=colors[:len(values)], edgecolor="white", linewidth=0.6)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("MAE (ppm)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(direction="in", width=1.1)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hop_ablation(summary_csv: str | Path, output_png: str | Path):
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    if "k_hop" not in df.columns or "mae" not in df.columns:
        return
    df = df.sort_values("k_hop")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(df["k_hop"], df["mae"], "o-", color="#2878B5", linewidth=1.8, markersize=6, label="MAE")
    if "rmse" in df.columns:
        ax.plot(df["k_hop"], df["rmse"], "s--", color="#DD8452", linewidth=1.4, markersize=5, label="RMSE")
    ax.set_xlabel("Subgraph radius k-hop", fontsize=12)
    ax.set_ylabel("Error (ppm)", fontsize=12)
    ax.set_xticks(df["k_hop"].astype(int).tolist())
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(direction="in", width=1.1)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
