"""Predict center-node chemical shifts with a trained G2S checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

try:
    from .g2s_model import LocalNMRDataset, load_raw_data
    from .g2s_utils import (
        compute_g2s_metrics,
        load_g2s_checkpoint,
        plot_error_histogram,
        plot_group_mae,
        plot_shift_regression,
        predict_loader,
        save_g2s_plot_data,
        save_json,
        save_prediction_points,
    )
    from ...shared.inverse_common import PPM_AXIS, lorentzian_spectrum, compute_r2_score
except ImportError:
    from model.forward.g2s.g2s_model import LocalNMRDataset, load_raw_data
    from model.forward.g2s.g2s_utils import (
        compute_g2s_metrics,
        load_g2s_checkpoint,
        plot_error_histogram,
        plot_group_mae,
        plot_shift_regression,
        predict_loader,
        save_g2s_plot_data,
        save_json,
        save_prediction_points,
    )
    from model.shared.inverse_common import PPM_AXIS, lorentzian_spectrum, compute_r2_score


def _mol_get(mol, key: str, default=None):
    if isinstance(mol, dict):
        return mol.get(key, default)
    return getattr(mol, key, default)


def _plot_single_molecule_spectrum(ppm_axis: torch.Tensor,
                                   true_spec: torch.Tensor,
                                   pred_spec: torch.Tensor,
                                   output_png: Path,
                                   title: str):
    import matplotlib.pyplot as plt

    ppm = ppm_axis.detach().cpu().numpy()
    true_np = true_spec.detach().cpu().numpy()
    pred_np = pred_spec.detach().cpu().numpy()
    diff_np = true_np - pred_np

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.2
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(ppm, true_np, color="#2878B5", linewidth=1.5, label="True")
    axes[0].plot(ppm, pred_np, color="#D55E00", linewidth=1.3, linestyle="--", label="Predicted")
    axes[0].set_ylabel("Intensity", fontsize=11)
    axes[0].set_title(title, fontsize=12)
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].tick_params(direction="in", width=1.0)
    axes[1].plot(ppm, diff_np, color="#444444", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Chemical shift (ppm)", fontsize=11)
    axes[1].set_ylabel("Diff.", fontsize=10)
    axes[1].tick_params(direction="in", width=1.0)
    for ax in axes:
        ax.set_xlim(float(ppm.max()), float(ppm.min()))
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_single_molecule_outputs(raw_list: list,
                                 prediction_data: dict,
                                 output_dir: Path,
                                 hwhm: float,
                                 intensity_mode: str):
    """Save molecule-level true/predicted spectra when a single .pt graph is used."""
    if len(raw_list) != 1:
        return
    mol = raw_list[0]
    y_true = _mol_get(mol, "y_spectrum", _mol_get(mol, "y_mol", None))
    real_intensity = _mol_get(mol, "real_intensity", None)
    if y_true is None or real_intensity is None:
        return

    center_ids = prediction_data["original_center_id"].detach().cpu().long()
    pred_mu = prediction_data["pred"].detach().cpu().float()
    true_mu = prediction_data["true"].detach().cpu().float()
    intensity_all = torch.as_tensor(real_intensity, dtype=torch.float).flatten()
    if str(intensity_mode).lower() == "ones":
        pi = torch.ones_like(pred_mu)
    else:
        pi = intensity_all[center_ids].float()

    pred_spec = lorentzian_spectrum(pred_mu, pi, PPM_AXIS.cpu(), hwhm=float(hwhm)).cpu()
    true_node_spec = lorentzian_spectrum(true_mu, pi, PPM_AXIS.cpu(), hwhm=float(hwhm)).cpu()
    y_true = torch.as_tensor(y_true, dtype=torch.float).flatten().cpu()
    n = min(int(y_true.numel()), int(pred_spec.numel()), int(true_node_spec.numel()), int(PPM_AXIS.numel()))
    y_true = y_true[:n]
    pred_spec = pred_spec[:n]
    true_node_spec = true_node_spec[:n]
    ppm_axis = PPM_AXIS.cpu()[:n]

    spectrum_r2 = compute_r2_score(y_true, pred_spec)
    node_spectrum_r2 = compute_r2_score(true_node_spec, pred_spec)
    spectrum_mae = float(torch.mean(torch.abs(y_true - pred_spec)).item())
    node_spectrum_mae = float(torch.mean(torch.abs(true_node_spec - pred_spec)).item())

    pd.DataFrame({
        "ppm": ppm_axis.numpy(),
        "true_spectrum": y_true.numpy(),
        "true_node_spectrum": true_node_spec.numpy(),
        "predicted_spectrum": pred_spec.numpy(),
        "diff_true_minus_pred": (y_true - pred_spec).numpy(),
        "diff_true_node_minus_pred": (true_node_spec - pred_spec).numpy(),
    }).to_csv(output_dir / "single_molecule_spectrum.csv", index=False)
    pd.DataFrame([{
        "n_centers": int(pred_mu.numel()),
        "shift_mae_ppm": float(torch.mean(torch.abs(pred_mu - true_mu)).item()),
        "shift_rmse_ppm": float(torch.sqrt(torch.mean((pred_mu - true_mu) ** 2)).item()),
        "spectrum_r2_vs_y_spectrum": float(spectrum_r2),
        "spectrum_mae_vs_y_spectrum": spectrum_mae,
        "spectrum_r2_vs_true_node_spectrum": float(node_spectrum_r2),
        "spectrum_mae_vs_true_node_spectrum": node_spectrum_mae,
        "hwhm": float(hwhm),
        "intensity_mode": str(intensity_mode),
    }]).to_csv(output_dir / "single_molecule_summary.csv", index=False)

    _plot_single_molecule_spectrum(
        ppm_axis,
        y_true,
        pred_spec,
        output_dir / "single_molecule_spectrum.png",
        title="G2S single molecule spectrum",
    )


def parse_args():
    parser = argparse.ArgumentParser("G2S chemical-shift prediction")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--input_path", type=Path, required=True, help="Directory or .pt molecular graph file.")
    parser.add_argument("--output_dir", type=Path, default=Path("g2s_prediction_results"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--k_hop", type=int, default=None, help="Override checkpoint k_hop.")
    parser.add_argument("--hwhm", type=float, default=1.0, help="Lorentzian HWHM used for single-molecule spectrum reconstruction.")
    parser.add_argument(
        "--intensity_mode",
        choices=["real", "ones"],
        default="real",
        help="Intensity used for single-molecule spectrum reconstruction. G2S itself predicts shifts only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_g2s_checkpoint(args.model_path, device=device)
    k_hop = int(args.k_hop) if args.k_hop is not None else int(config.k_hop)

    raw = load_raw_data(args.input_path)
    dataset = LocalNMRDataset(raw, k=k_hop)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    data = predict_loader(model, loader, device)
    metrics = compute_g2s_metrics(data["pred"], data["true"], data["center_su"])
    save_json(args.output_dir / "metrics.json", metrics)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    save_prediction_points(data, args.output_dir / "prediction_points.csv")
    save_g2s_plot_data(data, args.output_dir / "plot_data", prefix="prediction")
    plot_shift_regression(data["pred"], data["true"], args.output_dir / "regression.png", title=f"G2S prediction, k={k_hop}")
    plot_error_histogram(data["pred"], data["true"], args.output_dir / "error_histogram.png", title=f"G2S prediction errors, k={k_hop}")
    plot_group_mae(metrics, args.output_dir / "group_mae.png")
    save_single_molecule_outputs(raw, data, args.output_dir, hwhm=float(args.hwhm), intensity_mode=args.intensity_mode)
    print(pd.DataFrame([metrics])[["n", "mae", "rmse", "r2", "p95_ae"]].to_string(index=False))
    print(f"[done] prediction outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
