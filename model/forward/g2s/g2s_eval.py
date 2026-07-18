"""Evaluate a trained shift-only G2S checkpoint."""

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


def parse_args():
    parser = argparse.ArgumentParser("Evaluate shift-only G2S model")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("g2s_eval_results"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--k_hop", type=int, default=None, help="Override checkpoint k_hop when evaluating.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, metadata = load_g2s_checkpoint(args.model_path, device=device)
    k_hop = int(args.k_hop) if args.k_hop is not None else int(config.k_hop)
    raw = load_raw_data(args.input_dirs)
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
    save_g2s_plot_data(data, args.output_dir / "plot_data", prefix="eval")
    plot_shift_regression(data["pred"], data["true"], args.output_dir / "regression.png", title=f"G2S evaluation, k={k_hop}")
    plot_error_histogram(data["pred"], data["true"], args.output_dir / "error_histogram.png", title=f"G2S errors, k={k_hop}")
    plot_group_mae(metrics, args.output_dir / "group_mae.png")
    save_json(args.output_dir / "checkpoint_metadata.json", {
        "config": config.to_dict(),
        "metadata_keys": sorted(list(metadata.keys())),
        "model_path": str(args.model_path),
        "input_dirs": [str(p) for p in args.input_dirs],
        "evaluated_k_hop": k_hop,
    })
    print(pd.DataFrame([metrics])[["n", "mae", "rmse", "r2", "p95_ae"]].to_string(index=False))
    print(f"[done] results written to {args.output_dir}")


if __name__ == "__main__":
    main()
