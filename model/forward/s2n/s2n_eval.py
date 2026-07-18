"""Standard evaluation for S2N checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .s2n_model import S2NDataset, collect_pt_files
    from .s2n_utils import (
        compute_s2n_metrics,
        load_s2n_checkpoint,
        plot_regression,
        plot_total_su_regression,
        plot_su_bar,
        predict_loader,
        save_scale_stratified_s2n_eval,
        save_json,
        save_s2n_plot_data,
        save_prediction_table,
    )
except ImportError:
    from model.forward.s2n.s2n_model import S2NDataset, collect_pt_files
    from model.forward.s2n.s2n_utils import (
        compute_s2n_metrics,
        load_s2n_checkpoint,
        plot_regression,
        plot_total_su_regression,
        plot_su_bar,
        predict_loader,
        save_scale_stratified_s2n_eval,
        save_json,
        save_s2n_plot_data,
        save_prediction_table,
    )


def parse_args():
    parser = argparse.ArgumentParser("Evaluate an S2N checkpoint")
    parser.add_argument("--model_path", "--checkpoint", dest="model_path", type=Path, required=True)
    parser.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("s2n_eval_results"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--apply_mask", action="store_true", help="Apply element-feasibility mask before metrics.")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--examples_per_scale", type=int, default=2, help="Save this many worst examples from each scale bin.")
    parser.add_argument(
        "--scale_bins",
        type=float,
        nargs="*",
        default=[0, 80, 160, 260, 380, 1e9],
        help="Bin edges used for scale-stratified metrics."
    )
    parser.add_argument(
        "--scale_by",
        choices=["total_true_su", "carbon", "both"],
        default="both",
        help="Stratify metrics by total true SU count, carbon count, or both."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config, metadata = load_s2n_checkpoint(args.model_path, device=device)
    files = collect_pt_files(args.input_dirs)
    loader = DataLoader(
        S2NDataset(files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    result = predict_loader(model, loader, device, apply_mask=args.apply_mask)
    pred = result["pred"]
    true = result["true"]
    elements = result["elements"]
    paths = result["paths"]

    metrics = compute_s2n_metrics(pred, true, elements)
    save_json(args.output_dir / "metrics.json", metrics)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    save_s2n_plot_data(result, args.output_dir / "plot_data", prefix="eval")
    plot_regression(pred, true, args.output_dir / "regression.png")
    plot_total_su_regression(pred, true, args.output_dir / "total_su_regression.png")

    scale_dir = args.output_dir / "scale_eval"
    if args.scale_by in {"total_true_su", "both"}:
        save_scale_stratified_s2n_eval(
            result,
            scale_dir,
            prefix="eval",
            value_key="total_true_su",
            bins=args.scale_bins,
        )
    if args.scale_by in {"carbon", "both"}:
        save_scale_stratified_s2n_eval(
            result,
            scale_dir,
            prefix="eval",
            value_key="carbon",
            bins=args.scale_bins,
        )

    per_sample_mae = torch.abs(pred - true).mean(dim=1)
    sample_rows = []
    for i, path in enumerate(paths):
        sample_rows.append({
            "path": path,
            "mae": float(per_sample_mae[i].item()),
            "total_true": float(true[i].sum().item()),
            "total_pred": float(pred[i].sum().item()),
            "total_abs_error": float(abs(pred[i].sum().item() - true[i].sum().item())),
        })
    pd.DataFrame(sample_rows).to_csv(args.output_dir / "per_sample_errors.csv", index=False)

    order = torch.argsort(per_sample_mae, descending=True)
    example_dir = args.output_dir / "examples"
    example_dir.mkdir(exist_ok=True)
    for rank, idx_t in enumerate(order[: max(0, int(args.num_examples))]):
        idx = int(idx_t.item())
        stem = Path(paths[idx]).stem if idx < len(paths) else f"sample_{idx}"
        save_prediction_table(pred[idx], true[idx], example_dir / f"{rank+1:02d}_{stem}.csv")
        plot_su_bar(
            pred[idx],
            example_dir / f"{rank+1:02d}_{stem}.png",
            true=true[idx],
            title=f"S2N example: {stem}",
        )

    scale_example_dir = args.output_dir / "scale_examples"
    scale_example_dir.mkdir(exist_ok=True)
    scale_values = true.sum(dim=1)
    scale_bins = list(args.scale_bins)
    for bin_idx in range(len(scale_bins) - 1):
        lo = float(scale_bins[bin_idx])
        hi = float(scale_bins[bin_idx + 1])
        is_last = bin_idx == len(scale_bins) - 2
        mask = (scale_values >= lo) & ((scale_values <= hi) if is_last else (scale_values < hi))
        if not mask.any():
            continue
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        local_order = idxs[torch.argsort(per_sample_mae[idxs], descending=True)]
        label_hi = "inf" if hi >= 1e8 else f"{hi:g}"
        label = f"totalSU_{lo:g}_{label_hi}"
        for local_rank, idx_t in enumerate(local_order[: max(0, int(args.examples_per_scale))]):
            idx = int(idx_t.item())
            stem = Path(paths[idx]).stem if idx < len(paths) else f"sample_{idx}"
            out_stem = f"{label}_{local_rank+1:02d}_{stem}"
            save_prediction_table(pred[idx], true[idx], scale_example_dir / f"{out_stem}.csv")
            plot_su_bar(
                pred[idx],
                scale_example_dir / f"{out_stem}.png",
                true=true[idx],
                title=f"S2N {label}: {stem}",
            )

    save_json(args.output_dir / "checkpoint_config.json", config.to_dict())
    save_json(args.output_dir / "checkpoint_metadata_keys.json", {"keys": sorted(metadata.keys())})
    print(f"[done] S2N evaluation written to {args.output_dir}")


if __name__ == "__main__":
    main()
