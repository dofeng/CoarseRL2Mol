"""Run k-hop ablation experiments for shift-only G2S models."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

try:
    from .g2s_utils import plot_hop_ablation
except ImportError:
    from model.forward.g2s.g2s_utils import plot_hop_ablation


def parse_args():
    parser = argparse.ArgumentParser("Run G2S k-hop ablation")
    parser.add_argument("--train_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--val_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--test_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints_g2s_ablation"))
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--hid", type=int, default=384)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _paths(prefix: str, paths: list[Path]) -> list[str]:
    out = [prefix]
    out.extend(str(p) for p in paths)
    return out


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k_hop in args.hops:
        run_dir = args.output_dir / f"k{k_hop}"
        cmd = [
            sys.executable,
            "-m",
            "model.forward.g2s.g2s_train",
            *_paths("--train_dirs", args.train_dirs),
            *_paths("--val_dirs", args.val_dirs),
            *_paths("--test_dirs", args.test_dirs),
            "--output_dir",
            str(run_dir),
            "--k_hop",
            str(k_hop),
            "--hid",
            str(args.hid),
            "--latent_dim",
            str(args.latent_dim),
            "--n_layers",
            str(args.n_layers),
            "--dropout",
            str(args.dropout),
            "--beta",
            str(args.beta),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--num_workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
        print(f"\n--- Running G2S ablation k={k_hop} ---")
        subprocess.run(cmd, check=True)

        metrics_csv = run_dir / "eval" / "metrics_summary.csv"
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            if "split" in df.columns and "test" in set(df["split"]):
                row = df[df["split"] == "test"].iloc[0].to_dict()
            else:
                row = df.iloc[-1].to_dict()
            row["k_hop"] = int(k_hop)
            rows.append(row)

    if rows:
        summary = pd.DataFrame(rows).sort_values("k_hop")
        summary_path = args.output_dir / "hop_ablation_summary.csv"
        summary.to_csv(summary_path, index=False)
        key_cols = [
            "k_hop", "n", "mae", "rmse", "r2", "median_ae", "p95_ae",
            "region_aromatic_mae", "region_carbonyl_mae", "region_alkyl_mae",
            "group_aromatic_mae", "group_aliphatic_mae", "group_carbonyl_mae",
        ]
        available = [c for c in key_cols if c in summary.columns]
        summary[available].to_csv(args.output_dir / "hop_ablation_key_metrics.csv", index=False)
        plot_hop_ablation(summary_path, args.output_dir / "hop_ablation.png")
        print(f"[done] ablation summary written to {summary_path}")


if __name__ == "__main__":
    main()
