"""Train the S2N inventory model on leakage-safe split datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from model.forward.common import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .s2n_model import S2NConfig, S2NModel, S2NDataset, collect_pt_files, s2n_loss
    from .s2n_utils import (
        compute_s2n_metrics,
        metrics_to_dataframe,
        plot_regression,
        predict_loader,
        save_json,
        save_s2n_checkpoint,
        save_s2n_plot_data,
    )
except ImportError:
    from model.forward.s2n.s2n_model import S2NConfig, S2NModel, S2NDataset, collect_pt_files, s2n_loss
    from model.forward.s2n.s2n_utils import (
        compute_s2n_metrics,
        metrics_to_dataframe,
        plot_regression,
        predict_loader,
        save_json,
        save_s2n_checkpoint,
        save_s2n_plot_data,
    )


def build_loader(files: list[str],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int,
                 drop_last: bool) -> DataLoader:
    dataset = S2NDataset(files)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def run_epoch(model: S2NModel,
              loader: DataLoader,
              device: torch.device,
              config: S2NConfig,
              optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "count": 0.0,
        "absolute": 0.0,
        "element": 0.0,
        "total_su": 0.0,
        "group": 0.0,
        "absent": 0.0,
        "high_count": 0.0,
        "balance": 0.0,
    }
    count = 0
    desc = "Training" if is_train else "Validating"

    for batch in tqdm(loader, desc=desc):
        spectrum = batch["spectrum"].to(device)
        elements = batch["elements"].to(device)
        true = batch["n_su_true"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(spectrum, elements)
        loss, parts = s2n_loss(logits, true, elements, config=config)
        if torch.isnan(loss):
            continue
        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = spectrum.size(0)
        count += batch_size
        for key in totals:
            totals[key] += float(parts[key].detach().item()) * batch_size

    if count <= 0:
        raise RuntimeError(f"No valid batches processed during {desc.lower()}.")
    return {key: value / count for key, value in totals.items()}


def parse_args():
    parser = argparse.ArgumentParser("Train S2N model with explicit train/val/test directories")
    parser.add_argument("--train_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--val_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--test_dirs", type=Path, nargs="*", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints_s2n_v2"))
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--element_mode", choices=["scaled_ratios", "raw"], default="scaled_ratios")
    parser.add_argument("--carbon_scale", type=float, default=600.0)
    parser.add_argument("--use_spectrum_stats", action="store_true", help="Fuse explicit NMR region statistics with CNN and element embeddings.")
    parser.add_argument("--spectrum_stats_mode", choices=["regions_v1"], default="regions_v1")
    parser.add_argument("--target_mode", choices=["per_100c", "absolute"], default="per_100c")
    parser.add_argument("--abundance_scale", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--merged_repeat", type=int, default=1, help="Repeat files whose path contains 'merged' to increase large-sample weight.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rare_threshold", type=float, default=3000.0)
    parser.add_argument("--rare_max_weight", type=float, default=5.0)
    parser.add_argument("--w_count", type=float, default=1.0)
    parser.add_argument("--w_absolute", type=float, default=0.30)
    parser.add_argument("--w_element", type=float, default=0.35)
    parser.add_argument("--w_total", type=float, default=0.25)
    parser.add_argument("--w_group", type=float, default=0.20)
    parser.add_argument("--w_absent", type=float, default=0.10)
    parser.add_argument("--w_high_count", type=float, default=0.35)
    parser.add_argument("--high_count_threshold", type=float, default=50.0)
    parser.add_argument("--high_count_weight", type=float, default=4.0)
    parser.add_argument("--w_balance", type=float, default=0.0, help="Weight for aromatic-vs-aliphatic balance loss.")
    parser.add_argument("--balance_min_carbon", type=float, default=50.0, help="Apply balance loss only when C >= this threshold.")
    parser.add_argument("--balance_target", choices=["difference_per_100c", "ratio"], default="difference_per_100c")
    parser.add_argument("--apply_eval_mask", action="store_true", help="Apply element-feasibility mask for reported eval metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_files = collect_pt_files(args.train_dirs)
    val_files = collect_pt_files(args.val_dirs)
    test_files = collect_pt_files(args.test_dirs or []) if args.test_dirs else []
    if int(args.merged_repeat) > 1:
        repeat = int(args.merged_repeat)
        train_files = [
            file
            for file in train_files
            for _ in range(repeat if "merged" in str(file).lower() else 1)
        ]

    print("--- S2N dataset ---")
    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")

    config = S2NConfig(
        hid=args.hid,
        base_channels=args.base_channels,
        dropout=args.dropout,
        element_mode=args.element_mode,
        carbon_scale=args.carbon_scale,
        use_spectrum_stats=args.use_spectrum_stats,
        spectrum_stats_mode=args.spectrum_stats_mode,
        target_mode=args.target_mode,
        abundance_scale=args.abundance_scale,
        rare_threshold=args.rare_threshold,
        rare_max_weight=args.rare_max_weight,
        w_count=args.w_count,
        w_absolute=args.w_absolute,
        w_element=args.w_element,
        w_total=args.w_total,
        w_group=args.w_group,
        w_absent=args.w_absent,
        w_high_count=args.w_high_count,
        high_count_threshold=args.high_count_threshold,
        high_count_weight=args.high_count_weight,
        w_balance=args.w_balance,
        balance_min_carbon=args.balance_min_carbon,
        balance_target=args.balance_target,
    )
    save_json(args.output_dir / "config.json", config.to_dict())
    save_json(args.output_dir / "data_files.json", {
        "train_dirs": [str(p) for p in args.train_dirs],
        "val_dirs": [str(p) for p in args.val_dirs],
        "test_dirs": [str(p) for p in (args.test_dirs or [])],
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    })

    train_loader = build_loader(train_files, args.batch_size, True, args.num_workers, drop_last=True)
    val_loader = build_loader(val_files, args.batch_size, False, args.num_workers, drop_last=False)
    test_loader = build_loader(test_files, args.batch_size, False, args.num_workers, drop_last=False) if test_files else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S2NModel.from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    history = []
    best_val = float("inf")
    best_path = args.output_dir / "best_s2n_model.pt"
    last_path = args.output_dir / "last_s2n_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, config, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device, config, optimizer=None)
        scheduler.step(val_loss["loss"])

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_loss.items()})
        row.update({f"val_{k}": v for k, v in val_loss.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_loss['loss']:.4f} | Val {val_loss['loss']:.4f} | "
            f"Val count {val_loss['count']:.4f} | abs {val_loss['absolute']:.4f} | "
            f"elem {val_loss['element']:.4f} | total {val_loss['total_su']:.4f} | "
            f"group {val_loss['group']:.4f} | high {val_loss['high_count']:.4f} | "
            f"balance {val_loss['balance']:.4f}"
        )

        save_s2n_checkpoint(
            last_path,
            model,
            config,
            epoch=epoch,
            metrics={"train": train_loss, "val": val_loss},
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            extra={"data_files_json": str(args.output_dir / "data_files.json")},
        )
        if val_loss["loss"] < best_val:
            best_val = val_loss["loss"]
            save_s2n_checkpoint(
                best_path,
                model,
                config,
                epoch=epoch,
                metrics={"train": train_loss, "val": val_loss},
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                extra={"data_files_json": str(args.output_dir / "data_files.json")},
            )
            print(f"  [best] saved to {best_path}")

    # Final standardized evaluation with the best model.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics_by_split = {}
    eval_dir = args.output_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        data = predict_loader(model, loader, device, apply_mask=args.apply_eval_mask)
        metrics = compute_s2n_metrics(data["pred"], data["true"], data["elements"])
        metrics_by_split[split_name] = metrics
        save_s2n_plot_data(data, eval_dir / "plot_data", prefix=split_name)
        plot_regression(data["pred"], data["true"], eval_dir / f"{split_name}_regression.png", title=f"S2N {split_name}")
    metrics_to_dataframe(metrics_by_split).to_csv(eval_dir / "metrics_summary.csv", index=False)
    save_json(eval_dir / "metrics_summary.json", metrics_by_split)
    print(f"[done] best checkpoint: {best_path}")
    print(f"[done] evaluation written to: {eval_dir}")


if __name__ == "__main__":
    main()
