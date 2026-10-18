"""Train shift-only G2S models on explicit molecule splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from model.forward.common import set_seed
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    from .g2s_model import G2SConfig, LocalNMRDataset, NMR_VAE, collect_pt_files, g2s_loss_function, load_raw_data
    from .g2s_utils import (
        compute_g2s_metrics,
        metrics_to_dataframe,
        plot_error_histogram,
        plot_group_mae,
        plot_shift_regression,
        predict_loader,
        save_g2s_checkpoint,
        save_g2s_plot_data,
        save_json,
        save_prediction_points,
    )
except ImportError:
    from model.forward.g2s.g2s_model import G2SConfig, LocalNMRDataset, NMR_VAE, collect_pt_files, g2s_loss_function, load_raw_data
    from model.forward.g2s.g2s_utils import (
        compute_g2s_metrics,
        metrics_to_dataframe,
        plot_error_histogram,
        plot_group_mae,
        plot_shift_regression,
        predict_loader,
        save_g2s_checkpoint,
        save_g2s_plot_data,
        save_json,
        save_prediction_points,
    )


def build_dataset(paths: list[Path], k_hop: int) -> LocalNMRDataset:
    raw = load_raw_data(paths)
    return LocalNMRDataset(raw, k=k_hop)


def build_loader(dataset,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model: NMR_VAE,
              loader: DataLoader,
              device: torch.device,
              config: G2SConfig,
              optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "mu": 0.0, "kl": 0.0, "mae_ppm": 0.0}
    count = 0
    desc = "Training" if is_train else "Validating"
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred, latent = model(batch)
        loss, parts = g2s_loss_function(pred, latent, batch, beta=config.beta, loss_type=config.loss)
        if torch.isnan(loss):
            continue
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        batch_size = int(batch.real_ppm.numel())
        count += batch_size
        for key in totals:
            totals[key] += float(parts[key].detach().item()) * batch_size
    if count <= 0:
        raise RuntimeError(f"No valid batches processed during {desc.lower()}.")
    return {key: value / count for key, value in totals.items()}


def parse_args():
    parser = argparse.ArgumentParser("Train shift-only G2S model")
    parser.add_argument("--train_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--val_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--test_dirs", type=Path, nargs="*", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints_g2s_shift_k3"))
    parser.add_argument("--k_hop", type=int, default=3)
    parser.add_argument("--hid", type=int, default=384)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=1.0e-4)
    parser.add_argument("--loss", choices=["smooth_l1", "l1"], default="smooth_l1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_files = collect_pt_files(args.train_dirs)
    val_files = collect_pt_files(args.val_dirs)
    test_files = collect_pt_files(args.test_dirs or []) if args.test_dirs else []
    print("--- G2S dataset ---")
    print(f"Train molecules: {len(train_files)}")
    print(f"Val molecules:   {len(val_files)}")
    print(f"Test molecules:  {len(test_files)}")

    save_json(args.output_dir / "data_files.json", {
        "train_dirs": [str(p) for p in args.train_dirs],
        "val_dirs": [str(p) for p in args.val_dirs],
        "test_dirs": [str(p) for p in (args.test_dirs or [])],
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    })

    config = G2SConfig(
        k_hop=args.k_hop,
        hid=args.hid,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        beta=args.beta,
        loss=args.loss,
        predict_intensity=False,
    )
    save_json(args.output_dir / "config.json", config.to_dict())

    train_dataset = build_dataset(args.train_dirs, args.k_hop)
    val_dataset = build_dataset(args.val_dirs, args.k_hop)
    test_dataset = build_dataset(args.test_dirs, args.k_hop) if test_files else None
    print(f"Train subgraphs: {len(train_dataset)}")
    print(f"Val subgraphs:   {len(val_dataset)}")
    if test_dataset is not None:
        print(f"Test subgraphs:  {len(test_dataset)}")

    train_loader = build_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = build_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = build_loader(test_dataset, args.batch_size, False, args.num_workers) if test_dataset is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NMR_VAE.from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history = []
    best_val = float("inf")
    best_path = args.output_dir / "best_g2s_model.pt"
    last_path = args.output_dir / "last_g2s_model.pt"

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
            f"Train loss {train_loss['loss']:.5f}, MAE {train_loss['mae_ppm']:.3f} ppm | "
            f"Val loss {val_loss['loss']:.5f}, MAE {val_loss['mae_ppm']:.3f} ppm"
        )

        save_g2s_checkpoint(
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
            save_g2s_checkpoint(
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

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    eval_dir = args.output_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    metrics_by_split = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        data = predict_loader(model, loader, device)
        metrics = compute_g2s_metrics(data["pred"], data["true"], data["center_su"])
        metrics_by_split[split_name] = metrics
        save_prediction_points(data, eval_dir / f"{split_name}_points.csv")
        save_g2s_plot_data(data, eval_dir / "plot_data", prefix=split_name)
        plot_shift_regression(data["pred"], data["true"], eval_dir / f"{split_name}_regression.png", title=f"G2S {split_name}, k={args.k_hop}")
        plot_error_histogram(data["pred"], data["true"], eval_dir / f"{split_name}_error_hist.png", title=f"G2S {split_name} errors, k={args.k_hop}")
        plot_group_mae(metrics, eval_dir / f"{split_name}_group_mae.png", title=f"G2S {split_name} MAE by SU group")
    metrics_to_dataframe(metrics_by_split).to_csv(eval_dir / "metrics_summary.csv", index=False)
    save_json(eval_dir / "metrics_summary.json", metrics_by_split)
    print(f"[done] best checkpoint: {best_path}")
    print(f"[done] evaluation written to: {eval_dir}")


if __name__ == "__main__":
    main()
