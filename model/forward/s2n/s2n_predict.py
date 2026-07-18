"""Predict SU inventories with a trained S2N checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .s2n_model import S2NDataset, apply_element_feasibility_mask, extract_spectrum_stats
    from .s2n_utils import (
        compute_s2n_metrics,
        load_s2n_checkpoint,
        plot_su_bar,
        predict_loader,
        save_json,
        save_s2n_plot_data,
        save_prediction_table,
    )
    from ...shared.inverse_common import normalize_spectrum_to_carbon_count, resample_spectrum_to_ppm_axis
except ImportError:
    from model.forward.s2n.s2n_model import S2NDataset, apply_element_feasibility_mask, extract_spectrum_stats
    from model.forward.s2n.s2n_utils import (
        compute_s2n_metrics,
        load_s2n_checkpoint,
        plot_su_bar,
        predict_loader,
        save_json,
        save_s2n_plot_data,
        save_prediction_table,
    )
    from model.shared.inverse_common import normalize_spectrum_to_carbon_count, resample_spectrum_to_ppm_axis


def parse_elements(expr: str) -> torch.Tensor:
    matches = dict(re.findall(r"([CHONSX])\s*=\s*([0-9]+(?:\.[0-9]+)?)", expr.upper()))
    return torch.tensor([float(matches.get(sym, 0.0)) for sym in ["C", "H", "O", "N", "S", "X"]], dtype=torch.float)


def predict_from_csv(model_path: Path,
                     spectrum_csv: Path,
                     elements_str: str,
                     output_name: str,
                     output_dir: Path,
                     apply_mask: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _metadata = load_s2n_checkpoint(model_path, device=device)

    df_spec = pd.read_csv(spectrum_csv, sep=r"[;, \t]", engine="python", header=None)
    if df_spec.shape[1] < 2:
        raise ValueError("Spectrum CSV requires at least two columns: ppm, intensity")
    spectrum = resample_spectrum_to_ppm_axis(df_spec.iloc[:, 0].values, df_spec.iloc[:, 1].values)
    elements = parse_elements(elements_str)
    spectrum = normalize_spectrum_to_carbon_count(spectrum, float(elements[0].item()))
    spectrum_stats = extract_spectrum_stats(spectrum.unsqueeze(0), mode="regions_v1").squeeze(0).cpu()

    with torch.no_grad():
        pred = model.infer_su_hist(
            spectrum.unsqueeze(0).to(device),
            elements.unsqueeze(0).to(device),
            apply_mask=False,
        ).squeeze(0).cpu()
    if apply_mask:
        pred = apply_element_feasibility_mask(pred, elements)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_table(pred, None, output_dir / f"{output_name}_pred_su_hist.csv")
    plot_su_bar(pred, output_dir / f"{output_name}_pred_su_hist.png", title="Predicted Structural Unit Distribution")
    save_json(output_dir / f"{output_name}_config.json", config.to_dict())
    stat_names = [
        "frac_0_25", "frac_25_50", "frac_50_70", "frac_70_90",
        "frac_90_110", "frac_110_130", "frac_130_160",
        "frac_160_190", "frac_190_240", "frac_ordinary_0_50",
        "frac_oxygenated_50_90", "frac_aliphatic_0_90",
        "frac_aromatic_90_160", "frac_carbonyl_160_240",
        "log_area_scaled", "aromatic_minus_aliphatic",
    ]
    pd.DataFrame([{
        "feature": name,
        "value": float(spectrum_stats[i].item()),
    } for i, name in enumerate(stat_names)]).to_csv(output_dir / f"{output_name}_spectrum_stats.csv", index=False)

    df = pd.read_csv(output_dir / f"{output_name}_pred_su_hist.csv")
    nonzero = df[df["Predicted_Rounded"] > 0]
    print("\n--- S2N prediction ---")
    print(f"model: {model_path}")
    print(f"spectrum: {spectrum_csv}")
    print(f"elements: {elements.tolist()}")
    print(f"spectrum_stats_csv: {output_dir / f'{output_name}_spectrum_stats.csv'}")
    print(nonzero[["SU_Index", "SU_Name", "Predicted_Rounded"]].to_string(index=False))
    print(f"[done] results written to {output_dir}")


def validate_pt_inputs(model_path: Path,
                       input_path: Path,
                       output_dir: Path,
                       num_samples: int,
                       apply_mask: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _metadata = load_s2n_checkpoint(model_path, device=device)
    if input_path.is_dir():
        files = sorted(str(p) for p in input_path.glob("*.pt"))
    elif input_path.is_file():
        files = [str(input_path)]
    else:
        raise FileNotFoundError(f"Invalid input path: {input_path}")
    if not files:
        raise FileNotFoundError(f"No .pt files found in {input_path}")
    if num_samples > 0 and len(files) > num_samples:
        files = files[:num_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(S2NDataset(files), batch_size=16, shuffle=False)
    result = predict_loader(model, loader, device, apply_mask=apply_mask)
    metrics = compute_s2n_metrics(result["pred"], result["true"], result["elements"])
    save_json(output_dir / "metrics.json", metrics)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    save_s2n_plot_data(result, output_dir / "plot_data", prefix="validation")

    for i, path in enumerate(result["paths"]):
        stem = Path(path).stem
        save_prediction_table(result["pred"][i], result["true"][i], output_dir / f"{stem}_prediction.csv")
        plot_su_bar(
            result["pred"][i],
            output_dir / f"{stem}_prediction.png",
            true=result["true"][i],
            title=f"S2N validation: {stem}",
        )

    save_json(output_dir / "checkpoint_config.json", config.to_dict())
    print("\n--- S2N validation ---")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"[done] results written to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser("S2N prediction / quick validation")
    parser.add_argument("--model_path", "--checkpoint", dest="model_path", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spectrum_csv", type=Path)
    mode.add_argument("--input_path", type=Path)
    parser.add_argument("--elements", type=str, help="e.g. 'C=230,H=200,O=30,N=1,S=1,X=0'")
    parser.add_argument("--output_name", type=str, default="prediction")
    parser.add_argument("--output_dir", type=Path, default=Path("s2n_prediction_results"))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--apply_mask", action="store_true", help="Apply element-feasibility mask to predictions.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.spectrum_csv:
        if not args.elements:
            raise SystemExit("--elements is required with --spectrum_csv")
        predict_from_csv(
            args.model_path,
            args.spectrum_csv,
            args.elements,
            args.output_name,
            args.output_dir,
            apply_mask=args.apply_mask,
        )
    else:
        validate_pt_inputs(
            args.model_path,
            args.input_path,
            args.output_dir,
            args.num_samples,
            apply_mask=args.apply_mask,
        )


if __name__ == "__main__":
    main()
