"""Reproduce six evaluation panels from saved, unmodified model predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

try:
    from .plot import NatureStyleConfig, S2NFigure, summarize_s2n_errors
except ImportError:
    from plot import NatureStyleConfig, S2NFigure, summarize_s2n_errors


SCRIPT_DIR = Path(__file__).resolve().parent


def load_g2s_points(points_csv):
    """Keep every center and reject ambiguous, missing or invalid records."""
    points = pd.read_csv(points_csv)
    required = {'source_path', 'original_center_id', 'center_su', 'true_ppm', 'pred_ppm'}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f'Missing G2S columns: {sorted(missing)}')
    if points.empty or points[list(required)].isna().any().any():
        raise ValueError('G2S data must be nonempty and contain no missing values.')
    numeric = points[['original_center_id', 'center_su', 'true_ppm', 'pred_ppm']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError('G2S indices and shifts must be finite and nonnegative.')
    if not np.isin(points['center_su'], np.arange(26)).all():
        raise ValueError('G2S centers must be carbon-containing SU types 0–25.')
    center_indices = points['original_center_id'].to_numpy(dtype=float)
    if not np.equal(center_indices, np.floor(center_indices)).all():
        raise ValueError('G2S center indices must be integers.')
    if points['source_path'].astype(str).str.strip().eq('').any():
        raise ValueError('G2S source paths must not be empty.')
    if points.duplicated(['source_path', 'original_center_id']).any():
        raise ValueError('Duplicate G2S molecule/center records are not allowed.')
    return points


def select_example(s2n_points, g2s_points, sample_id='auto'):
    """Choose the common molecule nearest both median per-molecule MAE ranks."""
    s2n_working = s2n_points.assign(
        calculated_error=(s2n_points['pred_count'] - s2n_points['true_count']).abs()
    )
    g2s_working = g2s_points.assign(
        calculated_error=(g2s_points['pred_ppm'] - g2s_points['true_ppm']).abs()
    )
    candidates = s2n_working.groupby('path').agg(
        s2n_mae=('calculated_error', 'mean')
    ).join(g2s_working.groupby('source_path').agg(
        g2s_mae_ppm=('calculated_error', 'mean'),
        n_carbon_centers=('calculated_error', 'size'),
    ), how='inner').rename_axis('source_path').reset_index()
    if candidates.empty:
        raise ValueError('No common molecule source paths occur in the two prediction tables.')
    candidates['sample_id'] = candidates['source_path'].map(lambda value: Path(value).stem)
    candidates['s2n_mae_rank'] = candidates['s2n_mae'].rank(method='average', pct=True)
    candidates['g2s_mae_rank'] = candidates['g2s_mae_ppm'].rank(method='average', pct=True)
    candidates['median_rank_distance'] = (
        (candidates['s2n_mae_rank'] - 0.5).abs()
        + (candidates['g2s_mae_rank'] - 0.5).abs()
    )
    candidates = candidates.sort_values(
        ['median_rank_distance', 'source_path'], kind='stable'
    ).reset_index(drop=True)
    if sample_id == 'auto':
        chosen = candidates.iloc[0]
    else:
        matches = candidates[
            candidates['sample_id'].eq(str(sample_id))
            | candidates['source_path'].eq(str(sample_id))
        ]
        if len(matches) != 1:
            raise ValueError(
                f'Sample {sample_id!r} must uniquely occur in both evaluation tables; '
                'no substitute prediction will be fabricated. Use a complete source path if IDs are ambiguous.'
            )
        chosen = matches.iloc[0]
    candidates['selected'] = candidates['source_path'].eq(chosen['source_path'])
    su_example = s2n_points[s2n_points['path'].eq(chosen['source_path'])].sort_values('su_index').copy()
    shift_example = g2s_points[g2s_points['source_path'].eq(chosen['source_path'])].sort_values('original_center_id').copy()
    carbon_truth = su_example.set_index('su_index')['true_count'].reindex(range(26))
    labeled_carbons = shift_example['center_su'].value_counts().reindex(range(26), fill_value=0)
    if not np.array_equal(carbon_truth.to_numpy(), labeled_carbons.to_numpy()):
        raise ValueError('Selected molecule lacks complete, SU-consistent carbon-center labels; cannot show a complete reference spectrum.')
    return su_example, shift_example, candidates, chosen.to_dict()


def reconstruct_spectra(shift_points, hwhm=1.0, ppm_step=0.1):
    """Reconstruct both curves with identical unit weights and Lorentzian widths."""
    if not np.isfinite(hwhm) or hwhm <= 0:
        raise ValueError('HWHM must be positive and finite.')
    if not np.isfinite(ppm_step) or ppm_step <= 0:
        raise ValueError('PPM step must be positive and finite.')
    intervals = 240.0 / ppm_step
    if not np.isclose(intervals, round(intervals)) or intervals < 1:
        raise ValueError('PPM step must exactly divide the 0–240 ppm range.')
    reference_shifts = shift_points['true_ppm'].to_numpy(dtype=float)
    predicted_shifts = shift_points['pred_ppm'].to_numpy(dtype=float)
    if not reference_shifts.size or not np.isfinite([reference_shifts, predicted_shifts]).all():
        raise ValueError('Spectrum reconstruction requires nonempty, finite shift pairs.')
    if min(reference_shifts.min(), predicted_shifts.min()) < 0 or max(reference_shifts.max(), predicted_shifts.max()) > 240:
        raise ValueError('Shift outside the fixed 0–240 ppm plotting range; refusing to hide centers.')
    ppm_axis = np.linspace(0.0, 240.0, int(round(intervals)) + 1)
    reference = (hwhm ** 2 / ((ppm_axis[:, None] - reference_shifts) ** 2 + hwhm ** 2)).sum(axis=1)
    predicted = (hwhm ** 2 / ((ppm_axis[:, None] - predicted_shifts) ** 2 + hwhm ** 2)).sum(axis=1)
    common_scale = float(reference.max())
    spectrum = pd.DataFrame({
        'ppm': ppm_axis,
        'reference_reconstructed_raw': reference,
        'g2s_reconstructed_raw': predicted,
        'reference_relative_intensity': reference / common_scale,
        'g2s_relative_intensity': predicted / common_scale,
    })
    return spectrum, common_scale


def plot_su_example(axis, points, figmaker, sample_id):
    positions = points['su_index'].to_numpy()
    width = 0.40
    axis.bar(
        positions - width / 2, points['true_count'], width=width,
        color=figmaker.colors['blue'], edgecolor='black', linewidth=0.3,
        label='Reference',
    )
    axis.bar(
        positions + width / 2, points['pred_count'], width=width,
        color=figmaker.colors['orange'], edgecolor='black', linewidth=0.3,
        label='S2N',
    )
    axis.set_xlim(-0.8, 32.8)
    axis.set_xticks([0, 5, 10, 15, 20, 25, 32])
    axis.set_xlabel('SU type')
    axis.set_ylabel('SU count')
    maximum = float(points[['true_count', 'pred_count']].to_numpy().max())
    axis.set_ylim(0, max(1.0, maximum) * 1.4)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axis.legend(loc='upper right', frameon=False, fontsize=7, handlelength=1.1, labelspacing=0.25)
    axis.text(0.02, 0.95, f'Sample {sample_id}', transform=axis.transAxes, va='top', fontsize=7)


def plot_spectrum_example(axis, spectrum, figmaker, sample_id, shift_mae, filled=True):
    ppm_axis = spectrum['ppm'].to_numpy()
    reference = spectrum['reference_relative_intensity'].to_numpy()
    predicted = spectrum['g2s_relative_intensity'].to_numpy()
    if filled:
        axis.fill_between(ppm_axis, reference, color=figmaker.colors['blue'], alpha=0.18, linewidth=0)
        axis.fill_between(ppm_axis, predicted, color=figmaker.colors['orange'], alpha=0.24, linewidth=0)
    axis.plot(ppm_axis, reference, color=figmaker.colors['blue'], linewidth=0.9, label='Reference')
    axis.plot(ppm_axis, predicted, color=figmaker.colors['orange'], linewidth=0.9, linestyle='--', label='G2S')
    axis.set_xlim(240, 0)
    axis.set_ylim(0, max(reference.max(), predicted.max()) * 1.5)
    axis.set_xticks([240, 180, 120, 60, 0])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.set_xlabel('¹³C chemical shift (ppm)')
    axis.set_ylabel('Relative intensity')
    axis.legend(loc='upper left', frameon=False, fontsize=7, handlelength=1.6, labelspacing=0.25)
    axis.text(
        0.98, 0.95, f'Sample {sample_id}\nShift MAE = {shift_mae:.2f} ppm',
        transform=axis.transAxes, ha='right', va='top', fontsize=7,
    )


def create_composite(figmaker, s2n_points, group_errors, g2s_points, ablation,
                     su_example, spectrum, example_info, width_mm, height_mm, filled):
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = figure.add_gridspec(
        3, 2, left=0.10, right=0.89, bottom=0.07, top=0.97,
        hspace=0.68, wspace=0.62,
    )
    axes = [figure.add_subplot(grid[row_index, column_index])
            for row_index in range(3) for column_index in range(2)]
    truth = s2n_points['true_count'].to_numpy()
    predicted = s2n_points['pred_count'].to_numpy()
    figmaker._plot_regression(
        axes[0], truth, predicted, figmaker._compute_r2(truth, predicted),
        np.abs(truth - predicted).mean(),
    )
    truth_shifts = g2s_points['true_ppm'].to_numpy()
    predicted_shifts = g2s_points['pred_ppm'].to_numpy()
    figmaker._plot_regression(
        axes[1], truth_shifts, predicted_shifts,
        figmaker._compute_r2(truth_shifts, predicted_shifts),
        np.abs(truth_shifts - predicted_shifts).mean(),
        x_label='True chemical shift (ppm)', y_label='Predicted chemical shift (ppm)',
        mae_suffix=' ppm',
    )
    figmaker._plot_group_errors(axes[2], group_errors)
    figmaker._plot_hop_ablation(axes[3], ablation)
    plot_su_example(axes[4], su_example, figmaker, example_info['sample_id'])
    plot_spectrum_example(
        axes[5], spectrum, figmaker, example_info['sample_id'],
        example_info['g2s_mae_ppm'], filled=filled,
    )
    figure.canvas.draw()
    for axis, panel_label in zip(axes, 'abcdef'):
        bounds = axis.get_subplotspec().get_position(figure)
        figure.text(
            bounds.x0 - 0.055, bounds.y1 + 0.013, panel_label,
            fontsize=figmaker.config.FONT_SIZE_PANEL, fontweight='bold', va='bottom',
        )
    return figure


def file_record(source_path):
    source_path = Path(source_path).resolve()
    return {
        'path': str(source_path),
        'sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def write_source_data(output_dir, s2n_points, group_errors, per_su_errors,
                      g2s_points, ablation, su_example, shift_example,
                      spectrum, candidates, metadata):
    source_dir = output_dir / 'source_data'
    source_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        'S2N_regression_points.csv': s2n_points,
        'S2N_group_per_su_mae.csv': group_errors,
        'S2N_per_su_mae.csv': per_su_errors,
        'G2S_shift_points.csv': g2s_points,
        'G2S_hop_ablation.csv': ablation,
        'S2N_example_counts.csv': su_example,
        'G2S_example_shifts.csv': shift_example.assign(reconstruction_weight=1.0),
        'G2S_example_spectrum.csv': spectrum,
        'example_selection.csv': candidates,
    }
    for filename, table in tables.items():
        table.to_csv(source_dir / filename, index=False)
    (output_dir / 'figure_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8',
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s2n-points', type=Path, default=SCRIPT_DIR / 'checkpoints_s2n_v1/eval/plot_data/test_regression_points.csv')
    parser.add_argument('--g2s-points', type=Path, default=SCRIPT_DIR / 'checkpoints_g2s_ablation/k3/eval/test_points.csv')
    parser.add_argument('--g2s-ablation', type=Path, default=SCRIPT_DIR / 'checkpoints_g2s_ablation/hop_ablation_key_metrics.csv')
    parser.add_argument('--out-dir', type=Path, default=SCRIPT_DIR / 'output_figures_revised')
    parser.add_argument('--sample-id', default='auto', help='Common molecule ID/full source path; auto uses a deterministic median-rank rule.')
    parser.add_argument('--hwhm', type=float, default=1.0, help='Same Lorentzian HWHM for reference and G2S curves, in ppm.')
    parser.add_argument('--ppm-step', type=float, default=0.1)
    parser.add_argument('--curve-style', choices=['filled', 'lines'], default='filled')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--font', default='Arial')
    parser.add_argument('--width-mm', type=float, default=180.0)
    parser.add_argument('--height-mm', type=float, default=198.0)
    parser.add_argument('--example-width-mm', type=float, default=88.0)
    parser.add_argument('--example-height-mm', type=float, default=58.0)
    parser.add_argument('--preview-dpi', type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    dimensions = [args.width_mm, args.height_mm, args.example_width_mm, args.example_height_mm]
    if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
        raise ValueError('Figure dimensions must be positive and finite.')
    if args.dpi < 300 or args.preview_dpi <= 0:
        raise ValueError('Use at least 300 dpi for exports and a positive preview DPI.')
    font_path = font_manager.findfont(args.font, fallback_to_default=False)
    config = NatureStyleConfig()
    config.DPI = args.dpi
    config.FONT_NAME = args.font
    figmaker = S2NFigure(config)
    s2n_points = pd.read_csv(args.s2n_points)
    group_errors, per_su_errors = summarize_s2n_errors(s2n_points)
    g2s_points = load_g2s_points(args.g2s_points)
    ablation = figmaker.load_g2s_ablation_data(args.g2s_ablation)
    required_ablation = ['k_hop', 'n', 'mae', 'rmse']
    if ablation.empty or not np.isfinite(ablation[required_ablation].to_numpy(dtype=float)).all():
        raise ValueError('Ablation metrics and sample counts must be nonempty and finite.')
    if (ablation[required_ablation] < 0).any().any() or ablation['k_hop'].duplicated().any():
        raise ValueError('Ablation metrics must be nonnegative and hop radii unique.')
    if ablation['n'].nunique() != 1 or not ablation['n'].eq(len(g2s_points)).all():
        raise ValueError('Ablation rows and regression points must use the same evaluation sample count.')
    su_example, shift_example, candidates, example_info = select_example(s2n_points, g2s_points, args.sample_id)
    spectrum, common_scale = reconstruct_spectra(shift_example, args.hwhm, args.ppm_step)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'purpose': 'Global held-out errors and one matched, illustrative prediction example for S2N and G2S.',
        'archetype': 'quantitative grid',
        'inputs': {
            's2n_points': file_record(args.s2n_points),
            'g2s_points': file_record(args.g2s_points),
            'g2s_ablation': file_record(args.g2s_ablation),
        },
        'all_rows_retained': True,
        's2n_n_molecules': int(s2n_points['path'].nunique()),
        's2n_n_molecule_su_pairs': len(s2n_points),
        's2n_global_mae': float((s2n_points['pred_count'] - s2n_points['true_count']).abs().mean()),
        's2n_r2': figmaker._compute_r2(s2n_points['true_count'].to_numpy(), s2n_points['pred_count'].to_numpy()),
        's2n_group_metric': 'mean_{molecule,SU in group} |predicted count - true count|; zero counts included',
        'g2s_n_molecules': int(g2s_points['source_path'].nunique()),
        'g2s_n_carbon_centers': len(g2s_points),
        'g2s_global_shift_mae_ppm': float((g2s_points['pred_ppm'] - g2s_points['true_ppm']).abs().mean()),
        'g2s_r2': figmaker._compute_r2(g2s_points['true_ppm'].to_numpy(), g2s_points['pred_ppm'].to_numpy()),
        'example': example_info,
        'example_selection': {
            'requested': args.sample_id,
            'n_common_molecules': len(candidates),
            'rule': 'Minimize |S2N per-molecule MAE percentile rank - 0.5| + |G2S per-molecule shift MAE percentile rank - 0.5|; ties by source path.' if args.sample_id == 'auto' else 'Explicitly chosen by molecule ID/source path.',
            'scope': 'Common test records only; not a representative estimate of performance across all molecular sizes.',
        },
        'spectrum': {
            'reference_kind': 'Reconstructed from saved reference carbon shifts; not a measured experimental spectrum.',
            'prediction_kind': 'Saved G2S shift predictions; amplitudes are prescribed, not model-predicted.',
            'weights': 'One unit per labeled carbon center in both curves.',
            'line_shape': 'sum hwhm^2 / ((ppm - shift)^2 + hwhm^2)',
            'hwhm_ppm': args.hwhm,
            'ppm_range': [0.0, 240.0],
            'ppm_step': args.ppm_step,
            'normalization': 'Both curves divided by the same reference maximum; no separate peak normalization.',
            'shared_intensity_divisor': common_scale,
            'additional_smoothing_or_alignment': None,
        },
        'style': {
            'font': args.font, 'font_file': font_path,
            'body_font_pt': 7, 'panel_label_pt': 8,
            'dpi': args.dpi, 'composite_mm': [args.width_mm, args.height_mm],
            'example_mm': [args.example_width_mm, args.example_height_mm],
            'density_bins': 70, 'density_color_limit': 'full observed bin-count range; no percentile saturation',
            'fixed_canvas_no_tight_crop': True,
            'curve_style': args.curve_style,
        },
        'reporting_limits': [
            'Split labels are taken from the supplied evaluation exports; structural leakage was not re-audited.',
            'No multi-seed or multi-fold variability is supplied; no uncertainty bands are invented.',
            'S2N predicts counts, not a unique molecular graph; the G2S example uses its own reference graph.',
        ],
    }
    write_source_data(
        args.out_dir, s2n_points, group_errors, per_su_errors, g2s_points,
        ablation, su_example, shift_example, spectrum, candidates, metadata,
    )
    figures = []
    regression, _ = figmaker.create_regression_figure(
        s2n_points['true_count'].to_numpy(), s2n_points['pred_count'].to_numpy(),
        metadata['s2n_r2'], metadata['s2n_global_mae'],
    )
    figures.append(('S2N_regression', regression))
    group_figure, _ = figmaker.create_group_error_figure(group_errors)
    figures.append(('S2N_group_per_su_mae', group_figure))
    g2s_regression, _ = figmaker.create_g2s_regression_figure(
        g2s_points['true_ppm'].to_numpy(), g2s_points['pred_ppm'].to_numpy(),
        metadata['g2s_r2'], metadata['g2s_global_shift_mae_ppm'],
    )
    figures.append(('G2S_test_regression', g2s_regression))
    ablation_figure, _ = figmaker.create_g2s_ablation_figure(ablation)
    figures.append(('G2S_hop_ablation', ablation_figure))
    su_figure, su_axis = figmaker._create_canvas(
        args.example_width_mm, args.example_height_mm,
        left=0.16, bottom=0.21, right=0.97, top=0.96,
    )
    plot_su_example(su_axis, su_example, figmaker, example_info['sample_id'])
    figures.append((f'S2N_prediction_example_{example_info["sample_id"]}', su_figure))
    spectrum_figure, spectrum_axis = figmaker._create_canvas(
        args.example_width_mm, args.example_height_mm,
        left=0.17, bottom=0.21, right=0.97, top=0.96,
    )
    plot_spectrum_example(
        spectrum_axis, spectrum, figmaker, example_info['sample_id'],
        example_info['g2s_mae_ppm'], filled=args.curve_style == 'filled',
    )
    figures.append((f'G2S_spectrum_example_{example_info["sample_id"]}', spectrum_figure))
    composite = create_composite(
        figmaker, s2n_points, group_errors, g2s_points, ablation, su_example,
        spectrum, example_info, args.width_mm, args.height_mm,
        filled=args.curve_style == 'filled',
    )
    figures.append(('Fig3_forward_models_complete', composite))
    for basename, figure in figures:
        figmaker.save_figure(figure, args.out_dir, basename, dpi=args.dpi)
    composite.savefig(args.out_dir / 'Fig3_preview.png', dpi=args.preview_dpi, facecolor='white')
    for _, figure in figures:
        plt.close(figure)
    print(f'Example: {example_info["sample_id"]}; all 33 SU types and all carbon centers retained.')
    print(f'Figure bundle: {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()
