"""Export an optional larger S2N example without replacing the approved molecule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

try:
    from .plot import NatureStyleConfig, S2NFigure, validate_s2n_points
    from .molecule_structure import load_structure_manifest, verify_example_structure
    from .plot_molecule_panels import create_atomic_figure
    from .plot_predictions import file_record, load_g2s_points, plot_su_example, select_example
except ImportError:
    from plot import NatureStyleConfig, S2NFigure, validate_s2n_points
    from molecule_structure import load_structure_manifest, verify_example_structure
    from plot_molecule_panels import create_atomic_figure
    from plot_predictions import file_record, load_g2s_points, plot_su_example, select_example


SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-id', default='1588')
    parser.add_argument('--structure-manifest', type=Path, default=SCRIPT_DIR / 'larger_example_structures.json')
    parser.add_argument('--s2n-points', type=Path, default=SCRIPT_DIR / 'checkpoints_s2n_v1/eval/plot_data/test_regression_points.csv')
    parser.add_argument('--g2s-points', type=Path, default=SCRIPT_DIR / 'checkpoints_g2s_ablation/k3/eval/test_points.csv')
    parser.add_argument('--out-dir', type=Path, default=SCRIPT_DIR / 'output_s2n_alternative')
    parser.add_argument('--font', default='Arial')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    if args.sample_id == 'auto' or args.dpi < 300:
        raise ValueError('Select a molecule explicitly and export at least 300 dpi.')
    font_manager.findfont(args.font, fallback_to_default=False)
    config = NatureStyleConfig()
    config.FONT_NAME, config.DPI = args.font, args.dpi
    figmaker = S2NFigure(config)
    points = pd.read_csv(args.s2n_points)
    validate_s2n_points(points)
    shifts = load_g2s_points(args.g2s_points)
    structures = load_structure_manifest(args.structure_manifest)
    counts, shift_example, _, info = select_example(points, shifts, args.sample_id, structures)
    entry = structures[info['source_path']]
    structure = verify_example_structure(entry, counts, shift_example)
    structure_figure = create_atomic_figure(entry['smiles'], explicit_hydrogens=False)
    count_figure, axis = figmaker._create_canvas(180.0, 58.0, 0.08, 0.28, 0.985, 0.96)
    plot_su_example(axis, counts, figmaker)
    figures = {'Molecular_structure': structure_figure, 'S2N_count_comparison': count_figure}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.out_dir / 'source_data'
    source_dir.mkdir(exist_ok=True)
    counts.to_csv(source_dir / 'S2N_counts.csv', index=False)
    (source_dir / 'molecule.smi').write_text(entry['smiles'] + '\n', encoding='utf-8')
    metadata = {
        'purpose': 'Optional larger single-molecule S2N illustration, not a replacement for the approved default.',
        'selection': 'Reference size/type screening before inspecting prediction errors; not a best-fit example.',
        'structure': structure,
        'reference_total_su': float(counts['true_count'].sum()),
        'reference_present_types': int(counts['true_count'].gt(0).sum()),
        'max_reference_type_count': float(counts['true_count'].max()),
        's2n_mae_all_33_types': float((counts['pred_count'] - counts['true_count']).abs().mean()),
        'structure_display': 'Conventional skeletal formula with implicit hydrogens; stereochemical information only as supplied in source SMILES.',
        'sample_identifiers_drawn': False,
        'font': args.font,
        'export_dpi': args.dpi,
        'dimensions_mm': {name: (figure.get_size_inches() * 25.4).tolist() for name, figure in figures.items()},
        'inputs': {'structure_manifest': file_record(args.structure_manifest),
                   's2n_points': file_record(args.s2n_points), 'g2s_points': file_record(args.g2s_points)},
        'limitations': ['This is still a single molecule, not a representative high-count merged sample.',
                        'No claim that prediction quality is better; all 33 raw predictions remain visible.'],
    }
    (args.out_dir / 'figure_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8',
    )
    for name, figure in figures.items():
        figmaker.save_figure(figure, args.out_dir, name)
        figure.savefig(args.out_dir / f'{name}_preview.png', dpi=180, facecolor='white')
        plt.close(figure)


if __name__ == '__main__':
    main()
