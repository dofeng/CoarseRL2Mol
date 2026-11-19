"""Draw an identified molecule and its supplied spectrum, without inventing predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

try:
    from .plot import NatureStyleConfig, S2NFigure
    from .molecule_structure import draw_molecule
except ImportError:
    from plot import NatureStyleConfig, S2NFigure
    from molecule_structure import draw_molecule


SCRIPT_DIR = Path(__file__).resolve().parent


def load_reference(smiles_path, spectrum_path):
    """Preserve every supplied observation and reject ambiguous inputs."""
    smiles_path, spectrum_path = Path(smiles_path), Path(spectrum_path)
    if smiles_path.stem != spectrum_path.stem:
        raise ValueError('SMILES and spectrum filenames must identify the same sample.')
    smiles = smiles_path.read_text(encoding='utf-8').strip()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('The supplied SMILES cannot be parsed.')
    spectrum = pd.read_csv(spectrum_path, sep=None, engine='python')
    if list(spectrum.columns) != ['chemical_shift', 'intensity']:
        raise ValueError('Expected exactly chemical_shift,intensity; no columns are silently ignored.')
    if len(spectrum) < 2 or not np.isfinite(spectrum.to_numpy(dtype=float)).all():
        raise ValueError('The spectrum must contain at least two finite observations.')
    if not np.all(np.diff(spectrum['chemical_shift'].to_numpy(dtype=float)) > 0):
        raise ValueError('The supplied chemical-shift axis must be strictly increasing.')
    if (spectrum['intensity'] < 0).any() or spectrum['intensity'].max() <= 0:
        raise ValueError('Reference intensities must be nonnegative with a positive maximum.')
    structure = {
        'sample_id': smiles_path.stem,
        'smiles': smiles,
        'canonical_smiles': Chem.MolToSmiles(molecule),
        'molecular_formula': rdMolDescriptors.CalcMolFormula(molecule),
    }
    return structure, spectrum


def audit_su_coverage(smiles):
    """Inspect the existing assignment rules; never relabel unsupported atoms."""
    project_root = str(SCRIPT_DIR.parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.shared.coarse_graph import SU_DEFS, _assign_atoms_to_su

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Cannot audit invalid SMILES.')
    molecule = Chem.AddHs(molecule)
    matches, atom_assignments = _assign_atoms_to_su(molecule)
    heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1]
    unsupported = [atom for atom in heavy_atoms if atom_assignments[atom.GetIdx()] == -1]
    return {
        'vocabulary_size': len(SU_DEFS),
        'heavy_atom_count': len(heavy_atoms),
        'assigned_heavy_atom_count': len(heavy_atoms) - len(unsupported),
        'matched_su_count': len(matches),
        'all_heavy_atoms_assigned': not unsupported,
        'unassigned_atoms': [
            {
                'atom_index': atom.GetIdx(),
                'element': atom.GetSymbol(),
                'bonds': [
                    {
                        'neighbor_index': neighbor.GetIdx(),
                        'neighbor_element': neighbor.GetSymbol(),
                        'bond_type': str(molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()),
                    }
                    for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() != 1
                ],
            }
            for atom in unsupported
        ],
        'scope': 'Atom-assignment coverage only; not a complete model-input validation.',
    }


def create_reference_figure(figmaker, structure, spectrum, width_mm=180.0,
                            height_mm=68.0, filled=True):
    """Show only the supplied structure and curve, with no predicted or experimental label."""
    dimensions = [width_mm, height_mm]
    if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
        raise ValueError('Figure dimensions must be positive and finite.')
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    molecule_axis = figure.add_axes([0.035, 0.12, 0.40, 0.78])
    spectrum_axis = figure.add_axes([0.55, 0.23, 0.42, 0.64])
    draw_molecule(molecule_axis, structure['smiles'], font_size=figmaker.config.FONT_SIZE_AXIS)
    ppm_axis = spectrum['chemical_shift'].to_numpy(dtype=float)
    intensities = spectrum['intensity'].to_numpy(dtype=float)
    blue = figmaker.colors['blue']
    if filled:
        spectrum_axis.fill_between(ppm_axis, intensities, 0.0, color=blue, alpha=0.20, linewidth=0)
    spectrum_axis.plot(ppm_axis, intensities, color=blue, linewidth=0.9)
    spectrum_axis.set_xlim(ppm_axis[-1], ppm_axis[0])
    spectrum_axis.set_ylim(0.0, intensities.max() * 1.12)
    spectrum_axis.set_xlabel('Chemical shift (ppm)')
    spectrum_axis.set_ylabel('Reference intensity (a.u.)')
    figure.text(0.035, 0.95, 'a', fontsize=8, fontweight='bold')
    figure.text(0.50, 0.95, 'b', fontsize=8, fontweight='bold')
    figure.text(0.235, 0.06, f'Sample {structure["sample_id"]}', ha='center', fontsize=7)
    return figure


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smiles', type=Path, default=SCRIPT_DIR.parent / 'NMR示意/51121.txt')
    parser.add_argument('--spectrum', type=Path, default=SCRIPT_DIR.parent / 'NMR示意/51121.csv')
    parser.add_argument('--out-dir', type=Path, default=SCRIPT_DIR / 'output_reference_51121')
    parser.add_argument('--curve-style', choices=['filled', 'lines'], default='filled')
    parser.add_argument('--font', default='Arial')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--width-mm', type=float, default=180.0)
    parser.add_argument('--height-mm', type=float, default=68.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dpi < 300:
        raise ValueError('Use at least 300 dpi for manuscript exports.')
    font_path = font_manager.findfont(args.font, fallback_to_default=False)
    structure, spectrum = load_reference(args.smiles, args.spectrum)
    coverage = audit_su_coverage(structure['smiles'])
    config = NatureStyleConfig()
    config.FONT_NAME, config.DPI = args.font, args.dpi
    figmaker = S2NFigure(config)
    figure = create_reference_figure(
        figmaker, structure, spectrum, args.width_mm, args.height_mm, args.curve_style == 'filled',
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.out_dir / 'source_data'
    source_dir.mkdir(exist_ok=True)
    spectrum.to_csv(source_dir / 'reference_spectrum.csv', index=False)
    (source_dir / 'molecule.smi').write_text(
        f'{structure["smiles"]}\t{structure["sample_id"]}\n', encoding='utf-8',
    )
    metadata = {
        'core_conclusion': 'An identified molecular structure is paired with its supplied reference spectrum; no model-performance claim.',
        'archetype': 'asymmetric mixed-modality figure',
        'structure': structure,
        'inputs': {'smiles': file_record(args.smiles), 'spectrum': file_record(args.spectrum)},
        'spectrum_rows_input': len(spectrum),
        'spectrum_rows_plotted': len(spectrum),
        'chemical_shift_range_ppm': [float(spectrum['chemical_shift'].min()), float(spectrum['chemical_shift'].max())],
        'transformations': 'None: no smoothing, resampling, baseline correction, normalization or excluded observations.',
        'reference_origin': 'Unconfirmed: not labelled as experimental.',
        'prediction_status': 'Not generated: this figure contains only the supplied reference curve.',
        'model_evaluation_status': 'Not a test-set performance example; no MAE or held-out claim.',
        'su_coverage': coverage,
        'dimensions_mm': [args.width_mm, args.height_mm],
        'font': {'name': args.font, 'file': font_path, 'size_pt': config.FONT_SIZE_AXIS},
        'export_dpi': args.dpi,
        'curve_style': args.curve_style,
    }
    (args.out_dir / 'figure_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8',
    )
    figmaker.save_figure(figure, args.out_dir, f'Reference_structure_spectrum_{structure["sample_id"]}')
    figure.savefig(args.out_dir / 'reference_preview.png', dpi=180, facecolor='white')
    plt.close(figure)
    print('SU coverage:', json.dumps(coverage, ensure_ascii=False))
    print('Reference-only figure: no S2N or G2S prediction has been substituted or fabricated.')


if __name__ == '__main__':
    main()
