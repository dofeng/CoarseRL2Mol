"""Export four independent, unnumbered panels for one verified molecular example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor

try:
    from .plot import NatureStyleConfig, S2NFigure, validate_s2n_points
    from .molecule_structure import draw_molecule, load_structure_manifest, verify_example_structure
    from .plot_predictions import file_record, integerize_su_counts, load_g2s_points, plot_spectrum_example, plot_su_example, reconstruct_spectra, select_example, select_su_example_types
except ImportError:
    from plot import NatureStyleConfig, S2NFigure, validate_s2n_points
    from molecule_structure import draw_molecule, load_structure_manifest, verify_example_structure
    from plot_predictions import file_record, integerize_su_counts, load_g2s_points, plot_spectrum_example, plot_su_example, reconstruct_spectra, select_example, select_su_example_types


SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_NODE_RADIUS_PT = 7.0


def model_drawing_module():
    project_root = str(SCRIPT_DIR.parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.shared import coarse_graph

    return coarse_graph


def reference_drawing_module():
    project_root = str(SCRIPT_DIR.parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from RL_MTCS.io import output

    return output


def create_atomic_figure(smiles, width_mm=88.0, height_mm=88.0, explicit_hydrogens=True):
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    axis = figure.add_axes([0.035, 0.035, 0.93, 0.93])
    draw_molecule(
        axis, smiles, explicit_hydrogens=explicit_hydrogens,
        label_carbons=explicit_hydrogens,
    )
    return figure


def orient_coarse_positions(positions):
    """Rotate the unchanged model layout onto its long axis without reflecting it."""
    node_ids = sorted(positions)
    coordinates = np.asarray([positions[node_id] for node_id in node_ids], dtype=float)
    centered = coordinates - coordinates.mean(axis=0)
    if len(node_ids) < 2 or not np.any(centered):
        return dict(positions), 0.0
    _, _, principal_axes = np.linalg.svd(centered, full_matrices=False)
    direction = principal_axes[0]
    rotation = -float(np.degrees(np.arctan2(direction[1], direction[0])))
    oriented = model_drawing_module()._rotate_positions(positions, rotation)
    if oriented[node_ids[0]][0] > np.mean([point[0] for point in oriented.values()]):
        rotation += 180.0
        oriented = model_drawing_module()._rotate_positions(positions, rotation)
    return oriented, rotation


def regular_coarse_positions(node_count, edge_list, edge_length):
    """Depict the actual contracted topology with regular ring and branch coordinates."""
    topology = Chem.RWMol()
    for node_index in range(node_count):
        atom = Chem.Atom(0)
        atom.SetNoImplicit(True)
        topology.AddAtom(atom)
    edges = sorted({tuple(sorted(edge)) for edge in edge_list})
    for source, target in edges:
        topology.AddBond(source, target, Chem.BondType.SINGLE)
    molecule = topology.GetMol()
    Chem.GetSymmSSSR(molecule)
    rdDepictor.Compute2DCoords(molecule, bondLength=float(edge_length), forceRDKit=True)
    conformer = molecule.GetConformer()
    positions = {
        node_index: (float(conformer.GetAtomPosition(node_index).x),
                     float(conformer.GetAtomPosition(node_index).y))
        for node_index in range(node_count)
    }
    return positions, [list(ring) for ring in molecule.GetRingInfo().AtomRings()]


def create_coarse_figure(smiles, width_mm=88.0, height_mm=58.0, show_su_ids=True,
                         layout='regular'):
    """Preserve model topology and styling while independently laying out its nodes."""
    if layout not in {'regular', 'horizontal', 'model'}:
        raise ValueError('Coarse layout must be regular, horizontal or model.')
    model = model_drawing_module()
    reference = reference_drawing_module()
    graph, matches = model.smiles_to_coarse_grained_graph(
        smiles, min_su_nodes=1, save_visualization=False,
    )
    edge_list = graph.edge_index.t().tolist()
    presentation_rotation = 0.0
    rings = []
    if layout == 'regular':
        positions, rings = regular_coarse_positions(len(matches), edge_list, model.EDGE_LENGTH)
        label_rotation = 0.0
    else:
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.Compute2DCoords(molecule)
        rdkit_positions = model._compute_rdkit_su_positions(molecule, matches)
        positions = model._layout_on_hex_grid(
            edge_list, rdkit_positions, len(matches), mol=molecule, su_matches=matches,
        )
        positions = model._rotate_positions(positions, model.PLOT_ROTATION_DEGREES)
        label_rotation = model.LABEL_ROTATION_DEGREES
        if layout == 'horizontal':
            positions, presentation_rotation = orient_coarse_positions(positions)
            label_rotation = 0.0
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    axis = figure.add_axes([0.02, 0.02, 0.96, 0.96])
    coordinates = np.asarray([positions[index] for index in sorted(positions)], dtype=float)
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    node_radius_data = reference.REFERENCE_NODE_RADIUS
    axis.set_aspect('equal')
    for _ in range(32):
        margin = 3.0 * node_radius_data
        axis.set_xlim(x_min - margin, x_max + margin)
        axis.set_ylim(y_min - margin, y_max + margin)
        figure.canvas.draw()
        origin = axis.transData.transform((0.0, 0.0))
        x_scale_pt = np.linalg.norm(axis.transData.transform((1.0, 0.0)) - origin) * 72.0 / figure.dpi
        y_scale_pt = np.linalg.norm(axis.transData.transform((0.0, 1.0)) - origin) * 72.0 / figure.dpi
        next_radius_data = PANEL_NODE_RADIUS_PT / min(x_scale_pt, y_scale_pt)
        if np.isclose(next_radius_data, node_radius_data, rtol=1e-10, atol=1e-12):
            node_radius_data = next_radius_data
            break
        node_radius_data = next_radius_data
    margin = 3.0 * node_radius_data
    axis.set_xlim(x_min - margin, x_max + margin)
    axis.set_ylim(y_min - margin, y_max + margin)
    figure.canvas.draw()
    origin = axis.transData.transform((0.0, 0.0))
    points_per_unit = min(
        np.linalg.norm(axis.transData.transform((1.0, 0.0)) - origin),
        np.linalg.norm(axis.transData.transform((0.0, 1.0)) - origin),
    ) * 72.0 / figure.dpi
    node_radius_data = PANEL_NODE_RADIUS_PT / points_per_unit
    pairwise_distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    np.fill_diagonal(pairwise_distances, np.inf)
    if pairwise_distances.min() <= 2.0 * node_radius_data:
        raise ValueError('The fixed-size coarse nodes overlap at the requested panel dimensions.')
    edges = []
    observed_edges = set()
    for edge_index, (source, target) in enumerate(edge_list):
        key = tuple(sorted((source, target)))
        if key in observed_edges:
            continue
        observed_edges.add(key)
        start = np.asarray(positions[source], dtype=float)
        end = np.asarray(positions[target], dtype=float)
        displacement = end - start
        length = np.linalg.norm(displacement)
        if length <= 2 * node_radius_data:
            raise ValueError('Coarse nodes overlap; refusing to hide an edge underneath them.')
        offset = displacement / length * node_radius_data
        start, end = start + offset, end - offset
        axis.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=model.EDGE_COLOR, linewidth=model.EDGE_LW,
            zorder=1, solid_capstyle='round',
        )
        edges.append({
            'source': source, 'target': target,
            'bond_order': float(graph.edge_attr[edge_index, 0]),
            'ring_flag': float(graph.edge_attr[edge_index, 1]),
        })
    nodes = []
    for node_index, match in enumerate(matches):
        su_type = model.SU_NAME_TO_ID[match['name']]
        horizontal, vertical = positions[node_index]
        color = model._node_color(su_type)
        axis.add_patch(Circle(
            (horizontal, vertical), node_radius_data, facecolor=color,
            edgecolor='white', linewidth=0.8, alpha=model.NODE_ALPHA, zorder=3,
        ))
        if show_su_ids:
            axis.text(
                horizontal, vertical, str(su_type), ha='center', va='center',
                fontsize=reference.REFERENCE_FONT_SIZE, fontweight='bold', color=model.FONT_COLOR,
                rotation=label_rotation, rotation_mode='anchor', zorder=4,
            )
        nodes.append({
            'node_index': node_index, 'su_type': su_type, 'su_name': match['name'],
            'atom_indices': list(match['atoms']), 'layout_x': float(horizontal),
            'layout_y': float(vertical), 'rgb': list(color),
        })
    margin = 3.0 * node_radius_data
    axis.set_xlim(min(point[0] for point in positions.values()) - margin,
                  max(point[0] for point in positions.values()) + margin)
    axis.set_ylim(min(point[1] for point in positions.values()) - margin,
                  max(point[1] for point in positions.values()) + margin)
    axis.set_aspect('equal')
    axis.set_axis_off()
    style = {
        'parameter_source': 'model/shared/coarse_graph.py + RL_MTCS/io/output.py',
        'layout': layout,
        'rotation_degrees': None if layout == 'regular' else model.PLOT_ROTATION_DEGREES + presentation_rotation,
        'source_rotation_degrees': model.PLOT_ROTATION_DEGREES,
        'presentation_rotation_degrees': presentation_rotation,
        'label_rotation_degrees': label_rotation,
        'source_label_rotation_degrees': model.LABEL_ROTATION_DEGREES,
        'node_radius_data': node_radius_data, 'node_radius_pt': PANEL_NODE_RADIUS_PT,
        'node_alpha': model.NODE_ALPHA,
        'node_font_size_pt': reference.REFERENCE_FONT_SIZE, 'node_font_color': model.FONT_COLOR,
        'edge_color': list(model.EDGE_COLOR), 'edge_linewidth_pt': model.EDGE_LW,
        'edge_length': model.EDGE_LENGTH, 'show_su_ids': show_su_ids,
        'coordinate_source': 'RDKit 2D depiction of the contracted SU graph' if layout == 'regular' else 'Model hex layout',
        'coarse_ring_node_indices': rings,
        'adaptation': 'Regular mode recalculates coordinates from the unchanged SU adjacency, not from atomic ring sizes. Model SU colors/topology and edge styling are preserved; RL node proportions and label typography are adapted to a fixed physical point size; no SMILES title.',
    }
    return figure, {'nodes': nodes, 'edges': edges, 'style': style}


def create_prediction_figures(figmaker, counts, shifts, width_mm=88.0, height_mm=58.0,
                               hwhm=1.0, ppm_step=0.1, filled=True, show_metrics=False,
                               s2n_width_mm=88.0, s2n_present_only=True,
                               s2n_integer_predictions=True):
    spectrum, common_scale = reconstruct_spectra(shifts, hwhm=hwhm, ppm_step=ppm_step)
    g2s_figure, g2s_axis = figmaker._create_canvas(width_mm, height_mm, 0.18, 0.22, 0.96, 0.96)
    shift_mae = float((shifts['pred_ppm'] - shifts['true_ppm']).abs().mean())
    plot_spectrum_example(
        g2s_axis, spectrum, figmaker, shift_mae=shift_mae if show_metrics else None, filled=filled,
    )
    s2n_left = 0.08 if s2n_width_mm >= 140.0 else 0.16
    s2n_bottom = 0.20 if s2n_present_only else 0.28
    s2n_figure, s2n_axis = figmaker._create_canvas(s2n_width_mm, height_mm, s2n_left, s2n_bottom, 0.985, 0.96)
    plot_su_example(s2n_axis, counts, figmaker, present_only=s2n_present_only,
                    integer_predictions=s2n_integer_predictions)
    if show_metrics:
        predicted = integerize_su_counts(counts['pred_count']) if s2n_integer_predictions else counts['pred_count'].to_numpy()
        mae = float(np.abs(predicted - counts['true_count'].to_numpy()).mean())
        label = 'Integer MAE' if s2n_integer_predictions else 'Raw MAE'
        s2n_axis.text(0.02, 0.84, f'{label} = {mae:.3f} SU', transform=s2n_axis.transAxes, va='top', fontsize=7)
    return {'G2S_spectrum_comparison': g2s_figure, 'S2N_count_comparison': s2n_figure}, spectrum, common_scale


def summarize_s2n_sample_sizes(points):
    """Separate reference size and type coverage from prediction-based sample selection."""
    validate_s2n_points(points)
    errors = points.assign(absolute_error=(points['pred_count'] - points['true_count']).abs())
    summary = errors.groupby('path', sort=True).agg(
        total_su=('true_count', 'sum'),
        present_su_types=('true_count', lambda values: int(values.gt(0).sum())),
        max_type_count=('true_count', 'max'),
        mae_all_33_types=('absolute_error', 'mean'),
    )
    if summary['total_su'].le(0).any():
        raise ValueError('Sample-size auditing requires positive reference SU totals.')
    summary['zero_reference_types'] = 33 - summary['present_su_types']
    summary['source_subset'] = summary.index.map(lambda value: Path(value).parent.name)
    summary['carbon_count'] = errors[errors['su_index'].lt(26)].groupby('path')['true_count'].sum()
    summary['mae_present_types'] = errors[errors['true_count'].gt(0)].groupby('path')['absolute_error'].mean()
    summary['relative_l1_error'] = errors.groupby('path')['absolute_error'].sum() / summary['total_su']
    return summary.reset_index()


def describe_example_size(summary, source_path):
    selected = summary[summary['path'].eq(source_path)]
    if len(selected) != 1:
        raise ValueError('The selected example must occur exactly once in the size audit.')
    selected = selected.iloc[0]
    populations = {'all_test_samples': summary}
    populations.update({name: subset for name, subset in summary.groupby('source_subset', sort=True)})
    context = {}
    for name, population in populations.items():
        context[name] = {
            'n_samples': len(population),
            'median_total_su': float(population['total_su'].median()),
            'median_present_su_types': float(population['present_su_types'].median()),
            'fraction_with_total_su_at_most_example': float(population['total_su'].le(selected['total_su']).mean()),
        }
    return {
        'example_total_su': float(selected['total_su']),
        'example_present_su_types': int(selected['present_su_types']),
        'example_zero_reference_types': int(selected['zero_reference_types']),
        'example_max_type_count': float(selected['max_type_count']),
        'example_mae_present_types': float(selected['mae_present_types']),
        'populations': context,
        'interpretation': 'Source subsets are audited separately; a small single molecule is not equivalent to a high-count merged sample. No replacement is chosen by prediction error.',
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-id', default='15007', help='Internal data selector only; never printed on the panels.')
    parser.add_argument('--s2n-points', type=Path, default=SCRIPT_DIR / 'checkpoints_s2n_v1/eval/plot_data/test_regression_points.csv')
    parser.add_argument('--g2s-points', type=Path, default=SCRIPT_DIR / 'checkpoints_g2s_ablation/k3/eval/test_points.csv')
    parser.add_argument('--structure-manifest', type=Path, default=SCRIPT_DIR / 'example_structures.json')
    parser.add_argument('--out-dir', type=Path, default=SCRIPT_DIR / 'output_molecule_panels')
    parser.add_argument('--hydrogens', choices=['explicit', 'implicit'], default='explicit')
    parser.add_argument('--hide-su-ids', action='store_true')
    parser.add_argument('--show-metrics', action='store_true')
    parser.add_argument('--curve-style', choices=['filled', 'lines'], default='filled')
    parser.add_argument('--hwhm', type=float, default=1.0)
    parser.add_argument('--ppm-step', type=float, default=0.1)
    parser.add_argument('--width-mm', type=float, default=88.0)
    parser.add_argument('--s2n-width-mm', type=float, default=88.0, help='Width of the S2N category panel.')
    parser.add_argument('--s2n-types', choices=['present', 'all'], default='present',
                        help='Show true_count > 0 types by default; all retains all 33 categories.')
    parser.add_argument('--s2n-counts', choices=['integer', 'raw'], default='integer',
                        help='Integer counts use the Layer0 round-then-clamp rule; raw shows continuous model outputs.')
    parser.add_argument('--structure-height-mm', type=float, default=88.0)
    parser.add_argument('--coarse-height-mm', type=float, default=58.0)
    parser.add_argument('--coarse-layout', choices=['regular', 'horizontal', 'model'], default='regular')
    parser.add_argument('--prediction-height-mm', type=float, default=58.0)
    parser.add_argument('--font', default='Arial')
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    dimensions = [args.width_mm, args.s2n_width_mm, args.structure_height_mm,
                  args.coarse_height_mm, args.prediction_height_mm]
    if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
        raise ValueError('Figure dimensions must be positive and finite.')
    if args.dpi < 300:
        raise ValueError('Use at least 300 dpi for manuscript exports.')
    if args.sample_id == 'auto':
        raise ValueError('Select the molecule explicitly; no automatic replacement is allowed.')
    font_path = font_manager.findfont(args.font, fallback_to_default=False)
    config = NatureStyleConfig()
    config.FONT_NAME, config.DPI = args.font, args.dpi
    figmaker = S2NFigure(config)
    s2n_points = pd.read_csv(args.s2n_points)
    validate_s2n_points(s2n_points)
    g2s_points = load_g2s_points(args.g2s_points)
    structures = load_structure_manifest(args.structure_manifest)
    counts, shifts, _, example = select_example(s2n_points, g2s_points, args.sample_id, structures)
    size_summary = summarize_s2n_sample_sizes(s2n_points)
    size_context = describe_example_size(size_summary, example['source_path'])
    entry = structures[example['source_path']]
    structure = verify_example_structure(entry, counts, shifts)
    atomic_figure = create_atomic_figure(
        entry['smiles'], args.width_mm, args.structure_height_mm, args.hydrogens == 'explicit',
    )
    coarse_figure, coarse = create_coarse_figure(
        entry['smiles'], args.width_mm, args.coarse_height_mm, not args.hide_su_ids,
        args.coarse_layout,
    )
    figures = {'All_atom_structure': atomic_figure, 'Coarse_grained_structure': coarse_figure}
    prediction_figures, spectrum, common_scale = create_prediction_figures(
        figmaker, counts, shifts, args.width_mm, args.prediction_height_mm,
        args.hwhm, args.ppm_step, args.curve_style == 'filled', args.show_metrics,
        args.s2n_width_mm, args.s2n_types == 'present', args.s2n_counts == 'integer',
    )
    figures.update(prediction_figures)
    displayed_counts = select_su_example_types(counts, args.s2n_types == 'present')
    integer_counts = integerize_su_counts(counts['pred_count'])
    integer_mae = float(np.abs(integer_counts - counts['true_count'].to_numpy()).mean())
    s2n_axis = prediction_figures['S2N_count_comparison'].axes[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.out_dir / 'source_data'
    source_dir.mkdir(exist_ok=True)
    counts.to_csv(source_dir / 'S2N_counts.csv', index=False)
    counts.assign(
        pred_count_integer=integer_counts,
        abs_error_integer=np.abs(integer_counts - counts['true_count'].to_numpy()),
        displayed=counts['su_index'].isin(displayed_counts['su_index']),
    ).to_csv(source_dir / 'S2N_counts_integerized.csv', index=False)
    size_summary.to_csv(source_dir / 'S2N_test_sample_sizes.csv', index=False)
    shifts.assign(reconstruction_weight=1.0).to_csv(source_dir / 'G2S_shifts.csv', index=False)
    spectrum.to_csv(source_dir / 'G2S_spectra.csv', index=False)
    pd.DataFrame(coarse['nodes']).to_csv(source_dir / 'coarse_nodes.csv', index=False)
    pd.DataFrame(coarse['edges']).to_csv(source_dir / 'coarse_edges.csv', index=False)
    (source_dir / 'molecule.smi').write_text(entry['smiles'] + '\n', encoding='utf-8')
    atomistic_molecule = Chem.AddHs(Chem.MolFromSmiles(entry['smiles']))
    metadata = {
        'core_conclusion': 'One verified molecule connects the all-atom structure, model-defined coarse graph and matched S2N/G2S prediction comparisons.',
        'archetype': 'independent panels for author assembly',
        'selection': 'Explicit user-selected molecule; no automatic substitution.',
        'structure': structure,
        'all_atom_count_including_hydrogen': atomistic_molecule.GetNumAtoms(),
        'explicit_hydrogens_drawn': args.hydrogens == 'explicit',
        'sample_identifiers_drawn': False,
        'panel_letters_drawn': False,
        'inputs': {'s2n_points': file_record(args.s2n_points), 'g2s_points': file_record(args.g2s_points),
                   'structure_manifest': file_record(args.structure_manifest)},
        'sample_sizes': {'molecules': 1, 'su_types_in_source': len(counts),
                         'su_types_displayed': len(displayed_counts), 'carbon_centers': len(shifts)},
        'metrics': {'s2n_mae_per_su': float((counts['pred_count'] - counts['true_count']).abs().mean()),
                    's2n_integer_mae_per_su': integer_mae,
                    'g2s_shift_mae_ppm': float((shifts['pred_ppm'] - shifts['true_ppm']).abs().mean())},
        'metrics_drawn': args.show_metrics,
        'example_size_audit': size_context,
        'coarse_drawing': coarse['style'],
        'spectrum': {'hwhm_ppm': args.hwhm, 'ppm_step': args.ppm_step, 'rows': len(spectrum),
                     'carbon_weights': f'All {len(shifts)} carbon centers have unit weights on both curves; no intensity prediction.',
                     'scaling': 'One shared scale: maximum of the reference reconstruction.', 'common_scale': common_scale,
                     'reference': 'Reconstructed from the matched evaluation true_ppm values, not a separately measured experimental spectrum.',
                     'prediction': 'Reconstructed from saved G2S pred_ppm values using the identical kernel.',
                     'curve_style': args.curve_style, 'y_axis_drawn': False,
                     'legend_labels': ['True ¹³C NMR', 'G2S prediction'],
                     'true_curve_note': 'Ground-truth test-set chemical shifts reconstructed with the same Lorentzian kernel; not an independently acquired raw spectrum.'},
        's2n_axis': {'kind': 'categorical', 'categories': [f'SU{su_type}' for su_type in displayed_counts['su_index']],
                     'display_rule': 'true_count > 0' if args.s2n_types == 'present' else 'All 33 types',
                     'input_type_count': len(counts), 'displayed_type_count': len(displayed_counts),
                     'omitted_type_ids': sorted(set(counts['su_index']) - set(displayed_counts['su_index'])),
                     'all_33_ticks_drawn': len(displayed_counts) == 33,
                     'label_rotation_degrees': s2n_axis.get_xticklabels()[0].get_rotation(),
                     'legend_labels': ['True SU distribution', 'S2N prediction'],
                     'prediction_mode': args.s2n_counts,
                     'integerization': 'Round to nearest integer, ties to even, then clamp to zero; matches Layer0 initial torch.round().long().clamp(min=0).',
                     'postprocessing_scope': 'Initial integerization only; no Layer0 element/port/topology correction is included.',
                     'metric_scope': 'Raw and integer MAEs are recorded separately over all 33 types; original regression metrics and source counts are retained.'},
        'dimensions_mm': {name: (figure.get_size_inches() * 25.4).tolist() for name, figure in figures.items()},
        'font': {'name': args.font, 'file': font_path, 'body_size_pt': 7, 'coarse_su_size_pt': coarse['style']['node_font_size_pt']},
        'export_dpi': args.dpi,
        'exports': ['.pdf', '.svg', '.png', '.tif'],
        'limitations': ['Single illustrative test molecule; not evidence of overall performance.', structure['reference_version_note']],
    }
    (args.out_dir / 'figure_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8',
    )
    for name, figure in figures.items():
        figmaker.save_figure(figure, args.out_dir, name)
        figure.savefig(args.out_dir / f'{name}_preview.png', dpi=180, facecolor='white')
        plt.close(figure)
    print('Four independent panels exported without sample IDs, SMILES titles or panel letters.')


if __name__ == '__main__':
    main()
