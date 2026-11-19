"""Validate a matched example structure and draw its bonds as native vectors."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path

from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path
from matplotlib.path import Path as DrawingPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import ScaledTranslation
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_structure_manifest(manifest_path):
    """Load explicit source-path-to-structure mappings, never infer identity from IDs."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not manifest.get('molecules'):
        raise ValueError('A nonempty version-1 molecule structure manifest is required.')
    structures = {}
    for entry in manifest['molecules']:
        source_path = entry['source_path']
        if source_path in structures:
            raise ValueError(f'Duplicate structure mapping for {source_path}.')
        if not entry.get('smiles') or not entry.get('reference_carbon_centers'):
            raise ValueError(f'Structure and matched carbon-reference records are required for {source_path}.')
        structures[source_path] = entry
    return structures


def verify_example_structure(entry, su_points, shift_points):
    """Match SMILES-derived SU assignments and saved carbon references to both plots."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model.shared.coarse_graph import MoleculeProcessingError, smiles_to_coarse_grained_graph

    if not su_points['path'].eq(entry['source_path']).all():
        raise ValueError('The structure source path does not match the S2N example.')
    if not shift_points['source_path'].eq(entry['source_path']).all():
        raise ValueError('The structure source path does not match the G2S example.')
    molecule = Chem.MolFromSmiles(entry['smiles'])
    if molecule is None:
        raise ValueError('The example structure has invalid SMILES.')
    try:
        graph, _ = smiles_to_coarse_grained_graph(
            entry['smiles'], min_su_nodes=1, save_visualization=False,
        )
    except MoleculeProcessingError as error:
        raise ValueError(f'The example SMILES is not supported by the current SU vocabulary: {error}') from error
    reference_histogram = su_points.sort_values('su_index')['true_count'].to_numpy(dtype=float)
    if not np.array_equal(graph.su_hist.numpy(), reference_histogram):
        raise ValueError('The SMILES-derived SU counts do not match the S2N reference counts.')
    saved_histogram = np.asarray(entry['reference_su_histogram'], dtype=float)
    if not np.array_equal(saved_histogram, reference_histogram):
        raise ValueError('The structure audit snapshot does not match the S2N reference counts.')
    expected_centers = sorted(entry['reference_carbon_centers'], key=lambda center: center['original_center_id'])
    observed_centers = shift_points.sort_values('original_center_id')
    center_ids = observed_centers['original_center_id'].to_numpy(dtype=int)
    if len(expected_centers) != len(center_ids):
        raise ValueError('The structure audit snapshot has different carbon-center coverage.')
    if center_ids.max() >= graph.x.size(0):
        raise ValueError('Carbon-center IDs exceed the structure-derived graph size.')
    expected_ids = np.asarray([center['original_center_id'] for center in expected_centers])
    expected_types = np.asarray([center['center_su'] for center in expected_centers])
    expected_shifts = np.asarray([center['true_ppm'] for center in expected_centers], dtype=float)
    if not np.array_equal(expected_ids, center_ids):
        raise ValueError('The carbon-center IDs do not match the structure audit snapshot.')
    observed_types = observed_centers['center_su'].to_numpy(dtype=int)
    rebuilt_types = graph.x[center_ids, :33].argmax(dim=1).numpy()
    if not np.array_equal(expected_types, observed_types) or not np.array_equal(rebuilt_types, observed_types):
        raise ValueError('The structure-derived carbon-center SU types do not match the G2S references.')
    if not np.allclose(expected_shifts, observed_centers['true_ppm'], atol=1e-5, rtol=0):
        raise ValueError('The reference shifts do not match the structure audit snapshot.')
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    if entry.get('molecular_formula', formula) != formula:
        raise ValueError('The molecular formula does not match the example SMILES.')
    source_checks = {}
    for name, source in entry.get('provenance', {}).items():
        source_path = Path(source['path'])
        if source_path.is_file():
            checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if checksum != source['sha256']:
                raise ValueError(f'The original {name} file changed since the structure audit.')
            source_checks[name] = 'original file checksum verified'
        else:
            source_checks[name] = 'original file unavailable; using bundled audited snapshot'
    return {
        'source_path': entry['source_path'],
        'sample_id': Path(entry['source_path']).stem,
        'smiles': entry['smiles'],
        'canonical_isomeric_smiles': Chem.MolToSmiles(molecule, isomericSmiles=True),
        'molecular_formula': formula,
        'n_carbon_centers': int(len(center_ids)),
        'n_su_nodes': int(graph.x.size(0)),
        'su_histogram_matches': True,
        'carbon_center_types_match': True,
        'reference_shifts_match': True,
        'source_checks': source_checks,
        'provenance': entry.get('provenance', {}),
        'reference_version_note': entry.get('reference_version_note', ''),
    }


def _path_from_svg(path_string):
    pen = RecordingPen()
    parse_path(path_string, pen)
    vertices = []
    codes = []
    start_vertex = None
    for operation, points in pen.value:
        if operation == 'moveTo':
            start_vertex = points[0]
            vertices.extend(points)
            codes.append(DrawingPath.MOVETO)
        elif operation == 'lineTo':
            vertices.extend(points)
            codes.extend([DrawingPath.LINETO] * len(points))
        elif operation == 'curveTo':
            vertices.extend(points)
            codes.extend([DrawingPath.CURVE4] * len(points))
        elif operation == 'qCurveTo':
            vertices.extend(points)
            codes.extend([DrawingPath.CURVE3] * len(points))
        elif operation == 'closePath':
            vertices.append(start_vertex)
            codes.append(DrawingPath.CLOSEPOLY)
        elif operation != 'endPath':
            raise ValueError(f'Unsupported molecule path operation: {operation}.')
    return DrawingPath(vertices, codes)


def draw_molecule(axis, smiles, font_size=7.0, line_width=0.8,
                  explicit_hydrogens=False, label_carbons=False):
    """Transfer RDKit bonds, wedges and element labels to editable Matplotlib artists."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Cannot draw invalid SMILES.')
    if explicit_hydrogens:
        molecule = Chem.AddHs(molecule)
    axis.set_axis_off()
    figure = axis.figure
    figure.canvas.draw()
    bounds = axis.get_window_extent()
    pixels_per_point = 4.0
    width_points = bounds.width * 72.0 / figure.dpi
    height_points = bounds.height * 72.0 / figure.dpi
    width = max(1, int(round(width_points * pixels_per_point)))
    height = max(1, int(round(height_points * pixels_per_point)))
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height, -1, -1, True)
    options = drawer.drawOptions()
    options.fixedFontSize = int(round(font_size * pixels_per_point))
    options.bondLineWidth = line_width * pixels_per_point
    options.useBWAtomPalette()
    if label_carbons:
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 6:
                options.atomLabels[atom.GetIdx()] = 'C'
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    root = element_tree.fromstring(drawer.GetDrawingText())
    for element in root:
        tag = element.tag.rsplit('}', 1)[-1]
        style = dict(part.split(':', 1) for part in element.get('style', '').split(';') if ':' in part)
        if tag == 'path':
            path = _path_from_svg(element.attrib['d'])
            patch = PathPatch(
                path, facecolor=style.get('fill', 'none'),
                edgecolor=style.get('stroke', 'none'),
                linewidth=float(style.get('stroke-width', '0').removesuffix('px')) / pixels_per_point,
                capstyle='butt', joinstyle='miter',
            )
            axis.add_patch(patch)
        elif tag == 'text':
            axis.text(
                float(element.attrib['x']), float(element.attrib['y']), element.text or '',
                fontsize=float(style['font-size'].removesuffix('px')) / pixels_per_point,
                color=style.get('fill', 'black'), ha='left', va='baseline',
            )
        elif tag != 'rect':
            raise ValueError(f'Unsupported RDKit drawing element: {tag}.')
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    return axis


def plot_structure_panel(axis, structure, font_size=7.0):
    axis.set_axis_off()
    axis.text(
        0.02, 0.75, f'Sample {structure["sample_id"]}',
        transform=axis.transAxes, va='baseline', fontsize=font_size,
    )
    renderer = axis.figure.canvas.get_renderer()
    horizontal_offset = 0.0
    for token in re.findall(r'\d+|\D+', structure['molecular_formula']):
        subscript = token.isdigit()
        vertical_offset = -font_size * (1.8 if subscript else 1.6)
        offset = ScaledTranslation(
            horizontal_offset / 72.0, vertical_offset / 72.0,
            axis.figure.dpi_scale_trans,
        )
        label = axis.text(
            0.02, 0.75, token, transform=axis.transAxes + offset,
            va='baseline', fontsize=font_size * (0.8 if subscript else 1.0),
        )
        horizontal_offset += label.get_window_extent(renderer).width * 72.0 / axis.figure.dpi
    molecule_axis = axis.inset_axes([0.23, 0.0, 0.77, 1.0])
    draw_molecule(molecule_axis, structure['smiles'], font_size=font_size)
    return molecule_axis
