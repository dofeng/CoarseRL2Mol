"""Keep user-selected reference illustrations separate from model evaluation."""

from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot.fig3.plot import S2NFigure
from plot.fig3.plot_reference_example import audit_su_coverage, create_reference_figure, load_reference


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ReferenceExampleTests(unittest.TestCase):
    def load_selected(self):
        source = PROJECT_ROOT / 'plot/NMR示意'
        return load_reference(source / '51121.txt', source / '51121.csv')

    def test_selected_reference_preserves_all_observations(self):
        structure, spectrum = self.load_selected()
        self.assertEqual(structure['sample_id'], '51121')
        self.assertEqual(structure['molecular_formula'], 'C24H27N3O4')
        self.assertEqual(len(spectrum), 2401)
        self.assertEqual(spectrum['chemical_shift'].iloc[[0, -1]].tolist(), [0.0, 240.0])

    def test_selected_structure_has_two_unsupported_atoms(self):
        structure, _ = self.load_selected()
        coverage = audit_su_coverage(structure['smiles'])
        self.assertFalse(coverage['all_heavy_atoms_assigned'])
        self.assertEqual(coverage['heavy_atom_count'], 31)
        self.assertEqual(coverage['assigned_heavy_atom_count'], 29)
        self.assertEqual([(atom['atom_index'], atom['element']) for atom in coverage['unassigned_atoms']], [(6, 'C'), (7, 'N')])
        self.assertTrue(any(bond['bond_type'] == 'DOUBLE' for bond in coverage['unassigned_atoms'][0]['bonds']))

    def test_supported_structure_has_complete_atom_coverage(self):
        coverage = audit_su_coverage('CC')
        self.assertTrue(coverage['all_heavy_atoms_assigned'])
        self.assertEqual(coverage['unassigned_atoms'], [])

    def test_invalid_spectrum_is_not_silently_repaired(self):
        for spectrum in [
            {'chemical_shift': [0.0, 0.0], 'intensity': [1.0, 2.0]},
            {'chemical_shift': [0.0, 1.0], 'intensity': [1.0, -1.0]},
            {'chemical_shift': [0.0, 1.0], 'intensity': [1.0, np.inf]},
            {'chemical_shift': [0.0, 1.0], 'intensity': [0.0, 0.0]},
        ]:
            with self.subTest(spectrum=spectrum), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'molecule.txt').write_text('CC', encoding='utf-8')
                pd.DataFrame(spectrum).to_csv(root / 'molecule.csv', index=False)
                with self.assertRaises(ValueError):
                    load_reference(root / 'molecule.txt', root / 'molecule.csv')

    def test_mismatched_file_identifiers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same sample'):
            load_reference('51121.txt', '15007.csv')

    def test_figure_contains_only_the_supplied_curve(self):
        structure, spectrum = self.load_selected()
        figure = create_reference_figure(S2NFigure(), structure, spectrum)
        try:
            np.testing.assert_allclose(figure.get_size_inches() * 25.4, [180.0, 68.0])
            spectrum_axis = figure.axes[1]
            self.assertEqual(len(spectrum_axis.lines), 1)
            np.testing.assert_array_equal(spectrum_axis.lines[0].get_ydata(), spectrum['intensity'])
            self.assertEqual(spectrum_axis.get_xlim(), (240.0, 0.0))
            self.assertTrue(all(not axis.images for axis in figure.axes))
            with tempfile.TemporaryDirectory() as directory:
                svg_path = Path(directory) / 'reference.svg'
                figure.savefig(svg_path)
                svg = svg_path.read_text(encoding='utf-8')
                self.assertIn('<text', svg)
                self.assertNotIn('<image', svg)
        finally:
            plt.close(figure)


if __name__ == '__main__':
    unittest.main()
