"""Protect unnumbered molecular panels and the model-defined coarse drawing style."""

from collections import Counter
from pathlib import Path
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from plot.fig3.plot import S2NFigure
from plot.fig3.molecule_structure import load_structure_manifest
from plot.fig3.plot_molecule_panels import create_atomic_figure, create_coarse_figure, create_prediction_figures, describe_example_size, model_drawing_module, reference_drawing_module, summarize_s2n_sample_sizes


FIGURE_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = 'coarsedata_v1/test_single/15007.pt'


class MolecularPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = load_structure_manifest(FIGURE_DIR / 'example_structures.json')[SOURCE_PATH]
        counts = pd.read_csv(FIGURE_DIR / 'checkpoints_s2n_v1/eval/plot_data/test_regression_points.csv')
        shifts = pd.read_csv(FIGURE_DIR / 'checkpoints_g2s_ablation/k3/eval/test_points.csv')
        cls.counts = counts[counts['path'].eq(SOURCE_PATH)].sort_values('su_index')
        cls.shifts = shifts[shifts['source_path'].eq(SOURCE_PATH)].sort_values('original_center_id')

    def tearDown(self):
        plt.close('all')

    def test_all_atom_diagram_explicitly_contains_every_atom(self):
        S2NFigure()
        figure = create_atomic_figure(self.entry['smiles'])
        axis = figure.axes[0]
        labels = Counter(text.get_text() for text in axis.texts)
        self.assertEqual(labels, {'C': 12, 'H': 16, 'N': 2, 'O': 6})
        self.assertEqual(len(axis.images), 0)
        self.assertTrue(any(patch.get_facecolor()[3] > 0 for patch in axis.patches))
        np.testing.assert_allclose(figure.get_size_inches() * 25.4, [88, 88])

    def test_coarse_graph_reuses_model_palette_and_all_nodes(self):
        S2NFigure()
        model = model_drawing_module()
        reference = reference_drawing_module()
        figure, drawing = create_coarse_figure(self.entry['smiles'])
        axis = figure.axes[0]
        figure.canvas.draw()
        points_per_unit = np.linalg.norm(
            axis.transData.transform((1.0, 0.0)) - axis.transData.transform((0.0, 0.0))
        ) * 72.0 / figure.dpi
        circles = [patch for patch in axis.patches if isinstance(patch, Circle)]
        self.assertEqual(len(circles), 14)
        self.assertEqual(len(drawing['edges']), 15)
        self.assertEqual(Counter(node['su_type'] for node in drawing['nodes']),
                         Counter({0: 2, 2: 1, 11: 1, 13: 1, 19: 4, 22: 2, 23: 1, 28: 1, 29: 1}))
        for circle, node, text in zip(circles, drawing['nodes'], axis.texts):
            self.assertAlmostEqual(circle.radius, drawing['style']['node_radius_data'])
            self.assertAlmostEqual(circle.radius * points_per_unit, 7.0)
            np.testing.assert_allclose(circle.get_facecolor()[:3], model._node_color(node['su_type']))
            self.assertEqual(circle.get_alpha(), model.NODE_ALPHA)
            self.assertEqual(text.get_fontsize(), reference.REFERENCE_FONT_SIZE)
            self.assertEqual(text.get_rotation(), 0)
        for line in axis.lines:
            self.assertEqual(line.get_linewidth(), model.EDGE_LW)
            np.testing.assert_allclose(line.get_color(), model.EDGE_COLOR)
        self.assertEqual(axis.get_title(), '')
        self.assertEqual(drawing['style']['source_rotation_degrees'], model.PLOT_ROTATION_DEGREES)
        self.assertEqual(drawing['style']['layout'], 'regular')
        np.testing.assert_allclose(figure.get_size_inches() * 25.4, [88, 58])

    def test_coarse_node_physical_size_is_constant_across_panel_dimensions(self):
        small, small_drawing = create_coarse_figure(self.entry['smiles'], 88.0, 58.0)
        large, large_drawing = create_coarse_figure(self.entry['smiles'], 110.0, 72.5)
        for figure, drawing in [(small, small_drawing), (large, large_drawing)]:
            axis = figure.axes[0]
            figure.canvas.draw()
            points_per_unit = np.linalg.norm(
                axis.transData.transform((1.0, 0.0)) - axis.transData.transform((0.0, 0.0))
            ) * 72.0 / figure.dpi
            self.assertAlmostEqual(
                drawing['style']['node_radius_data'] * points_per_unit,
                drawing['style']['node_radius_pt'],
            )
            self.assertEqual(drawing['style']['node_font_size_pt'], 7)
        self.assertAlmostEqual(small_drawing['style']['node_radius_pt'], large_drawing['style']['node_radius_pt'])

    def test_horizontal_layout_is_only_a_rigid_rotation(self):
        _, horizontal = create_coarse_figure(self.entry['smiles'], layout='horizontal')
        legacy_figure, legacy = create_coarse_figure(self.entry['smiles'], layout='model')
        horizontal_positions = np.array([[node['layout_x'], node['layout_y']] for node in horizontal['nodes']])
        legacy_positions = np.array([[node['layout_x'], node['layout_y']] for node in legacy['nodes']])
        horizontal_distances = np.linalg.norm(horizontal_positions[:, None] - horizontal_positions[None, :], axis=2)
        legacy_distances = np.linalg.norm(legacy_positions[:, None] - legacy_positions[None, :], axis=2)
        np.testing.assert_allclose(horizontal_distances, legacy_distances, atol=1e-12)
        self.assertEqual(horizontal['edges'], legacy['edges'])
        self.assertTrue(np.ptp(horizontal_positions[:, 0]) > np.ptp(horizontal_positions[:, 1]))
        centered = horizontal_positions - horizontal_positions.mean(axis=0)
        self.assertAlmostEqual((centered.T @ centered)[0, 1], 0.0, places=10)
        self.assertEqual(legacy['style']['rotation_degrees'], model_drawing_module().PLOT_ROTATION_DEGREES)
        self.assertTrue(all(text.get_rotation() == 270 for text in legacy_figure.axes[0].texts))

    def test_regular_coordinates_preserve_topology_and_have_equal_bond_lengths(self):
        _, regular = create_coarse_figure(self.entry['smiles'])
        _, legacy = create_coarse_figure(self.entry['smiles'], layout='model')
        self.assertEqual(regular['edges'], legacy['edges'])
        self.assertEqual([node['atom_indices'] for node in regular['nodes']], [node['atom_indices'] for node in legacy['nodes']])
        positions = np.array([[node['layout_x'], node['layout_y']] for node in regular['nodes']])
        for edge in regular['edges']:
            self.assertAlmostEqual(np.linalg.norm(positions[edge['source']] - positions[edge['target']]), regular['style']['edge_length'])
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreater(distances.min(), 2 * regular['style']['node_radius_data'])
        self.assertEqual(sorted(len(ring) for ring in regular['style']['coarse_ring_node_indices']), [4, 5])

    def test_coarse_cycles_are_regular_polygons_not_forced_atomic_rings(self):
        _, drawing = create_coarse_figure(self.entry['smiles'])
        positions = np.array([[node['layout_x'], node['layout_y']] for node in drawing['nodes']])
        for ring in drawing['style']['coarse_ring_node_indices']:
            vertices = positions[ring]
            centered = vertices - vertices.mean(axis=0)
            radii = np.linalg.norm(centered, axis=1)
            np.testing.assert_allclose(radii, radii[0], atol=1e-10)
            directions = centered / radii[:, None]
            neighbor_cosines = np.sum(directions * np.roll(directions, 1, axis=0), axis=1)
            np.testing.assert_allclose(neighbor_cosines, np.cos(2 * np.pi / len(ring)), atol=1e-10)

    def test_sample_size_audit_separates_single_and_merged_populations(self):
        first = self.counts.assign(path='test_single/example.pt', sample_index=0)
        second = self.counts.assign(path='test_single/second.pt', sample_index=1,
                                    true_count=self.counts['true_count'] * 2, pred_count=self.counts['true_count'] * 2)
        merged = self.counts.assign(path='test_merged/combined.pt', sample_index=2,
                                    true_count=self.counts['true_count'] * 10, pred_count=self.counts['true_count'] * 10)
        summary = summarize_s2n_sample_sizes(pd.concat([first, second, merged], ignore_index=True))
        context = describe_example_size(summary, 'test_single/example.pt')
        self.assertEqual(context['example_total_su'], 14)
        self.assertEqual(context['example_present_su_types'], 9)
        self.assertEqual(context['example_zero_reference_types'], 24)
        self.assertEqual(context['populations']['test_single']['median_total_su'], 21)
        self.assertEqual(context['populations']['test_merged']['median_total_su'], 140)
        self.assertEqual(context['populations']['test_single']['fraction_with_total_su_at_most_example'], 0.5)
        expected = (self.counts.loc[self.counts['true_count'].gt(0), 'pred_count'] - self.counts.loc[self.counts['true_count'].gt(0), 'true_count']).abs().mean()
        self.assertAlmostEqual(context['example_mae_present_types'], expected)

    def test_larger_candidate_structures_match_their_saved_test_references(self):
        from plot.fig3.molecule_structure import verify_example_structure

        entries = load_structure_manifest(FIGURE_DIR / 'larger_example_structures.json')
        counts = pd.read_csv(FIGURE_DIR / 'checkpoints_s2n_v1/eval/plot_data/test_regression_points.csv')
        shifts = pd.read_csv(FIGURE_DIR / 'checkpoints_g2s_ablation/k3/eval/test_points.csv')
        for source_path, expected_total in [('coarsedata_v1/test_single/1588.pt', 56),
                                            ('coarsedata_v1/test_single/51277.pt', 28)]:
            with self.subTest(source_path=source_path):
                selected_counts = counts[counts['path'].eq(source_path)]
                selected_shifts = shifts[shifts['source_path'].eq(source_path)]
                structure = verify_example_structure(entries[source_path], selected_counts, selected_shifts)
                self.assertEqual(structure['n_su_nodes'], expected_total)
                self.assertEqual(selected_counts['true_count'].gt(0).sum(), 11)

    def test_edges_stop_at_node_boundaries(self):
        figure, drawing = create_coarse_figure(self.entry['smiles'])
        positions = {node['node_index']: np.array([node['layout_x'], node['layout_y']]) for node in drawing['nodes']}
        for line, edge in zip(figure.axes[0].lines, drawing['edges']):
            ends = np.column_stack([line.get_xdata(), line.get_ydata()])
            self.assertAlmostEqual(np.linalg.norm(ends[0] - positions[edge['source']]), drawing['style']['node_radius_data'])
            self.assertAlmostEqual(np.linalg.norm(ends[1] - positions[edge['target']]), drawing['style']['node_radius_data'])

    def test_su_labels_can_be_hidden_without_removing_nodes(self):
        figure, drawing = create_coarse_figure(self.entry['smiles'], show_su_ids=False)
        self.assertEqual(len(figure.axes[0].patches), 14)
        self.assertEqual(len(figure.axes[0].texts), 0)
        self.assertFalse(drawing['style']['show_su_ids'])

    def test_comparisons_preserve_data_and_have_no_identifiers(self):
        figures, spectrum, _ = create_prediction_figures(S2NFigure(), self.counts, self.shifts)
        self.assertEqual(len(spectrum), 2401)
        s2n_axis = figures['S2N_count_comparison'].axes[0]
        present = self.counts[self.counts['true_count'].gt(0)]
        self.assertEqual(len(s2n_axis.patches), 2 * len(present))
        np.testing.assert_allclose([patch.get_height() for patch in s2n_axis.patches[:len(present)]], present['true_count'])
        np.testing.assert_array_equal([patch.get_height() for patch in s2n_axis.patches[len(present):]],
                                      np.maximum(np.rint(present['pred_count']), 0))
        g2s_axis = figures['G2S_spectrum_comparison'].axes[0]
        self.assertEqual(len(g2s_axis.lines), 2)
        np.testing.assert_allclose(g2s_axis.lines[0].get_ydata(), spectrum['reference_relative_intensity'])
        np.testing.assert_allclose(g2s_axis.lines[1].get_ydata(), spectrum['g2s_relative_intensity'])
        for figure in figures.values():
            np.testing.assert_allclose(figure.get_size_inches() * 25.4, [88, 58])
            self.assertEqual(len(figure.texts), 0)
            for axis in figure.axes:
                self.assertFalse(any('15007' in text.get_text() or 'Sample' in text.get_text() for text in axis.texts))
                self.assertEqual(len(axis.texts), 0)

    def test_g2s_spectrum_omits_y_axis_and_preserves_x_axis(self):
        figures, _, _ = create_prediction_figures(S2NFigure(), self.counts, self.shifts)
        axis = figures['G2S_spectrum_comparison'].axes[0]
        self.assertEqual(axis.get_ylabel(), '')
        self.assertEqual(len(axis.get_yticks()), 0)
        self.assertFalse(axis.spines['left'].get_visible())
        self.assertTrue(axis.spines['bottom'].get_visible())
        self.assertEqual(axis.get_xlabel(), '¹³C chemical shift (ppm)')
        self.assertEqual(axis.get_xlim(), (240, 0))
        np.testing.assert_array_equal(axis.get_xticks(), [240, 180, 120, 60, 0])
        self.assertEqual(
            [line.get_label() for line in axis.lines],
            ['True ¹³C NMR', 'G2S prediction'],
        )

    def test_only_true_present_su_types_are_categorical_and_visible(self):
        figures, _, _ = create_prediction_figures(S2NFigure(), self.counts, self.shifts)
        figure = figures['S2N_count_comparison']
        axis = figure.axes[0]
        np.testing.assert_array_equal(axis.get_xticks(), np.arange(9))
        labels = axis.get_xticklabels()
        self.assertEqual([label.get_text() for label in labels],
                         ['SU0', 'SU2', 'SU11', 'SU13', 'SU19', 'SU22', 'SU23', 'SU28', 'SU29'])
        self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                         ['True SU distribution', 'S2N prediction'])
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = [label.get_window_extent(renderer) for label in labels]
        for left, right in zip(boxes, boxes[1:]):
            self.assertLess(left.x1, right.x0)
        for label, bounds in zip(labels, boxes):
            self.assertGreaterEqual(label.get_fontsize(), 7)
            self.assertGreaterEqual(bounds.y0, 0)
            self.assertLessEqual(bounds.x1, figure.bbox.width)

    def test_s2n_width_can_be_adjusted_without_dropping_present_types(self):
        figures, _, _ = create_prediction_figures(S2NFigure(), self.counts, self.shifts, s2n_width_mm=120)
        figure = figures['S2N_count_comparison']
        np.testing.assert_allclose(figure.get_size_inches() * 25.4, [120, 58])
        self.assertEqual(len(figure.axes[0].get_xticklabels()), 9)

    def test_s2n_all_types_option_preserves_all_zero_and_nonzero_predictions(self):
        figures, _, _ = create_prediction_figures(
            S2NFigure(), self.counts, self.shifts, s2n_width_mm=180, s2n_present_only=False,
        )
        axis = figures['S2N_count_comparison'].axes[0]
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                         [f'SU{su_type}' for su_type in range(33)])
        self.assertEqual(len(axis.patches), 66)
        np.testing.assert_array_equal([patch.get_height() for patch in axis.patches[33:]],
                                      np.maximum(np.rint(self.counts['pred_count']), 0))

    def test_raw_s2n_predictions_remain_available_without_mutating_the_source(self):
        before = self.counts.copy(deep=True)
        figures, _, _ = create_prediction_figures(
            S2NFigure(), self.counts, self.shifts, s2n_integer_predictions=False,
        )
        axis = figures['S2N_count_comparison'].axes[0]
        present = self.counts[self.counts['true_count'].gt(0)]
        np.testing.assert_allclose([bar.get_height() for bar in axis.patches[len(present):]], present['pred_count'])
        pd.testing.assert_frame_equal(self.counts, before)

    def test_integer_count_display_does_not_force_the_prediction_to_match_truth(self):
        figures, _, _ = create_prediction_figures(S2NFigure(), self.counts, self.shifts)
        axis = figures['S2N_count_comparison'].axes[0]
        predicted = [bar.get_height() for bar in axis.patches[9:]]
        self.assertEqual(predicted, [0, 0, 0, 1, 5, 2, 1, 2, 2])
        self.assertEqual(sum(predicted), 13)
        self.assertEqual(self.counts['true_count'].sum(), 14)

    def test_present_only_filter_does_not_modify_full_counts_or_hide_present_type_errors(self):
        before = self.counts.copy(deep=True)
        counts = self.counts.copy(deep=True)
        counts.loc[counts['su_index'].eq(0), 'pred_count'] = 0.0
        counts.loc[counts['su_index'].eq(1), 'pred_count'] = 100.0
        requested_counts = counts.copy(deep=True)
        figures, _, _ = create_prediction_figures(S2NFigure(), counts, self.shifts)
        axis = figures['S2N_count_comparison'].axes[0]
        self.assertEqual(len(axis.patches), 18)
        self.assertEqual(axis.patches[9].get_height(), 0.0)
        self.assertNotIn('SU1', [label.get_text() for label in axis.get_xticklabels()])
        pd.testing.assert_frame_equal(counts, requested_counts)
        pd.testing.assert_frame_equal(self.counts, before)

    def test_metric_annotation_is_optional_and_still_unnumbered(self):
        figures, _, _ = create_prediction_figures(S2NFigure(), self.counts, self.shifts, show_metrics=True)
        for figure in figures.values():
            texts = [text.get_text() for axis in figure.axes for text in axis.texts]
            self.assertEqual(len(texts), 1)
            self.assertIn('MAE', texts[0])
            self.assertNotIn('15007', texts[0])


if __name__ == '__main__':
    unittest.main()
