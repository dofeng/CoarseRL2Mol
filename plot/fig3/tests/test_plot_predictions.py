"""Integrity and export regression checks for the manuscript figure bundle."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from plot.fig3.plot import NatureStyleConfig, S2NFigure, summarize_s2n_errors
from plot.fig3.plot_predictions import create_composite, load_g2s_points, reconstruct_spectra, select_example


def make_s2n_points():
    records = []
    for sample_index, sample_name in enumerate(['alpha', 'beta', 'gamma', 'delta']):
        for su_index in range(33):
            true_count = 2.0 if su_index == 13 else 0.0
            records.append({
                'sample_index': sample_index,
                'path': f'test/{sample_name}.pt',
                'su_index': su_index,
                'true_count': true_count,
                'pred_count': true_count + (sample_index + 1) / 10,
            })
    return pd.DataFrame(records)


def make_g2s_points():
    records = []
    for sample_index, sample_name in enumerate(['alpha', 'beta', 'gamma', 'delta']):
        for center_index, shift in enumerate([100.0, 125.0]):
            records.append({
                'source_path': f'test/{sample_name}.pt',
                'original_center_id': center_index,
                'center_su': 13,
                'true_ppm': shift,
                'pred_ppm': shift + sample_index + 1,
            })
    return pd.DataFrame(records)


class CountErrorTests(unittest.TestCase):
    def test_group_weighted_mean_equals_global_mae(self):
        points = make_s2n_points()
        groups, per_su = summarize_s2n_errors(points)
        expected = (points['pred_count'] - points['true_count']).abs().mean()
        self.assertAlmostEqual(np.average(groups['per_su_mae'], weights=groups['n_su_types']), expected)
        self.assertAlmostEqual(per_su['mae'].mean(), expected)
        self.assertEqual(len(per_su), 33)

    def test_within_group_cancellation_does_not_hide_errors(self):
        points = make_s2n_points().query('sample_index == 0').copy()
        points['pred_count'] = points['true_count']
        points.loc[points['su_index'].isin([0, 1]), 'true_count'] = 2.0
        points.loc[points['su_index'].eq(0), 'pred_count'] = 3.0
        points.loc[points['su_index'].eq(1), 'pred_count'] = 1.0
        groups, _ = summarize_s2n_errors(points)
        carbonyl = points[points['su_index'] < 5]
        self.assertAlmostEqual(abs(carbonyl['pred_count'].sum() - carbonyl['true_count'].sum()), 0.0)
        self.assertAlmostEqual(groups.set_index('group').loc['carbonyl', 'per_su_mae'], 0.4)

    def test_missing_zero_count_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'all 33'):
            summarize_s2n_errors(make_s2n_points().iloc[1:])

    def test_duplicate_and_nonfinite_rows_are_rejected(self):
        points = make_s2n_points()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize_s2n_errors(pd.concat([points, points.iloc[[0]]]))
        points.loc[0, 'pred_count'] = np.inf
        with self.assertRaisesRegex(ValueError, 'finite'):
            summarize_s2n_errors(points)

    def test_r2_is_not_clipped_to_zero(self):
        self.assertLess(S2NFigure._compute_r2(np.array([1., 2.]), np.array([10., 20.])), 0.0)


class PredictionExampleTests(unittest.TestCase):
    def test_auto_example_is_deterministic_and_not_best_case(self):
        points = make_s2n_points()
        shifts = make_g2s_points()
        _, _, candidates, chosen = select_example(points, shifts)
        _, _, _, reordered = select_example(points.iloc[::-1], shifts.iloc[::-1])
        self.assertEqual(chosen['sample_id'], 'beta')
        self.assertEqual(chosen['source_path'], reordered['source_path'])
        self.assertEqual(candidates['selected'].sum(), 1)

    def test_unavailable_sample_is_not_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, 'no substitute'):
            select_example(make_s2n_points(), make_g2s_points(), '51121')

    def test_incomplete_carbon_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete'):
            select_example(make_s2n_points(), make_g2s_points().iloc[1:], 'alpha')

    def test_common_spectrum_scale_preserves_amplitude_differences(self):
        shifts = pd.DataFrame({'true_ppm': [80., 120.], 'pred_ppm': [100., 100.]})
        spectrum, scale = reconstruct_spectra(shifts)
        np.testing.assert_allclose(spectrum['g2s_relative_intensity'], spectrum['g2s_reconstructed_raw'] / scale)
        self.assertGreater(spectrum['g2s_relative_intensity'].max(), 1.9)
        self.assertEqual(len(spectrum), 2401)

    def test_matching_shifts_have_matching_reconstructions(self):
        shifts = pd.DataFrame({'true_ppm': [80., 120.], 'pred_ppm': [80., 120.]})
        spectrum, _ = reconstruct_spectra(shifts, hwhm=1.2)
        np.testing.assert_allclose(spectrum['reference_relative_intensity'], spectrum['g2s_relative_intensity'])

    def test_invalid_linewidth_and_out_of_range_shifts_are_rejected(self):
        shifts = make_g2s_points()
        for invalid_width in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                reconstruct_spectra(shifts, hwhm=invalid_width)
        shifts.loc[0, 'pred_ppm'] = 241
        with self.assertRaisesRegex(ValueError, 'outside'):
            reconstruct_spectra(shifts)

    def test_g2s_duplicate_centers_are_rejected(self):
        shifts = make_g2s_points()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'points.csv'
            pd.concat([shifts, shifts.iloc[[0]]]).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_g2s_points(source)

    def test_fixed_canvas_and_editable_text_exports(self):
        config = NatureStyleConfig()
        config.FONT_NAME = 'DejaVu Sans'
        plt.rcParams['savefig.bbox'] = 'tight'
        figmaker = S2NFigure(config)
        figure, axis = figmaker._create_canvas(88., 58., 0.2, 0.2, 0.95, 0.95)
        axis.set_xlabel('Editable label')
        with tempfile.TemporaryDirectory() as temp_dir:
            figmaker.save_figure(figure, temp_dir, 'example', dpi=300)
            with Image.open(Path(temp_dir) / 'example.png') as image:
                self.assertEqual(image.size, (int(88 / 25.4 * 300), int(58 / 25.4 * 300)))
            self.assertIn('<text', (Path(temp_dir) / 'example.svg').read_text())
        plt.close(figure)

    def test_composite_labels_and_colorbars_fit_inside_canvas(self):
        config = NatureStyleConfig()
        config.FONT_NAME = 'DejaVu Sans'
        figmaker = S2NFigure(config)
        points = make_s2n_points()
        shifts = make_g2s_points()
        groups, _ = summarize_s2n_errors(points)
        su_example, shift_example, _, info = select_example(points, shifts)
        spectrum, _ = reconstruct_spectra(shift_example)
        ablation = pd.DataFrame({'k_hop': [1, 2, 3, 4], 'mae': [2., 1.5, 1., 0.8], 'rmse': [3., 2., 1.8, 1.5]})
        figure = create_composite(
            figmaker, points, groups, shifts, ablation, su_example,
            spectrum, info, 180, 198, True,
        )
        renderer = figure.canvas.get_renderer()
        texts = list(figure.texts)
        for axis in figure.axes:
            texts.extend([axis.xaxis.label, axis.yaxis.label])
        for text in texts:
            if text.get_text():
                bounds = text.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, 0, text.get_text())
                self.assertLessEqual(bounds.x1, figure.bbox.width, text.get_text())
                self.assertGreaterEqual(bounds.y0, 0, text.get_text())
                self.assertLessEqual(bounds.y1, figure.bbox.height, text.get_text())
        plt.close(figure)


if __name__ == '__main__':
    unittest.main()
