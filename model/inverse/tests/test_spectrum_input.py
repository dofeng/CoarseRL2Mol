import math

import numpy as np
import pandas as pd
import pytest
import torch

from model.inverse.inverse_pipeline import load_target_spectrum_csv, read_spectrum_csv
from model.shared.coarse_graph import PPM_AXIS, PPM_STEP
from model.shared.inverse_common import normalize_spectrum_to_carbon_count


def test_baseline_preserves_broad_and_weak_peaks_before_area_normalization(tmp_path):
    ppm = PPM_AXIS.numpy().astype(np.float64)
    broad = 20 * np.exp(-0.5 * ((ppm - 130) / 18) ** 2)
    weak = 0.5 * np.exp(-0.5 * ((ppm - 70) / 3) ** 2)
    baseline = 1.5 + 0.004 * ppm
    raw = broad + weak + baseline
    source = tmp_path / 'spectrum.csv'
    report = tmp_path / 'run' / 'input_spectrum.csv'
    np.savetxt(source, np.column_stack([ppm, raw])[::-1], delimiter=';')

    target = load_target_spectrum_csv(
        source, 460, baseline_windows=[(0, 7), (223, 240)], diagnostic_path=report,
    )

    data = pd.read_csv(report)
    np.testing.assert_allclose(data.raw, raw, atol=3e-6)
    np.testing.assert_allclose(data.baseline, baseline, atol=0.04)
    np.testing.assert_allclose(data.corrected, broad + weak, atol=0.04)
    np.testing.assert_allclose(data.target, target.numpy(), atol=2e-6)
    assert float(target.sum()) * PPM_STEP == pytest.approx(math.pi * 460, rel=1e-6)


def test_default_anchors_follow_smoothed_piecewise_background(tmp_path):
    ppm = PPM_AXIS.numpy().astype(np.float64)
    baseline = np.interp(ppm, [0, 80, 190, 240], [1, 2, 2, 0.5])
    signal = 12 * np.exp(-0.5 * ((ppm - 130) / 8) ** 2)
    signal += 1.5 * np.exp(-0.5 * ((ppm - 173) / 3) ** 2)
    noise = 0.05 * (-1.0) ** np.arange(ppm.size)
    source = tmp_path / 'spectrum.csv'
    report = tmp_path / 'input_spectrum.csv'
    np.savetxt(source, np.column_stack([ppm, baseline + signal + noise]), delimiter=',')

    load_target_spectrum_csv(source, 460, diagnostic_path=report)

    data = pd.read_csv(report)
    assert data.baseline_anchor.sum() == 4
    assert np.std(data.smoothed - baseline - signal) < np.std(noise) / 5
    np.testing.assert_allclose(data.baseline, baseline, atol=0.25)
    np.testing.assert_allclose(data.corrected, signal, atol=0.25)


def test_padding_stays_zero_when_a_cropped_spectrum_has_a_signed_background(tmp_path):
    ppm = PPM_AXIS.numpy().astype(np.float64)
    observed = (ppm >= 60) & (ppm <= 180)
    signal = 3 * np.exp(-0.5 * ((ppm - 120) / 5) ** 2)
    source = tmp_path / 'cropped.csv'
    report = tmp_path / 'input_spectrum.csv'
    np.savetxt(source, np.column_stack([ppm[observed], signal[observed] - 2]), delimiter=',')

    target = load_target_spectrum_csv(source, 20, diagnostic_path=report)

    data = pd.read_csv(report)
    np.testing.assert_array_equal(data.observed, observed)
    np.testing.assert_allclose(data.baseline[observed], -2, atol=0.02)
    assert (data.loc[~observed, ['raw', 'baseline', 'corrected', 'target']] == 0).all().all()
    assert (target >= 0).all()


def test_disabled_baseline_uses_original_normalization(tmp_path):
    source = tmp_path / 'signed.csv'
    source.write_text('0.1,-1\n0.2,4\n0.3,1\n0.4,0\n', encoding='ascii')

    original = normalize_spectrum_to_carbon_count(read_spectrum_csv(source), 3)
    disabled = load_target_spectrum_csv(source, 3, baseline_strength=0, smooth_window_ppm=0)

    assert torch.equal(disabled, original)
    assert disabled[0] < 0


def test_pure_linear_baseline_does_not_normalize_roundoff_into_peaks(tmp_path):
    source = tmp_path / 'linear.csv'
    ppm = PPM_AXIS.numpy().astype(np.float64)
    np.savetxt(source, np.column_stack([ppm, 1.5 + 0.004 * ppm]), delimiter=',')

    target = load_target_spectrum_csv(source, 460)

    assert torch.count_nonzero(target) == 0
