"""Read and prepare experimental CSV spectra for inverse inference."""

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..shared.coarse_graph import PPM_AXIS, PPM_STEP
from ..shared.inverse_common import normalize_spectrum_to_carbon_count, resample_spectrum_to_ppm_axis

DEFAULT_SMOOTH_WINDOW_PPM = 1.5
DEFAULT_BASELINE_WINDOWS = ((0.0, 7.0), (65.0, 95.0), (186.0, 194.0), (223.0, 240.0))


def _read_resampled_spectrum(path: str | Path) -> tuple[torch.Tensor, np.ndarray]:
    data = pd.read_csv(path, sep=r'[;, \t]', engine='python', header=None)
    if data.shape[1] < 2:
        raise ValueError('Spectrum CSV must contain ppm and intensity columns')
    ppm = pd.to_numeric(data.iloc[:, 0], errors='coerce').to_numpy(dtype=np.float64)
    intensity = pd.to_numeric(data.iloc[:, 1], errors='coerce').to_numpy(dtype=np.float64)
    valid = np.isfinite(ppm) & np.isfinite(intensity)
    ppm, intensity = ppm[valid], intensity[valid]
    raw = resample_spectrum_to_ppm_axis(ppm, intensity, ppm_axis=PPM_AXIS)
    axis = PPM_AXIS.numpy().astype(np.float64)
    observed = (axis >= ppm.min()) & (axis <= ppm.max()) if ppm.size else np.zeros(axis.size, dtype=bool)
    return raw, observed


def read_spectrum_csv(path: str | Path) -> torch.Tensor:
    """Read and resample a CSV without baseline correction or area scaling."""
    raw, _ = _read_resampled_spectrum(path)
    return raw


def _smooth_spectrum(signal: np.ndarray, window_ppm: float) -> np.ndarray:
    if window_ppm == 0 or signal.size < 3:
        return signal
    from scipy.signal import savgol_filter

    points = round(min(window_ppm / PPM_STEP, signal.size))
    points -= (points + 1) % 2
    return savgol_filter(signal, points, polyorder=2) if points >= 3 else signal


def _estimate_baseline(
    ppm: np.ndarray,
    signal: np.ndarray,
    windows: Sequence[tuple[float, float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if windows is None:
        fit_windows = DEFAULT_BASELINE_WINDOWS
    else:
        fit_windows = sorted((float(lo), float(hi)) for lo, hi in windows)
        if (
            len(fit_windows) < 2
            or any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in fit_windows)
            or any(left[1] >= right[0] for left, right in zip(fit_windows, fit_windows[1:]))
        ):
            raise ValueError('Provide at least two separate, increasing baseline windows in ppm')

    def low_anchor(indices: np.ndarray) -> tuple[float, float]:
        count = max(1, math.ceil(0.20 * indices.size))
        lowest = indices[np.argsort(signal[indices], kind='stable')[:count]]
        return float(np.median(ppm[lowest])), float(np.median(signal[lowest]))

    anchors = []
    for lo, hi in fit_windows:
        indices = np.flatnonzero((ppm >= lo) & (ppm <= hi))
        if indices.size >= 3:
            anchors.append(low_anchor(indices))
    if len(anchors) < 2:
        if windows is not None:
            raise ValueError('At least two baseline windows must overlap the measured spectrum')
        edge_size = max(1, min(round(5.0 / PPM_STEP), signal.size // 4))
        anchors = [
            low_anchor(np.arange(edge_size)),
            low_anchor(np.arange(signal.size - edge_size, signal.size)),
        ]

    anchor_ppm, anchor_intensity = np.asarray(anchors).T
    baseline = np.interp(ppm, anchor_ppm, anchor_intensity)
    for i, j, outside in ((0, 1, ppm < anchor_ppm[0]), (-2, -1, ppm > anchor_ppm[-1])):
        slope = (anchor_intensity[j] - anchor_intensity[i]) / (anchor_ppm[j] - anchor_ppm[i])
        baseline[outside] = anchor_intensity[i] + slope * (ppm[outside] - anchor_ppm[i])
    return baseline, anchor_ppm


def load_target_spectrum_csv(
    path: str | Path,
    carbon_count: float,
    *,
    baseline_strength: float = 1.0,
    smooth_window_ppm: float = DEFAULT_SMOOTH_WINDOW_PPM,
    baseline_windows: Sequence[tuple[float, float]] | None = None,
    diagnostic_path: str | Path | None = None,
) -> torch.Tensor:
    """Smooth, subtract a piecewise linear baseline and normalize to pi * carbon_count.

    Specify two baseline windows for a single line. Set both baseline_strength
    and smooth_window_ppm to zero to reproduce area-only normalization.
    """
    strength = float(baseline_strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError('baseline_strength must be between 0 and 1')
    smooth_window = float(smooth_window_ppm)
    if not math.isfinite(smooth_window) or smooth_window < 0.0:
        raise ValueError('smooth_window_ppm must be finite and non-negative')

    raw, observed = _read_resampled_spectrum(path)
    ppm = PPM_AXIS.numpy().astype(np.float64)
    raw_values = raw.numpy().astype(np.float64)
    smoothed = raw_values.copy()
    smoothed[observed] = _smooth_spectrum(raw_values[observed], smooth_window)
    baseline = np.zeros_like(raw_values)
    baseline_anchor = np.zeros(ppm.size, dtype=bool)
    corrected = torch.from_numpy(smoothed).to(raw.dtype)
    if strength > 0.0 and observed.sum() >= 3:
        estimate, anchor_ppm = _estimate_baseline(ppm[observed], smoothed[observed], baseline_windows)
        baseline[observed] = strength * estimate
        baseline_anchor[np.abs(ppm[:, None] - anchor_ppm).argmin(axis=0)] = True
        residual = np.maximum(smoothed - baseline, 0.0)
        # Area normalization would amplify subtraction roundoff into peaks.
        precision = 8 * torch.finfo(raw.dtype).eps * np.max(np.abs(raw_values))
        if residual.max() <= precision:
            residual.fill(0.0)
        corrected = torch.from_numpy(residual).to(raw.dtype)

    target = normalize_spectrum_to_carbon_count(corrected, carbon_count)
    if diagnostic_path is not None:
        destination = Path(diagnostic_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'ppm': ppm,
            'observed': observed,
            'raw': raw_values,
            'smoothed': smoothed,
            'baseline': baseline,
            'baseline_anchor': baseline_anchor,
            'corrected': corrected.numpy().astype(np.float64),
            'target': target.numpy(),
        }).to_csv(destination, index=False)
        print(
            f'[Spectrum Input] smooth_window_ppm={smooth_window:g} baseline_strength={strength:g} '
            f'anchor_ppm={ppm[baseline_anchor].round(2).tolist()} diagnostics={destination}'
        )
    return target
