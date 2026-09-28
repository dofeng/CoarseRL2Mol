from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ....shared.coarse_graph import get_su_ppm_range
from ....shared.inverse_common import reconstruction_peak_scale
from ....shared.aromatic_plan import su12_transition_diagnostics
from .conversion_scoring import evaluate_flex_impact
from .histogram import _h_total
from .profiles import MIN_STAGE3_AROMATIC_RESOURCE_COUNT
from ..stage3_models import Stage3SpectrumCandidate


SPECTRUM_RESOURCE_TYPES = (11, 12, 13, 22, 23, 24)
# SU14/15/16 are intentionally kept outside the ordinary atomic spectrum
# resource set.  They may only enter through one of the explicit paired
# conversion rules below; adding them to SPECTRUM_RESOURCE_TYPES would reopen
# arbitrary single-node conversions in the legacy candidate generator.
UNSATURATED_SPECTRUM_TYPES = (14, 15, 16)
UNSATURATED_PAIR_SOURCE_BY_TARGET = {
    14: 24,
    15: 23,
    16: 22,
}
UNSATURATED_DIFF_WINDOWS: Dict[int, Tuple[float, float]] = {
    14: (125.0, 150.0),
    15: (120.0, 140.0),
    16: (104.0, 125.0),
}
# Canonical forward rules are aliphatic -> unsaturated.  Reverse candidates
# are generated from the same rules so no unlisted pair (notably 22+22) can
# enter the spectrum-only path.
SPECTRUM_PAIR_RULES: Tuple[Tuple[Tuple[int, int], Tuple[int, int], str], ...] = (
    ((22, 23), (16, 15), "22+23_to_16+15"),
    ((23, 23), (15, 15), "23+23_to_15+15"),
    ((22, 24), (16, 14), "22+24_to_16+14"),
)
SPECTRUM_UNSATURATED_CAP_RATIO = 0.10
SPECTRUM_ALIPHATIC_NEGATIVE_CARBON_THRESHOLD = 12.0
SPECTRUM_UNIT_PEAK_HWHM = 1.0
DEFAULT_SPECTRUM_WINDOWS: Dict[int, Tuple[float, float, float, float]] = {
    int(su_type): (*get_su_ppm_range(int(su_type)), *get_su_ppm_range(int(su_type)))
    for su_type in SPECTRUM_RESOURCE_TYPES
}

SPECTRUM_H_SOFT_LOWER_RATIO = -0.03
SPECTRUM_H_SOFT_UPPER_RATIO = 0.03
SPECTRUM_H_OUTSIDE_WORSENING_WEIGHT = 3.0
SPECTRUM_FLEX_DEVIATION_WORSENING_PENALTY = 0.02
SPECTRUM_FLEX_INVALID_WINDOW_PENALTY = 0.05
SPECTRUM_PEAK_URGENCY_WEIGHT = 0.60
SPECTRUM_AREA_URGENCY_WEIGHT = 0.40
SPECTRUM_BATCH_CARBON_RATIO = 0.02


def spectrum_h_soft_bounds(target_h: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if target_h is None or float(target_h) <= 0.0:
        return None, None
    return (
        float(target_h) * (1.0 + float(SPECTRUM_H_SOFT_LOWER_RATIO)),
        float(target_h) * (1.0 + float(SPECTRUM_H_SOFT_UPPER_RATIO)),
    )


def spectrum_candidate_balance_penalty(
    *,
    H_before: torch.Tensor,
    H_after: torch.Tensor,
    E_target: Optional[torch.Tensor],
    special_degree_meta: Optional[Dict[str, Any]] = None,
    flex_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return an R2-scale soft penalty for H and flex degradation."""
    target_h: Optional[float]
    try:
        target_h = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        target_h = None
    if target_h is not None and float(target_h) <= 0.0:
        target_h = None

    h_before = float(_h_total(H_before, special_degree_meta))
    h_after = float(_h_total(H_after, special_degree_meta))
    lower, upper = spectrum_h_soft_bounds(target_h)
    if target_h is None:
        error_before = error_after = outside_before = outside_after = 0.0
    else:
        error_before = abs(float(h_before) - float(target_h)) / float(target_h)
        error_after = abs(float(h_after) - float(target_h)) / float(target_h)
        ratio_before = (float(h_before) - float(target_h)) / float(target_h)
        ratio_after = (float(h_after) - float(target_h)) / float(target_h)

        def _outside(ratio: float) -> float:
            if float(ratio) < float(SPECTRUM_H_SOFT_LOWER_RATIO):
                return float(SPECTRUM_H_SOFT_LOWER_RATIO) - float(ratio)
            if float(ratio) > float(SPECTRUM_H_SOFT_UPPER_RATIO):
                return float(ratio) - float(SPECTRUM_H_SOFT_UPPER_RATIO)
            return 0.0

        outside_before = _outside(ratio_before)
        outside_after = _outside(ratio_after)

    outside_worsening = max(0.0, float(outside_after) - float(outside_before))
    h_penalty = float(outside_worsening) * float(SPECTRUM_H_OUTSIDE_WORSENING_WEIGHT)

    flex_impact = evaluate_flex_impact(H_before, H_after, flex_audit)
    flex_before = dict(flex_impact.before or {})
    flex_after = dict(flex_impact.after or {})
    flex_deviation_worsening = max(
        0,
        int(flex_after.get("deviation", 0) or 0)
        - int(flex_before.get("deviation", 0) or 0),
    )
    flex_penalty = (
        float(flex_deviation_worsening)
        * float(SPECTRUM_FLEX_DEVIATION_WORSENING_PENALTY)
    )
    if (
        str(flex_after.get("scenario", "")) == "invalid"
        and str(flex_before.get("scenario", "")) != "invalid"
    ):
        flex_penalty += float(SPECTRUM_FLEX_INVALID_WINDOW_PENALTY)

    return {
        "penalty": float(h_penalty + flex_penalty),
        "h_penalty": float(h_penalty),
        "flex_penalty": float(flex_penalty),
        "h_before": float(h_before),
        "h_after": float(h_after),
        "target_h": None if target_h is None else float(target_h),
        "soft_lower": None if lower is None else float(lower),
        "soft_upper": None if upper is None else float(upper),
        "h_error_before": float(error_before),
        "h_error_after": float(error_after),
        "h_outside_before": float(outside_before),
        "h_outside_after": float(outside_after),
        "h_compensation_required": bool(float(outside_after) > 1e-12),
        "h_compensation_suppressed_within_window": bool(float(outside_after) <= 1e-12),
        "spectrum_h_soft_lower_ratio": float(SPECTRUM_H_SOFT_LOWER_RATIO),
        "spectrum_h_soft_upper_ratio": float(SPECTRUM_H_SOFT_UPPER_RATIO),
        "flex_before": flex_before,
        "flex_after": flex_after,
        "soft_constraint": True,
    }


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _window_abs(ppm: np.ndarray, diff: np.ndarray, lo: float, hi: float) -> float:
    if ppm.size == 0 or diff.size == 0 or ppm.size != diff.size:
        return 0.0
    mask = (ppm >= float(lo)) & (ppm <= float(hi))
    return float(np.sum(np.abs(diff[mask]))) if np.any(mask) else 0.0


def _window_signed(ppm: np.ndarray, diff: np.ndarray, lo: float, hi: float) -> Dict[str, float]:
    if ppm.size == 0 or diff.size == 0 or ppm.size != diff.size:
        return {
            "positive": 0.0,
            "negative": 0.0,
            "signed": 0.0,
            "absolute": 0.0,
            "positive_peak": 0.0,
            "negative_peak": 0.0,
            "positive_peak_ppm": None,
            "negative_peak_ppm": None,
        }
    mask = (ppm >= float(lo)) & (ppm <= float(hi))
    values = diff[mask] if np.any(mask) else np.asarray([], dtype=np.float64)
    ppm_values = ppm[mask] if np.any(mask) else np.asarray([], dtype=np.float64)
    positive = float(np.sum(values[values > 0])) if values.size else 0.0
    negative = float(-np.sum(values[values < 0])) if values.size else 0.0
    positive_index = int(np.argmax(values)) if values.size else -1
    negative_index = int(np.argmin(values)) if values.size else -1
    positive_peak = max(0.0, float(values[positive_index])) if positive_index >= 0 else 0.0
    negative_peak = max(0.0, float(-values[negative_index])) if negative_index >= 0 else 0.0
    return {
        "positive": positive,
        "negative": negative,
        "signed": positive - negative,
        "absolute": positive + negative,
        "positive_peak": float(positive_peak),
        "negative_peak": float(negative_peak),
        "positive_peak_ppm": (
            float(ppm_values[positive_index]) if positive_peak > 0.0 else None
        ),
        "negative_peak_ppm": (
            float(ppm_values[negative_index]) if negative_peak > 0.0 else None
        ),
    }


def spectrum_window_audit(
    difference_spectrum: Optional[Dict[str, Any]],
    *,
    windows: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
) -> Dict[str, Any]:
    data = dict(difference_spectrum or {})
    ppm = _array(data.get("ppm"))
    diff = _array(data.get("diff"))
    resolved = dict(windows or DEFAULT_SPECTRUM_WINDOWS)
    rows: Dict[str, Any] = {}
    for su_type in SPECTRUM_RESOURCE_TYPES:
        narrow_lo, narrow_hi, wide_lo, wide_hi = resolved[int(su_type)]
        narrow = _window_signed(ppm, diff, narrow_lo, narrow_hi)
        wide = _window_signed(ppm, diff, wide_lo, wide_hi)
        rows[str(su_type)] = {
            "narrow": narrow,
            "wide": wide,
            "need": float(narrow["signed"]),
            "window": [float(narrow_lo), float(narrow_hi)],
            "wide_window": [float(wide_lo), float(wide_hi)],
        }
    return {
        "valid": bool(ppm.size > 0 and ppm.size == diff.size),
        "ppm_size": int(ppm.size),
        "r2": data.get("r2"),
        "by_su": rows,
    }


def spectrum_window_loss(
    difference_spectrum: Optional[Dict[str, Any]],
    su_types: Optional[Iterable[int]] = None,
    *,
    windows: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
) -> float:
    data = dict(difference_spectrum or {})
    ppm = _array(data.get("ppm"))
    diff = _array(data.get("diff"))
    resolved = dict(windows or DEFAULT_SPECTRUM_WINDOWS)
    selected = sorted({int(value) for value in list(su_types or SPECTRUM_RESOURCE_TYPES)})
    total = 0.0
    for su_type in selected:
        if int(su_type) not in resolved:
            continue
        narrow_lo, narrow_hi, wide_lo, wide_hi = resolved[int(su_type)]
        total += _window_abs(ppm, diff, narrow_lo, narrow_hi)
        if (float(wide_lo), float(wide_hi)) != (float(narrow_lo), float(narrow_hi)):
            total += 0.35 * _window_abs(ppm, diff, wide_lo, wide_hi)
    return float(total)


def _signature(operations: Sequence[Dict[str, Any]]) -> str:
    payload = [
        {
            "node_id": int(dict(op or {}).get("node_id", -1)),
            "from_su": int(dict(op or {}).get("from_su", -1)),
            "to_su": int(dict(op or {}).get("to_su", -1)),
        }
        for op in list(operations or [])
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inverse_bundle_signature(operations: Sequence[Dict[str, Any]]) -> str:
    inverse = [
        {
            "node_id": int(dict(op or {}).get("node_id", -1)),
            "from_su": int(dict(op or {}).get("to_su", -1)),
            "to_su": int(dict(op or {}).get("from_su", -1)),
        }
        for op in reversed(list(operations or []))
    ]
    return _signature(inverse)


def _histogram_after_operations(
    H: torch.Tensor,
    operations: Sequence[Dict[str, Any]],
) -> torch.Tensor:
    candidate = torch.clamp(H.detach().cpu(), min=0).long().clone()
    for raw in list(operations or []):
        op = dict(raw or {})
        source = int(op.get("from_su", -1))
        target = int(op.get("to_su", -1))
        if source < 0 or target < 0 or source >= candidate.numel() or target >= candidate.numel():
            continue
        candidate[source] -= 1
        candidate[target] += 1
    return candidate


def histogram_signature_after_operations(H: torch.Tensor, operations: Sequence[Dict[str, Any]]) -> str:
    candidate = _histogram_after_operations(H, operations)
    encoded = json.dumps(candidate.tolist(), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _spectrum_candidate_h_rank(balance: Dict[str, Any]) -> int:
    """Rank optional spectrum trials without turning the H window into a hard gate."""
    target_h = balance.get("target_h")
    if target_h is None:
        return 0
    outside_before = max(0.0, float(balance.get("h_outside_before", 0.0) or 0.0))
    outside_after = max(0.0, float(balance.get("h_outside_after", 0.0) or 0.0))
    error_before = max(0.0, float(balance.get("h_error_before", 0.0) or 0.0))
    error_after = max(0.0, float(balance.get("h_error_after", 0.0) or 0.0))
    epsilon = 1e-12
    if outside_after <= epsilon:
        return 0 if error_after <= error_before + epsilon else 1
    if outside_after + epsilon < outside_before:
        return 2
    if abs(outside_after - outside_before) <= epsilon:
        return 3
    return 4


def _spectrum_urgency_profiles(audit: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, float]]:
    rows = dict(audit.get("by_su", {}) or {})
    area_scale = max(
        [
            float(dict(dict(rows.get(str(su_type), {}) or {}).get("narrow", {}) or {}).get(key, 0.0) or 0.0)
            for su_type in SPECTRUM_RESOURCE_TYPES
            for key in ("positive", "negative")
        ]
        or [0.0]
    )
    peak_scale = max(
        [
            float(dict(dict(rows.get(str(su_type), {}) or {}).get("narrow", {}) or {}).get(key, 0.0) or 0.0)
            for su_type in SPECTRUM_RESOURCE_TYPES
            for key in ("positive_peak", "negative_peak")
        ]
        or [0.0]
    )
    area_scale = max(float(area_scale), 1e-12)
    peak_scale = max(float(peak_scale), 1e-12)
    profiles: Dict[int, Dict[str, Any]] = {}
    for su_type in SPECTRUM_RESOURCE_TYPES:
        row = dict(rows.get(str(su_type), {}) or {})
        narrow = dict(row.get("narrow", {}) or {})
        positive_area = max(0.0, float(narrow.get("positive", 0.0) or 0.0))
        negative_area = max(0.0, float(narrow.get("negative", 0.0) or 0.0))
        positive_peak = max(0.0, float(narrow.get("positive_peak", 0.0) or 0.0))
        negative_peak = max(0.0, float(narrow.get("negative_peak", 0.0) or 0.0))
        positive_area_norm = float(positive_area) / float(area_scale)
        negative_area_norm = float(negative_area) / float(area_scale)
        positive_peak_norm = float(positive_peak) / float(peak_scale)
        negative_peak_norm = float(negative_peak) / float(peak_scale)
        deficit_score = (
            float(SPECTRUM_PEAK_URGENCY_WEIGHT) * float(positive_peak_norm)
            + float(SPECTRUM_AREA_URGENCY_WEIGHT) * float(positive_area_norm)
        )
        surplus_score = (
            float(SPECTRUM_PEAK_URGENCY_WEIGHT) * float(negative_peak_norm)
            + float(SPECTRUM_AREA_URGENCY_WEIGHT) * float(negative_area_norm)
        )
        epsilon = 1e-12
        if deficit_score > surplus_score + epsilon:
            direction = "deficit"
        elif surplus_score > deficit_score + epsilon:
            direction = "surplus"
        else:
            signed_need = float(row.get("need", 0.0) or 0.0)
            direction = "deficit" if signed_need > 0.0 else "surplus" if signed_need < 0.0 else "none"
        profiles[int(su_type)] = {
            "direction": str(direction),
            "need": float(row.get("need", 0.0) or 0.0),
            "deficit_score": float(deficit_score),
            "surplus_score": float(surplus_score),
            "positive_area": float(positive_area),
            "negative_area": float(negative_area),
            "positive_peak": float(positive_peak),
            "negative_peak": float(negative_peak),
            "positive_peak_ppm": narrow.get("positive_peak_ppm"),
            "negative_peak_ppm": narrow.get("negative_peak_ppm"),
            "positive_area_norm": float(positive_area_norm),
            "negative_area_norm": float(negative_area_norm),
            "positive_peak_norm": float(positive_peak_norm),
            "negative_peak_norm": float(negative_peak_norm),
        }
    dominant_max = max(
        [
            max(float(profile["deficit_score"]), float(profile["surplus_score"]))
            for profile in profiles.values()
        ]
        or [0.0]
    )
    return profiles, {
        "area_scale": float(area_scale),
        "peak_scale": float(peak_scale),
        "score_floor": max(1e-12, 0.01 * float(dominant_max)),
    }


def _difference_ppm_step(ppm: np.ndarray) -> float:
    if int(ppm.size) < 2:
        return 0.1
    differences = np.abs(np.diff(ppm.astype(np.float64, copy=False)))
    finite = differences[np.isfinite(differences) & (differences > 1e-12)]
    return float(np.median(finite)) if finite.size else 0.1


def spectrum_signed_region_audit(
    difference_spectrum: Optional[Dict[str, Any]],
    lo: float,
    hi: float,
) -> Dict[str, Any]:
    """Integrate positive and negative difference-spectrum lobes in one region."""
    data = dict(difference_spectrum or {})
    ppm = _array(data.get("ppm"))
    diff = _array(data.get("diff"))
    valid = bool(ppm.size > 0 and ppm.size == diff.size)
    if not valid:
        return {
            "valid": False,
            "window": [float(lo), float(hi)],
            "positive_area": 0.0,
            "negative_area": 0.0,
            "signed_area": 0.0,
            "absolute_area": 0.0,
            "negative_fraction": 0.0,
            "positive_fraction": 0.0,
            "net_surplus_area": 0.0,
            "net_deficit_area": 0.0,
            "ppm_step": 0.0,
        }
    row = _window_signed(ppm, diff, float(lo), float(hi))
    step = float(_difference_ppm_step(ppm))
    positive = float(row["positive"]) * step
    negative = float(row["negative"]) * step
    absolute = float(positive + negative)
    return {
        "valid": True,
        "window": [float(lo), float(hi)],
        "positive_area": float(positive),
        "negative_area": float(negative),
        "signed_area": float(positive - negative),
        "absolute_area": float(absolute),
        "negative_fraction": (
            float(negative) / float(absolute) if absolute > 1e-12 else 0.0
        ),
        "positive_fraction": (
            float(positive) / float(absolute) if absolute > 1e-12 else 0.0
        ),
        "net_surplus_area": max(0.0, float(negative - positive)),
        "net_deficit_area": max(0.0, float(positive - negative)),
        "positive_peak": float(row["positive_peak"]),
        "negative_peak": float(row["negative_peak"]),
        "positive_peak_ppm": row["positive_peak_ppm"],
        "negative_peak_ppm": row["negative_peak_ppm"],
        "ppm_step": float(step),
    }


def aliphatic_negative_excess_audit(
    difference_spectrum: Optional[Dict[str, Any]],
    *,
    threshold_carbon_count: float = SPECTRUM_ALIPHATIC_NEGATIVE_CARBON_THRESHOLD,
) -> Dict[str, Any]:
    """Audit the local 0--90 ppm negative lobe used by spectrum rebalancing.

    ``difference = target - reconstruction``; consequently a negative lobe in
    this interval means that the current reconstruction contains too much
    aliphatic signal.  The project uses unit Lorentzian peaks with HWHM=1 and
    normalizes one carbon to an area of pi, so the equivalent-carbon estimate
    is intentionally based on that same convention rather than on the raw
    grid-point sum.
    """
    data = dict(difference_spectrum or {})
    row = spectrum_signed_region_audit(data, 0.0, 90.0)
    unit_area = float(math.pi) * float(SPECTRUM_UNIT_PEAK_HWHM)
    negative_area = max(0.0, float(row.get("negative_area", 0.0) or 0.0))
    positive_area = max(0.0, float(row.get("positive_area", 0.0) or 0.0))
    equivalent_carbon = (
        float(negative_area) / float(unit_area)
        if float(unit_area) > 1e-12
        else 0.0
    )
    threshold_area = max(0.0, float(threshold_carbon_count)) * float(unit_area)
    layer2_contract = bool(
        str(data.get("source_stage", "") or "") == "layer2"
        and str(data.get("spectral_contract", "") or "")
        == "template_center_mu"
    )
    area_threshold_triggered = bool(
        row.get("valid", False)
        and float(negative_area) > float(threshold_area) + 1e-12
    )
    return {
        "valid": bool(row.get("valid", False)),
        "window": [0.0, 90.0],
        "negative_area_0_90": float(negative_area),
        "positive_area_0_90": float(positive_area),
        "net_surplus_area_0_90": max(0.0, float(negative_area - positive_area)),
        "unit_peak_area": float(unit_area),
        "negative_equivalent_carbon": float(equivalent_carbon),
        "threshold_carbon_count": float(threshold_carbon_count),
        "threshold_area": float(threshold_area),
        "area_threshold_triggered": bool(area_threshold_triggered),
        "triggered": bool(area_threshold_triggered and layer2_contract),
        "aliphatic_excess_triggered": bool(
            area_threshold_triggered and layer2_contract
        ),
        "source_stage": str(data.get("source_stage", "") or ""),
        "spectral_contract": str(data.get("spectral_contract", "") or ""),
        "layer2_contract": bool(layer2_contract),
        "region": dict(row),
    }


def unsaturated_window_audit(
    difference_spectrum: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return the explicit SU14/15/16 windows used by pair candidates."""
    return {
        str(su_type): dict(
            spectrum_signed_region_audit(
                difference_spectrum,
                *UNSATURATED_DIFF_WINDOWS[int(su_type)],
            )
        )
        for su_type in UNSATURATED_SPECTRUM_TYPES
    }


def fixed_flex_capacity_spectrum_audit(
    difference_spectrum: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the spectral evidence required specifically for SU23 -> SU13."""
    data = dict(difference_spectrum or {})
    source_stage = str(data.get("source_stage", "") or "")
    spectral_contract = str(data.get("spectral_contract", "") or "")
    layer2_contract = bool(
        source_stage == "layer2"
        and spectral_contract == "template_center_mu"
    )
    aliphatic = spectrum_signed_region_audit(data, 0.0, 90.0)
    su13_lo, su13_hi = get_su_ppm_range(13)
    su13 = spectrum_signed_region_audit(data, su13_lo, su13_hi)
    donor_regions = {
        str(su_type): spectrum_signed_region_audit(
            data,
            *get_su_ppm_range(int(su_type)),
        )
        for su_type in (10, 11, 12, 23)
    }
    aliphatic_epsilon = max(
        1e-12,
        1e-6 * float(aliphatic.get("absolute_area", 0.0) or 0.0),
    )
    su13_epsilon = max(
        1e-12,
        1e-6 * float(su13.get("absolute_area", 0.0) or 0.0),
    )
    aliphatic_surplus = bool(
        aliphatic.get("valid", False)
        and float(aliphatic.get("net_surplus_area", 0.0) or 0.0)
        > float(aliphatic_epsilon)
    )
    su13_deficit = bool(
        su13.get("valid", False)
        and float(su13.get("net_deficit_area", 0.0) or 0.0)
        > float(su13_epsilon)
    )
    donor_surplus = {
        str(su_type): bool(
            float(donor_regions[str(su_type)].get("net_surplus_area", 0.0) or 0.0)
            > max(
                1e-12,
                1e-6
                * float(
                    donor_regions[str(su_type)].get("absolute_area", 0.0)
                    or 0.0
                ),
            )
        )
        for su_type in (10, 11, 12, 23)
    }
    window_data_valid = bool(
        aliphatic.get("valid", False) and su13.get("valid", False)
    )
    return {
        "valid": bool(window_data_valid and layer2_contract),
        "window_data_valid": bool(window_data_valid),
        "layer2_contract": bool(layer2_contract),
        "source_stage": source_stage,
        "spectral_contract": spectral_contract,
        "r2": data.get("r2"),
        "aliphatic_0_90": dict(aliphatic),
        "su13_target": dict(su13),
        "donor_regions": donor_regions,
        "aliphatic_surplus": bool(aliphatic_surplus),
        "su13_deficit": bool(su13_deficit),
        "donor_surplus": donor_surplus,
        "eligible": bool(
            window_data_valid
            and layer2_contract
            and aliphatic_surplus
            and su13_deficit
        ),
    }


def _deoverlapped_lobe_areas(
    ppm: np.ndarray,
    diff: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    areas = {
        int(su_type): {"positive": 0.0, "negative": 0.0}
        for su_type in SPECTRUM_RESOURCE_TYPES
    }
    if ppm.size == 0 or ppm.size != diff.size:
        return areas
    support = np.zeros((len(SPECTRUM_RESOURCE_TYPES), int(ppm.size)), dtype=np.float64)
    for row_index, su_type in enumerate(SPECTRUM_RESOURCE_TYPES):
        lo, hi, _, _ = DEFAULT_SPECTRUM_WINDOWS[int(su_type)]
        width = max(float(hi) - float(lo), 1e-12)
        support[row_index, (ppm >= float(lo)) & (ppm <= float(hi))] = 1.0 / float(width)
    denominator = np.sum(support, axis=0)
    covered = denominator > 0.0
    if not np.any(covered):
        return areas
    shares = np.zeros_like(support)
    shares[:, covered] = support[:, covered] / denominator[covered]
    step = _difference_ppm_step(ppm)
    positive = np.maximum(diff, 0.0)
    negative = np.maximum(-diff, 0.0)
    for row_index, su_type in enumerate(SPECTRUM_RESOURCE_TYPES):
        areas[int(su_type)] = {
            "positive": float(np.sum(positive * shares[row_index]) * float(step)),
            "negative": float(np.sum(negative * shares[row_index]) * float(step)),
        }
    return areas


def _apportion_integer_counts(
    total: int,
    weights: Dict[int, float],
    capacities: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    allocation = {int(key): 0 for key in weights}
    capacity_by_key = {
        int(key): max(0, int(dict(capacities or {}).get(int(key), int(total))))
        for key in weights
    }
    for _ in range(max(0, int(total))):
        eligible = [
            int(key)
            for key, weight in weights.items()
            if float(weight) > 0.0
            and int(allocation[int(key)]) < int(capacity_by_key[int(key)])
        ]
        if not eligible:
            break
        selected = max(
            eligible,
            key=lambda key: (
                float(weights[int(key)]) / float(allocation[int(key)] + 1),
                float(weights[int(key)]),
                -int(key),
            ),
        )
        allocation[int(selected)] += 1
    return {
        int(key): int(value)
        for key, value in allocation.items()
        if int(value) > 0
    }


def _direction_weights(
    su_types: Sequence[int],
    urgency_profiles: Dict[int, Dict[str, Any]],
    lobe_areas: Dict[int, Dict[str, float]],
    *,
    direction: str,
) -> Dict[int, float]:
    area_key = "negative" if str(direction) == "surplus" else "positive"
    peak_key = "negative_peak" if str(direction) == "surplus" else "positive_peak"
    area_total = sum(
        max(0.0, float(lobe_areas[int(su_type)][area_key]))
        for su_type in su_types
    )
    peak_total = sum(
        max(0.0, float(urgency_profiles[int(su_type)][peak_key]))
        for su_type in su_types
    )
    weights: Dict[int, float] = {}
    for su_type in su_types:
        area_share = (
            max(0.0, float(lobe_areas[int(su_type)][area_key])) / float(area_total)
            if float(area_total) > 1e-12
            else 0.0
        )
        peak_share = (
            max(0.0, float(urgency_profiles[int(su_type)][peak_key])) / float(peak_total)
            if float(peak_total) > 1e-12
            else 0.0
        )
        weights[int(su_type)] = (
            float(SPECTRUM_PEAK_URGENCY_WEIGHT) * float(peak_share)
            + float(SPECTRUM_AREA_URGENCY_WEIGHT) * float(area_share)
        )
    return weights


def _transport_operations(
    *,
    batch_size: int,
    source_weights: Dict[int, float],
    target_weights: Dict[int, float],
    available_ids_by_su: Dict[int, Sequence[int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[int, int], Dict[int, int]]:
    source_counts = _apportion_integer_counts(
        int(batch_size),
        source_weights,
        capacities={
            int(su_type): len(list(available_ids_by_su.get(int(su_type), []) or []))
            for su_type in source_weights
        },
    )
    actual_batch = sum(int(value) for value in source_counts.values())
    target_counts = _apportion_integer_counts(int(actual_batch), target_weights)
    source_remaining = dict(source_counts)
    target_remaining = dict(target_counts)
    source_offsets = {int(key): 0 for key in source_counts}
    operations: List[Dict[str, Any]] = []
    conversion_counts: Dict[str, int] = {}
    while source_remaining and target_remaining:
        source = max(
            source_remaining,
            key=lambda key: (
                float(source_weights.get(int(key), 0.0)),
                int(source_remaining[int(key)]),
                -int(key),
            ),
        )
        target = max(
            target_remaining,
            key=lambda key: (
                float(target_weights.get(int(key), 0.0)),
                int(target_remaining[int(key)]),
                -int(key),
            ),
        )
        amount = min(
            int(source_remaining[int(source)]),
            int(target_remaining[int(target)]),
        )
        source_ids = list(available_ids_by_su.get(int(source), []) or [])
        start = int(source_offsets[int(source)])
        for node_id in source_ids[start:start + int(amount)]:
            operations.append({
                "op": "convert",
                "node_id": int(node_id),
                "from_su": int(source),
                "to_su": int(target),
                "reason": "remaining_pool_difference_spectrum_proportional_bundle",
            })
        source_offsets[int(source)] += int(amount)
        conversion_key = f"{int(source)}->{int(target)}"
        conversion_counts[conversion_key] = int(conversion_counts.get(conversion_key, 0)) + int(amount)
        source_remaining[int(source)] -= int(amount)
        target_remaining[int(target)] -= int(amount)
        if int(source_remaining[int(source)]) <= 0:
            source_remaining.pop(int(source), None)
        if int(target_remaining[int(target)]) <= 0:
            target_remaining.pop(int(target), None)
    return operations, conversion_counts, source_counts, target_counts


def _bundle_delta(operations: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    delta: Dict[int, int] = {}
    for raw in operations:
        op = dict(raw or {})
        source = int(op.get("from_su", -1))
        target = int(op.get("to_su", -1))
        delta[source] = int(delta.get(source, 0)) - 1
        delta[target] = int(delta.get(target, 0)) + 1
    return {int(key): int(value) for key, value in sorted(delta.items()) if int(value) != 0}


def _su12_cap_for_operations(
    H: torch.Tensor,
    operations: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate every SU12-generating step in a spectrum candidate."""
    state = torch.clamp(H.detach().cpu(), min=0).long().clone()
    for raw in list(operations or []):
        operation = dict(raw or {})
        source = int(operation.get("from_su", -1))
        target = int(operation.get("to_su", -1))
        if not (0 <= source < int(state.numel()) and 0 <= target < int(state.numel())):
            continue
        count = max(1, int(operation.get("count", 1) or 1))
        for _ in range(count):
            if int(state[source].item()) <= 0:
                return {
                    "generation_allowed": False,
                    "reason": "spectrum_candidate_histogram_underflow",
                    "from_su": source,
                    "to_su": target,
                }
            before = state.detach().clone()
            state[source] -= 1
            state[target] += 1
            transition = su12_transition_diagnostics(before, state)
            if not bool(transition.get("generation_allowed", True)):
                return {
                    **dict(transition),
                    "generation_allowed": False,
                    "reason": "su12_generation_cap_reached",
                    "from_su": source,
                    "to_su": target,
                }
    return {
        **dict(su12_transition_diagnostics(H, state)),
        "generation_allowed": True,
        "reason": "ok",
    }


def _append_low_h_compensation(
    *,
    H: torch.Tensor,
    primary_operations: Sequence[Dict[str, Any]],
    available_ids_by_su: Dict[int, Sequence[int]],
    E_target: Optional[torch.Tensor],
    special_degree_meta: Optional[Dict[str, Any]],
    max_h_compensation_ops: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Add the minimum SU11->SU23 compensation needed to re-enter the H window."""
    primary = [dict(op or {}) for op in list(primary_operations or [])]
    primary_H = _histogram_after_operations(H, primary)
    h_before = float(_h_total(H, special_degree_meta))
    h_after_primary = float(_h_total(primary_H, special_degree_meta))
    try:
        target_h = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        target_h = None
    if target_h is not None and float(target_h) <= 0.0:
        target_h = None
    lower, upper = spectrum_h_soft_bounds(target_h)
    diagnostics: Dict[str, Any] = {
        "policy": "minimum_su11_to_su23_until_spectrum_h_window",
        "h_before": float(h_before),
        "h_after_primary": float(h_after_primary),
        "h_after_compensation": float(h_after_primary),
        "target_h": None if target_h is None else float(target_h),
        "soft_lower": None if lower is None else float(lower),
        "soft_upper": None if upper is None else float(upper),
        "required_count": 0,
        "applied_count": 0,
        "available_su11_count": 0,
        "max_count": max(0, int(max_h_compensation_ops)),
        "complete": True,
        "reason": "not_required",
    }
    if lower is None or float(h_after_primary) >= float(lower) - 1e-12:
        return primary, diagnostics

    used_node_ids = {
        int(op.get("node_id", -1))
        for op in primary
        if int(op.get("node_id", -1)) >= 0
    }
    available_su11_ids = [
        int(node_id)
        for node_id in list(available_ids_by_su.get(11, []) or [])
        if int(node_id) not in used_node_ids
    ]
    diagnostics["available_su11_count"] = len(available_su11_ids)

    if int(primary_H.numel()) <= 23 or int(primary_H[11].item()) <= 0:
        diagnostics.update({"complete": False, "reason": "su11_histogram_unavailable"})
        return primary, diagnostics
    probe_H = primary_H.clone()
    probe_H[11] -= 1
    probe_H[23] += 1
    h_per_op = float(_h_total(probe_H, special_degree_meta)) - float(h_after_primary)
    diagnostics["h_gain_per_operation"] = float(h_per_op)
    if float(h_per_op) <= 0.0:
        diagnostics.update({"complete": False, "reason": "nonpositive_su11_to_su23_h_gain"})
        return primary, diagnostics

    required_count = max(
        0,
        int(math.ceil((float(lower) - float(h_after_primary) - 1e-12) / float(h_per_op))),
    )
    applied_count = min(
        int(required_count),
        max(0, int(max_h_compensation_ops)),
        len(available_su11_ids),
        int(primary_H[11].item()),
    )
    compensation = [
        {
            "op": "convert",
            "node_id": int(node_id),
            "from_su": 11,
            "to_su": 23,
            "reason": "remaining_pool_spectrum_h_compensation",
            "operation_kind": "h_compensation",
        }
        for node_id in available_su11_ids[:applied_count]
    ]
    combined = primary + compensation
    compensated_H = _histogram_after_operations(H, combined)
    h_after_compensation = float(_h_total(compensated_H, special_degree_meta))
    complete = bool(float(h_after_compensation) >= float(lower) - 1e-12)
    diagnostics.update({
        "required_count": int(required_count),
        "applied_count": int(applied_count),
        "h_after_compensation": float(h_after_compensation),
        "complete": bool(complete),
        "reason": (
            "restored_to_window"
            if complete
            else "compensation_limit_reached"
            if int(applied_count) >= max(0, int(max_h_compensation_ops))
            else "insufficient_available_su11"
        ),
    })
    return combined, diagnostics


_SPECTRUM_PROFILE_FIELDS = (
    "mu_median",
    "mu_common_min",
    "mu_common_max",
    "mu_q05",
    "mu_q95",
)
_SPECTRUM_GAIN_EPSILON = 1e-10


def _validated_spectrum_profiles(
    profile_stats: Optional[Dict[int, Dict[str, Any]]],
    su_types: Optional[Iterable[int]] = None,
) -> Tuple[Optional[Dict[int, Dict[str, float]]], Dict[str, Any]]:
    profiles: Dict[int, Dict[str, float]] = {}
    missing: List[int] = []
    invalid: Dict[str, str] = {}
    raw_stats = dict(profile_stats or {})
    required_types = tuple(
        sorted({
            int(value) for value in list(
                SPECTRUM_RESOURCE_TYPES if su_types is None else su_types
            )
        })
    )
    for su_type in required_types:
        raw = raw_stats.get(int(su_type), raw_stats.get(str(int(su_type))))
        if not isinstance(raw, dict):
            missing.append(int(su_type))
            continue
        try:
            row = {
                field: float(raw[field])
                for field in _SPECTRUM_PROFILE_FIELDS
            }
        except (KeyError, TypeError, ValueError) as exc:
            invalid[str(int(su_type))] = f"missing_or_invalid_field:{exc}"
            continue
        if not all(np.isfinite(value) for value in row.values()):
            invalid[str(int(su_type))] = "nonfinite_profile_value"
            continue
        lo = float(row["mu_common_min"])
        q05 = float(row["mu_q05"])
        median = float(row["mu_median"])
        q95 = float(row["mu_q95"])
        hi = float(row["mu_common_max"])
        if not (lo < q05 <= median <= q95 < hi):
            invalid[str(int(su_type))] = "invalid_profile_order"
            continue
        profiles[int(su_type)] = row
    diagnostics = {
        "profile_source": "su_nmr_common_range_filtered.csv",
        "required_su_types": list(required_types),
        "missing_su_types": missing,
        "invalid_profiles": invalid,
    }
    if missing or invalid or len(profiles) != len(required_types):
        return None, {**diagnostics, "reason": "spectrum_profile_unavailable"}
    return profiles, {**diagnostics, "reason": "ok"}


def _spectrum_profile_weights(
    ppm: np.ndarray,
    profiles: Dict[int, Dict[str, float]],
) -> np.ndarray:
    """Return normalized trapezoid support for the six adjustable SU types."""
    raw = np.zeros((len(SPECTRUM_RESOURCE_TYPES), int(ppm.size)), dtype=np.float64)
    for row_index, su_type in enumerate(SPECTRUM_RESOURCE_TYPES):
        profile = profiles[int(su_type)]
        lo = float(profile["mu_common_min"])
        q05 = float(profile["mu_q05"])
        q95 = float(profile["mu_q95"])
        hi = float(profile["mu_common_max"])
        rising = (ppm > lo) & (ppm < q05)
        core = (ppm >= q05) & (ppm <= q95)
        falling = (ppm > q95) & (ppm < hi)
        raw[row_index, rising] = (ppm[rising] - lo) / (q05 - lo)
        raw[row_index, core] = 1.0
        raw[row_index, falling] = (hi - ppm[falling]) / (hi - q95)
    denominator = np.sum(raw, axis=0)
    normalized = raw.copy()
    overlap = np.count_nonzero(raw > 0.0, axis=0) > 1
    normalized[:, overlap] = raw[:, overlap] / denominator[overlap]
    return normalized


def _lorentzian_peak(
    ppm: np.ndarray,
    mu: float,
    hwhm: float = 1.0,
    *,
    su_type: Optional[int] = None,
    aromatic_response_scale: float = 1.0,
) -> np.ndarray:
    hwhm_sq = float(hwhm) ** 2
    amplitude = reconstruction_peak_scale(
        int(su_type) if su_type is not None else -1,
        aromatic_response_scale,
    )
    return float(amplitude) * hwhm_sq / ((ppm - float(mu)) ** 2 + hwhm_sq)


def _profile_support_at_mu(profile: Dict[str, float], mu: float) -> float:
    lo = float(profile["mu_common_min"])
    q05 = float(profile["mu_q05"])
    q95 = float(profile["mu_q95"])
    hi = float(profile["mu_common_max"])
    value = float(mu)
    if value <= lo or value >= hi:
        return 0.0
    if value < q05:
        return float((value - lo) / (q05 - lo))
    if value <= q95:
        return 1.0
    return float((hi - value) / (hi - q95))


def _validated_hop1_template_options(
    options_by_su: Optional[Dict[int, Any]],
    profiles: Dict[int, Dict[str, float]],
    su_types: Optional[Iterable[int]] = None,
) -> Tuple[Optional[Dict[int, List[Dict[str, Any]]]], Dict[str, Any]]:
    """Normalize filtered-library 1-hop peaks for spectrum-only prediction."""
    normalized: Dict[int, List[Dict[str, Any]]] = {}
    missing: List[int] = []
    invalid_counts: Dict[str, int] = {}
    raw_options = dict(options_by_su or {})
    required_types = tuple(
        sorted({
            int(value) for value in list(
                SPECTRUM_RESOURCE_TYPES if su_types is None else su_types
            )
        })
    )
    for su_type in required_types:
        raw_rows = raw_options.get(int(su_type), raw_options.get(str(int(su_type)), []))
        unique: Dict[Tuple[Tuple[int, ...], float], Dict[str, Any]] = {}
        invalid_count = 0
        for raw in list(raw_rows or []):
            row = dict(raw or {})
            try:
                hop1_tuple = tuple(sorted(int(value) for value in tuple(row["hop1_tuple"])))
                mu = float(row["mu_median"])
                sample_weight = float(row.get("sample_weight_sum", 0.0) or 0.0)
                n_templates = int(row.get("n_templates", 0) or 0)
            except (KeyError, TypeError, ValueError):
                invalid_count += 1
                continue
            support = _profile_support_at_mu(profiles[int(su_type)], mu)
            explicit_window = UNSATURATED_DIFF_WINDOWS.get(int(su_type))
            outside_explicit_window = bool(
                explicit_window is not None
                and not (
                    float(explicit_window[0]) <= float(mu)
                    <= float(explicit_window[1])
                )
            )
            if (
                not hop1_tuple
                or not np.isfinite(mu)
                or mu == 0.0
                or support <= 0.0
                or outside_explicit_window
            ):
                invalid_count += 1
                continue
            item = {
                "su_type": int(su_type),
                "hop1_tuple": hop1_tuple,
                "mu_median": float(mu),
                "sample_weight_sum": float(max(0.0, sample_weight)),
                "n_templates": int(max(0, n_templates)),
                "profile_support": float(support),
            }
            key = (hop1_tuple, float(mu))
            previous = unique.get(key)
            if previous is None or (
                float(item["sample_weight_sum"]), int(item["n_templates"])
            ) > (
                float(previous["sample_weight_sum"]), int(previous["n_templates"])
            ):
                unique[key] = item
        rows = sorted(
            unique.values(),
            key=lambda item: (
                float(item["mu_median"]),
                tuple(item["hop1_tuple"]),
                -float(item["sample_weight_sum"]),
            ),
        )
        if not rows:
            missing.append(int(su_type))
        else:
            normalized[int(su_type)] = rows
        if invalid_count:
            invalid_counts[str(int(su_type))] = int(invalid_count)
    diagnostics = {
        "hop1_profile_source": "filtered_template_library_exact_hop1",
        "hop1_missing_su_types": missing,
        "hop1_invalid_option_counts": invalid_counts,
        "hop1_option_counts": {
            str(su_type): len(normalized.get(int(su_type), []))
            for su_type in required_types
        },
    }
    if missing:
        return None, {**diagnostics, "reason": "spectrum_hop1_profile_unavailable"}
    return normalized, {**diagnostics, "reason": "ok"}


def _profile_loss_from_arrays(
    diff: np.ndarray,
    normalized_weights: np.ndarray,
    su_types: Optional[Iterable[int]] = None,
) -> float:
    selected = set(int(value) for value in list(su_types or SPECTRUM_RESOURCE_TYPES))
    rows = [
        row_index
        for row_index, su_type in enumerate(SPECTRUM_RESOURCE_TYPES)
        if int(su_type) in selected
    ]
    if not rows:
        return 0.0
    support = np.sum(normalized_weights[rows, :], axis=0)
    return float(np.sum(np.abs(diff) * support))


def spectrum_profile_loss(
    difference_spectrum: Optional[Dict[str, Any]],
    profile_stats: Optional[Dict[int, Dict[str, Any]]],
    su_types: Optional[Iterable[int]] = None,
) -> float:
    profiles, _ = _validated_spectrum_profiles(profile_stats)
    data = dict(difference_spectrum or {})
    ppm = _array(data.get("ppm"))
    diff = _array(data.get("diff"))
    if profiles is None or ppm.size == 0 or ppm.size != diff.size:
        return float("inf")
    weights = _spectrum_profile_weights(ppm, profiles)
    return _profile_loss_from_arrays(diff, weights, su_types)


def _atomic_score_state(
    *,
    residual: np.ndarray,
    donor_rows: Sequence[Dict[str, Any]],
    donor_peaks: np.ndarray,
    target_rows: Sequence[Dict[str, Any]],
    target_peaks: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Score physical peak coverage directly on the current residual.

    The CSV profiles validate that a target template belongs to its SU range;
    they must not attenuate either the donor or target coverage score.
    """
    negative = np.maximum(-residual, 0.0)
    positive = np.maximum(residual, 0.0)
    donor_coverage_peaks = np.where(donor_peaks >= 0.5, donor_peaks, 0.0)
    target_coverage_peaks = np.where(target_peaks >= 0.5, target_peaks, 0.0)
    return {
        "donor_surplus": donor_coverage_peaks @ negative,
        "target_fill": target_coverage_peaks @ positive,
        "residual_source_dot": donor_peaks @ residual,
        "residual_target_dot": target_peaks @ residual,
    }


def _ranked_atomics_for_direction(
    *,
    residual: np.ndarray,
    source_su: int,
    target_su: int,
    donor_rows: Sequence[Dict[str, Any]],
    donor_peaks: np.ndarray,
    donor_peak_norms: np.ndarray,
    target_rows: Sequence[Dict[str, Any]],
    target_peaks: np.ndarray,
    target_peak_norms: np.ndarray,
    donor_target_dot: np.ndarray,
    score_state: Dict[str, np.ndarray],
    used_node_ids: set[int],
    H_state: torch.Tensor,
    max_results: int,
) -> List[Dict[str, Any]]:
    if int(source_su) == int(target_su):
        return []
    if (
        int(source_su) in {12, 13}
        and int(H_state[int(source_su)].item())
        <= int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT)
    ):
        return []
    eligible_indices = [
        index
        for index, row in enumerate(donor_rows)
        if int(row["su_type"]) == int(source_su)
        and int(row["node_id"]) not in used_node_ids
    ]
    if not eligible_indices:
        return []
    eligible_targets = [
        index for index, row in enumerate(target_rows)
        if int(row["su_type"]) == int(target_su)
    ]
    if not eligible_targets:
        return []
    donor_surplus = score_state["donor_surplus"][eligible_indices]
    valid_surplus = donor_surplus > _SPECTRUM_GAIN_EPSILON
    if not np.any(valid_surplus):
        return []
    target_fill = score_state["target_fill"][eligible_targets]
    valid_fill = target_fill > _SPECTRUM_GAIN_EPSILON
    if not np.any(valid_fill):
        return []
    residual_source_dot = score_state["residual_source_dot"][eligible_indices]
    residual_target_dot = score_state["residual_target_dot"][eligible_targets]
    source_norms = donor_peak_norms[eligible_indices]
    target_norms = target_peak_norms[eligible_targets]
    source_target_dot = donor_target_dot[np.ix_(eligible_indices, eligible_targets)]
    gains = (
        -2.0 * (
            residual_source_dot[:, None] - residual_target_dot[None, :]
        )
        - (
            source_norms[:, None] + target_norms[None, :]
            - 2.0 * source_target_dot
        )
    )
    valid = (
        valid_surplus[:, None]
        & valid_fill[None, :]
        & (gains > _SPECTRUM_GAIN_EPSILON)
    )
    if not np.any(valid):
        return []

    result_limit = max(0, int(max_results))
    if result_limit <= 0:
        return []
    valid_count = int(np.count_nonzero(valid))
    result_limit = min(int(result_limit), int(valid_count))
    if result_limit == 1:
        threshold_gain = float(np.max(np.where(valid, gains, -np.inf)))
    else:
        valid_gains = gains[valid]
        threshold_gain = float(
            np.partition(valid_gains, int(valid_gains.size) - result_limit)[
                int(valid_gains.size) - result_limit
            ]
        )
    # Include the complete threshold tie so secondary keys remain exact.
    valid_pairs = np.argwhere(valid & (gains >= threshold_gain))
    ranked_pairs = sorted(
        ((int(pair[0]), int(pair[1])) for pair in valid_pairs),
        key=lambda pair: (
            -float(gains[pair[0], pair[1]]),
            -float(target_fill[pair[1]]),
            -float(target_rows[eligible_targets[pair[1]]]["sample_weight_sum"]),
            int(source_su),
            int(target_su),
            int(donor_rows[eligible_indices[pair[0]]]["node_id"]),
            tuple(target_rows[eligible_targets[pair[1]]]["hop1_tuple"]),
        ),
    )[:result_limit]
    operations: List[Dict[str, Any]] = []
    for donor_local, target_local in ranked_pairs:
        donor_index = int(eligible_indices[donor_local])
        target_index = int(eligible_targets[target_local])
        node_id = int(donor_rows[donor_index]["node_id"])
        source_peak = donor_peaks[donor_index]
        target_peak = target_peaks[target_index]
        target_row = dict(target_rows[target_index])
        operations.append({
            "op": "convert",
            "node_id": int(node_id),
            "from_su": int(source_su),
            "to_su": int(target_su),
            "reason": "remaining_pool_difference_spectrum_nodewise_greedy",
            "source_mu": float(donor_rows[donor_index]["mu"]),
            "target_mu_goal": float(target_row["mu_median"]),
            "simulated_target_hop1": [
                int(value) for value in tuple(target_row["hop1_tuple"])
            ],
            "target_template_sample_weight": float(
                target_row["sample_weight_sum"]
            ),
            "target_template_count": int(target_row["n_templates"]),
            "target_profile_support": float(target_row["profile_support"]),
            "predicted_sse_gain": float(gains[donor_local, target_local]),
            "predicted_target_score": float(target_fill[target_local]),
            "donor_surplus_score": float(donor_surplus[donor_local]),
            "_next_residual": residual + source_peak - target_peak,
        })
    return operations


def _best_atomic_for_direction(**kwargs: Any) -> Optional[Dict[str, Any]]:
    ranked = _ranked_atomics_for_direction(**kwargs, max_results=1)
    return ranked[0] if ranked else None


def _operation_sort_key(operation: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(operation.get("predicted_sse_gain", 0.0) or 0.0),
        -float(operation.get("predicted_target_score", 0.0) or 0.0),
        -float(operation.get("target_template_sample_weight", 0.0) or 0.0),
        int(operation.get("from_su", -1)),
        int(operation.get("to_su", -1)),
        int(operation.get("node_id", -1)),
        tuple(int(value) for value in operation.get("simulated_target_hop1", []) or []),
    )


def _su12_bias_score(
    operations: Sequence[Dict[str, Any]],
    aliphatic_audit: Optional[Dict[str, Any]],
) -> float:
    """Return the structural 13->12 tie-break score for one candidate.

    This is intentionally a secondary score.  The spectrum SSE/R2 gain stays
    authoritative; the score only prefers a candidate that consumes more
    SU13 into SU12 when the local aliphatic excess audit is active.
    """
    audit = dict(aliphatic_audit or {})
    if not bool(audit.get("triggered", False)):
        return 0.0
    excess = max(
        0.0,
        float(audit.get("negative_equivalent_carbon", 0.0) or 0.0)
        - float(audit.get("threshold_carbon_count", SPECTRUM_ALIPHATIC_NEGATIVE_CARBON_THRESHOLD) or 0.0),
    )
    count = sum(
        1
        for raw in list(operations or [])
        if int(dict(raw or {}).get("from_su", -1)) == 13
        and int(dict(raw or {}).get("to_su", -1)) == 12
    )
    return float(count) * float(excess)


def _spectrum_node_provenance(node: Any) -> Dict[str, Any]:
    """Capture topology ownership metadata for reversible spectrum trials."""
    locked = getattr(node, "stage3_locked_neighbor_ids", set())
    try:
        locked_ids = sorted(int(value) for value in list(locked or []))
    except (TypeError, ValueError):
        locked_ids = []
    owner_id = getattr(node, "stage3_topology_owner_id", None)
    try:
        owner_id = None if owner_id is None else int(owner_id)
    except (TypeError, ValueError):
        owner_id = None
    return {
        "source_owner_product_id": owner_id,
        "source_topology_role": str(
            getattr(node, "stage3_topology_role", "") or ""
        ),
        "source_locked_neighbor_ids": locked_ids,
        "source_resource_claimed": bool(
            getattr(node, "stage3_resource_claimed", False)
        ),
    }


def _candidate_from_path(
    *,
    H: torch.Tensor,
    baseline_diff: np.ndarray,
    normalized_weights: np.ndarray,
    seed_direction: Tuple[int, int],
    operations: Sequence[Dict[str, Any]],
    residual: np.ndarray,
    pair_groups: Optional[Sequence[Dict[str, Any]]] = None,
    su12_bias_score: float = 0.0,
    spectrum_audit: Optional[Dict[str, Any]] = None,
) -> Stage3SpectrumCandidate:
    public_operations = [
        {key: value for key, value in dict(operation).items() if not str(key).startswith("_")}
        for operation in operations
    ]
    bundle_signature = _signature(public_operations)
    primary_inverse_signature = inverse_bundle_signature(public_operations)
    histogram_signature = histogram_signature_after_operations(H, public_operations)
    target_types = sorted({int(op["to_su"]) for op in public_operations})
    predicted_sse_gain = float(
        np.dot(baseline_diff, baseline_diff) - np.dot(residual, residual)
    )
    predicted_window_gain = float(
        _profile_loss_from_arrays(baseline_diff, normalized_weights, target_types)
        - _profile_loss_from_arrays(residual, normalized_weights, target_types)
    )
    operation_gains = [
        float(op.get("predicted_sse_gain", 0.0) or 0.0)
        for op in public_operations
    ]
    conversion_counts: Dict[str, int] = {}
    for op in public_operations:
        key = f"{int(op['from_su'])}->{int(op['to_su'])}"
        conversion_counts[key] = int(conversion_counts.get(key, 0)) + 1
    candidate = Stage3SpectrumCandidate(
        candidate_id=(
            f"spectrum:nodewise_greedy:{len(public_operations)}:"
            f"{int(seed_direction[0])}-{int(seed_direction[1])}:{bundle_signature[:10]}"
        ),
        histogram_signature=str(histogram_signature),
        inverse_signature=str(primary_inverse_signature),
        operations=public_operations,
        pair_groups=[dict(group or {}) for group in list(pair_groups or [])],
        target_su_types=sorted({
            int(value)
            for op in public_operations
            for value in (op["from_su"], op["to_su"])
        }),
        predicted_window_gain=float(predicted_window_gain),
        primary_adjustment_count=len(public_operations),
        diagnostics={
            "bundle_signature": str(bundle_signature),
            "bundle_strategy": "deterministic_multistart_nodewise_greedy",
            "bundle_kind": "nodewise_greedy",
            "batch_size": len(public_operations),
            "total_operation_count": len(public_operations),
            "h_compensation_count": 0,
            "h_compensation_conversion_counts": {},
            "h_compensation": {"enabled": False, "reason": "spectrum_first_policy"},
            "seed_direction": [int(seed_direction[0]), int(seed_direction[1])],
            "primary_histogram_signature": str(histogram_signature),
            "primary_inverse_signature": str(primary_inverse_signature),
            "predicted_global_sse_gain": float(predicted_sse_gain),
            "predicted_target_window_gain": float(predicted_window_gain),
            "primary_target_su_types": target_types,
            "operation_predicted_gains": operation_gains,
            "source_mu": [float(op["source_mu"]) for op in public_operations],
            "target_mu_goal": [float(op["target_mu_goal"]) for op in public_operations],
            "simulated_target_hop1": [
                list(op.get("simulated_target_hop1", []) or [])
                for op in public_operations
            ],
            "greedy_depth": len(public_operations),
            "profile_source": "su_nmr_common_range_filtered.csv",
            "target_peak_source": "filtered_template_library_exact_hop1",
            "conversion_counts": conversion_counts,
            "pair_count": len(list(pair_groups or [])),
            "pair_groups": [dict(group or {}) for group in list(pair_groups or [])],
            "su12_bias_score": float(su12_bias_score),
            "aliphatic_excess_audit": dict(
                dict(spectrum_audit or {}).get("aliphatic_excess", {}) or {}
            ),
            "delta_by_su": {
                str(key): int(value)
                for key, value in _bundle_delta(public_operations).items()
            },
            "final_balance_operations": [],
            "does_not_reserve_su24": True,
        },
    )
    if list(pair_groups or []):
        candidate.diagnostics["bundle_kind"] = "spectrum_pair_seed"
        candidate.diagnostics["bundle_strategy"] = "deterministic_pair_seed_greedy"
    return candidate


def _pair_window_bounds(su_type: int) -> Tuple[float, float]:
    if int(su_type) in UNSATURATED_DIFF_WINDOWS:
        return tuple(UNSATURATED_DIFF_WINDOWS[int(su_type)])
    return tuple(get_su_ppm_range(int(su_type)))


def _pair_direction_evidence(
    difference_spectrum: Optional[Dict[str, Any]],
    source_types: Sequence[int],
    target_types: Sequence[int],
    *,
    ppm: Optional[np.ndarray] = None,
    source_pair: Optional[Sequence[Dict[str, Any]]] = None,
    target_pair: Optional[Sequence[Dict[str, Any]]] = None,
    source_local_surplus: Optional[Sequence[float]] = None,
    target_local_deficit: Optional[Sequence[float]] = None,
    aromatic_response_scale: float = 1.0,
) -> Dict[str, Any]:
    """Return diagnostic direction evidence for one pair rule.

    Direction is evaluated at the concrete node/template peaks when pairs are
    generated.  The old per-SU net-window gate is retained only as a
    diagnostic fallback for callers that do not provide concrete rows; it is
    no longer the acceptance criterion for a pair candidate.
    """
    source_rows = {
        int(su_type): spectrum_signed_region_audit(
            difference_spectrum,
            *_pair_window_bounds(int(su_type)),
        )
        for su_type in sorted(set(int(value) for value in source_types))
    }
    target_rows = {
        int(su_type): spectrum_signed_region_audit(
            difference_spectrum,
            *_pair_window_bounds(int(su_type)),
        )
        for su_type in sorted(set(int(value) for value in target_types))
    }
    source_surplus = {
        int(su_type): max(
            0.0,
            float(dict(source_rows[int(su_type)]).get("negative_area", 0.0) or 0.0)
            - float(dict(source_rows[int(su_type)]).get("positive_area", 0.0) or 0.0),
        )
        for su_type in source_rows
    }
    target_deficit = {
        int(su_type): max(
            0.0,
            float(dict(target_rows[int(su_type)]).get("positive_area", 0.0) or 0.0)
            - float(dict(target_rows[int(su_type)]).get("negative_area", 0.0) or 0.0),
        )
        for su_type in target_rows
    }
    local_source: List[float] = []
    local_target: List[float] = []
    data = dict(difference_spectrum or {})
    diff = _array(data.get("diff"))
    grid = _array(ppm)
    step = _difference_ppm_step(grid) if grid.size == diff.size else 1.0
    if grid.size == diff.size and diff.size:
        negative = np.maximum(-diff, 0.0)
        positive = np.maximum(diff, 0.0)
        if source_local_surplus is not None:
            local_source = [float(value) for value in source_local_surplus]
        else:
            for row in list(source_pair or []):
                peak = _lorentzian_peak(
                    grid,
                    float(row.get("mu", 0.0)),
                    su_type=int(row.get("su_type", -1)),
                    aromatic_response_scale=aromatic_response_scale,
                )
                local_source.append(float(np.dot(peak, negative) * step))
        if target_local_deficit is not None:
            local_target = [float(value) for value in target_local_deficit]
        else:
            for row in list(target_pair or []):
                peak = _lorentzian_peak(
                    grid,
                    float(row.get("mu_median", 0.0)),
                    su_type=int(row.get("su_type", -1)),
                    aromatic_response_scale=aromatic_response_scale,
                )
                local_target.append(float(np.dot(peak, positive) * step))
    pair_source_evidence = float(sum(local_source))
    pair_target_evidence = float(sum(local_target))
    if local_source or local_target:
        valid = bool(
            pair_source_evidence > 1e-12
            and pair_target_evidence > 1e-12
        )
    else:
        valid = bool(
            sum(float(value) for value in source_surplus.values()) > 1e-12
            and sum(float(value) for value in target_deficit.values()) > 1e-12
        )
    return {
        "valid": bool(valid),
        "source_surplus": {
            str(key): float(value) for key, value in source_surplus.items()
        },
        "target_deficit": {
            str(key): float(value) for key, value in target_deficit.items()
        },
        "source_windows": {
            str(key): list(_pair_window_bounds(int(key)))
            for key in source_rows
        },
        "target_windows": {
            str(key): list(_pair_window_bounds(int(key)))
            for key in target_rows
        },
        "source_local_surplus": [float(value) for value in local_source],
        "target_local_deficit": [float(value) for value in local_target],
        "pair_source_evidence": float(pair_source_evidence),
        "pair_target_evidence": float(pair_target_evidence),
        "evidence_mode": "node_mu_local" if (local_source or local_target) else "su_window_diagnostic",
    }


def _pair_source_combinations(
    rows_by_su: Dict[int, Sequence[Dict[str, Any]]],
    source_types: Sequence[int],
    *,
    limit_per_type: int = 8,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return bounded, deterministic real-node pairs for one pair rule."""
    first_type, second_type = (int(source_types[0]), int(source_types[1]))
    first_rows = list(rows_by_su.get(first_type, []) or [])[: int(limit_per_type)]
    second_rows = list(rows_by_su.get(second_type, []) or [])[: int(limit_per_type)]
    if first_type == second_type:
        return [
            (dict(left), dict(right))
            for left, right in combinations(first_rows, 2)
            if int(left.get("node_id", -1)) != int(right.get("node_id", -1))
        ]
    return [
        (dict(left), dict(right))
        for left, right in product(first_rows, second_rows)
        if int(left.get("node_id", -1)) != int(right.get("node_id", -1))
    ]


def _pair_donor_sort_key(
    row: Dict[str, Any],
    *,
    ppm: np.ndarray,
    diff: np.ndarray,
    aromatic_response_scale: float = 1.0,
) -> Tuple[float, float, int]:
    """Rank concrete donors by local negative-lobe evidence, deterministically."""
    peak = _lorentzian_peak(
        ppm,
        float(row.get("mu", 0.0)),
        su_type=int(row.get("su_type", -1)),
        aromatic_response_scale=aromatic_response_scale,
    )
    step = _difference_ppm_step(ppm) if ppm.size == diff.size else 1.0
    negative = np.maximum(-diff, 0.0)
    surplus = float(np.dot(peak, negative) * step)
    negative_indices = np.flatnonzero(negative > 0.0)
    if negative_indices.size:
        weights = negative[negative_indices]
        center = float(np.average(ppm[negative_indices], weights=weights))
        distance = abs(float(row.get("mu", 0.0)) - center)
    else:
        distance = float("inf")
    return (-surplus, distance, int(row.get("node_id", -1)))


def _pair_target_combinations(
    rows_by_su: Dict[int, Sequence[Dict[str, Any]]],
    target_types: Sequence[int],
    *,
    limit_per_type: int = 8,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    first_type, second_type = (int(target_types[0]), int(target_types[1]))
    def _in_explicit_window(su_type: int, row: Dict[str, Any]) -> bool:
        bounds = UNSATURATED_DIFF_WINDOWS.get(int(su_type))
        if bounds is None:
            return True
        try:
            mu = float(row.get("mu_median"))
        except (TypeError, ValueError):
            return False
        return float(bounds[0]) <= mu <= float(bounds[1])

    first_rows = [
        dict(row) for row in list(rows_by_su.get(first_type, []) or [])
        if _in_explicit_window(first_type, dict(row))
    ][: int(limit_per_type)]
    second_rows = [
        dict(row) for row in list(rows_by_su.get(second_type, []) or [])
        if _in_explicit_window(second_type, dict(row))
    ][: int(limit_per_type)]
    if first_type == second_type:
        # Two physical nodes may legitimately use the same exact-hop1 median
        # template, so combinations with replacement are intentional here.
        return [
            (dict(left), dict(right))
            for left in first_rows
            for right in first_rows
        ]
    return [
        (dict(left), dict(right))
        for left, right in product(first_rows, second_rows)
    ]


def _generate_spectrum_pair_candidates(
    *,
    H: torch.Tensor,
    difference_spectrum: Dict[str, Any],
    ppm: np.ndarray,
    donor_rows: Sequence[Dict[str, Any]],
    target_rows_by_su: Dict[int, Sequence[Dict[str, Any]]],
    max_primary_adjustments: int,
    unsaturated_cap: Optional[int],
    normalized_weights: np.ndarray,
    rejected_histogram_signatures: Iterable[str],
    rejected_bundle_signatures: Iterable[str],
    aliphatic_audit: Optional[Dict[str, Any]] = None,
    aromatic_response_scale: float = 1.0,
) -> Tuple[List[Stage3SpectrumCandidate], Dict[str, Any]]:
    """Generate only the explicitly allowed 2-node unsaturated pair bundles."""
    donor_rows_by_su: Dict[int, List[Dict[str, Any]]] = {}
    for row in list(donor_rows or []):
        donor_rows_by_su.setdefault(int(row["su_type"]), []).append(dict(row))
    for rows in donor_rows_by_su.values():
        rows.sort(key=lambda item: int(item.get("node_id", -1)))
    # The bounded pair search should inspect chemically/spectrally useful
    # donors first.  The final node-id tie break keeps replay deterministic.
    for rows in donor_rows_by_su.values():
        rows.sort(key=lambda item: _pair_donor_sort_key(
            item,
            ppm=ppm,
            diff=_array(difference_spectrum.get("diff")),
            aromatic_response_scale=aromatic_response_scale,
        ))

    # Index peaks by the physical node id, never by ``id(row)``.  Pair rows
    # are deliberately copied by the combination helpers, so Python object
    # identity is not stable across the generator.
    donor_peaks_by_node_id = {
        int(row["node_id"]): _lorentzian_peak(
            ppm,
            float(row["mu"]),
            su_type=int(row.get("su_type", -1)),
            aromatic_response_scale=aromatic_response_scale,
        )
        for rows in donor_rows_by_su.values()
        for row in rows
    }
    target_rows_indexed: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {
        int(su_type): [
            (int(index), dict(row))
            for index, row in enumerate(list(rows or []))
        ]
        for su_type, rows in target_rows_by_su.items()
    }
    target_peaks = {
        (int(su_type), int(index)): _lorentzian_peak(
            ppm,
            float(row["mu_median"]),
            su_type=int(su_type),
            aromatic_response_scale=aromatic_response_scale,
        )
        for su_type, indexed_rows in target_rows_indexed.items()
        for index, row in indexed_rows
    }
    diff_array = _array(difference_spectrum.get("diff"))
    negative_diff = np.maximum(-diff_array, 0.0)
    positive_diff = np.maximum(diff_array, 0.0)
    diff_step = _difference_ppm_step(ppm) if ppm.size == diff_array.size else 1.0
    source_local_cache: Dict[int, float] = {
        int(row["node_id"]): float(
            np.dot(donor_peaks_by_node_id[int(row["node_id"])], negative_diff)
            * diff_step
        )
        for rows in donor_rows_by_su.values()
        for row in rows
    }
    target_local_cache: Dict[Tuple[int, int], float] = {
        (int(su_type), int(index)): float(
            np.dot(target_peaks[(int(su_type), int(index))], positive_diff)
            * diff_step
        )
        for su_type, indexed_rows in target_rows_indexed.items()
        for index, _row in indexed_rows
    }
    atomic_donor_rows = [
        row for rows in donor_rows_by_su.values()
        for row in rows
        if int(row.get("su_type", -1)) in SPECTRUM_RESOURCE_TYPES
    ]
    atomic_donor_peaks = np.stack([
        _lorentzian_peak(
            ppm,
            float(row["mu"]),
            su_type=int(row.get("su_type", -1)),
            aromatic_response_scale=aromatic_response_scale,
        )
        for row in atomic_donor_rows
    ]) if atomic_donor_rows else np.empty((0, int(ppm.size)), dtype=np.float64)
    atomic_target_rows = [
        dict(row)
        for su_type in SPECTRUM_RESOURCE_TYPES
        for row in list(target_rows_by_su.get(int(su_type), []) or [])
    ]
    atomic_target_peaks = np.stack([
        _lorentzian_peak(
            ppm,
            float(row["mu_median"]),
            su_type=int(row.get("su_type", -1)),
            aromatic_response_scale=aromatic_response_scale,
        )
        for row in atomic_target_rows
    ]) if atomic_target_rows else np.empty((0, int(ppm.size)), dtype=np.float64)
    rejected_hist = {str(value) for value in list(rejected_histogram_signatures or [])}
    rejected_bundles = {str(value) for value in list(rejected_bundle_signatures or [])}
    candidates: List[Stage3SpectrumCandidate] = []
    attempts: List[Dict[str, Any]] = []
    if int(max_primary_adjustments) < 2 or int(H.numel()) <= max(
        UNSATURATED_SPECTRUM_TYPES + (22, 23, 24)
    ):
        return [], {"pair_candidate_count": 0, "pair_attempts": []}

    for forward_sources, forward_targets, rule_name in SPECTRUM_PAIR_RULES:
        for reverse in (False, True):
            source_types = forward_targets if reverse else forward_sources
            target_types = forward_sources if reverse else forward_targets
            direction = "unsaturated_to_aliphatic" if reverse else "aliphatic_to_unsaturated"
            source_pairs = _pair_source_combinations(
                donor_rows_by_su,
                source_types,
            )
            target_pairs = _pair_target_combinations(
                target_rows_by_su,
                target_types,
            )
            if not source_pairs or not target_pairs:
                attempts.append({
                    "rule": str(rule_name),
                    "direction": str(direction),
                    "reason": "pair_source_or_target_template_unavailable",
                    "source_count": len(source_pairs),
                    "target_count": len(target_pairs),
                })
                continue
            for source_pair in source_pairs:
                source_peak = (
                    donor_peaks_by_node_id[int(source_pair[0]["node_id"])]
                    + donor_peaks_by_node_id[int(source_pair[1]["node_id"])]
                )
                for target_pair in target_pairs:
                    target_index_0 = next(
                        index
                        for index, row in target_rows_indexed[int(target_types[0])]
                        if row == target_pair[0]
                    )
                    target_index_1 = next(
                        index
                        for index, row in target_rows_indexed[int(target_types[1])]
                        if row == target_pair[1]
                    )
                    # Direction is a property of this concrete node/template
                    # pair, not of the entire SU window.  In particular,
                    # aromatic overlap can make one member's broad window
                    # have the opposite net sign while the actual Lorentzian
                    # replacement remains beneficial.
                    evidence = _pair_direction_evidence(
                        difference_spectrum,
                        source_types,
                        target_types,
                        ppm=ppm,
                        source_pair=source_pair,
                        target_pair=target_pair,
                        source_local_surplus=[
                            source_local_cache[int(source_pair[0]["node_id"])],
                            source_local_cache[int(source_pair[1]["node_id"])],
                        ],
                        target_local_deficit=[
                            target_local_cache[(int(target_types[0]), int(target_index_0))],
                            target_local_cache[(int(target_types[1]), int(target_index_1))],
                        ],
                        aromatic_response_scale=aromatic_response_scale,
                    )
                    if not bool(evidence.get("valid", False)):
                        attempts.append({
                            "rule": str(rule_name),
                            "direction": str(direction),
                            "reason": "pair_local_direction_not_supported",
                            "source_node_ids": [
                                int(source_pair[0]["node_id"]),
                                int(source_pair[1]["node_id"]),
                            ],
                            "target_mu": [
                                float(target_pair[0]["mu_median"]),
                                float(target_pair[1]["mu_median"]),
                            ],
                            "evidence": evidence,
                        })
                        continue
                    target_peak = (
                        target_peaks[
                            (int(target_types[0]), int(target_index_0))
                        ]
                        + target_peaks[
                            (int(target_types[1]), int(target_index_1))
                        ]
                    )
                    operations: List[Dict[str, Any]] = []
                    pair_id_seed = {
                        "rule": str(rule_name),
                        "direction": str(direction),
                        "node_ids": sorted({
                            int(source_pair[0]["node_id"]),
                            int(source_pair[1]["node_id"]),
                        }),
                        "source": [int(value) for value in source_types],
                        "target": [int(value) for value in target_types],
                    }
                    pair_id = "pair:" + hashlib.sha1(
                        json.dumps(pair_id_seed, sort_keys=True).encode("utf-8")
                    ).hexdigest()[:12]
                    for source_row, target_row, source_su, target_su in zip(
                        source_pair,
                        target_pair,
                        source_types,
                        target_types,
                    ):
                        operations.append({
                            "op": "convert",
                            "node_id": int(source_row["node_id"]),
                            "from_su": int(source_su),
                            "to_su": int(target_su),
                            "source_original_su_type": int(source_su),
                            "reason": "remaining_pool_difference_spectrum_pair",
                            "operation_kind": "spectrum_pair_primary",
                            "pair_id": str(pair_id),
                            "pair_rule": str(rule_name),
                            "pair_direction": str(direction),
                            "source_mu": float(source_row["mu"]),
                            "target_mu_goal": float(target_row["mu_median"]),
                            "simulated_target_hop1": [
                                int(value)
                                for value in tuple(target_row["hop1_tuple"])
                            ],
                            **{
                                key: value
                                for key, value in source_row.items()
                                if str(key).startswith("source_")
                            },
                        })
                    residual = np.asarray(
                        np.asarray(difference_spectrum.get("diff", []), dtype=np.float64)
                        + source_peak
                        - target_peak,
                        dtype=np.float64,
                    )
                    pair_predicted_gain = float(
                        np.dot(
                            np.asarray(difference_spectrum.get("diff", []), dtype=np.float64),
                            np.asarray(difference_spectrum.get("diff", []), dtype=np.float64),
                        )
                        - np.dot(residual, residual)
                    )
                    # The two-node pair must improve the spectrum as a unit;
                    # an appended ordinary conversion may not rescue a pair
                    # whose own joint replacement is harmful.
                    if pair_predicted_gain <= 1e-10:
                        attempts.append({
                            "rule": str(rule_name),
                            "direction": str(direction),
                            "reason": "pair_predicted_sse_gain_not_positive",
                            "pair_predicted_sse_gain": float(pair_predicted_gain),
                            "evidence": evidence,
                        })
                        continue
                    # A pair is the indivisible seed.  Ordinary six-type
                    # conversions may be appended while budget remains, but
                    # neither member of the pair can be selected alone.
                    used_node_ids = {
                        int(operation["node_id"]) for operation in operations
                    }
                    H_state = _histogram_after_operations(H, operations)
                    atomic_donor_peak_norms = np.sum(
                        atomic_donor_peaks * atomic_donor_peaks, axis=1
                    )
                    atomic_target_peak_norms = np.sum(
                        atomic_target_peaks * atomic_target_peaks, axis=1
                    )
                    atomic_donor_target_dot = (
                        atomic_donor_peaks @ atomic_target_peaks.T
                        if atomic_donor_peaks.size and atomic_target_peaks.size
                        else np.empty((len(atomic_donor_rows), len(atomic_target_rows)))
                    )
                    while len(operations) < int(max_primary_adjustments):
                        if not atomic_donor_rows or not atomic_target_rows:
                            break
                        score_state = _atomic_score_state(
                            residual=residual,
                            donor_rows=atomic_donor_rows,
                            donor_peaks=atomic_donor_peaks,
                            target_rows=atomic_target_rows,
                            target_peaks=atomic_target_peaks,
                        )
                        next_operations: List[Dict[str, Any]] = []
                        for next_source in SPECTRUM_RESOURCE_TYPES:
                            for next_target in SPECTRUM_RESOURCE_TYPES:
                                atomic = _best_atomic_for_direction(
                                    residual=residual,
                                    source_su=int(next_source),
                                    target_su=int(next_target),
                                    donor_rows=atomic_donor_rows,
                                    donor_peaks=atomic_donor_peaks,
                                    donor_peak_norms=atomic_donor_peak_norms,
                                    target_rows=atomic_target_rows,
                                    target_peaks=atomic_target_peaks,
                                    target_peak_norms=atomic_target_peak_norms,
                                    donor_target_dot=atomic_donor_target_dot,
                                    score_state=score_state,
                                    used_node_ids=used_node_ids,
                                    H_state=H_state,
                                )
                                if atomic is not None:
                                    next_operations.append(atomic)
                        if not next_operations:
                            break
                        next_operation = min(next_operations, key=_operation_sort_key)
                        operations.append(next_operation)
                        used_node_ids.add(int(next_operation["node_id"]))
                        residual = np.asarray(
                            next_operation["_next_residual"], dtype=np.float64
                        )
                        H_state = _histogram_after_operations(H, operations)
                    candidate_H = _histogram_after_operations(H, operations)
                    su12_cap = _su12_cap_for_operations(H, operations)
                    if not bool(su12_cap.get("generation_allowed", True)):
                        attempts.append({
                            "rule": str(rule_name),
                            "direction": str(direction),
                            "reason": "su12_generation_cap_reached",
                            "su12_cap": dict(su12_cap),
                        })
                        continue
                    if (
                        unsaturated_cap is not None
                        and int(candidate_H[14].item())
                        + int(candidate_H[15].item())
                        + int(candidate_H[16].item())
                        > int(unsaturated_cap)
                    ):
                        attempts.append({
                            "rule": str(rule_name),
                            "direction": str(direction),
                            "reason": "unsaturated_cap_exceeded",
                        })
                        continue
                    predicted_gain = float(
                        np.dot(
                            np.asarray(difference_spectrum.get("diff", []), dtype=np.float64),
                            np.asarray(difference_spectrum.get("diff", []), dtype=np.float64),
                        )
                        - np.dot(residual, residual)
                    )
                    if predicted_gain <= 1e-10:
                        continue
                    pair_group = {
                        "pair_id": str(pair_id),
                        "pair_rule": str(rule_name),
                        "pair_rules_used": [str(rule_name)],
                        "pair_direction": str(direction),
                        "source_su_types": [int(value) for value in source_types],
                        "target_su_types": [int(value) for value in target_types],
                        "source_node_ids": [
                            int(source_pair[0]["node_id"]),
                            int(source_pair[1]["node_id"]),
                        ],
                        "source_provenance": [
                            {
                                str(key): value
                                for key, value in source_row.items()
                                if str(key).startswith("source_")
                            }
                            for source_row in source_pair
                        ],
                        "target_template_options": [
                            {
                                "su_type": int(target_types[index]),
                                "hop1_tuple": list(target_pair[index]["hop1_tuple"]),
                                "mu_median": float(target_pair[index]["mu_median"]),
                            }
                            for index in range(2)
                        ],
                    }
                    candidate = _candidate_from_path(
                        H=H,
                        baseline_diff=np.asarray(
                            difference_spectrum.get("diff", []),
                            dtype=np.float64,
                        ),
                        normalized_weights=normalized_weights,
                        seed_direction=(
                            int(source_types[0]),
                            int(target_types[0]),
                        ),
                        operations=operations,
                        residual=residual,
                        pair_groups=[pair_group],
                        su12_bias_score=_su12_bias_score(
                            operations,
                            aliphatic_audit,
                        ),
                        spectrum_audit={
                            "aliphatic_excess": dict(aliphatic_audit or {}),
                        },
                    )
                    candidate.diagnostics.update({
                        "pair_rule": str(rule_name),
                        "pair_direction": str(direction),
                        "source_su_types": [int(value) for value in source_types],
                        "target_su_types": [int(value) for value in target_types],
                        "pair_count": 1,
                        "pair_operations": [dict(op) for op in operations[:2]],
                        "pair_operation_count": 2,
                        "h_before": float(_h_total(H, None)),
                        "h_after_predicted": float(_h_total(candidate_H, None)),
                        "unsaturated_count_before": int(
                            sum(int(H[index].item()) for index in (14, 15, 16))
                        ),
                        "unsaturated_count_after": int(
                            sum(int(candidate_H[index].item()) for index in (14, 15, 16))
                        ),
                        "unsaturated_cap": (
                            None if unsaturated_cap is None else int(unsaturated_cap)
                        ),
                        "h_delta": int(
                            -2 if str(direction) == "aliphatic_to_unsaturated" else 2
                        ),
                        "pair_direction_evidence": dict(evidence),
                        "pair_source_evidence": float(
                            evidence.get("pair_source_evidence", 0.0) or 0.0
                        ),
                        "pair_target_evidence": float(
                            evidence.get("pair_target_evidence", 0.0) or 0.0
                        ),
                        "pair_predicted_sse_gain": float(pair_predicted_gain),
                        "rejected_pair_reasons": [],
                    })
                    histogram_signature = str(candidate.histogram_signature or "")
                    bundle_signature = str(
                        candidate.diagnostics.get("bundle_signature", "") or ""
                    )
                    if (
                        histogram_signature in rejected_hist
                        or bundle_signature in rejected_bundles
                    ):
                        continue
                    candidates.append(candidate)
                    # Keep pair trial generation bounded. The normal candidate
                    # limit remains the final cross-family limit.
                    if len(candidates) >= 4 * max(1, int(max_primary_adjustments)):
                        return candidates, {
                            "pair_candidate_count": len(candidates),
                            "pair_attempts": attempts[-64:],
                        }
    return candidates, {
        "pair_candidate_count": len(candidates),
        "pair_attempts": attempts[-64:],
    }


def generate_spectrum_candidates(
    *,
    H: torch.Tensor,
    difference_spectrum: Optional[Dict[str, Any]],
    spectrum_node_ids_by_su: Dict[int, Sequence[int]],
    aromatic_donor_ids_by_su: Dict[int, Sequence[int]],
    nodes: Optional[Sequence[Any]] = None,
    profile_stats: Optional[Dict[int, Dict[str, Any]]] = None,
    hop1_template_options: Optional[Dict[int, Any]] = None,
    candidate_limit: int = 8,
    max_primary_adjustments: int = 10,
    max_h_compensation_ops: int = 8,
    rejected_histogram_signatures: Optional[Iterable[str]] = None,
    rejected_bundle_signatures: Optional[Iterable[str]] = None,
    forbidden_inverse_signatures: Optional[Iterable[str]] = None,
    E_target: Optional[torch.Tensor] = None,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    flex_audit: Optional[Dict[str, Any]] = None,
    temporary_unsaturated_ids_by_su: Optional[Dict[int, Sequence[int]]] = None,
    aromatic_response_scale: float = 1.0,
) -> Tuple[List[Stage3SpectrumCandidate], Dict[str, Any]]:
    del special_degree_meta, flex_audit, max_h_compensation_ops
    data = dict(difference_spectrum or {})
    ppm = _array(data.get("ppm"))
    diff = _array(data.get("diff"))
    if ppm.size == 0 or ppm.size != diff.size:
        return [], {
            "valid": False,
            "ppm_size": int(ppm.size),
            "reason": "missing_or_invalid_difference_spectrum",
        }
    profiles, profile_audit = _validated_spectrum_profiles(profile_stats)
    if profiles is None:
        return [], {
            "valid": True,
            "ppm_size": int(ppm.size),
            **profile_audit,
        }
    target_options, hop1_audit = _validated_hop1_template_options(
        hop1_template_options,
        profiles,
    )
    if target_options is None:
        return [], {
            "valid": True,
            "ppm_size": int(ppm.size),
            **profile_audit,
            **hop1_audit,
        }
    # SU14/15/16 are optional pair-only targets.  Their absence must disable
    # only the corresponding pair candidates, never the ordinary six-type
    # spectrum path.
    # Validate optional types independently.  A missing SU14 profile/template
    # disables only the 22+24<->16+14 rule; SU15/SU16 remain eligible.
    unsaturated_options: Dict[int, List[Dict[str, Any]]] = {}
    unsaturated_profile_audit: Dict[str, Any] = {}
    unsaturated_hop1_audit: Dict[str, Any] = {
        "hop1_optional_su_types": list(UNSATURATED_SPECTRUM_TYPES),
        "hop1_optional_available": False,
        "hop1_optional_missing_su_types": [],
    }
    for optional_su in UNSATURATED_SPECTRUM_TYPES:
        one_profile, one_profile_audit = _validated_spectrum_profiles(
            profile_stats,
            su_types=(int(optional_su),),
        )
        unsaturated_profile_audit[str(int(optional_su))] = dict(
            one_profile_audit or {}
        )
        if one_profile is None or int(optional_su) not in one_profile:
            unsaturated_hop1_audit["hop1_optional_missing_su_types"].append(
                int(optional_su)
            )
            continue
        one_options, one_audit = _validated_hop1_template_options(
            hop1_template_options,
            {int(optional_su): one_profile[int(optional_su)]},
            su_types=(int(optional_su),),
        )
        if one_options is None or not one_options.get(int(optional_su)):
            unsaturated_hop1_audit["hop1_optional_missing_su_types"].append(
                int(optional_su)
            )
            continue
        unsaturated_options[int(optional_su)] = list(one_options[int(optional_su)])
        unsaturated_hop1_audit.setdefault("by_su", {})[str(int(optional_su))] = dict(
            one_audit or {}
        )
    unsaturated_hop1_audit["hop1_optional_available"] = bool(unsaturated_options)
    aliphatic_audit = aliphatic_negative_excess_audit(data)
    unsaturated_audit = unsaturated_window_audit(data)
    normalized_weights = _spectrum_profile_weights(ppm, profiles)
    available: Dict[int, List[int]] = {}
    for source in (spectrum_node_ids_by_su, aromatic_donor_ids_by_su):
        for su_type, values in dict(source or {}).items():
            if int(su_type) not in SPECTRUM_RESOURCE_TYPES:
                continue
            available.setdefault(int(su_type), []).extend(
                int(value) for value in list(values or [])
            )
    available = {
        int(su_type): sorted(set(int(value) for value in values))
        for su_type, values in available.items()
    }
    for su_type in SPECTRUM_RESOURCE_TYPES:
        available.setdefault(int(su_type), [])
    node_lookup = {
        int(getattr(node, "global_id", -1)): node
        for node in list(nodes or [])
    }
    donor_rows: List[Dict[str, Any]] = []
    for su_type in SPECTRUM_RESOURCE_TYPES:
        for node_id in available[int(su_type)]:
            node = node_lookup.get(int(node_id))
            try:
                node_su = int(getattr(node, "su_type", -1))
                node_mu = float(getattr(node, "mu", float("nan")))
            except (TypeError, ValueError):
                continue
            if node_su != int(su_type) or not np.isfinite(node_mu) or node_mu == 0.0:
                continue
            donor_rows.append({
                "node_id": int(node_id),
                "su_type": int(su_type),
                "mu": float(node_mu),
                **_spectrum_node_provenance(node),
            })
    # Fixed SU14/15/16 nodes are temporary spectrum donors only.  They are
    # never inserted into the ordinary resource pool and their live topology
    # ownership is untouched until a candidate survives rebuild/validation.
    for su_type in UNSATURATED_SPECTRUM_TYPES:
        for node_id in sorted({
            int(value)
            for value in list(
                dict(temporary_unsaturated_ids_by_su or {}).get(
                    int(su_type), []
                )
                or []
            )
        }):
            node = node_lookup.get(int(node_id))
            try:
                node_su = int(getattr(node, "su_type", -1))
                node_mu = float(getattr(node, "mu", float("nan")))
            except (TypeError, ValueError):
                continue
            if node_su != int(su_type) or not np.isfinite(node_mu) or node_mu == 0.0:
                continue
            donor_rows.append({
                "node_id": int(node_id),
                "su_type": int(su_type),
                "mu": float(node_mu),
                "temporary_fixed_donor": True,
                **_spectrum_node_provenance(node),
            })
    donor_rows.sort(key=lambda row: (int(row["su_type"]), int(row["node_id"])))
    if not donor_rows:
        return [], {
            "valid": True,
            "ppm_size": int(ppm.size),
            **profile_audit,
            "reason": "no_layer2_resource_donor_nodes",
            "available_ids_by_su": {
                str(key): list(values) for key, values in sorted(available.items())
            },
        }
    donor_peaks = np.stack([
        _lorentzian_peak(
            ppm,
            float(row["mu"]),
            su_type=int(row.get("su_type", -1)),
            aromatic_response_scale=aromatic_response_scale,
        )
        for row in donor_rows
    ])
    donor_peak_norms = np.sum(donor_peaks * donor_peaks, axis=1)
    target_rows = [
        dict(row)
        for su_type in SPECTRUM_RESOURCE_TYPES
        for row in list(target_options[int(su_type)] or [])
    ]
    pair_target_rows_by_su: Dict[int, List[Dict[str, Any]]] = {
        int(su_type): [dict(row) for row in list(target_options[int(su_type)] or [])]
        for su_type in SPECTRUM_RESOURCE_TYPES
    }
    if unsaturated_options is not None:
        pair_target_rows_by_su.update({
            int(su_type): [
                dict(row)
                for row in list(unsaturated_options.get(int(su_type), []) or [])
            ]
            for su_type in UNSATURATED_SPECTRUM_TYPES
            if int(su_type) in unsaturated_options
        })
    target_peaks = np.stack([
        _lorentzian_peak(
            ppm,
            float(row["mu_median"]),
            su_type=int(row.get("su_type", -1)),
            aromatic_response_scale=aromatic_response_scale,
        )
        for row in target_rows
    ])
    target_peak_norms = np.sum(target_peaks * target_peaks, axis=1)
    donor_target_dot = donor_peaks @ target_peaks.T
    rejected = {str(value) for value in list(rejected_histogram_signatures or [])}
    rejected_bundles = {str(value) for value in list(rejected_bundle_signatures or [])}
    forbidden_inverse = {str(value) for value in list(forbidden_inverse_signatures or [])}
    limit = max(0, int(candidate_limit))
    primary_cap = max(0, int(max_primary_adjustments))
    if limit <= 0 or primary_cap <= 0:
        return [], {
            "valid": True,
            "ppm_size": int(ppm.size),
            **profile_audit,
            "reason": "spectrum_candidate_budget_zero",
        }

    unsaturated_cap: Optional[int] = None
    try:
        carbon_target = float(E_target.detach().cpu().float()[0].item())
        if np.isfinite(carbon_target) and carbon_target >= 0.0:
            unsaturated_cap = int(
                math.floor(carbon_target * SPECTRUM_UNSATURATED_CAP_RATIO)
            )
    except Exception:
        unsaturated_cap = None
    pair_candidates: List[Stage3SpectrumCandidate] = []
    pair_audit: Dict[str, Any] = {
        "pair_candidate_count": 0,
        "pair_attempts": [],
        "unsaturated_cap": unsaturated_cap,
    }
    if unsaturated_options is not None and all(
        int(index) < int(H.numel()) for index in (14, 15, 16, 22, 23, 24)
    ):
        pair_candidates, pair_audit = _generate_spectrum_pair_candidates(
            H=H,
            difference_spectrum=data,
            ppm=ppm,
            donor_rows=donor_rows,
            target_rows_by_su=pair_target_rows_by_su,
            max_primary_adjustments=primary_cap,
            unsaturated_cap=unsaturated_cap,
            normalized_weights=normalized_weights,
            rejected_histogram_signatures=rejected,
            rejected_bundle_signatures=rejected_bundles,
            aliphatic_audit=aliphatic_audit,
            aromatic_response_scale=aromatic_response_scale,
        )

    baseline_score_state = _atomic_score_state(
        residual=diff,
        donor_rows=donor_rows,
        donor_peaks=donor_peaks,
        target_rows=target_rows,
        target_peaks=target_peaks,
    )
    seed_atomics: List[Dict[str, Any]] = []
    for source_su in SPECTRUM_RESOURCE_TYPES:
        if not any(int(row["su_type"]) == int(source_su) for row in donor_rows):
            continue
        for target_su in SPECTRUM_RESOURCE_TYPES:
            if int(source_su) == int(target_su):
                continue
            seed_atomics.extend(_ranked_atomics_for_direction(
                residual=diff,
                source_su=int(source_su),
                target_su=int(target_su),
                donor_rows=donor_rows,
                donor_peaks=donor_peaks,
                donor_peak_norms=donor_peak_norms,
                target_rows=target_rows,
                target_peaks=target_peaks,
                target_peak_norms=target_peak_norms,
                donor_target_dot=donor_target_dot,
                score_state=baseline_score_state,
                used_node_ids=set(),
                H_state=torch.clamp(H.detach().cpu(), min=0).long(),
                max_results=max(1, int(limit)),
            ))
    # Spectrum improvement remains the primary ordering criterion.  The
    # aliphatic-excess SU12 preference is deliberately only a tie-break among
    # operations with essentially the same predicted gain; it must not push a
    # materially worse SSE candidate ahead of a better ordinary conversion.
    seed_atomics.sort(key=lambda operation: (
        round(-float(operation.get("predicted_sse_gain", 0.0) or 0.0), 6),
        -float(
            1.0
            if bool(aliphatic_audit.get("triggered", False))
            and int(operation.get("from_su", -1)) == 13
            and int(operation.get("to_su", -1)) == 12
            else 0.0
        ),
        _operation_sort_key(operation),
    ))

    selected: List[Stage3SpectrumCandidate] = []
    selected_histograms: set[str] = set()
    selected_bundles: set[str] = set()
    seed_attempt_count = 0
    for seed in seed_atomics:
        seed_attempt_count += 1
        direction = (int(seed["from_su"]), int(seed["to_su"]))
        operations: List[Dict[str, Any]] = [seed]
        residual = np.asarray(seed["_next_residual"], dtype=np.float64)
        used_node_ids = {int(seed["node_id"])}
        H_state = _histogram_after_operations(H, operations)
        while len(operations) < int(primary_cap):
            score_state = _atomic_score_state(
                residual=residual,
                donor_rows=donor_rows,
                donor_peaks=donor_peaks,
                target_rows=target_rows,
                target_peaks=target_peaks,
            )
            next_operations: List[Dict[str, Any]] = []
            for next_source in SPECTRUM_RESOURCE_TYPES:
                for next_target in SPECTRUM_RESOURCE_TYPES:
                    atomic = _best_atomic_for_direction(
                        residual=residual,
                        source_su=int(next_source),
                        target_su=int(next_target),
                        donor_rows=donor_rows,
                        donor_peaks=donor_peaks,
                        donor_peak_norms=donor_peak_norms,
                        target_rows=target_rows,
                        target_peaks=target_peaks,
                        target_peak_norms=target_peak_norms,
                        donor_target_dot=donor_target_dot,
                        score_state=score_state,
                        used_node_ids=used_node_ids,
                        H_state=H_state,
                    )
                    if atomic is not None:
                        next_operations.append(atomic)
            if not next_operations:
                break
            best_next = min(next_operations, key=_operation_sort_key)
            operations.append(best_next)
            used_node_ids.add(int(best_next["node_id"]))
            residual = np.asarray(best_next["_next_residual"], dtype=np.float64)
            H_state = _histogram_after_operations(H, operations)

        su12_cap = _su12_cap_for_operations(H, operations)
        if not bool(su12_cap.get("generation_allowed", True)):
            continue
        candidate = _candidate_from_path(
            H=H,
            baseline_diff=diff,
            normalized_weights=normalized_weights,
            seed_direction=direction,
            operations=operations,
            residual=residual,
            su12_bias_score=_su12_bias_score(operations, aliphatic_audit),
            spectrum_audit={
                "aliphatic_excess": dict(aliphatic_audit),
                "unsaturated_windows": dict(unsaturated_audit),
            },
        )
        diagnostics = dict(candidate.diagnostics or {})
        histogram_signature = str(candidate.histogram_signature or "")
        bundle_signature = str(diagnostics.get("bundle_signature", "") or "")
        if (
            not histogram_signature
            or histogram_signature in selected_histograms
            or histogram_signature in rejected
            or bundle_signature in selected_bundles
            or bundle_signature in rejected_bundles
            or bundle_signature in forbidden_inverse
        ):
            continue
        selected.append(candidate)
        selected_histograms.add(histogram_signature)
        selected_bundles.add(bundle_signature)
        if len(selected) >= max(int(limit), 4 * int(limit)):
            break

    # Merge ordinary, SU12-biased and pair bundles into one candidate set.
    # Pair bundles remain indivisible through ``pair_groups`` and are ranked
    # with the same predicted spectrum gain as ordinary candidates.
    merged_candidates: List[Stage3SpectrumCandidate] = []
    merged_histograms: set[str] = set()
    merged_bundles: set[str] = set()
    for candidate in list(selected) + list(pair_candidates):
        histogram_signature = str(getattr(candidate, "histogram_signature", "") or "")
        diagnostics = dict(getattr(candidate, "diagnostics", {}) or {})
        bundle_signature = str(diagnostics.get("bundle_signature", "") or "")
        inverse_signature = str(
            diagnostics.get(
                "primary_inverse_signature",
                getattr(candidate, "inverse_signature", ""),
            )
            or ""
        )
        if (
            not histogram_signature
            or histogram_signature in merged_histograms
            or histogram_signature in rejected
            or bundle_signature in merged_bundles
            or bundle_signature in rejected_bundles
            or inverse_signature in forbidden_inverse
        ):
            continue
        merged_candidates.append(candidate)
        merged_histograms.add(histogram_signature)
        if bundle_signature:
            merged_bundles.add(bundle_signature)

    merged_candidates.sort(key=lambda candidate: (
        -round(float(dict(getattr(candidate, "diagnostics", {}) or {}).get(
            "predicted_global_sse_gain", 0.0
        ) or 0.0), 6),
        -float(dict(getattr(candidate, "diagnostics", {}) or {}).get(
            "su12_bias_score", 0.0
        ) or 0.0),
        -float(getattr(candidate, "predicted_window_gain", 0.0) or 0.0),
        str(getattr(candidate, "candidate_id", "")),
    ))
    selected = merged_candidates[: int(limit)]
    for rank, candidate in enumerate(selected, start=1):
        candidate.diagnostics["spectral_priority_rank"] = int(rank)
    return selected, {
        "valid": True,
        "ppm_size": int(ppm.size),
        "r2": data.get("r2"),
        **profile_audit,
        **hop1_audit,
        "reason": "ok" if selected else "no_ranked_spectrum_candidate",
        "available_ids_by_su": {str(key): list(values) for key, values in sorted(available.items())},
        "eligible_donor_count": len(donor_rows),
        "eligible_target_hop1_count": len(target_rows),
        "seed_atomic_count": len(seed_atomics),
        "seed_attempt_count": int(seed_attempt_count),
        "seed_direction_count": len({
            (int(operation["from_su"]), int(operation["to_su"]))
            for operation in seed_atomics
        }),
        "pair_candidate_count": int(len(pair_candidates)),
        "pair_audit": dict(pair_audit or {}),
        "rejected_pair_reasons": [
            dict(item or {}) for item in list(
                dict(pair_audit or {}).get("pair_attempts", []) or []
            )
        ],
        "aliphatic_excess": dict(aliphatic_audit),
        "unsaturated_windows": dict(unsaturated_audit),
        "unsaturated_profile_audit": dict(unsaturated_profile_audit or {}),
        "unsaturated_hop1_audit": dict(unsaturated_hop1_audit or {}),
        "candidate_count_before_limit": int(len(merged_candidates)),
        "candidate_limit": int(limit),
        "max_primary_adjustments": int(primary_cap),
        "max_h_compensation_ops": 0,
        "h_compensation_enabled": False,
        "ranking_policy": "deterministic_exact_atomic_multistart_hop1_peak_greedy",
        "spectrum_first_ranking": True,
        "rejected_histogram_signature_count": len(rejected),
        "rejected_bundle_signature_count": len(rejected_bundles),
        "forbidden_inverse_signature_count": len(forbidden_inverse),
        "does_not_reserve_su24": True,
    }


def apply_spectrum_candidate(
    *,
    nodes: Sequence[Any],
    H: torch.Tensor,
    candidate: Stage3SpectrumCandidate,
) -> Tuple[Optional[List[Any]], Optional[torch.Tensor], Dict[str, Any]]:
    nodes_out = deepcopy(list(nodes or []))
    by_id = {int(getattr(node, "global_id", -1)): node for node in nodes_out}
    H_out = torch.clamp(H.detach().cpu(), min=0).long().clone()
    operations = [dict(raw or {}) for raw in list(candidate.operations or [])]
    su12_cap = _su12_cap_for_operations(H_out, operations)
    if not bool(su12_cap.get("generation_allowed", True)):
        return None, None, {
            "reason": "su12_generation_cap_reached",
            "su12_cap": dict(su12_cap),
        }
    pair_groups = [
        dict(group or {})
        for group in list(
            getattr(candidate, "pair_groups", None)
            or dict(getattr(candidate, "diagnostics", {}) or {}).get(
                "pair_groups", []
            )
            or []
        )
    ]
    pair_operation_ids: set[int] = set()
    allowed_pair_rules = {
        (tuple(sorted(forward_source)), tuple(sorted(forward_target)))
        for forward_source, forward_target, _ in SPECTRUM_PAIR_RULES
    } | {
        (tuple(sorted(forward_target)), tuple(sorted(forward_source)))
        for forward_source, forward_target, _ in SPECTRUM_PAIR_RULES
    }
    # Type-set validation alone is insufficient for the asymmetric rules.
    # For example, {22,23}->{15,16} is the same set as the valid rule, but
    # swapping the two concrete mappings would put the wrong target template
    # on each donor.  Keep the operation-level mapping canonical and validate
    # it independently of list/order metadata.
    allowed_pair_mappings = {
        tuple(sorted(zip(forward_source, forward_target)))
        for forward_source, forward_target, _ in SPECTRUM_PAIR_RULES
    } | {
        tuple(sorted(zip(forward_target, forward_source)))
        for forward_source, forward_target, _ in SPECTRUM_PAIR_RULES
    }
    operations_by_pair: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for index, op in enumerate(operations):
        pair_id = str(op.get("pair_id", "") or "")
        if pair_id:
            operations_by_pair.setdefault(pair_id, []).append((index, op))
        if int(op.get("from_su", -1)) in UNSATURATED_SPECTRUM_TYPES or int(
            op.get("to_su", -1)
        ) in UNSATURATED_SPECTRUM_TYPES:
            if not pair_id:
                return None, None, {
                    "reason": "spectrum_unsaturated_conversion_requires_pair",
                    "operation": dict(op),
                }
    declared_pair_ids: set[str] = set()
    for group in pair_groups:
        pair_id = str(group.get("pair_id", "") or "")
        if pair_id in declared_pair_ids:
            return None, None, {
                "reason": "spectrum_pair_metadata_duplicate",
                "pair_id": pair_id,
            }
        declared_pair_ids.add(pair_id)
        grouped = list(operations_by_pair.get(pair_id, []))
        if not pair_id or len(grouped) != 2:
            return None, None, {
                "reason": "spectrum_pair_not_atomic",
                "pair_id": pair_id,
                "operation_count": len(grouped),
            }
        source_types = tuple(
            sorted(int(op.get("from_su", -1)) for _, op in grouped)
        )
        target_types = tuple(
            sorted(int(op.get("to_su", -1)) for _, op in grouped)
        )
        operation_mapping = tuple(sorted(
            (
                int(op.get("from_su", -1)),
                int(op.get("to_su", -1)),
            )
            for _, op in grouped
        ))
        if (source_types, target_types) not in allowed_pair_rules:
            return None, None, {
                "reason": "spectrum_pair_rule_forbidden",
                "pair_id": pair_id,
                "source_su_types": list(source_types),
                "target_su_types": list(target_types),
            }
        if operation_mapping not in allowed_pair_mappings:
            return None, None, {
                "reason": "spectrum_pair_operation_mapping_forbidden",
                "pair_id": pair_id,
                "operation_mapping": [list(item) for item in operation_mapping],
            }
        declared_sources = tuple(sorted(int(value) for value in list(
            group.get("source_su_types", []) or []
        )))
        declared_targets = tuple(sorted(int(value) for value in list(
            group.get("target_su_types", []) or []
        )))
        if declared_sources and declared_sources != source_types:
            return None, None, {
                "reason": "spectrum_pair_source_metadata_mismatch",
                "pair_id": pair_id,
            }
        declared_node_ids = tuple(sorted(int(value) for value in list(
            group.get("source_node_ids", []) or []
        )))
        actual_node_ids = tuple(sorted(
            int(op.get("node_id", -1)) for _, op in grouped
        ))
        if declared_node_ids and declared_node_ids != actual_node_ids:
            return None, None, {
                "reason": "spectrum_pair_node_metadata_mismatch",
                "pair_id": pair_id,
                "declared_node_ids": list(declared_node_ids),
                "actual_node_ids": list(actual_node_ids),
            }
        if declared_targets and declared_targets != target_types:
            return None, None, {
                "reason": "spectrum_pair_target_metadata_mismatch",
                "pair_id": pair_id,
            }
        grouped_indexes = {index for index, _ in grouped}
        if pair_operation_ids.intersection(grouped_indexes):
            return None, None, {
                "reason": "spectrum_pair_operation_reused",
                "pair_id": pair_id,
            }
        pair_operation_ids.update(grouped_indexes)
    # A conversion involving SU14/15/16 must be covered by exactly one pair;
    # this rejects accidental 22->16, 23->15, 24->14 single-node fallbacks.
    for index, op in enumerate(operations):
        if (
            int(op.get("from_su", -1)) in UNSATURATED_SPECTRUM_TYPES
            or int(op.get("to_su", -1)) in UNSATURATED_SPECTRUM_TYPES
        ) and index not in pair_operation_ids:
            return None, None, {
                "reason": "spectrum_unsaturated_operation_outside_pair",
                "operation": dict(op),
            }
    seen_ids: set[int] = set()
    for raw in operations:
        op = dict(raw or {})
        node_id = int(op.get("node_id", -1))
        source = int(op.get("from_su", -1))
        target = int(op.get("to_su", -1))
        node = by_id.get(int(node_id))
        if (
            node_id in seen_ids
            or node is None
            or int(getattr(node, "su_type", -1)) != int(source)
        ):
            return None, None, {
                "reason": "spectrum_candidate_real_id_mismatch",
                "node_id": int(node_id),
                "expected_su": int(source),
                "actual_su": None if node is None else int(getattr(node, "su_type", -1)),
            }
        if source < 0 or target < 0 or source >= H_out.numel() or target >= H_out.numel() or int(H_out[source].item()) <= 0:
            return None, None, {"reason": "spectrum_candidate_histogram_invalid", "operation": op}
        # Spectrum operations still obey the aromatic reserve. A parity/role
        # operation appended at the rebuild boundary is different: it is the
        # deliberate consumer of one reserved SU12/SU13 donor; a later Stage3
        # structural-unit adjustment is expected to restore the staging minimum.
        balance_operation = str(op.get("operation_kind", "") or "") in {
            "parity_balance",
            "port_role_balance",
        }
        if (
            source in {12, 13}
            and int(H_out[source].item())
            <= int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT)
            and not balance_operation
        ):
            return None, None, {
                "reason": "spectrum_candidate_aromatic_minimum_preserved",
                "node_id": int(node_id),
                "from_su": int(source),
                "minimum": int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT),
            }
        seen_ids.add(int(node_id))
        H_out[source] -= 1
        H_out[target] += 1
        setattr(node, "su_type", int(target))
    candidate_diagnostics = dict(getattr(candidate, "diagnostics", {}) or {})
    unsaturated_cap = candidate_diagnostics.get(
        "unsaturated_cap",
        dict(candidate_diagnostics.get("pair_diagnostics", {}) or {}).get(
            "unsaturated_cap"
        ),
    )
    if unsaturated_cap is not None and all(
        int(index) < int(H_out.numel()) for index in UNSATURATED_SPECTRUM_TYPES
    ):
        unsaturated_after = sum(
            int(H_out[index].item()) for index in UNSATURATED_SPECTRUM_TYPES
        )
        if int(unsaturated_after) > int(unsaturated_cap):
            return None, None, {
                "reason": "spectrum_unsaturated_cap_exceeded",
                "unsaturated_count_after": int(unsaturated_after),
                "unsaturated_cap": int(unsaturated_cap),
            }
    if int(H_out.sum().item()) != int(torch.clamp(H.detach().cpu(), min=0).long().sum().item()):
        return None, None, {"reason": "spectrum_candidate_su_count_not_conserved"}
    return nodes_out, H_out, {}
