from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ....shared.coarse_graph import get_su_ppm_range
from .conversion_scoring import evaluate_flex_impact
from .histogram import _h_total
from .profiles import MIN_STAGE3_AROMATIC_RESOURCE_COUNT
from ..stage3_models import Stage3SpectrumCandidate


SPECTRUM_RESOURCE_TYPES = (11, 12, 13, 22, 23, 24)
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
) -> Tuple[Optional[Dict[int, Dict[str, float]]], Dict[str, Any]]:
    profiles: Dict[int, Dict[str, float]] = {}
    missing: List[int] = []
    invalid: Dict[str, str] = {}
    raw_stats = dict(profile_stats or {})
    for su_type in SPECTRUM_RESOURCE_TYPES:
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
        "required_su_types": list(SPECTRUM_RESOURCE_TYPES),
        "missing_su_types": missing,
        "invalid_profiles": invalid,
    }
    if missing or invalid or len(profiles) != len(SPECTRUM_RESOURCE_TYPES):
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


def _lorentzian_peak(ppm: np.ndarray, mu: float, hwhm: float = 1.0) -> np.ndarray:
    hwhm_sq = float(hwhm) ** 2
    return hwhm_sq / ((ppm - float(mu)) ** 2 + hwhm_sq)


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
) -> Tuple[Optional[Dict[int, List[Dict[str, Any]]]], Dict[str, Any]]:
    """Normalize filtered-library 1-hop peaks for spectrum-only prediction."""
    normalized: Dict[int, List[Dict[str, Any]]] = {}
    missing: List[int] = []
    invalid_counts: Dict[str, int] = {}
    raw_options = dict(options_by_su or {})
    for su_type in SPECTRUM_RESOURCE_TYPES:
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
            if not hop1_tuple or not np.isfinite(mu) or mu == 0.0 or support <= 0.0:
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
            for su_type in SPECTRUM_RESOURCE_TYPES
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


def _candidate_from_path(
    *,
    H: torch.Tensor,
    baseline_diff: np.ndarray,
    normalized_weights: np.ndarray,
    seed_direction: Tuple[int, int],
    operations: Sequence[Dict[str, Any]],
    residual: np.ndarray,
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
            "delta_by_su": {
                str(key): int(value)
                for key, value in _bundle_delta(public_operations).items()
            },
            "final_balance_operations": [],
            "does_not_reserve_su24": True,
        },
    )
    return candidate


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
) -> Tuple[List[Stage3SpectrumCandidate], Dict[str, Any]]:
    del E_target, special_degree_meta, flex_audit, max_h_compensation_ops
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
        _lorentzian_peak(ppm, float(row["mu"]))
        for row in donor_rows
    ])
    donor_peak_norms = np.sum(donor_peaks * donor_peaks, axis=1)
    target_rows = [
        dict(row)
        for su_type in SPECTRUM_RESOURCE_TYPES
        for row in list(target_options[int(su_type)] or [])
    ]
    target_peaks = np.stack([
        _lorentzian_peak(ppm, float(row["mu_median"]))
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
    seed_atomics.sort(key=_operation_sort_key)

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

        candidate = _candidate_from_path(
            H=H,
            baseline_diff=diff,
            normalized_weights=normalized_weights,
            seed_direction=direction,
            operations=operations,
            residual=residual,
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
        if len(selected) >= int(limit):
            break

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
        "candidate_count_before_limit": int(seed_attempt_count),
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
    seen_ids: set[int] = set()
    for raw in list(candidate.operations or []):
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
        if (
            source in {12, 13}
            and int(H_out[source].item())
            <= int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT)
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
    if int(H_out.sum().item()) != int(torch.clamp(H.detach().cpu(), min=0).long().sum().item()):
        return None, None, {"reason": "spectrum_candidate_su_count_not_conserved"}
    return nodes_out, H_out, {}
