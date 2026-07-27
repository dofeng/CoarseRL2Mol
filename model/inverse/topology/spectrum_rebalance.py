from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ...shared.coarse_graph import get_su_ppm_range
from .conversion_scoring import evaluate_flex_impact
from .histogram_adjustment import _h_total
from .stage3_models import Stage3SpectrumCandidate


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


def generate_spectrum_candidates(
    *,
    H: torch.Tensor,
    difference_spectrum: Optional[Dict[str, Any]],
    closure_pool_ids_by_su: Dict[int, Sequence[int]],
    aromatic_donor_ids_by_su: Dict[int, Sequence[int]],
    candidate_limit: int = 2,
    max_primary_adjustments: int = 10,
    max_h_compensation_ops: int = 8,
    rejected_histogram_signatures: Optional[Iterable[str]] = None,
    forbidden_inverse_signatures: Optional[Iterable[str]] = None,
    E_target: Optional[torch.Tensor] = None,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    flex_audit: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Stage3SpectrumCandidate], Dict[str, Any]]:
    audit = spectrum_window_audit(difference_spectrum)
    if not bool(audit.get("valid", False)):
        return [], {**audit, "reason": "missing_or_invalid_difference_spectrum"}
    available: Dict[int, List[int]] = {}
    for source in (closure_pool_ids_by_su, aromatic_donor_ids_by_su):
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
    urgency_profiles, urgency_scales = _spectrum_urgency_profiles(audit)
    score_floor = float(urgency_scales["score_floor"])
    deficits = [
        int(su_type)
        for su_type in SPECTRUM_RESOURCE_TYPES
        if str(urgency_profiles[int(su_type)]["direction"]) == "deficit"
        and float(urgency_profiles[int(su_type)]["deficit_score"]) >= float(score_floor)
    ]
    surpluses = [
        int(su_type)
        for su_type in SPECTRUM_RESOURCE_TYPES
        if str(urgency_profiles[int(su_type)]["direction"]) == "surplus"
        and float(urgency_profiles[int(su_type)]["surplus_score"]) >= float(score_floor)
        and bool(available[int(su_type)])
    ]
    rejected = {str(value) for value in list(rejected_histogram_signatures or [])}
    forbidden_inverse = {str(value) for value in list(forbidden_inverse_signatures or [])}
    limit = max(0, int(candidate_limit))
    configured_cap = max(0, int(max_primary_adjustments))
    try:
        target_carbon = float(E_target.detach().cpu().float()[0].item())
    except Exception:
        target_carbon = float(torch.clamp(H.detach().cpu(), min=0).sum().item())
    carbon_cap = max(
        0,
        int(math.floor(float(SPECTRUM_BATCH_CARBON_RATIO) * target_carbon)),
    )
    batch_cap = min(int(configured_cap), int(carbon_cap))
    ppm = _array(dict(difference_spectrum or {}).get("ppm"))
    diff = _array(dict(difference_spectrum or {}).get("diff"))
    lobe_areas = _deoverlapped_lobe_areas(ppm, diff)
    source_signal_count = sum(
        float(lobe_areas[int(su_type)]["negative"]) / float(math.pi)
        for su_type in surpluses
    )
    target_signal_count = sum(
        float(lobe_areas[int(su_type)]["positive"]) / float(math.pi)
        for su_type in deficits
    )
    estimated_signal_count = max(
        0,
        int(math.floor(min(float(source_signal_count), float(target_signal_count)) + 0.5)),
    )
    available_source_count = sum(len(available[int(su_type)]) for su_type in surpluses)
    full_batch = min(
        int(batch_cap),
        int(estimated_signal_count),
        int(available_source_count),
    )
    source_weights = _direction_weights(
        surpluses,
        urgency_profiles,
        lobe_areas,
        direction="surplus",
    )
    target_weights = _direction_weights(
        deficits,
        urgency_profiles,
        lobe_areas,
        direction="deficit",
    )
    requested_batches: List[int] = []
    if int(full_batch) > 0 and int(limit) > 0:
        if int(limit) == 1 or int(full_batch) == 1:
            requested_batches = [1]
        else:
            requested_batches = [1]
            for index in range(1, int(limit) - 1):
                fraction = float(index) / float(max(1, int(limit) - 1))
                requested_batches.append(
                    max(1, int(math.floor(1.0 + fraction * float(int(full_batch) - 1) + 0.5)))
                )
            requested_batches.append(int(full_batch))
            requested_batches = list(dict.fromkeys(requested_batches))
    candidates: List[Stage3SpectrumCandidate] = []
    for proposal_index, requested_batch in enumerate(requested_batches[:limit]):
        primary_operations, conversion_counts, source_counts, target_counts = _transport_operations(
            batch_size=int(requested_batch),
            source_weights=source_weights,
            target_weights=target_weights,
            available_ids_by_su=available,
        )
        if not primary_operations:
            continue
        operations, compensation_diagnostics = _append_low_h_compensation(
            H=H,
            primary_operations=primary_operations,
            available_ids_by_su=available,
            E_target=E_target,
            special_degree_meta=special_degree_meta,
            max_h_compensation_ops=int(max_h_compensation_ops),
        )
        compensation_operations = [
            dict(op or {})
            for op in operations
            if str(dict(op or {}).get("operation_kind", "")) == "h_compensation"
        ]
        bundle_signature = _signature(operations)
        inverse_signature = inverse_bundle_signature(operations)
        histogram_signature = histogram_signature_after_operations(H, operations)
        if histogram_signature in rejected or bundle_signature in forbidden_inverse:
            continue
        candidate_H = _histogram_after_operations(H, operations)
        balance = spectrum_candidate_balance_penalty(
            H_before=H,
            H_after=candidate_H,
            E_target=E_target,
            special_degree_meta=special_degree_meta,
            flex_audit=flex_audit,
        )
        delta = _bundle_delta(operations)
        bundle_kind = (
            "atomic"
            if int(requested_batch) == 1
            else "full"
            if int(requested_batch) == int(full_batch)
            else "progressive"
        )
        candidates.append(Stage3SpectrumCandidate(
            candidate_id=(
                f"spectrum:proportional_{bundle_kind}:{len(primary_operations)}:"
                f"{bundle_signature[:10]}"
            ),
            histogram_signature=str(histogram_signature),
            inverse_signature=str(inverse_signature),
            operations=operations,
            target_su_types=sorted({
                int(value)
                for op in operations
                for value in (op["from_su"], op["to_su"])
            }),
            predicted_window_gain=float(
                min(float(source_signal_count), float(target_signal_count))
            ),
            primary_adjustment_count=len(primary_operations),
            diagnostics={
                "bundle_signature": str(bundle_signature),
                "bundle_strategy": "direct_difference_spectrum_proportion",
                "bundle_kind": str(bundle_kind),
                "batch_size": len(primary_operations),
                "total_operation_count": len(operations),
                "h_compensation_count": len(compensation_operations),
                "h_compensation_conversion_counts": (
                    {"11->23": len(compensation_operations)}
                    if compensation_operations else {}
                ),
                "h_compensation": dict(compensation_diagnostics),
                "batch_carbon_ratio": float(SPECTRUM_BATCH_CARBON_RATIO),
                "target_carbon_count": float(target_carbon),
                "carbon_batch_cap": int(carbon_cap),
                "configured_batch_cap": int(configured_cap),
                "effective_batch_cap": int(batch_cap),
                "estimated_signal_count": int(estimated_signal_count),
                "source_signal_count": float(source_signal_count),
                "target_signal_count": float(target_signal_count),
                "available_source_count": int(available_source_count),
                "source_counts": {str(key): int(value) for key, value in source_counts.items()},
                "target_counts": {str(key): int(value) for key, value in target_counts.items()},
                "conversion_counts": dict(conversion_counts),
                "delta_by_su": {str(key): int(value) for key, value in delta.items()},
                "source_weights": {str(key): float(value) for key, value in source_weights.items()},
                "target_weights": {str(key): float(value) for key, value in target_weights.items()},
                "preflight_balance": dict(balance),
                "h_priority_rank": int(_spectrum_candidate_h_rank(balance)),
                "does_not_reserve_su24": True,
            },
        ))
    for rank, candidate in enumerate(candidates, start=1):
        candidate.diagnostics["spectral_priority_rank"] = int(rank)
    return candidates[:limit], {
        **audit,
        "reason": "ok" if candidates else "no_ranked_spectrum_candidate",
        "available_ids_by_su": {str(key): list(values) for key, values in sorted(available.items())},
        "candidate_count_before_limit": len(candidates),
        "candidate_limit": int(limit),
        "max_primary_adjustments": int(configured_cap),
        "max_h_compensation_ops": max(0, int(max_h_compensation_ops)),
        "h_soft_window_aware_ranking": bool(E_target is not None),
        "ranking_policy": "atomic_then_progressive_direct_proportional_bundles",
        "spectrum_first_ranking": True,
        "batch_carbon_ratio": float(SPECTRUM_BATCH_CARBON_RATIO),
        "target_carbon_count": float(target_carbon),
        "carbon_batch_cap": int(carbon_cap),
        "effective_batch_cap": int(batch_cap),
        "estimated_signal_count": int(estimated_signal_count),
        "source_signal_count": float(source_signal_count),
        "target_signal_count": float(target_signal_count),
        "available_source_count": int(available_source_count),
        "deoverlapped_lobe_areas": {
            str(key): dict(value) for key, value in sorted(lobe_areas.items())
        },
        "urgency_scales": dict(urgency_scales),
        "urgency_by_su": {
            str(key): dict(value) for key, value in sorted(urgency_profiles.items())
        },
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
        if node_id in seen_ids or node is None or int(getattr(node, "su_type", -1)) != int(source):
            return None, None, {
                "reason": "spectrum_candidate_real_id_mismatch",
                "node_id": int(node_id),
                "expected_su": int(source),
                "actual_su": None if node is None else int(getattr(node, "su_type", -1)),
            }
        if source < 0 or target < 0 or source >= H_out.numel() or target >= H_out.numel() or int(H_out[source].item()) <= 0:
            return None, None, {"reason": "spectrum_candidate_histogram_invalid", "operation": op}
        seen_ids.add(int(node_id))
        H_out[source] -= 1
        H_out[target] += 1
        setattr(node, "su_type", int(target))
        history = list(getattr(node, "stage3_conversion_history", []) or [])
        history.append({
            "from_su": int(source),
            "to_su": int(target),
            "reason": str(op.get("reason", "remaining_pool_difference_spectrum_rebalance")),
        })
        setattr(node, "stage3_conversion_history", history)
    if int(H_out.sum().item()) != int(torch.clamp(H.detach().cpu(), min=0).long().sum().item()):
        return None, None, {"reason": "spectrum_candidate_su_count_not_conserved"}
    return nodes_out, H_out, {}
