from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .stage3_models import Stage3SpectrumCandidate


SPECTRUM_RESOURCE_TYPES = (11, 12, 13, 22, 23, 24)
DEFAULT_SPECTRUM_WINDOWS: Dict[int, Tuple[float, float, float, float]] = {
    # narrow_lo, narrow_hi, wide_lo, wide_hi
    11: (105.0, 122.0, 95.0, 130.0),
    12: (118.0, 134.0, 108.0, 143.0),
    13: (126.0, 143.0, 115.0, 152.0),
    22: (14.0, 26.0, 8.0, 32.0),
    23: (22.0, 36.0, 15.0, 45.0),
    24: (32.0, 48.0, 24.0, 58.0),
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
        return {"positive": 0.0, "negative": 0.0, "signed": 0.0, "absolute": 0.0}
    mask = (ppm >= float(lo)) & (ppm <= float(hi))
    values = diff[mask] if np.any(mask) else np.asarray([], dtype=np.float64)
    positive = float(np.sum(values[values > 0])) if values.size else 0.0
    negative = float(-np.sum(values[values < 0])) if values.size else 0.0
    return {
        "positive": positive,
        "negative": negative,
        "signed": positive - negative,
        "absolute": positive + negative,
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


def histogram_signature_after_operations(H: torch.Tensor, operations: Sequence[Dict[str, Any]]) -> str:
    candidate = torch.clamp(H.detach().cpu(), min=0).long().clone()
    for raw in list(operations or []):
        op = dict(raw or {})
        source = int(op.get("from_su", -1))
        target = int(op.get("to_su", -1))
        if source < 0 or target < 0 or source >= candidate.numel() or target >= candidate.numel():
            continue
        candidate[source] -= 1
        candidate[target] += 1
    encoded = json.dumps(candidate.tolist(), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_spectrum_candidates(
    *,
    H: torch.Tensor,
    difference_spectrum: Optional[Dict[str, Any]],
    closure_pool_ids_by_su: Dict[int, Sequence[int]],
    aromatic_donor_ids_by_su: Dict[int, Sequence[int]],
    candidate_limit: int = 8,
    max_primary_adjustments: int = 10,
    rejected_histogram_signatures: Optional[Iterable[str]] = None,
    forbidden_inverse_signatures: Optional[Iterable[str]] = None,
) -> Tuple[List[Stage3SpectrumCandidate], Dict[str, Any]]:
    audit = spectrum_window_audit(difference_spectrum)
    if not bool(audit.get("valid", False)):
        return [], {**audit, "reason": "missing_or_invalid_difference_spectrum"}
    available: Dict[int, List[int]] = {
        int(su_type): sorted({int(value) for value in list(values or [])})
        for su_type, values in {
            **dict(closure_pool_ids_by_su or {}),
            **dict(aromatic_donor_ids_by_su or {}),
        }.items()
        if int(su_type) in SPECTRUM_RESOURCE_TYPES
    }
    for su_type in SPECTRUM_RESOURCE_TYPES:
        available.setdefault(int(su_type), [])
    needs = {
        int(su_type): float(dict(dict(audit.get("by_su", {}) or {}).get(str(su_type), {}) or {}).get("need", 0.0) or 0.0)
        for su_type in SPECTRUM_RESOURCE_TYPES
    }
    total_abs = sum(abs(value) for value in needs.values())
    noise_floor = max(1e-12, 0.01 * float(total_abs))
    deficits = [su for su in SPECTRUM_RESOURCE_TYPES if needs[int(su)] > noise_floor]
    surpluses = [su for su in SPECTRUM_RESOURCE_TYPES if needs[int(su)] < -noise_floor and available[int(su)]]
    rejected = {str(value) for value in list(rejected_histogram_signatures or [])}
    forbidden_inverse = {str(value) for value in list(forbidden_inverse_signatures or [])}
    candidates: List[Stage3SpectrumCandidate] = []
    seen_bundle_signatures: set[str] = set()
    limit = max(0, int(candidate_limit))
    primary_limit = max(0, min(10, int(max_primary_adjustments)))
    for target in sorted(deficits, key=lambda su: (-needs[int(su)], int(su))):
        for source in sorted(surpluses, key=lambda su: (needs[int(su)], int(su))):
            if int(source) == int(target):
                continue
            source_ids = list(available[int(source)])
            maximum = min(primary_limit, len(source_ids))
            for count in range(1, maximum + 1):
                operations = [
                    {
                        "op": "convert",
                        "node_id": int(node_id),
                        "from_su": int(source),
                        "to_su": int(target),
                        "reason": "remaining_pool_difference_spectrum_rebalance",
                    }
                    for node_id in source_ids[:count]
                ]
                bundle_signature = _signature(operations)
                inverse_signature = inverse_bundle_signature(operations)
                histogram_signature = histogram_signature_after_operations(H, operations)
                if bundle_signature in seen_bundle_signatures:
                    continue
                if histogram_signature in rejected or bundle_signature in forbidden_inverse:
                    continue
                seen_bundle_signatures.add(bundle_signature)
                predicted_gain = float(count) * min(abs(needs[int(source)]), abs(needs[int(target)]))
                candidates.append(Stage3SpectrumCandidate(
                    candidate_id=f"spectrum:{int(source)}:{int(target)}:{int(count)}",
                    histogram_signature=str(histogram_signature),
                    inverse_signature=str(inverse_signature),
                    operations=operations,
                    target_su_types=sorted({int(source), int(target)}),
                    predicted_window_gain=float(predicted_gain),
                    primary_adjustment_count=int(count),
                    diagnostics={
                        "bundle_signature": str(bundle_signature),
                        "source_need": float(needs[int(source)]),
                        "target_need": float(needs[int(target)]),
                        "does_not_reserve_su24": True,
                    },
                ))
    candidates.sort(key=lambda item: (-float(item.predicted_window_gain), int(item.primary_adjustment_count), str(item.candidate_id)))
    return candidates[:limit], {
        **audit,
        "reason": "ok" if candidates else "no_ranked_spectrum_candidate",
        "available_ids_by_su": {str(key): list(values) for key, values in sorted(available.items())},
        "candidate_count_before_limit": len(candidates),
        "candidate_limit": int(limit),
        "max_primary_adjustments": int(primary_limit),
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
