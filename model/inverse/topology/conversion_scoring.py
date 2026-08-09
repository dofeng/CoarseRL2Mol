from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from .aromatic_clusters import generate_aromatic_clusters
from .resource_profiles import (
    DEFAULT_STAGE3_H_HARD_TOLERANCE,
    SU11_AROMATIC_ROTATION,
    h_hard_bounds,
    h_within_hard_bounds,
)
from .resource_accounting import positive_resource_gap
from .stage3_models import HistogramConversion


RESOURCE_TYPES = (11, 22, 23, 24)
AROMATIC_TYPES = (12, 13)
CONVERSION_SCORING_VERSION = "hard_h_flex_balance_v6"
INFEASIBLE_SCORE = 1.0e12


@dataclass(frozen=True)
class FlexImpact:
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    penalty: float = 0.0
    scenario_before: str = ""
    scenario_after: str = ""


@dataclass(frozen=True)
class ConversionContext:
    inventory: Dict[int, int]
    demand: Dict[int, int]
    current_H: torch.Tensor
    target_h: Optional[float] = None
    h_tolerance: float = DEFAULT_STAGE3_H_HARD_TOLERANCE
    preferred_demand: Dict[int, int] = field(default_factory=dict)
    flex_audit: Dict[str, Any] = field(default_factory=dict)
    protected_resource_counts: Dict[int, int] = field(default_factory=dict)
    su11_rotation_index: int = 0
    phase: str = ""
    reason: str = ""
    allow_h_outside_nonworsening: bool = False


@dataclass(frozen=True)
class ConversionImpact:
    op: HistogramConversion
    resource_gap_before: Dict[int, int]
    resource_gap_after: Dict[int, int]
    resource_gap_gain: int
    h_before: Optional[float]
    h_after: Optional[float]
    h_delta: float
    h_bound_penalty: float
    h_drift_penalty: float
    flex_before: Dict[str, Any]
    flex_after: Dict[str, Any]
    flex_penalty: float
    aromatic_penalty: float
    donor_criticality_penalty: float
    topology_gain: float
    total_score: float
    feasible: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> Tuple[Any, ...]:
        source = int(self.op.from_su)
        target = int(self.op.to_su)
        return (
            0 if self.feasible else 1,
            round(float(self.total_score), 9),
            round(float(self.h_bound_penalty), 9),
            round(float(self.flex_penalty), 9),
            round(float(self.h_drift_penalty), 9),
            round(float(self.donor_criticality_penalty), 9),
            int(source in AROMATIC_TYPES),
            int(source),
            int(target),
        )


@dataclass(frozen=True)
class ConversionBundleScore:
    ops: Tuple[HistogramConversion, ...]
    H_after: Optional[torch.Tensor]
    inventory_after: Dict[int, int]
    resource_gap_before: Dict[int, int]
    resource_gap_after: Dict[int, int]
    resource_gap_gain: int
    h_before: Optional[float]
    h_after: Optional[float]
    h_delta: float
    h_bound_penalty: float
    h_drift_penalty: float
    flex_impact: FlexImpact
    aromatic_penalty: float
    donor_criticality_penalty: float
    topology_gain: float
    operation_penalty: float
    total_score: float
    feasible: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> Tuple[Any, ...]:
        primary = self.ops[0] if self.ops else HistogramConversion(-1, -1, 1, "")
        return (
            0 if self.feasible else 1,
            round(float(self.total_score), 9),
            round(float(self.h_bound_penalty), 9),
            round(float(self.flex_impact.penalty), 9),
            round(float(self.h_drift_penalty), 9),
            round(float(self.donor_criticality_penalty), 9),
            len(self.ops),
            int(primary.from_su in AROMATIC_TYPES),
            int(primary.from_su),
            int(primary.to_su),
        )


def _positive_gap(demand: Dict[int, int], inventory: Dict[int, int]) -> Dict[int, int]:
    return positive_resource_gap(demand, inventory, resource_types=RESOURCE_TYPES)


def _gap_total(gap: Dict[int, int]) -> int:
    return sum(max(0, int(value)) for value in dict(gap or {}).values())


def _hist_counts(H: torch.Tensor) -> Dict[int, int]:
    return {
        int(index): int(value)
        for index, value in enumerate(H.detach().cpu().long().tolist())
        if int(value) > 0
    }


def _apply_conversion_to_H(
    H: torch.Tensor,
    conversion: HistogramConversion,
) -> Optional[torch.Tensor]:
    candidate = H.detach().clone().long()
    source = int(conversion.from_su)
    target = int(conversion.to_su)
    count = max(1, int(conversion.count))
    if not (0 <= source < int(candidate.numel()) and 0 <= target < int(candidate.numel())):
        return None
    if int(candidate[source].item()) < int(count):
        return None
    candidate[source] -= int(count)
    candidate[target] += int(count)
    return candidate


def apply_conversions_to_H(
    H: torch.Tensor,
    conversions: Sequence[HistogramConversion],
) -> Optional[torch.Tensor]:
    candidate = H.detach().clone().long()
    for conversion in list(conversions or []):
        candidate = _apply_conversion_to_H(candidate, conversion)
        if candidate is None:
            return None
    return candidate


def apply_conversions_to_inventory(
    inventory: Dict[int, int],
    conversions: Sequence[HistogramConversion],
) -> Optional[Dict[int, int]]:
    work = {int(key): int(value) for key, value in dict(inventory or {}).items()}
    for conversion in list(conversions or []):
        source = int(conversion.from_su)
        target = int(conversion.to_su)
        count = max(1, int(conversion.count))
        if int(work.get(source, 0)) < int(count):
            return None
        work[source] = int(work.get(source, 0)) - int(count)
        work[target] = int(work.get(target, 0)) + int(count)
    return work


def _target_h_from_context(context: ConversionContext) -> Optional[float]:
    if context.target_h is None:
        return None
    try:
        value = float(context.target_h)
    except Exception:
        return None
    return value if value > 0 else None


def _ratio(current_h: Optional[float], target_h: Optional[float]) -> Optional[float]:
    if current_h is None or target_h is None or float(target_h) <= 0:
        return None
    return (float(current_h) - float(target_h)) / float(target_h)


def _outside_distance(
    ratio: Optional[float],
    tolerance: float = DEFAULT_STAGE3_H_HARD_TOLERANCE,
) -> float:
    if ratio is None:
        return 0.0
    lower_ratio = -min(1.0, max(0.0, float(tolerance)))
    upper_ratio = min(1.0, max(0.0, float(tolerance)))
    if float(ratio) < float(lower_ratio):
        return float(lower_ratio) - float(ratio)
    if float(ratio) > float(upper_ratio):
        return float(ratio) - float(upper_ratio)
    return 0.0


def _h_penalty(
    *,
    h_before: Optional[float],
    h_after: Optional[float],
    target_h: Optional[float],
    tolerance: float = DEFAULT_STAGE3_H_HARD_TOLERANCE,
) -> Tuple[float, float, Dict[str, Any]]:
    if h_before is None or h_after is None:
        return 0.0, 0.0, {"mode": "unconstrained"}
    h_delta = float(h_after) - float(h_before)
    if target_h is None or float(target_h) <= 0:
        return abs(float(h_delta)) * 0.4, abs(float(h_delta)) * 0.4, {
            "mode": "no_target_h",
            "h_delta": float(h_delta),
        }

    ratio_before = _ratio(h_before, target_h)
    ratio_after = _ratio(h_after, target_h)
    effective_tolerance = min(1.0, max(0.0, float(tolerance)))
    outside_before = _outside_distance(ratio_before, effective_tolerance)
    outside_after = _outside_distance(ratio_after, effective_tolerance)
    current_inside = h_within_hard_bounds(
        float(h_before), float(target_h), effective_tolerance
    )
    after_inside = h_within_hard_bounds(
        float(h_after), float(target_h), effective_tolerance
    )
    lower, upper = h_hard_bounds(float(target_h), effective_tolerance)
    bound_penalty = 0.0
    drift_penalty = 0.0

    # Stage3 resource planning has only a hard H boundary. Values inside that
    # boundary are equivalent for H scoring; temporary violations remain
    # rankable so compensation search can restore the complete bundle.
    target_scale = max(1.0, float(target_h))
    bound_penalty += outside_after * target_scale * 35.0
    if outside_after > outside_before:
        bound_penalty += (outside_after - outside_before) * target_scale * 80.0
    if not after_inside and current_inside:
        bound_penalty += 20.0

    diagnostics = {
        "mode": "stage3_hard_bound_score",
        "h_before": float(h_before),
        "h_after": float(h_after),
        "h_delta": float(h_delta),
        "target_h": float(target_h),
        "hard_tolerance": float(effective_tolerance),
        "ratio_before": float(ratio_before) if ratio_before is not None else None,
        "ratio_after": float(ratio_after) if ratio_after is not None else None,
        "hard_lower": float(lower) if lower is not None else None,
        "hard_upper": float(upper) if upper is not None else None,
        "hard_valid_after": bool(after_inside),
        "outside_before": float(outside_before),
        "outside_after": float(outside_after),
    }
    return float(bound_penalty), float(drift_penalty), diagnostics


def h_within_context_hard_bounds(
    context: ConversionContext,
    current_h: Optional[float],
    h_before: Optional[float] = None,
) -> bool:
    if current_h is None:
        return False
    target_h = _target_h_from_context(context)
    if target_h is None:
        return True
    tolerance = float(context.h_tolerance)
    if not bool(context.allow_h_outside_nonworsening) or h_before is None:
        return h_within_hard_bounds(float(current_h), target_h, tolerance)
    before_outside = _outside_distance(
        _ratio(float(h_before), target_h),
        tolerance,
    )
    after_outside = _outside_distance(
        _ratio(float(current_h), target_h),
        tolerance,
    )
    if float(before_outside) > 0.0:
        return float(after_outside) <= float(before_outside) + 1e-12
    return float(after_outside) <= 1e-12


def _adjust_bridgehead_info(
    flex_audit: Dict[str, Any],
    H_before: torch.Tensor,
    H_after: torch.Tensor,
) -> Tuple[int, int, int, int, int, int, int]:
    values = tuple(int(value) for value in tuple(dict(flex_audit or {}).get("bridgehead_info", ()) or ()))
    if len(values) >= 7:
        x, y, z, m, n, p, q = values[:7]
    else:
        before_12 = int(H_before[12].item()) if 12 < int(H_before.numel()) else 0
        x, y, z, m, n, p, q = 0, 0, 0, 0, 0, 0, before_12
    before_12 = int(H_before[12].item()) if 12 < int(H_before.numel()) else 0
    after_12 = int(H_after[12].item()) if 12 < int(H_after.numel()) else 0
    baseline_12 = int(
        dict(flex_audit or {}).get("histogram_su12_count", before_12) or 0
    )
    q = max(0, int(q) + int(after_12) - int(baseline_12))
    return int(x), int(y), int(z), int(m), int(n), int(p), int(q)


def _flex_window_from_H(H: torch.Tensor, bridgehead_info: Tuple[int, ...], explicit_count: int) -> Dict[str, Any]:
    counts = _hist_counts(H)
    clusters = generate_aromatic_clusters(counts, bridgehead_info=tuple(bridgehead_info or ()))
    cluster_count = int(len(clusters))
    rigid10_pairs = max(0, int(counts.get(10, 0)) // 2)
    z_value = int(cluster_count) - int(rigid10_pairs)
    flex_lower = int(z_value + 1)
    flex_upper = int(math.floor(0.8 * float(max(0, int(cluster_count)))))
    if cluster_count <= 0:
        scenario = "not_applicable"
    elif flex_lower > flex_upper:
        scenario = "invalid"
    elif int(explicit_count) < int(flex_lower):
        scenario = "below_range"
    elif int(explicit_count) > int(flex_upper):
        scenario = "above_range"
    else:
        scenario = "in_range"
    if scenario in {"not_applicable", "invalid"}:
        deviation = 0 if scenario == "not_applicable" else int(flex_lower - flex_upper)
    elif explicit_count < flex_lower:
        deviation = int(flex_lower) - int(explicit_count)
    elif explicit_count > flex_upper:
        deviation = int(explicit_count) - int(flex_upper)
    else:
        deviation = 0
    return {
        "X": int(cluster_count),
        "Y": int(rigid10_pairs),
        "Z": int(z_value),
        "flex_lower": int(flex_lower),
        "flex_upper": int(flex_upper),
        "explicit_flex_count": int(explicit_count),
        "maximum_flex_capacity": max(0, int(flex_upper) - int(explicit_count)),
        "minimum_flex_gap": max(0, int(flex_lower) - int(explicit_count)),
        "scenario": str(scenario),
        "deviation": int(deviation),
    }


def _flex_transition_penalty(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> float:
    before_deviation = int(before.get("deviation", 0) or 0)
    after_deviation = int(after.get("deviation", 0) or 0)
    penalty = 0.0

    # A conversion that enables one flex product but raises the lower bound by
    # the same amount has made no net progress.  Flex convergence therefore
    # has to outweigh the H soft-window preference; otherwise repeated
    # SU12->SU11 conversions can keep moving the target away indefinitely.
    if after_deviation > before_deviation:
        drift = float(after_deviation - before_deviation)
        penalty += 14.0 * drift + 4.0 * drift * drift
    elif after_deviation < before_deviation:
        penalty -= float(before_deviation - after_deviation) * 120.0
    if str(after.get("scenario", "")) == "invalid":
        penalty += 45.0
    if int(after.get("maximum_flex_capacity", 0) or 0) < int(before.get("maximum_flex_capacity", 0) or 0):
        penalty += float(int(before["maximum_flex_capacity"]) - int(after["maximum_flex_capacity"])) * 3.0
    if int(after.get("minimum_flex_gap", 0) or 0) > int(before.get("minimum_flex_gap", 0) or 0):
        penalty += float(int(after["minimum_flex_gap"]) - int(before["minimum_flex_gap"])) * 8.0
    if (
        int(after.get("X", 0) or 0) < int(before.get("X", 0) or 0)
        and after_deviation >= before_deviation
    ):
        penalty += float(int(before["X"]) - int(after["X"])) * 4.0
    return float(penalty)


def evaluate_flex_impact(
    H_before: torch.Tensor,
    H_after: torch.Tensor,
    flex_audit: Optional[Dict[str, Any]],
) -> FlexImpact:
    audit = dict(flex_audit or {})
    if not audit:
        return FlexImpact()
    explicit_before = int(
        audit.get(
            "effective_flex_count",
            audit.get("explicit_flex_count", audit.get("explicit_flex_count_raw", 0)),
        )
        or 0
    )
    expected_product_gain = max(0, int(audit.get("expected_flex_product_gain", 0) or 0))
    explicit_after = int(explicit_before) + int(expected_product_gain)
    before_bridgehead_info = _adjust_bridgehead_info(audit, H_before, H_before)
    before = _flex_window_from_H(H_before, before_bridgehead_info, explicit_before)
    after_bridgehead_info = _adjust_bridgehead_info(audit, H_before, H_after)
    after = _flex_window_from_H(H_after, after_bridgehead_info, explicit_after)
    before["expected_flex_product_gain"] = int(expected_product_gain)
    after["expected_flex_product_gain"] = int(expected_product_gain)
    penalty = _flex_transition_penalty(before, after)
    return FlexImpact(
        before=before,
        after=after,
        penalty=float(penalty),
        scenario_before=str(before.get("scenario", "")),
        scenario_after=str(after.get("scenario", "")),
    )


def _aromatic_rotation_offset(cursor: int, su_type: int) -> int:
    for offset in range(len(SU11_AROMATIC_ROTATION)):
        position = (int(cursor) + int(offset)) % len(SU11_AROMATIC_ROTATION)
        if int(SU11_AROMATIC_ROTATION[position]) == int(su_type):
            return int(offset)
    return len(SU11_AROMATIC_ROTATION)


def _aromatic_penalty(
    context: ConversionContext,
    ops: Sequence[HistogramConversion],
    flex_impact: FlexImpact,
) -> float:
    penalty = 0.0
    cursor = int(context.su11_rotation_index)
    for op in list(ops or []):
        source = int(op.from_su)
        target = int(op.to_su)
        if source in AROMATIC_TYPES and target != 11:
            penalty += 1.4
        elif source == 11 and target in AROMATIC_TYPES:
            penalty += 0.8
        if source in AROMATIC_TYPES and (target == 11 or op.via_su is not None and int(op.via_su) == 11):
            offset = _aromatic_rotation_offset(cursor, source)
            penalty += float(offset) * 0.01
            cursor += int(offset) + 1
        elif source == 11 and target in AROMATIC_TYPES:
            offset = _aromatic_rotation_offset(cursor, target)
            penalty += float(offset) * 0.005
            cursor += int(offset) + 1
    return float(penalty)


def _donor_criticality_penalty(
    context: ConversionContext,
    ops: Sequence[HistogramConversion],
    inventory_after: Dict[int, int],
) -> float:
    demand = {int(key): int(value) for key, value in dict(context.demand or {}).items()}
    preferred = {int(key): int(value) for key, value in dict(context.preferred_demand or {}).items()}
    protected = {int(key): int(value) for key, value in dict(context.protected_resource_counts or {}).items()}
    penalty = 0.0
    source_weights = {22: 4.0, 23: 3.0, 24: 1.5, 11: 2.0, 12: 1.2, 13: 0.8}
    for op in list(ops or []):
        source = int(op.from_su)
        target = int(op.to_su)
        count = max(1, int(op.count))
        reserved = max(
            int(demand.get(source, 0)),
            int(preferred.get(source, 0)),
            int(protected.get(source, 0)),
        )
        remaining = int(inventory_after.get(source, 0)) - int(reserved)
        if remaining < 0:
            penalty += abs(int(remaining)) * source_weights.get(source, 1.0) * 3.0
        elif remaining == 0 and source in {22, 23, 24}:
            penalty += source_weights.get(source, 1.0) * 0.8
        if target == 11 and source in {22, 23}:
            penalty += source_weights.get(source, 1.0) * float(count)
        if target == 11 and source == 24:
            penalty += 0.8 * float(count)

    required_11 = max(int(demand.get(11, 0)), int(preferred.get(11, 0)), int(protected.get(11, 0)))
    excess_11 = int(inventory_after.get(11, 0)) - int(required_11)
    if excess_11 > 0:
        penalty += float(excess_11) * 1.5
    return float(penalty)


def score_conversion_bundle(
    context: ConversionContext,
    conversions: Sequence[HistogramConversion],
    *,
    h_total_fn: Callable[[torch.Tensor, Optional[Dict[str, Any]]], float],
    special_degree_meta: Optional[Dict[str, Any]] = None,
) -> ConversionBundleScore:
    ops = tuple(conversions or ())
    gap_before = _positive_gap(context.demand, context.inventory)
    inventory_after = apply_conversions_to_inventory(context.inventory, ops)
    H_after = apply_conversions_to_H(context.current_H, ops)
    if inventory_after is None or H_after is None:
        return ConversionBundleScore(
            ops=ops,
            H_after=None,
            inventory_after=dict(context.inventory),
            resource_gap_before=gap_before,
            resource_gap_after=gap_before,
            resource_gap_gain=0,
            h_before=None,
            h_after=None,
            h_delta=0.0,
            h_bound_penalty=INFEASIBLE_SCORE,
            h_drift_penalty=INFEASIBLE_SCORE,
            flex_impact=FlexImpact(penalty=INFEASIBLE_SCORE),
            aromatic_penalty=INFEASIBLE_SCORE,
            donor_criticality_penalty=INFEASIBLE_SCORE,
            topology_gain=0.0,
            operation_penalty=INFEASIBLE_SCORE,
            total_score=INFEASIBLE_SCORE,
            feasible=False,
            diagnostics={"reason": "negative_inventory_or_histogram"},
        )
    gap_after = _positive_gap(context.demand, inventory_after)
    gap_gain = int(_gap_total(gap_before) - _gap_total(gap_after))
    target_h = _target_h_from_context(context)
    h_before = h_total_fn(context.current_H, special_degree_meta)
    h_after = h_total_fn(H_after, special_degree_meta)
    h_delta = float(h_after) - float(h_before)
    h_bound_penalty, h_drift_penalty, h_diag = _h_penalty(
        h_before=float(h_before),
        h_after=float(h_after),
        target_h=target_h,
        tolerance=float(context.h_tolerance),
    )
    flex_impact = evaluate_flex_impact(context.current_H, H_after, context.flex_audit)
    aromatic_penalty = _aromatic_penalty(context, ops, flex_impact)
    donor_penalty = _donor_criticality_penalty(context, ops, inventory_after)
    operation_penalty = float(len(ops)) * 0.25 + sum(0.4 for op in ops if op.via_su is not None)
    topology_gain = float(gap_gain) * 100.0
    resource_penalty = float(_gap_total(gap_after)) * 22.0
    if _gap_total(gap_after) > _gap_total(gap_before):
        resource_penalty += 250.0
    total_score = (
        resource_penalty
        - topology_gain
        + float(h_bound_penalty)
        + float(h_drift_penalty)
        + float(flex_impact.penalty)
        + float(aromatic_penalty)
        + float(donor_penalty)
        + float(operation_penalty)
    )
    diagnostics = {
        "scoring_version": CONVERSION_SCORING_VERSION,
        "h": {
            **dict(h_diag or {}),
            "hard_bound_penalty": float(h_bound_penalty),
            "drift_penalty": float(h_drift_penalty),
        },
        "flex": {
            "before": dict(flex_impact.before or {}),
            "after": dict(flex_impact.after or {}),
            "penalty": float(flex_impact.penalty),
        },
        "resource_gap_before": dict(gap_before),
        "resource_gap_after": dict(gap_after),
        "resource_gap_gain": int(gap_gain),
        "aromatic_penalty": float(aromatic_penalty),
        "donor_criticality_penalty": float(donor_penalty),
        "operation_penalty": float(operation_penalty),
        "topology_gain": float(topology_gain),
        "total_score": float(total_score),
    }
    return ConversionBundleScore(
        ops=ops,
        H_after=H_after,
        inventory_after=inventory_after,
        resource_gap_before=gap_before,
        resource_gap_after=gap_after,
        resource_gap_gain=int(gap_gain),
        h_before=float(h_before),
        h_after=float(h_after),
        h_delta=float(h_delta),
        h_bound_penalty=float(h_bound_penalty),
        h_drift_penalty=float(h_drift_penalty),
        flex_impact=flex_impact,
        aromatic_penalty=float(aromatic_penalty),
        donor_criticality_penalty=float(donor_penalty),
        topology_gain=float(topology_gain),
        operation_penalty=float(operation_penalty),
        total_score=float(total_score),
        feasible=True,
        diagnostics=diagnostics,
    )


def score_conversion(
    context: ConversionContext,
    conversion: HistogramConversion,
    *,
    h_total_fn: Callable[[torch.Tensor, Optional[Dict[str, Any]]], float],
    special_degree_meta: Optional[Dict[str, Any]] = None,
) -> ConversionImpact:
    bundle = score_conversion_bundle(
        context,
        [conversion],
        h_total_fn=h_total_fn,
        special_degree_meta=special_degree_meta,
    )
    return ConversionImpact(
        op=conversion,
        resource_gap_before=dict(bundle.resource_gap_before),
        resource_gap_after=dict(bundle.resource_gap_after),
        resource_gap_gain=int(bundle.resource_gap_gain),
        h_before=bundle.h_before,
        h_after=bundle.h_after,
        h_delta=float(bundle.h_delta),
        h_bound_penalty=float(bundle.h_bound_penalty),
        h_drift_penalty=float(bundle.h_drift_penalty),
        flex_before=dict(bundle.flex_impact.before or {}),
        flex_after=dict(bundle.flex_impact.after or {}),
        flex_penalty=float(bundle.flex_impact.penalty),
        aromatic_penalty=float(bundle.aromatic_penalty),
        donor_criticality_penalty=float(bundle.donor_criticality_penalty),
        topology_gain=float(bundle.topology_gain),
        total_score=float(bundle.total_score),
        feasible=bool(bundle.feasible),
        diagnostics=dict(bundle.diagnostics or {}),
    )


def rank_conversions(
    context: ConversionContext,
    candidates: Iterable[HistogramConversion],
    *,
    h_total_fn: Callable[[torch.Tensor, Optional[Dict[str, Any]]], float],
    special_degree_meta: Optional[Dict[str, Any]] = None,
) -> List[ConversionImpact]:
    impacts = [
        score_conversion(
            context,
            candidate,
            h_total_fn=h_total_fn,
            special_degree_meta=special_degree_meta,
        )
        for candidate in list(candidates or [])
    ]
    return sorted(impacts, key=lambda item: item.sort_key)
