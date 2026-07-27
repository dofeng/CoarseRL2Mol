from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ...shared.coarse_graph import E_SU
from ...shared.inverse_common import get_effective_hist_element_vector
from .stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    HistogramConversion,
    Stage3AdjustmentApplyResult,
    Stage3AdjustmentPlan,
)
from .conversion_scoring import (
    ConversionBundleScore,
    ConversionContext,
    apply_conversions_to_H,
    apply_conversions_to_inventory,
    rank_conversions,
    score_conversion_bundle,
)
from .resource_profiles import (
    H_SOFT_LOWER_RATIO,
    H_SOFT_UPPER_RATIO,
    MAX_FLEX_LOWER_INCREASE_PER_ADJUSTMENT_PLAN,
    SU11_AROMATIC_ROTATION,
    h_within_soft_bounds,
)
from .resource_accounting import (
    positive_resource_gap,
    preflight_histogram_accounting,
    resource_gap_does_not_worsen,
)


RESOURCE_TYPES = (11, 23, 22, 24)
TARGET_PRIORITY = {23: 0, 11: 1, 22: 2, 24: 3}
AROMATIC_PATTERN = (12, 13)


def _counts(H: torch.Tensor, keys: Iterable[int]) -> Dict[int, int]:
    return {
        int(key): int(H[int(key)].item()) if 0 <= int(key) < int(H.numel()) else 0
        for key in keys
    }


def _h_total(H: torch.Tensor, special_degree_meta: Optional[Dict[str, Any]]) -> float:
    H_effective = torch.zeros(int(E_SU.size(0)), dtype=torch.long)
    copied = min(int(H.numel()), int(H_effective.numel()))
    H_effective[:copied] = H.detach().cpu().long()[:copied]
    vector = get_effective_hist_element_vector(
        H_effective,
        special_degree_meta=special_degree_meta,
        E_SU_tensor=E_SU.detach().cpu(),
    )
    return float(vector[1].detach().cpu().item())


def _conversion_delta(from_su: int, to_su: int, count: int = 1) -> Dict[int, int]:
    return {int(from_su): -int(count), int(to_su): int(count)}


def _apply_delta(H: torch.Tensor, delta: Dict[int, int]) -> Optional[torch.Tensor]:
    candidate = H.detach().clone().long()
    for su_type, value in dict(delta or {}).items():
        if not 0 <= int(su_type) < int(candidate.numel()):
            return None
        candidate[int(su_type)] += int(value)
        if int(candidate[int(su_type)].item()) < 0:
            return None
    return candidate


def _merge_delta(ops: Iterable[HistogramConversion]) -> Dict[int, int]:
    delta: Dict[int, int] = {}
    for op in ops:
        delta[int(op.from_su)] = int(delta.get(int(op.from_su), 0)) - int(op.count)
        delta[int(op.to_su)] = int(delta.get(int(op.to_su), 0)) + int(op.count)
    return {int(key): int(value) for key, value in delta.items() if int(value) != 0}


def _iter_plan_ops(plan: Stage3AdjustmentPlan) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for bundle in list(plan.adjustment_bundles or []):
        bundle_dict = bundle.to_dict() if hasattr(bundle, "to_dict") else dict(bundle or {})
        for raw in list(bundle_dict.get("bundle_ops", []) or []):
            op = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw or {})
            ops.append(op)
    if not ops:
        for raw in list(plan.planned_conversions or []):
            op = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw or {})
            ops.append(op)
    return ops


def _phase_inventory(plan: Stage3AdjustmentPlan, H: torch.Tensor) -> Dict[int, int]:
    runtime = dict(plan.runtime or {})
    assembly = dict(runtime.get("assembly_before_adjustment", {}) or {})
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    counts = dict(ledger.get("convertible_counts", {}) or {})
    if not counts:
        counts = dict(ledger.get("remaining_counts_by_su", {}) or {})
    if not counts:
        counts = dict(ledger.get("remaining_counts", {}) or {})
    inventory = {
        int(key): int(value)
        for key, value in counts.items()
        if str(key).lstrip("-").isdigit()
    }
    for su_type in set(RESOURCE_TYPES) | {10, 12, 13}:
        inventory.setdefault(int(su_type), 0)
    donor_counts = dict(ledger.get("su11_donor_counts", {}) or {})
    for su_type in (10, 12, 13):
        inventory[int(su_type)] = int(donor_counts.get(str(su_type), donor_counts.get(int(su_type), inventory[int(su_type)])) or 0)
    return inventory


def _phase_demand(plan: Stage3AdjustmentPlan, inventory: Dict[int, int]) -> Dict[int, int]:
    request = dict(plan.hard_required_counts or plan.minimum_resource_request or {})
    if not request:
        request = dict(plan.resource_deficit or {})
        return {
            int(su): int(inventory.get(int(su), 0)) + max(0, int(value))
            for su, value in request.items()
            if str(su).lstrip("-").isdigit()
        }
    demand = {
        int(su): max(0, int(value))
        for su, value in request.items()
        if str(su).lstrip("-").isdigit()
    }
    for su_type in RESOURCE_TYPES:
        demand.setdefault(int(su_type), 0)
    return demand


def _preferred_phase_demand(plan: Stage3AdjustmentPlan) -> Dict[int, int]:
    return {
        int(su): max(0, int(value))
        for su, value in dict(plan.preferred_required_counts or {}).items()
        if str(su).lstrip("-").isdigit()
    }


def _runtime_flex_audit(plan: Stage3AdjustmentPlan) -> Dict[str, Any]:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    audit: Dict[str, Any] = {}
    if phase_eval is not None:
        audit = dict(getattr(phase_eval, "flex_audit", {}) or {})
    phase_eval_dict = dict(runtime.get("phase_evaluation_dict", {}) or {})
    if not audit:
        audit = dict(phase_eval_dict.get("flex_audit", {}) or {})
    if not audit:
        audit = dict(runtime.get("flex_audit", {}) or {})
    if not audit:
        return {}

    flex_counts = (
        dict(getattr(phase_eval, "flex_counts", {}) or {})
        if phase_eval is not None
        else dict(phase_eval_dict.get("flex_counts", {}) or {})
    )
    if flex_counts:
        audit["effective_flex_count"] = int(
            flex_counts.get(
                "final",
                audit.get("explicit_flex_count", audit.get("explicit_flex_count_raw", 0)),
            )
            or 0
        )
    active_roles = {
        str(dict(contract or {}).get("target_role", "") or "")
        for contract in list(getattr(plan, "requirement_contracts", []) or [])
        if dict(contract or {})
    }
    audit["expected_flex_product_gain"] = int(
        str(getattr(plan, "phase", "") or "") == "external_flex_closure"
        and "flex_chain" in active_roles
    )
    return audit


def _runtime_protected_counts(plan: Stage3AdjustmentPlan) -> Dict[int, int]:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    assembly = dict(runtime.get("assembly_before_adjustment", {}) or {})
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    protected = dict(ledger.get("protected_counts", {}) or {})
    return {
        int(key): max(0, int(value))
        for key, value in protected.items()
        if str(key).lstrip("-").isdigit()
    }


def _flex_lower_increase_allowed(
    plan: Stage3AdjustmentPlan,
    score: ConversionBundleScore,
) -> bool:
    consumes_su12_via_su11 = any(
        int(op.from_su) == 12
        and (
            int(op.to_su) == 11
            or op.via_su is not None and int(op.via_su) == 11
        )
        for op in score.ops
    )
    if not consumes_su12_via_su11:
        return True
    audit = _runtime_flex_audit(plan)
    phase = str(getattr(plan, "phase", "") or "")
    if phase == "flex_range_reconciliation" or not bool(audit.get("enforced", False)):
        return True
    if audit.get("flex_lower") is None:
        return True
    flex_after = dict(getattr(score.flex_impact, "after", {}) or {})
    if flex_after.get("flex_lower") is None:
        return True
    baseline_lower = int(audit.get("flex_lower", 0) or 0)
    candidate_lower = int(flex_after.get("flex_lower", baseline_lower) or baseline_lower)
    maximum_lower = (
        int(baseline_lower)
        + int(MAX_FLEX_LOWER_INCREASE_PER_ADJUSTMENT_PLAN)
    )
    flex_diagnostics = dict(score.diagnostics.get("flex", {}) or {})
    flex_diagnostics.update({
        "baseline_flex_lower": int(baseline_lower),
        "maximum_allowed_flex_lower": int(maximum_lower),
        "candidate_flex_lower": int(candidate_lower),
        "lower_increase_allowed": bool(candidate_lower <= maximum_lower),
    })
    score.diagnostics["flex"] = flex_diagnostics
    allowed = bool(candidate_lower <= maximum_lower)
    if not allowed:
        rejection = {
            "ops": [_conversion_signature(op) for op in score.ops],
            "baseline_flex_lower": int(baseline_lower),
            "maximum_allowed_flex_lower": int(maximum_lower),
            "candidate_flex_lower": int(candidate_lower),
        }
        diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
        rejections = list(diagnostics.get("flex_lower_increase_rejections", []) or [])
        if rejection not in rejections:
            rejections.append(rejection)
        diagnostics["flex_lower_increase_rejections"] = rejections
        plan.diagnostics = diagnostics
    return allowed


def _scoring_context(
    plan: Stage3AdjustmentPlan,
    H: torch.Tensor,
    inventory: Dict[int, int],
    demand: Dict[int, int],
    preferred_demand: Optional[Dict[int, int]],
    target_h: Optional[float],
    aromatic_cursor: int,
) -> ConversionContext:
    return ConversionContext(
        inventory={int(key): int(value) for key, value in dict(inventory or {}).items()},
        demand={int(key): int(value) for key, value in dict(demand or {}).items()},
        current_H=H.detach().clone().long(),
        target_h=target_h,
        preferred_demand={
            int(key): int(value)
            for key, value in dict(preferred_demand or {}).items()
            if str(key).lstrip("-").isdigit()
        },
        flex_audit=_runtime_flex_audit(plan),
        protected_resource_counts=_runtime_protected_counts(plan),
        su11_rotation_index=int(aromatic_cursor),
        phase=str(getattr(plan, "phase", "") or ""),
        reason=str(getattr(plan, "reason", "") or ""),
    )


def _surplus(
    inventory: Dict[int, int],
    demand: Dict[int, int],
    su_type: int,
) -> int:
    reserved = int(demand.get(int(su_type), 0)) if int(su_type) in RESOURCE_TYPES else 0
    return int(inventory.get(int(su_type), 0)) - int(reserved)


def _direct_gap_candidates(
    target: int,
    inventory: Dict[int, int],
    demand: Dict[int, int],
) -> List[HistogramConversion]:
    candidates: List[HistogramConversion] = []
    donor_types = list(RESOURCE_TYPES)
    if int(target) == 11:
        donor_types = donor_types + [12, 13]
    for source in donor_types:
        if int(source) == int(target) or _surplus(inventory, demand, int(source)) <= 0:
            continue
        candidates.append(HistogramConversion(
            from_su=int(source),
            to_su=int(target),
            count=1,
            reason="h_direction_aware_resource_rebalance",
        ))
    return candidates


def _conversion_signature(conversion: HistogramConversion) -> str:
    via = (
        f"@{int(conversion.via_su)}"
        if conversion.via_su is not None
        else ""
    )
    return f"{int(conversion.from_su)}>{int(conversion.to_su)}{via}"


def _forbidden_primary_signatures(plan: Stage3AdjustmentPlan) -> set[str]:
    return {
        str(value)
        for value in list(
            dict(getattr(plan, "diagnostics", {}) or {}).get(
                "forbidden_primary_conversion_signatures",
                [],
            )
            or []
        )
        if str(value)
    }


def _atomic_aromatic_bridge_candidates(
    target: int,
    inventory: Dict[int, int],
    demand: Dict[int, int],
) -> List[HistogramConversion]:
    if int(target) == 11:
        return []
    return [
        HistogramConversion(
            from_su=int(source),
            to_su=int(target),
            count=1,
            reason=f"atomic_aromatic_via_su11_to_su{int(target)}",
            via_su=11,
        )
        for source in AROMATIC_PATTERN
        if _surplus(inventory, demand, int(source)) > 0
    ]


def _aromatic_rotation_offset(aromatic_cursor: int, su_type: int) -> Optional[int]:
    for offset in range(len(SU11_AROMATIC_ROTATION)):
        position = (int(aromatic_cursor) + int(offset)) % len(SU11_AROMATIC_ROTATION)
        if int(SU11_AROMATIC_ROTATION[position]) == int(su_type):
            return int(offset)
    return None


def advance_su11_aromatic_rotation(aromatic_cursor: int, su_type: int) -> int:
    offset = _aromatic_rotation_offset(int(aromatic_cursor), int(su_type))
    if offset is None:
        return int(aromatic_cursor)
    return int(aromatic_cursor) + int(offset) + 1


def next_su11_aromatic_target(
    aromatic_cursor: int,
    available_targets: Iterable[int] = AROMATIC_PATTERN,
) -> Tuple[Optional[int], int]:
    available = {int(value) for value in list(available_targets or [])}
    if not available:
        return None, int(aromatic_cursor)
    for offset in range(len(SU11_AROMATIC_ROTATION)):
        position = (int(aromatic_cursor) + int(offset)) % len(SU11_AROMATIC_ROTATION)
        target = int(SU11_AROMATIC_ROTATION[position])
        if target in available:
            return target, int(aromatic_cursor) + int(offset) + 1
    return None, int(aromatic_cursor)


def _advance_cursor_for_conversion(aromatic_cursor: int, conversion: HistogramConversion) -> int:
    if int(conversion.from_su) in AROMATIC_PATTERN and (
        int(conversion.to_su) == 11
        or conversion.via_su is not None and int(conversion.via_su) == 11
    ):
        return advance_su11_aromatic_rotation(
            int(aromatic_cursor),
            int(conversion.from_su),
        )
    if int(conversion.from_su) == 11 and int(conversion.to_su) in AROMATIC_PATTERN:
        return advance_su11_aromatic_rotation(
            int(aromatic_cursor),
            int(conversion.to_su),
        )
    return int(aromatic_cursor)


def _h_compensation_candidates(
    inventory: Dict[int, int],
    demand: Dict[int, int],
    protected_counts: Optional[Dict[int, int]] = None,
) -> List[HistogramConversion]:
    candidates: List[HistogramConversion] = []
    donor_types = list(RESOURCE_TYPES) + [12, 13]
    safe_nondeficit_pairs = {
        (22, 23),
        (23, 22),
        (24, 22),
        (24, 23),
        (11, 23),
        (23, 11),
        (24, 11),
        (12, 23),
        (13, 23),
    }
    protected = {
        int(key): max(0, int(value))
        for key, value in dict(protected_counts or {}).items()
    }
    for source in donor_types:
        source_reserve = max(
            int(demand.get(int(source), 0)),
            int(protected.get(int(source), 0)),
        )
        source_margin = int(inventory.get(int(source), 0)) - int(source_reserve)
        if int(source_margin) <= 0:
            continue
        targets = list(RESOURCE_TYPES)
        for target in targets:
            if int(source) == int(target):
                continue
            if (int(source), int(target)) not in safe_nondeficit_pairs:
                continue
            # H compensation is not a second topology-gap fill. Once H is near
            # or outside its soft window, a conversion such as SU23->SU22 must
            # remain available even when SU22 already satisfies its contract
            # reserve. The bundle filters below still require the donor to be
            # surplus, preserve the primary conversion, and leave every
            # positive topology resource gap unchanged.
            candidates.append(HistogramConversion(
                from_su=int(source),
                to_su=int(target),
                count=1,
                reason="soft_h_balance_compensation",
                via_su=11 if int(source) in AROMATIC_PATTERN and int(target) != 11 else None,
            ))
    return candidates


def _h_ratio(value: float, target_h: float) -> float:
    return (float(value) - float(target_h)) / float(target_h)


def _h_outside_distance(value: float, target_h: float) -> float:
    ratio = _h_ratio(value, target_h)
    if ratio < float(H_SOFT_LOWER_RATIO):
        return float(H_SOFT_LOWER_RATIO) - float(ratio)
    if ratio > float(H_SOFT_UPPER_RATIO):
        return float(ratio) - float(H_SOFT_UPPER_RATIO)
    return 0.0


def _h_nearest_margin(value: float, target_h: float) -> float:
    ratio = _h_ratio(value, target_h)
    return min(
        float(ratio) - float(H_SOFT_LOWER_RATIO),
        float(H_SOFT_UPPER_RATIO) - float(ratio),
    )


def _compensation_improves_h_state(
    *,
    primary_score: ConversionBundleScore,
    bundle_score: ConversionBundleScore,
    target_h: Optional[float],
) -> bool:
    if target_h is None or float(target_h) <= 0:
        return False
    if primary_score.h_before is None or primary_score.h_after is None or bundle_score.h_after is None:
        return False
    h_before = float(primary_score.h_before)
    h_primary = float(primary_score.h_after)
    h_bundle = float(bundle_score.h_after)
    primary_inside = h_within_soft_bounds(h_primary, float(target_h))
    bundle_inside = h_within_soft_bounds(h_bundle, float(target_h))
    window_width = float(H_SOFT_UPPER_RATIO) - float(H_SOFT_LOWER_RATIO)
    critical_margin = 0.25 * float(window_width)

    if primary_inside:
        before_inside = h_within_soft_bounds(h_before, float(target_h))
        before_margin = _h_nearest_margin(h_before, float(target_h))
        primary_margin = _h_nearest_margin(h_primary, float(target_h))
        # A primary that leaves the current soft-window margin unchanged must
        # remain primary-only.  Compensation is justified inside the window
        # only when the topology conversion itself moves H toward a critical
        # boundary; merely starting near a boundary is not enough.
        if before_inside and float(primary_margin) >= float(before_margin) - 1e-12:
            return False
        if float(primary_margin) >= float(critical_margin):
            return False
        return bool(
            bundle_inside
            and _h_nearest_margin(h_bundle, float(target_h)) > float(primary_margin) + 1e-12
        )

    primary_outside = _h_outside_distance(h_primary, float(target_h))
    bundle_outside = _h_outside_distance(h_bundle, float(target_h))
    if bundle_outside + 1e-12 < primary_outside:
        return True
    if not h_within_soft_bounds(h_before, float(target_h)):
        return abs(h_bundle - float(target_h)) + 1e-9 < abs(h_primary - float(target_h))
    return False


def _bundle_preserves_primary_effect(ops: Iterable[HistogramConversion]) -> bool:
    bundle = list(ops or [])
    if not bundle:
        return False
    primary = bundle[0]
    source = int(primary.from_su)
    target = int(primary.to_su)
    count = max(1, int(primary.count))
    if source == target:
        return False
    delta = _merge_delta(bundle)
    return (
        int(delta.get(source, 0)) <= -int(count)
        and int(delta.get(target, 0)) >= int(count)
    )


def _enumerate_scored_bundles(
    primary_candidates: List[HistogramConversion],
    *,
    context: ConversionContext,
    max_h_compensation_ops: int,
    special_degree_meta: Optional[Dict[str, Any]],
) -> List[ConversionBundleScore]:
    bundles: List[List[HistogramConversion]] = [[candidate] for candidate in list(primary_candidates or [])]
    compensation_depth = max(0, int(max_h_compensation_ops))
    if compensation_depth <= 0:
        return [
            score_conversion_bundle(
                context,
                bundle,
                h_total_fn=_h_total,
                special_degree_meta=special_degree_meta,
            )
            for bundle in bundles
        ]

    expanded: List[List[HistogramConversion]] = list(bundles)
    frontier: List[List[HistogramConversion]] = list(bundles)
    seen_bundle_signatures = {
        tuple(
            (int(op.from_su), int(op.to_su), int(op.count), op.via_su)
            for op in bundle
        )
        for bundle in bundles
    }
    beam_width = 24
    for _ in range(compensation_depth):
        next_candidates: List[Tuple[ConversionBundleScore, List[HistogramConversion]]] = []
        for bundle in frontier:
            primary_score = score_conversion_bundle(
                context,
                bundle,
                h_total_fn=_h_total,
                special_degree_meta=special_degree_meta,
            )
            if not primary_score.feasible:
                continue
            H_after = apply_conversions_to_H(context.current_H, bundle)
            inventory_after = apply_conversions_to_inventory(context.inventory, bundle)
            if H_after is None or inventory_after is None:
                continue
            bundle_context = ConversionContext(
                inventory=inventory_after,
                demand=context.demand,
                current_H=H_after,
                target_h=context.target_h,
                preferred_demand=context.preferred_demand,
                flex_audit=context.flex_audit,
                protected_resource_counts=context.protected_resource_counts,
                su11_rotation_index=context.su11_rotation_index,
                phase=context.phase,
                reason=context.reason,
            )
            ranked_compensation = rank_conversions(
                bundle_context,
                _h_compensation_candidates(
                    inventory_after,
                    context.demand,
                    protected_counts=context.protected_resource_counts,
                ),
                h_total_fn=_h_total,
                special_degree_meta=special_degree_meta,
            )
            for impact in ranked_compensation[:6]:
                candidate = list(bundle) + [impact.op]
                if not _bundle_preserves_primary_effect(candidate):
                    continue
                if not _bundle_uses_only_initial_donors(context.inventory, candidate):
                    continue
                if not _bundle_preserves_donor_reserves(context, candidate):
                    continue
                score = score_conversion_bundle(
                    context,
                    candidate,
                    h_total_fn=_h_total,
                    special_degree_meta=special_degree_meta,
                )
                if not score.feasible:
                    continue
                if not _compensation_improves_h_state(
                    primary_score=primary_score,
                    bundle_score=score,
                    target_h=context.target_h,
                ):
                    continue
                if dict(score.resource_gap_after) != dict(
                    primary_score.resource_gap_after
                ):
                    continue
                if _gap_total(score.resource_gap_after) > _gap_total(score.resource_gap_before):
                    continue
                signature = tuple(
                    (int(op.from_su), int(op.to_su), int(op.count), op.via_su)
                    for op in candidate
                )
                if signature in seen_bundle_signatures:
                    continue
                seen_bundle_signatures.add(signature)
                next_candidates.append((score, candidate))
        next_candidates.sort(key=lambda item: item[0].sort_key)
        frontier = [candidate for _, candidate in next_candidates[:beam_width]]
        expanded.extend(frontier)
        if not frontier:
            break
    return [
        score_conversion_bundle(
            context,
            bundle,
            h_total_fn=_h_total,
            special_degree_meta=special_degree_meta,
        )
        for bundle in expanded
    ]


def _choose_bundle_from_gap(
    H: torch.Tensor,
    inventory: Dict[int, int],
    demand: Dict[int, int],
    aromatic_cursor: int,
    *,
    plan: Stage3AdjustmentPlan,
    preferred_demand: Optional[Dict[int, int]],
    target_h: Optional[float],
    special_degree_meta: Optional[Dict[str, Any]],
    max_h_compensation_ops: int,
) -> Tuple[List[HistogramConversion], int, Optional[ConversionBundleScore]]:
    gap = {su: int(demand.get(su, 0)) - int(inventory.get(su, 0)) for su in RESOURCE_TYPES}
    targets = [su for su in RESOURCE_TYPES if int(gap.get(su, 0)) > 0]
    if not targets:
        return [], int(aromatic_cursor), None
    target = sorted(targets, key=lambda su: (-int(gap[su]), TARGET_PRIORITY[int(su)]))[0]
    candidates = _direct_gap_candidates(int(target), inventory, demand)
    candidates.extend(_atomic_aromatic_bridge_candidates(int(target), inventory, demand))
    forbidden = _forbidden_primary_signatures(plan)
    candidates = [
        candidate
        for candidate in candidates
        if _conversion_signature(candidate) not in forbidden
    ]
    context = _scoring_context(
        plan,
        H,
        inventory,
        demand,
        preferred_demand,
        target_h,
        aromatic_cursor,
    )
    ranked_primary = rank_conversions(
        context,
        candidates,
        h_total_fn=_h_total,
        special_degree_meta=special_degree_meta,
    )
    primary_candidates = [impact.op for impact in ranked_primary if impact.feasible]
    if not primary_candidates:
        return [], int(aromatic_cursor), None
    scored_bundles = _enumerate_scored_bundles(
        primary_candidates,
        context=context,
        max_h_compensation_ops=max_h_compensation_ops,
        special_degree_meta=special_degree_meta,
    )
    feasible = [
        score for score in scored_bundles
        if score.feasible
        and _bundle_preserves_primary_effect(score.ops)
        and _gap_total(score.resource_gap_after) < _gap_total(score.resource_gap_before)
        and _does_not_worsen_resource_gap(context.demand, context.inventory, score.inventory_after)
        and _bundle_uses_only_initial_donors(context.inventory, score.ops)
        and _bundle_preserves_donor_reserves(context, score.ops)
        and _flex_lower_increase_allowed(plan, score)
    ]
    if not feasible:
        return [], int(aromatic_cursor), None
    selected = min(feasible, key=lambda item: item.sort_key)
    cursor = int(aromatic_cursor)
    for op in selected.ops:
        cursor = _advance_cursor_for_conversion(cursor, op)
    return list(selected.ops), int(cursor), selected


def _choose_from_gap(
    H: torch.Tensor,
    inventory: Dict[int, int],
    demand: Dict[int, int],
    aromatic_cursor: int,
    current_h: Optional[float] = None,
    target_h: Optional[float] = None,
    plan: Optional[Stage3AdjustmentPlan] = None,
    preferred_demand: Optional[Dict[int, int]] = None,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    max_h_compensation_ops: int = 0,
) -> Tuple[Optional[HistogramConversion], int]:
    del current_h
    scoring_plan = plan or Stage3AdjustmentPlan(phase="")
    bundle, cursor, _ = _choose_bundle_from_gap(
        H,
        inventory,
        demand,
        aromatic_cursor,
        plan=scoring_plan,
        preferred_demand=preferred_demand,
        target_h=target_h,
        special_degree_meta=special_degree_meta,
        max_h_compensation_ops=max_h_compensation_ops,
    )
    if not bundle:
        return None, int(aromatic_cursor)
    return bundle[0], int(cursor)


def _positive_gap(demand: Dict[int, int], inventory: Dict[int, int]) -> Dict[int, int]:
    return positive_resource_gap(demand, inventory, resource_types=RESOURCE_TYPES)


def _gap_total(gap: Dict[int, int]) -> int:
    return sum(max(0, int(value)) for value in dict(gap or {}).values())


def _does_not_worsen_resource_gap(
    demand: Dict[int, int],
    inventory_before: Dict[int, int],
    inventory_after: Dict[int, int],
) -> bool:
    return resource_gap_does_not_worsen(
        demand,
        inventory_before,
        inventory_after,
        resource_types=RESOURCE_TYPES,
    )


def _bundle_uses_only_initial_donors(
    inventory: Dict[int, int],
    ops: Iterable[HistogramConversion],
) -> bool:
    required: Dict[int, int] = {}
    for op in list(ops or []):
        required[int(op.from_su)] = int(required.get(int(op.from_su), 0)) + max(1, int(op.count))
    return all(
        int(count) <= int(inventory.get(int(su_type), 0))
        for su_type, count in dict(required or {}).items()
    )


def _bundle_preserves_donor_reserves(
    context: ConversionContext,
    ops: Iterable[HistogramConversion],
) -> bool:
    consumed: Dict[int, int] = {}
    for op in list(ops or []):
        source = int(op.from_su)
        consumed[source] = int(consumed.get(source, 0)) + max(1, int(op.count))
    for su_type, count in consumed.items():
        reserve = max(
            int(context.demand.get(int(su_type), 0)),
            int(context.protected_resource_counts.get(int(su_type), 0)),
        )
        available = int(context.inventory.get(int(su_type), 0)) - int(reserve)
        if int(count) > max(0, int(available)):
            return False
    return True


def build_histogram_adjustment_plan(
    plan: Stage3AdjustmentPlan,
    H: torch.Tensor,
    E_target: Any = None,
    max_primary_adjustments: int = 10,
    max_h_compensation_ops: int = 8,
    h_tolerance: float = 0.06,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    su11_rotation_index: int = 0,
) -> HistogramAdjustmentPlan:
    if H is None:
        runtime = dict(plan.runtime or {})
        classification = runtime.get("classification_before_adjustment")
        nodes = list(getattr(classification, "nodes", []) or [])
        H_before = torch.zeros(int(E_SU.size(0)), dtype=torch.long)
        for node in nodes:
            su_type = int(getattr(node, "su_type", -1))
            if 0 <= su_type < int(H_before.numel()):
                H_before[su_type] += 1
    else:
        H_before = torch.clamp(H.detach().clone(), min=0).long()
    H_work = H_before.detach().clone()
    inventory = _phase_inventory(plan, H_before)
    demand = _phase_demand(plan, inventory)
    preferred_demand = _preferred_phase_demand(plan)
    inventory_before = dict(inventory)
    demand_before = dict(demand)
    groups: List[AtomicAdjustmentGroup] = []
    aromatic_cursor = max(0, int(su11_rotation_index))
    aromatic_cursor_before = int(aromatic_cursor)
    soft_direction_blocked = False
    explicit_ops = _iter_plan_ops(plan)
    explicit_index = 0
    used_gap_selector = False
    rejected_primary_states = set()
    repeated_primary_blocked = False
    target_h = None
    if E_target is not None:
        try:
            target_h = float(E_target.detach().cpu().float()[1].item())
        except Exception:
            target_h = None

    while len(groups) < max(0, int(max_primary_adjustments)):
        primary: Optional[HistogramConversion] = None
        selected_bundle: List[HistogramConversion] = []
        selected_bundle_score: Optional[ConversionBundleScore] = None
        candidate_aromatic_cursor = int(aromatic_cursor)
        hard_gap_present = any(
            int(demand.get(int(su_type), 0)) > int(inventory.get(int(su_type), 0))
            for su_type in RESOURCE_TYPES
        )
        if hard_gap_present:
            used_gap_selector = True
            selected_bundle, candidate_aromatic_cursor, selected_bundle_score = _choose_bundle_from_gap(
                H_work,
                inventory,
                demand,
                aromatic_cursor,
                plan=plan,
                preferred_demand=preferred_demand,
                target_h=target_h,
                special_degree_meta=special_degree_meta,
                max_h_compensation_ops=max_h_compensation_ops,
            )
            primary = selected_bundle[0] if selected_bundle else None
            if primary is None and target_h is not None:
                soft_direction_blocked = True
        if not hard_gap_present and preferred_demand:
            preferred_absolute = {
                int(su): max(int(demand.get(int(su), 0)), int(value))
                for su, value in preferred_demand.items()
            }
            for su_type in RESOURCE_TYPES:
                preferred_absolute.setdefault(int(su_type), int(demand.get(int(su_type), 0)))
            preferred_gap_present = any(
                int(preferred_absolute.get(int(su_type), 0)) > int(inventory.get(int(su_type), 0))
                for su_type in RESOURCE_TYPES
            )
            if preferred_gap_present:
                used_gap_selector = True
                selected_bundle, candidate_aromatic_cursor, selected_bundle_score = _choose_bundle_from_gap(
                    H_work,
                    inventory,
                    preferred_absolute,
                    aromatic_cursor,
                    plan=plan,
                    preferred_demand=preferred_absolute,
                    target_h=target_h,
                    special_degree_meta=special_degree_meta,
                    max_h_compensation_ops=max_h_compensation_ops,
                )
                primary = selected_bundle[0] if selected_bundle else None
                if primary is None and target_h is not None:
                    soft_direction_blocked = True
        if primary is None and not hard_gap_present and not used_gap_selector:
            while explicit_index < len(explicit_ops) and primary is None:
                op = explicit_ops[explicit_index]
                explicit_index += 1
                if op.get("from_su") is None or op.get("to_su") is None:
                    continue
                candidate = HistogramConversion(
                    from_su=int(op["from_su"]),
                    to_su=int(op["to_su"]),
                    count=max(1, int(op.get("count", 1) or 1)),
                    node_ids=[int(op["node_id"])] if op.get("node_id") is not None else [],
                    reason=str(op.get("reason", "stage_planner_conversion")),
                    via_su=int(op["via_su"]) if op.get("via_su") is not None else None,
                )
                if _conversion_signature(candidate) in _forbidden_primary_signatures(plan):
                    continue
                context = _scoring_context(
                    plan,
                    H_work,
                    inventory,
                    demand,
                    preferred_demand,
                    target_h,
                    aromatic_cursor,
                )
                scored = _enumerate_scored_bundles(
                    [candidate],
                    context=context,
                    max_h_compensation_ops=max_h_compensation_ops,
                    special_degree_meta=special_degree_meta,
                )
                selected_score = min(
                    [
                        item for item in scored
                        if item.feasible
                        and _bundle_preserves_primary_effect(item.ops)
                        and _does_not_worsen_resource_gap(demand, inventory, item.inventory_after)
                        and _bundle_uses_only_initial_donors(inventory, item.ops)
                        and _bundle_preserves_donor_reserves(context, item.ops)
                        and _flex_lower_increase_allowed(plan, item)
                    ],
                    key=lambda item: item.sort_key,
                    default=None,
                )
                if selected_score is None:
                    continue
                selected_bundle = list(selected_score.ops)
                selected_bundle_score = selected_score
                primary = selected_bundle[0]
                candidate_aromatic_cursor = int(aromatic_cursor)
                for bundle_op in selected_bundle:
                    candidate_aromatic_cursor = _advance_cursor_for_conversion(
                        int(candidate_aromatic_cursor),
                        bundle_op,
                    )
        if primary is None:
            break
        compensation = list(selected_bundle[1:]) if selected_bundle else []
        primary_state = (
            tuple(int(value) for value in H_work.detach().cpu().long().tolist()),
            tuple(sorted((int(key), int(value)) for key, value in inventory.items())),
            tuple(sorted((int(key), int(value)) for key, value in demand.items())),
            int(primary.from_su),
            int(primary.to_su),
            int(primary.count),
            int(primary.via_su) if primary.via_su is not None else -1,
            tuple(
                (int(op.from_su), int(op.to_su), int(op.count), int(op.via_su) if op.via_su is not None else -1)
                for op in compensation
            ),
        )
        if primary_state in rejected_primary_states:
            repeated_primary_blocked = True
            break
        primary_inventory_after = dict(inventory)
        primary_inventory_after[int(primary.from_su)] = int(primary_inventory_after.get(int(primary.from_su), 0)) - int(primary.count)
        primary_inventory_after[int(primary.to_su)] = int(primary_inventory_after.get(int(primary.to_su), 0)) + int(primary.count)
        if int(inventory.get(int(primary.from_su), 0)) < int(primary.count):
            rejected_primary_states.add(primary_state)
            continue
        if not _does_not_worsen_resource_gap(demand, inventory, primary_inventory_after):
            rejected_primary_states.add(primary_state)
            continue
        primary_delta = _conversion_delta(primary.from_su, primary.to_su, primary.count)
        H_primary = _apply_delta(H_work, primary_delta)
        if H_primary is None:
            rejected_primary_states.add(primary_state)
            continue
        before_gap = {su: int(demand.get(su, 0)) - int(inventory.get(su, 0)) for su in RESOURCE_TYPES}
        h_before = _h_total(H_work, special_degree_meta)
        H_primary_step = H_primary
        if primary.via_su is not None:
            H_primary_step = _apply_delta(
                H_work,
                _conversion_delta(primary.from_su, int(primary.via_su), primary.count),
            )
            if H_primary_step is None:
                rejected_primary_states.add(primary_state)
                continue
        h_after_primary = _h_total(H_primary_step, special_degree_meta)
        all_ops = [primary] + compensation
        combined_delta = _merge_delta(all_ops)
        if not combined_delta:
            rejected_primary_states.add(primary_state)
            continue
        H_final = apply_conversions_to_H(H_work, all_ops)
        inventory_after = apply_conversions_to_inventory(inventory, all_ops)
        if H_final is None or inventory_after is None:
            rejected_primary_states.add(primary_state)
            continue
        after_gap = {su: int(demand.get(su, 0)) - int(inventory_after.get(su, 0)) for su in RESOURCE_TYPES}
        if not _does_not_worsen_resource_gap(demand, inventory, inventory_after):
            rejected_primary_states.add(primary_state)
            continue
        groups.append(AtomicAdjustmentGroup(
            primary_op=primary,
            h_compensation_ops=compensation,
            combined_delta=combined_delta,
            demand_before=before_gap,
            demand_after=after_gap,
            h_before=h_before,
            h_after_primary=h_after_primary,
            h_after_final=_h_total(H_final, special_degree_meta),
            scoring_diagnostics=(
                dict(selected_bundle_score.diagnostics or {})
                if selected_bundle_score is not None
                else {}
            ),
        ))
        aromatic_cursor = int(candidate_aromatic_cursor)
        H_work = H_final
        inventory = inventory_after

    unresolved = {
        int(su): max(0, int(demand.get(su, 0)) - int(inventory.get(su, 0)))
        for su in RESOURCE_TYPES
        if int(demand.get(su, 0)) - int(inventory.get(su, 0)) > 0
    }
    return HistogramAdjustmentPlan(
        phase=str(plan.phase or ""),
        groups=groups,
        primary_adjustment_count=len(groups),
        physical_conversion_count=sum(
            (2 if group.primary_op.via_su is not None else 1)
            + sum(2 if op.via_su is not None else 1 for op in group.h_compensation_ops)
            for group in groups
        ),
        H_before=H_before,
        H_after=H_work,
        demand_before=demand_before,
        demand_after=dict(demand),
        inventory_before=inventory_before,
        inventory_after=inventory,
        protected_resource_counts=_runtime_protected_counts(plan),
        accounting_diagnostics={
            "mode": "authorized_inventory_planning",
            "h_tolerance_deprecated": True,
            "received_h_tolerance": float(h_tolerance),
            "flex_lower_increase_limit": int(
                MAX_FLEX_LOWER_INCREASE_PER_ADJUSTMENT_PLAN
            ),
            "flex_lower_increase_rejections": list(
                dict(getattr(plan, "diagnostics", {}) or {}).get(
                    "flex_lower_increase_rejections",
                    [],
                )
                or []
            ),
        },
        unresolved_counts=unresolved,
        blocked_reason=(
            "resource_conversion_scoring_no_feasible_candidate"
            if not groups and soft_direction_blocked
            else "repeated_primary_candidate_no_progress"
            if not groups and repeated_primary_blocked
            else "" if groups
            else str(plan.blocked_reason or plan.reason or "histogram_adjustment_impossible")
        ),
        su11_rotation_index_before=int(aromatic_cursor_before),
        su11_rotation_index_after=int(aromatic_cursor),
    )


def apply_histogram_adjustment_plan(
    H: torch.Tensor,
    histogram_plan: HistogramAdjustmentPlan,
    nodes: Optional[List[Any]] = None,
) -> Stage3AdjustmentApplyResult:
    H_before = torch.clamp(H.detach().clone(), min=0).long()
    H_out = H_before.detach().clone()
    nodes_out = copy.deepcopy(list(nodes or []))
    nodes_by_id = {
        int(getattr(node, "global_id", -1)): node
        for node in nodes_out
        if int(getattr(node, "global_id", -1)) >= 0
    }
    applied_ops: List[Dict[str, Any]] = []
    rejected_ops: List[Dict[str, Any]] = []
    initial_total = int(H_before.sum().item())
    convertible_inventory = {
        int(key): int(value)
        for key, value in dict(histogram_plan.inventory_before or {}).items()
    }
    phase_demand = {
        int(key): int(value)
        for key, value in dict(histogram_plan.demand_before or {}).items()
    }
    aggregate_mode = bool(
        dict(getattr(histogram_plan, "aggregate_inventory_before", {}) or {})
        or dict(getattr(histogram_plan, "aggregate_inventory_after", {}) or {})
        or dict(getattr(histogram_plan, "aggregate_required_counts", {}) or {})
    )
    accounting_ok, accounting_proof = preflight_histogram_accounting(histogram_plan)
    if not accounting_ok:
        return Stage3AdjustmentApplyResult(
            nodes=copy.deepcopy(list(nodes or [])),
            H=H_before,
            applied_ops=[],
            rejected_ops=[{
                "reason": "aggregate_resource_preflight_failed",
                "accounting": accounting_proof,
            }],
            changed=False,
            conservation_before={"su_count": initial_total},
            conservation_after={"su_count": initial_total},
        )
    for group in list(histogram_plan.groups or []):
        required_donors: Dict[int, int] = {}
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            required_donors[int(op.from_su)] = int(required_donors.get(int(op.from_su), 0)) + int(op.count)
        unavailable = {
            int(su_type): int(count) - int(convertible_inventory.get(int(su_type), 0))
            for su_type, count in required_donors.items()
            if int(count) > int(convertible_inventory.get(int(su_type), 0))
        }
        if unavailable:
            rejected_ops.append({
                "group": group.to_dict(),
                "reason": "donor_not_in_authorized_stage3_pool",
                "unavailable_counts": unavailable,
            })
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=applied_ops,
                rejected_ops=rejected_ops, changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": initial_total},
            )
        candidate_inventory = dict(convertible_inventory)
        for su_type, value in dict(group.combined_delta or {}).items():
            candidate_inventory[int(su_type)] = int(candidate_inventory.get(int(su_type), 0)) + int(value)
        if (
            not aggregate_mode
            and not _does_not_worsen_resource_gap(
                phase_demand,
                convertible_inventory,
                candidate_inventory,
            )
        ):
            rejected_ops.append({
                "group": group.to_dict(),
                "reason": "bundle_would_worsen_phase_resource_deficit",
                "gap_before": _positive_gap(phase_demand, convertible_inventory),
                "gap_after": _positive_gap(phase_demand, candidate_inventory),
            })
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                rejected_ops=rejected_ops, changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": initial_total},
            )
        candidate = _apply_delta(H_out, dict(group.combined_delta or {}))
        if candidate is None or int(candidate.sum().item()) != initial_total:
            rejected_ops.append({"group": group.to_dict(), "reason": "histogram_conservation_failed"})
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=applied_ops,
                rejected_ops=rejected_ops, changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": int(H_out.sum().item())},
            )
        H_out = candidate
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            node_ids = [int(value) for value in list(getattr(op, "node_ids", []) or [])]
            if len(node_ids) != int(op.count):
                rejected_ops.append({
                    "group": group.to_dict(),
                    "reason": "authorized_real_donor_ids_missing",
                    "from_su": int(op.from_su),
                    "required": int(op.count),
                    "provided_node_ids": list(node_ids),
                })
                return Stage3AdjustmentApplyResult(
                    nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                    rejected_ops=rejected_ops, changed=False,
                    conservation_before={"su_count": initial_total},
                    conservation_after={"su_count": initial_total},
                )
            for node_id in node_ids:
                node = nodes_by_id.get(int(node_id))
                if node is None or int(getattr(node, "su_type", -1)) != int(op.from_su):
                    rejected_ops.append({
                        "group": group.to_dict(),
                        "reason": "real_donor_node_type_mismatch",
                        "node_id": int(node_id),
                        "from_su": int(op.from_su),
                    })
                    return Stage3AdjustmentApplyResult(
                        nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                        rejected_ops=rejected_ops, changed=False,
                        conservation_before={"su_count": initial_total},
                        conservation_after={"su_count": initial_total},
                    )
                history = list(getattr(node, "stage3_conversion_history", []) or [])
                if op.via_su is not None:
                    history.extend([
                        {
                            "from_su": int(op.from_su),
                            "to_su": int(op.via_su),
                            "reason": str(op.reason),
                        },
                        {
                            "from_su": int(op.via_su),
                            "to_su": int(op.to_su),
                            "reason": str(op.reason),
                        },
                    ])
                else:
                    history.append({
                        "from_su": int(op.from_su),
                        "to_su": int(op.to_su),
                        "reason": str(op.reason),
                    })
                setattr(node, "su_type", int(op.to_su))
                setattr(node, "stage3_conversion_history", history)
        convertible_inventory = candidate_inventory
        applied_ops.append(group.to_dict())
    planned_h_after = getattr(histogram_plan, "H_after", None)
    if planned_h_after is not None:
        expected_h = torch.clamp(planned_h_after.detach().cpu(), min=0).long()
        actual_h = torch.clamp(H_out.detach().cpu(), min=0).long()
        if not torch.equal(expected_h, actual_h):
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                rejected_ops=[{
                    "reason": "applied_histogram_differs_from_plan",
                    "expected_h": expected_h.tolist(),
                    "actual_h": actual_h.tolist(),
                }],
                changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": initial_total},
            )
    node_histogram = torch.zeros_like(H_out)
    for node in nodes_out:
        su_type = int(getattr(node, "su_type", -1))
        if 0 <= su_type < int(node_histogram.numel()):
            node_histogram[su_type] += 1
    if nodes_out and not torch.equal(node_histogram, H_out):
        return Stage3AdjustmentApplyResult(
            nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
            rejected_ops=[{"reason": "real_node_histogram_mismatch"}], changed=False,
            conservation_before={"su_count": initial_total},
            conservation_after={"su_count": int(node_histogram.sum().item())},
        )
    return Stage3AdjustmentApplyResult(
        nodes=nodes_out,
        H=H_out,
        applied_ops=applied_ops,
        rejected_ops=rejected_ops,
        changed=bool(not torch.equal(H_before, H_out)),
        requested_counts={str(k): int(v) for k, v in histogram_plan.demand_before.items()},
        unresolved_counts={str(k): int(v) for k, v in histogram_plan.unresolved_counts.items()},
        conservation_before={"su_count": initial_total},
        conservation_after={"su_count": int(H_out.sum().item())},
    )
