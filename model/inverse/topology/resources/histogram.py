from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ....shared.coarse_graph import E_SU
from ....shared.inverse_common import get_effective_hist_element_vector
from ....shared.aromatic_plan import su12_transition_diagnostics
from ..stage3_models import (
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
from .conversion_paths import (
    ConversionPathError,
    audit_conversion_paths,
    bind_conversion_paths,
)
from .profiles import (
    DEFAULT_STAGE3_H_SOFT_TOLERANCE,
    MAX_FLEX_LOWER_INCREASE_PER_ADJUSTMENT_PLAN,
    MIN_STAGE3_AROMATIC_RESOURCE_COUNT,
    SU11_AROMATIC_ROTATION,
)
from .accounting import (
    positive_resource_gap,
    preflight_histogram_accounting,
    resource_gap_does_not_worsen,
)
from .types import (
    EXTERNAL_ADJUSTMENT_RESOURCE_TYPES,
    GENERIC_CLOSURE_RESOURCE_TYPES,
    SU24_RING_RESOURCE_TYPE,
)


ADJUSTABLE_CLOSURE_TARGET_TYPES = GENERIC_CLOSURE_RESOURCE_TYPES
RESIDUAL_CLOSURE_TYPES = GENERIC_CLOSURE_RESOURCE_TYPES
TARGET_PRIORITY = {23: 0, 11: 1, 22: 2}
AROMATIC_PATTERN = (12, 13)


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


def _phase_inventory(plan: Stage3AdjustmentPlan) -> Dict[int, int]:
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
    for su_type in EXTERNAL_ADJUSTMENT_RESOURCE_TYPES:
        inventory.setdefault(int(su_type), 0)
    donor_counts = dict(ledger.get("su11_donor_counts", {}) or {})
    for su_type in (12, 13):
        inventory[int(su_type)] = int(
            donor_counts.get(
                str(su_type),
                donor_counts.get(int(su_type), inventory[int(su_type)]),
            )
            or 0
        )
    # SU24 is never an ordinary histogram donor.  A released SU24 has already
    # been reclassified to SU23 by pure-ring recovery and is included above
    # through the ordinary SU23 bucket.
    inventory.pop(SU24_RING_RESOURCE_TYPE, None)
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
    for su_type in ADJUSTABLE_CLOSURE_TARGET_TYPES:
        demand.setdefault(int(su_type), 0)
    return demand


def _preferred_phase_demand(plan: Stage3AdjustmentPlan) -> Dict[int, int]:
    return {
        int(su): max(0, int(value))
        for su, value in dict(plan.preferred_required_counts or {}).items()
        if str(su).lstrip("-").isdigit()
    }


def _required_residual_flex(plan: Stage3AdjustmentPlan) -> bool:
    return bool(
        str(plan.phase) == "residual_resource_self_closure"
        and any(
            contract.get("target_role") == "flex_chain"
            and dict(contract.get("diagnostics", {}) or {}).get("required_flex", False)
            for contract in plan.requirement_contracts
        )
    )


def _residual_closure_only(plan: Stage3AdjustmentPlan) -> bool:
    return bool(
        str(plan.phase) == "residual_resource_self_closure"
        and not _required_residual_flex(plan)
    )


def _runtime_flex_audit(plan: Stage3AdjustmentPlan) -> Dict[str, Any]:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    audit: Dict[str, Any] = {}
    if phase_eval is not None:
        audit = dict(getattr(phase_eval, "flex_audit", {}) or {})
    if not audit:
        audit = dict(runtime.get("flex_audit", {}) or {})
    if not audit:
        return {}

    flex_counts = dict(getattr(phase_eval, "flex_counts", {}) or {})
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
        (str(getattr(plan, "phase", "") or "") == "external_flex_closure"
         and "flex_chain" in active_roles)
        or _required_residual_flex(plan)
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


def _bundle_has_feasible_aromatic_staging(
    score: ConversionBundleScore,
) -> bool:
    """Allow an odd SU12 prefix that the rebuild-boundary balancer can repair.

    Physical conversions can continue within one external generation before
    port balance runs. Rejecting these prefixes here would prevent that
    mandatory balance step from ever seeing them. Other generator failures
    remain infeasible, and this check never authorizes a final graph export.
    """
    flex_after = dict(getattr(score.flex_impact, "after", {}) or {})
    return bool(
        flex_after.get("aromatic_staging_valid", True)
        or flex_after.get("aromatic_staging_projected", False)
    )


def _scoring_context(
    plan: Stage3AdjustmentPlan,
    H: torch.Tensor,
    inventory: Dict[int, int],
    demand: Dict[int, int],
    preferred_demand: Optional[Dict[int, int]],
    target_h: Optional[float],
    aromatic_cursor: int,
    h_tolerance: float,
) -> ConversionContext:
    allowed_types = tuple(EXTERNAL_ADJUSTMENT_RESOURCE_TYPES)
    return ConversionContext(
        inventory={int(key): int(value) for key, value in dict(inventory or {}).items()},
        demand={int(key): int(value) for key, value in dict(demand or {}).items()},
        current_H=H.detach().clone().long(),
        target_h=target_h,
        h_tolerance=float(h_tolerance),
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
        allowed_conversion_types=allowed_types,
    )


def _surplus(
    inventory: Dict[int, int],
    demand: Dict[int, int],
    su_type: int,
) -> int:
    reserved = (
        int(demand.get(int(su_type), 0))
        if int(su_type) in ADJUSTABLE_CLOSURE_TARGET_TYPES
        else 0
    )
    return int(inventory.get(int(su_type), 0)) - int(reserved)


def _direct_gap_candidates(
    target: int,
    inventory: Dict[int, int],
    demand: Dict[int, int],
    *,
    residual_closure_only: bool = False,
) -> List[HistogramConversion]:
    candidates: List[HistogramConversion] = []
    donor_types = list(
        RESIDUAL_CLOSURE_TYPES
        if residual_closure_only
        else ADJUSTABLE_CLOSURE_TARGET_TYPES
    )
    if int(target) == 11 and not residual_closure_only:
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


def _conversion_bundle_signature(
    conversions: Iterable[HistogramConversion],
) -> str:
    ops = list(conversions or [])
    if not ops:
        return ""

    def op_signature(op: HistogramConversion) -> str:
        count = max(1, int(op.count))
        suffix = f"x{count}" if count != 1 else ""
        return f"{_conversion_signature(op)}{suffix}"

    primary = op_signature(ops[0])
    compensation = sorted(op_signature(op) for op in ops[1:])
    if compensation:
        primary = f"{primary}|{','.join(compensation)}"
    return f"bundle:{primary}"


def _stateful_conversion_bundle_signature(
    H: torch.Tensor,
    conversions: Iterable[HistogramConversion],
) -> str:
    bundle = _conversion_bundle_signature(conversions)
    if not bundle:
        return ""
    state = ",".join(
        str(int(value))
        for value in torch.clamp(H.detach().cpu(), min=0).long().tolist()
    )
    return f"state:{state}|{bundle}"


def _bundle_is_forbidden(
    H: torch.Tensor,
    conversions: Iterable[HistogramConversion],
    forbidden: Iterable[str],
) -> bool:
    blocked = {str(value) for value in list(forbidden or []) if str(value)}
    ops = list(conversions or [])
    return bool(
        _conversion_bundle_signature(ops) in blocked
        or _stateful_conversion_bundle_signature(H, ops) in blocked
    )


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


def _forbidden_conversion_su_types(plan: Stage3AdjustmentPlan) -> set[int]:
    return {
        int(value)
        for value in list(
            dict(getattr(plan, "diagnostics", {}) or {}).get(
                "forbidden_conversion_su_types", []
            )
            or []
        )
        if str(value).lstrip("-").isdigit()
    }


def _conversion_uses_forbidden_su(
    op: HistogramConversion,
    forbidden_su_types: Iterable[int],
    allowed_conversion_types: Optional[Iterable[int]] = None,
) -> bool:
    forbidden = {
        int(value) for value in list(forbidden_su_types or [])
    }
    allowed = set(EXTERNAL_ADJUSTMENT_RESOURCE_TYPES)
    if allowed_conversion_types is not None:
        allowed = {int(value) for value in list(allowed_conversion_types or [])}
    if SU24_RING_RESOURCE_TYPE not in allowed:
        forbidden.add(SU24_RING_RESOURCE_TYPE)
    forbidden.update(
        int(value) for value in list(forbidden_su_types or [])
    )
    used_types = {int(op.from_su), int(op.to_su)}
    if op.via_su is not None:
        used_types.add(int(op.via_su))
    return bool(
        used_types & forbidden
        or not used_types.issubset(allowed)
    )


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


def _aromatic_rotation_offset(
    aromatic_cursor: int,
    su_type: int,
    rotation: Tuple[int, ...] = SU11_AROMATIC_ROTATION,
) -> Optional[int]:
    for offset in range(len(rotation)):
        position = (int(aromatic_cursor) + int(offset)) % len(rotation)
        if int(rotation[position]) == int(su_type):
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
    *,
    residual_closure_only: bool = False,
) -> List[HistogramConversion]:
    candidates: List[HistogramConversion] = []
    donor_types = (
        list(RESIDUAL_CLOSURE_TYPES)
        if residual_closure_only
        else list(ADJUSTABLE_CLOSURE_TARGET_TYPES) + [12, 13]
    )
    safe_nondeficit_pairs = {
        (22, 23),
        (23, 22),
        (11, 23),
        (23, 11),
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
        targets = list(
            RESIDUAL_CLOSURE_TYPES
            if residual_closure_only
            else ADJUSTABLE_CLOSURE_TARGET_TYPES
        )
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
                reason="stage3_h_soft_compensation",
                via_su=11 if int(source) in AROMATIC_PATTERN and int(target) != 11 else None,
            ))
    return candidates


def _h_ratio(value: float, target_h: float) -> float:
    return (float(value) - float(target_h)) / float(target_h)


def _stage3_h_soft_tolerance(value: float) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError):
        tolerance = float(DEFAULT_STAGE3_H_SOFT_TOLERANCE)
    return min(1.0, max(0.0, float(tolerance)))


def _h_compensation_outside_distance(
    value: float,
    target_h: float,
    tolerance: float,
) -> float:
    ratio = _h_ratio(value, target_h)
    return max(0.0, abs(float(ratio)) - float(tolerance))


def _compensation_improves_h_state(
    *,
    primary_score: ConversionBundleScore,
    bundle_score: ConversionBundleScore,
    target_h: Optional[float],
    h_tolerance: float = DEFAULT_STAGE3_H_SOFT_TOLERANCE,
) -> bool:
    if target_h is None or float(target_h) <= 0:
        return False
    if primary_score.h_before is None or primary_score.h_after is None or bundle_score.h_after is None:
        return False
    h_primary = float(primary_score.h_after)
    h_bundle = float(bundle_score.h_after)
    tolerance = _stage3_h_soft_tolerance(h_tolerance)
    primary_ratio = _h_ratio(h_primary, float(target_h))
    bundle_ratio = _h_ratio(h_bundle, float(target_h))

    if abs(float(primary_ratio)) <= float(tolerance) + 1e-12:
        return False
    if float(primary_ratio) < -float(tolerance):
        if float(bundle_ratio) <= float(primary_ratio) + 1e-12:
            return False
        if float(bundle_ratio) > float(tolerance) + 1e-12:
            return False
    else:
        if float(bundle_ratio) >= float(primary_ratio) - 1e-12:
            return False
        if float(bundle_ratio) < -float(tolerance) - 1e-12:
            return False

    primary_outside = _h_compensation_outside_distance(
        h_primary,
        float(target_h),
        tolerance,
    )
    bundle_outside = _h_compensation_outside_distance(
        h_bundle,
        float(target_h),
        tolerance,
    )
    return bool(float(bundle_outside) + 1e-12 < float(primary_outside))


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
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    residual_closure_only: bool,
    forbidden_su_types: Optional[Iterable[int]] = None,
) -> List[ConversionBundleScore]:
    forbidden = {int(value) for value in list(forbidden_su_types or [])}
    bundles: List[List[HistogramConversion]] = [
        [candidate]
        for candidate in list(primary_candidates or [])
        if not _conversion_uses_forbidden_su(
            candidate,
            forbidden,
            allowed_conversion_types=context.allowed_conversion_types,
        )
    ]
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
                h_tolerance=float(context.h_tolerance),
                preferred_demand=context.preferred_demand,
                flex_audit=context.flex_audit,
                protected_resource_counts=context.protected_resource_counts,
                su11_rotation_index=context.su11_rotation_index,
                phase=context.phase,
                reason=context.reason,
                allowed_conversion_types=context.allowed_conversion_types,
            )
            ranked_compensation = rank_conversions(
                bundle_context,
                [
                    candidate
                    for candidate in _h_compensation_candidates(
                        inventory_after,
                        context.demand,
                        protected_counts=context.protected_resource_counts,
                        residual_closure_only=residual_closure_only,
                    )
                    if not _conversion_uses_forbidden_su(
                        candidate,
                        forbidden,
                        allowed_conversion_types=bundle_context.allowed_conversion_types,
                    )
                ],
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
                    h_tolerance=h_tolerance,
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
    h_tolerance: float,
) -> Tuple[List[HistogramConversion], int, Optional[ConversionBundleScore]]:
    gap = {
        su: int(demand.get(su, 0)) - int(inventory.get(su, 0))
        for su in ADJUSTABLE_CLOSURE_TARGET_TYPES
    }
    targets = [
        su
        for su in ADJUSTABLE_CLOSURE_TARGET_TYPES
        if int(gap.get(su, 0)) > 0
    ]
    if not targets:
        return [], int(aromatic_cursor), None
    target = sorted(targets, key=lambda su: (-int(gap[su]), TARGET_PRIORITY[int(su)]))[0]
    residual_closure_only = _residual_closure_only(plan)
    direct_candidates = _direct_gap_candidates(
        int(target),
        inventory,
        demand,
        residual_closure_only=residual_closure_only,
    )
    aromatic_candidates = (
        []
        if residual_closure_only
        else _atomic_aromatic_bridge_candidates(int(target), inventory, demand)
    )
    forbidden = _forbidden_primary_signatures(plan)
    forbidden_su_types = _forbidden_conversion_su_types(plan)
    context = _scoring_context(
        plan,
        H,
        inventory,
        demand,
        preferred_demand,
        target_h,
        aromatic_cursor,
        h_tolerance,
    )
    def _select_from(candidates: List[HistogramConversion]):
        filtered = [
            candidate
            for candidate in candidates
            if _conversion_signature(candidate) not in forbidden
            and not _conversion_uses_forbidden_su(
                candidate,
                forbidden_su_types,
                allowed_conversion_types=context.allowed_conversion_types,
            )
        ]
        ranked_primary = rank_conversions(
            context,
            filtered,
            h_total_fn=_h_total,
            special_degree_meta=special_degree_meta,
        )
        primary_candidates = [
            impact.op for impact in ranked_primary if impact.feasible
        ]
        if not primary_candidates:
            return []
        scored_bundles = _enumerate_scored_bundles(
            primary_candidates,
            context=context,
            max_h_compensation_ops=max_h_compensation_ops,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            residual_closure_only=residual_closure_only,
            forbidden_su_types=forbidden_su_types,
        )
        return [
            score for score in scored_bundles
            if score.feasible
            and _bundle_has_feasible_aromatic_staging(score)
            and not _bundle_is_forbidden(H, score.ops, forbidden)
            and _bundle_preserves_primary_effect(score.ops)
            and _gap_total(score.resource_gap_after) < _gap_total(score.resource_gap_before)
            and _does_not_worsen_resource_gap(
                context.demand, context.inventory, score.inventory_after
            )
            and _bundle_uses_only_initial_donors(context.inventory, score.ops)
            and _bundle_preserves_donor_reserves(context, score.ops)
            and _flex_lower_increase_allowed(plan, score)
        ]

    # Aromatic donors supplement required topology only after ordinary
    # closure conversions fail. This includes a mandatory residual flex
    # chain needed to reach the lower bound.
    feasible = _select_from(direct_candidates)
    if not feasible and aromatic_candidates:
        feasible = _select_from(direct_candidates + aromatic_candidates)
    if not feasible:
        return [], int(aromatic_cursor), None
    selected = min(feasible, key=lambda item: item.sort_key)
    cursor = int(aromatic_cursor)
    for op in selected.ops:
        cursor = _advance_cursor_for_conversion(cursor, op)
    return list(selected.ops), int(cursor), selected


def _positive_gap(demand: Dict[int, int], inventory: Dict[int, int]) -> Dict[int, int]:
    return positive_resource_gap(
        demand,
        inventory,
        resource_types=ADJUSTABLE_CLOSURE_TARGET_TYPES,
    )


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
        resource_types=ADJUSTABLE_CLOSURE_TARGET_TYPES,
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
        if int(su_type) in {12, 13}:
            physical_available = max(
                0,
                int(context.current_H[int(su_type)].item())
                - int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT),
            )
            if int(count) > int(physical_available):
                return False
    return True


def build_histogram_adjustment_plan(
    plan: Stage3AdjustmentPlan,
    H: torch.Tensor,
    E_target: Any = None,
    max_primary_adjustments: int = 10,
    max_h_compensation_ops: int = 8,
    h_tolerance: float = DEFAULT_STAGE3_H_SOFT_TOLERANCE,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    su11_rotation_index: int = 0,
    node_pool: Optional[Dict[int, List[int]]] = None,
    protected_node_ids: Optional[List[int]] = None,
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
    inventory = _phase_inventory(plan)
    demand = _phase_demand(plan, inventory)
    preferred_demand = _preferred_phase_demand(plan)
    inventory_before = dict(inventory)
    demand_before = dict(demand)
    groups: List[AtomicAdjustmentGroup] = []
    concrete_pool = copy.deepcopy(node_pool)
    primary_node_ids: set[int] = set()
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

    def _primary_units() -> int:
        if concrete_pool is not None:
            return len(primary_node_ids)
        return sum(
            len(list(getattr(group.primary_op, "node_ids", []) or []))
            or max(0, int(getattr(group.primary_op, "count", 0) or 0))
            for group in groups
        )

    while int(max_primary_adjustments) > 0:
        primary: Optional[HistogramConversion] = None
        selected_bundle: List[HistogramConversion] = []
        selected_bundle_score: Optional[ConversionBundleScore] = None
        candidate_aromatic_cursor = int(aromatic_cursor)
        hard_gap_present = any(
            int(demand.get(int(su_type), 0)) > int(inventory.get(int(su_type), 0))
            for su_type in ADJUSTABLE_CLOSURE_TARGET_TYPES
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
                h_tolerance=h_tolerance,
            )
            primary = selected_bundle[0] if selected_bundle else None
            if primary is None and target_h is not None:
                soft_direction_blocked = True
        if not hard_gap_present and preferred_demand:
            preferred_absolute = {
                int(su): max(int(demand.get(int(su), 0)), int(value))
                for su, value in preferred_demand.items()
            }
            for su_type in ADJUSTABLE_CLOSURE_TARGET_TYPES:
                preferred_absolute.setdefault(int(su_type), int(demand.get(int(su_type), 0)))
            preferred_gap_present = any(
                int(preferred_absolute.get(int(su_type), 0)) > int(inventory.get(int(su_type), 0))
                for su_type in ADJUSTABLE_CLOSURE_TARGET_TYPES
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
                    h_tolerance=h_tolerance,
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
                if _conversion_uses_forbidden_su(
                    candidate,
                    _forbidden_conversion_su_types(plan),
                    allowed_conversion_types=_scoring_context(
                        plan,
                        H_work,
                        inventory,
                        demand,
                        preferred_demand,
                        target_h,
                        aromatic_cursor,
                        h_tolerance,
                    ).allowed_conversion_types,
                ):
                    continue
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
                    h_tolerance,
                )
                scored = _enumerate_scored_bundles(
                    [candidate],
                    context=context,
                    max_h_compensation_ops=max_h_compensation_ops,
                    h_tolerance=h_tolerance,
                    special_degree_meta=special_degree_meta,
                    residual_closure_only=_residual_closure_only(plan),
                    forbidden_su_types=_forbidden_conversion_su_types(plan),
                )
                selected_score = min(
                    [
                        item for item in scored
                        if item.feasible
                        and _bundle_has_feasible_aromatic_staging(item)
                        and not _bundle_is_forbidden(
                            H_work,
                            item.ops,
                            _forbidden_primary_signatures(plan),
                        )
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
        before_gap = {
            su: int(demand.get(su, 0)) - int(inventory.get(su, 0))
            for su in ADJUSTABLE_CLOSURE_TARGET_TYPES
        }
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
        after_gap = {
            su: int(demand.get(su, 0)) - int(inventory_after.get(su, 0))
            for su in ADJUSTABLE_CLOSURE_TARGET_TYPES
        }
        if not _does_not_worsen_resource_gap(demand, inventory, inventory_after):
            rejected_primary_states.add(primary_state)
            continue
        candidate_primary_units = len(
            list(getattr(primary, "node_ids", []) or [])
        ) or max(0, int(getattr(primary, "count", 0) or 0))
        candidate_group = AtomicAdjustmentGroup(
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
        )
        if concrete_pool is not None:
            try:
                candidate_pool, path_proof = bind_conversion_paths(
                    [candidate_group], concrete_pool,
                    protected_node_ids=protected_node_ids,
                )
            except ConversionPathError:
                rejected_primary_states.add(primary_state)
                continue
            candidate_ids = primary_node_ids | set(path_proof["primary_node_ids"])
            if len(candidate_ids) > max(0, int(max_primary_adjustments)):
                break
            concrete_pool = candidate_pool
            primary_node_ids = candidate_ids
        elif _primary_units() + int(candidate_primary_units) > max(0, int(max_primary_adjustments)):
            break
        groups.append(candidate_group)
        aromatic_cursor = int(candidate_aromatic_cursor)
        H_work = H_final
        inventory = inventory_after

    unresolved = {
        int(su): max(0, int(demand.get(su, 0)) - int(inventory.get(su, 0)))
        for su in ADJUSTABLE_CLOSURE_TARGET_TYPES
        if int(demand.get(su, 0)) - int(inventory.get(su, 0)) > 0
    }
    return HistogramAdjustmentPlan(
        phase=str(plan.phase or ""),
        groups=groups,
        # External budgets are charged per concrete primary node, not per
        # histogram group.  A group containing six node IDs therefore costs
        # six units; H-compensation/port-balance operations stay out of this
        # count.
        primary_adjustment_count=_primary_units(),
        physical_conversion_count=sum(
            int(group.primary_op.count) * (2 if group.primary_op.via_su is not None else 1)
            + sum(int(op.count) * (2 if op.via_su is not None else 1) for op in group.h_compensation_ops)
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
            "h_tolerance_mode": "stage3_symmetric_soft_window",
            "effective_h_tolerance": float(_stage3_h_soft_tolerance(h_tolerance)),
            "stage3_soft_lower_ratio": -float(_stage3_h_soft_tolerance(h_tolerance)),
            "stage3_soft_upper_ratio": float(_stage3_h_soft_tolerance(h_tolerance)),
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
    path_groups = list(histogram_plan.groups or [])
    path_proof: Dict[str, Any] = {"node_paths": {}}
    if nodes_by_id or any(
        list(getattr(operation, "node_ids", []) or [])
        for group in path_groups
        for operation in [group.primary_op] + list(group.h_compensation_ops or [])
    ):
        initial_ids_by_su: Dict[int, List[int]] = {}
        for node_id, node in nodes_by_id.items():
            initial_ids_by_su.setdefault(int(node.su_type), []).append(node_id)
        path_proof = audit_conversion_paths(
            path_groups, initial_ids_by_su=initial_ids_by_su if nodes_by_id else None,
        )
        if not path_proof["ok"]:
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before,
                applied_ops=[], rejected_ops=[{
                    "reason": "invalid_conversion_path", "proof": path_proof,
                }], changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": initial_total},
            )
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
    planned_h_after = getattr(histogram_plan, "H_after", None)
    if (
        list(histogram_plan.groups or [])
        and isinstance(planned_h_after, torch.Tensor)
        and torch.equal(
            torch.clamp(planned_h_after.detach().cpu(), min=0).long(),
            H_before.detach().cpu(),
        )
    ):
        return Stage3AdjustmentApplyResult(
            nodes=copy.deepcopy(list(nodes or [])),
            H=H_before,
            applied_ops=[],
            rejected_ops=[{"reason": "histogram_plan_has_no_net_effect"}],
            changed=False,
            conservation_before={"su_count": initial_total},
            conservation_after={"su_count": initial_total},
        )
    used_conversion_node_ids = {int(node_id) for node_id in path_proof["node_paths"]}
    # Port-balance operations are part of the same atomic Stage3 transaction.
    # Do not reject a primary group merely because its intermediate histogram
    # is over the SU12 cap; the final cap check below evaluates the complete
    # group + balance batch.
    has_planned_balance = bool(
        getattr(histogram_plan, "parity_balance_op", None) is not None
        or getattr(histogram_plan, "port_role_balance_op", None) is not None
    )
    planned_balance_count = sum(
        getattr(histogram_plan, field, None) is not None
        for field in ("parity_balance_op", "port_role_balance_op")
    )
    used_balance_node_ids: set[int] = set()
    for balance_field in ("parity_balance_op", "port_role_balance_op"):
        balance_op = getattr(histogram_plan, balance_field, None)
        if balance_op is None:
            continue
        for node_id in [int(value) for value in list(balance_op.node_ids or [])]:
            if node_id in used_conversion_node_ids or node_id in used_balance_node_ids:
                return Stage3AdjustmentApplyResult(
                    nodes=copy.deepcopy(list(nodes or [])),
                    H=H_before,
                    applied_ops=[],
                    rejected_ops=[{
                        "reason": "duplicate_real_conversion_node_id",
                        "node_id": int(node_id),
                        "operation": str(balance_field),
                    }],
                    changed=False,
                    conservation_before={"su_count": initial_total},
                    conservation_after={"su_count": initial_total},
                )
            used_balance_node_ids.add(int(node_id))
    for group in list(histogram_plan.groups or []):
        required_donors: Dict[int, int] = {}
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            required_donors[int(op.from_su)] = int(required_donors.get(int(op.from_su), 0)) + int(op.count)
        unavailable = {
            int(su_type): int(count) - int(convertible_inventory.get(int(su_type), 0))
            for su_type, count in required_donors.items()
            if int(count) > int(convertible_inventory.get(int(su_type), 0))
        }
        for su_type in (12, 13):
            required = int(required_donors.get(int(su_type), 0))
            available = max(
                0,
                int(H_out[int(su_type)].item())
                - int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT),
            )
            if required > available:
                unavailable[int(su_type)] = max(
                    int(unavailable.get(int(su_type), 0)),
                    int(required - available),
                )
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
        su12_cap = su12_transition_diagnostics(H_out, candidate)
        if (
            not has_planned_balance
            and not bool(su12_cap.get("generation_allowed", True))
        ):
            rejected_ops.append({
                "group": group.to_dict(),
                "reason": "su12_generation_cap_reached",
                "su12_cap": dict(su12_cap),
            })
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
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
                setattr(node, "su_type", int(op.to_su))
        convertible_inventory = candidate_inventory
        applied_ops.append(group.to_dict())

    for balance_field in ("parity_balance_op", "port_role_balance_op"):
        balance_op = getattr(histogram_plan, balance_field, None)
        if balance_op is None:
            continue
        reason_prefix = (
            "parity_balance"
            if balance_field == "parity_balance_op"
            else "port_role_balance"
        )
        source = int(balance_op.from_su)
        target = int(balance_op.to_su)
        count = max(1, int(balance_op.count))
        node_ids = [int(value) for value in list(balance_op.node_ids or [])]
        # SU12/SU13 are protected for ordinary resource conversions, but the
        # explicit port-balance operations are allowed to consume that reserve
        # temporarily. A later Stage3 structural-unit adjustment restores the
        # aromatic staging pool, so this lower-bound check must not reject the
        # balance transaction itself.
        if (
            source in {12, 13}
            and count
            > max(
                0,
                int(H_out[source].item())
                - int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT),
            )
            and balance_field not in {"parity_balance_op", "port_role_balance_op"}
        ):
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                rejected_ops=[{
                    "reason": f"{reason_prefix}_aromatic_minimum_preserved",
                    "from_su": source,
                    "minimum": int(MIN_STAGE3_AROMATIC_RESOURCE_COUNT),
                    "available": int(H_out[source].item()),
                }],
                changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": initial_total},
            )
        if int(convertible_inventory.get(source, 0)) < count:
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                rejected_ops=[{
                    "reason": f"{reason_prefix}_donor_not_in_authorized_stage3_pool",
                    "from_su": source,
                    "required": count,
                    "available": int(convertible_inventory.get(source, 0)),
                }],
                changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": initial_total},
            )
        if len(node_ids) != count:
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                rejected_ops=[{
                    "reason": f"{reason_prefix}_real_donor_ids_missing",
                    "from_su": source,
                    "required": count,
                    "provided_node_ids": node_ids,
                }],
                changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": initial_total},
            )
        candidate = _apply_delta(H_out, _conversion_delta(source, target, count))
        if candidate is None or int(candidate.sum().item()) != initial_total:
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                rejected_ops=[{
                    "reason": f"{reason_prefix}_histogram_conservation_failed"
                }],
                changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": initial_total},
            )
        su12_cap = su12_transition_diagnostics(H_out, candidate)
        if (
            planned_balance_count <= 1
            and not bool(su12_cap.get("generation_allowed", True))
        ):
            return Stage3AdjustmentApplyResult(
                nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                rejected_ops=[{
                    "reason": f"{reason_prefix}_su12_generation_cap_reached",
                    "su12_cap": dict(su12_cap),
                }],
                changed=False,
                conservation_before={"su_count": initial_total},
                conservation_after={"su_count": int(H_out.sum().item())},
            )
        for node_id in node_ids:
            node = nodes_by_id.get(int(node_id))
            if node is None or int(getattr(node, "su_type", -1)) != source:
                return Stage3AdjustmentApplyResult(
                    nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
                    rejected_ops=[{
                        "reason": f"{reason_prefix}_real_donor_node_type_mismatch",
                        "node_id": int(node_id),
                        "from_su": source,
                    }],
                    changed=False,
                    conservation_before={"su_count": initial_total},
                    conservation_after={"su_count": initial_total},
                )
            setattr(node, "su_type", target)
        H_out = candidate
        convertible_inventory[source] = int(convertible_inventory.get(source, 0)) - count
        convertible_inventory[target] = int(convertible_inventory.get(target, 0)) + count
        applied_ops.append({balance_field: balance_op.to_dict()})

    final_su12_cap = su12_transition_diagnostics(H_before, H_out)
    if not bool(final_su12_cap.get("generation_allowed", True)):
        return Stage3AdjustmentApplyResult(
            nodes=copy.deepcopy(list(nodes or [])), H=H_before, applied_ops=[],
            rejected_ops=[{
                "reason": "su12_generation_cap_reached",
                "su12_cap": dict(final_su12_cap),
            }],
            changed=False,
            conservation_before={"su_count": initial_total},
            conservation_after={"su_count": initial_total},
        )
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
