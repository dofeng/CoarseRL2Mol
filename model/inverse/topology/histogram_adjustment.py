from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ...shared.coarse_graph import E_SU
from ...shared.inverse_common import get_effective_hist_element_vector
from .conversion_solver import is_request_only_stage3_op
from .stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    HistogramConversion,
    Stage3AdjustmentApplyResult,
    Stage3AdjustmentPlan,
)


RESOURCE_TYPES = (11, 23, 22, 24)
TARGET_PRIORITY = {23: 0, 11: 1, 22: 2, 24: 3}
AROMATIC_PATTERN = (12, 13)
SU11_AROMATIC_ROTATION = (13, 13, 12)
H_SOFT_LOWER_RATIO = -0.02
H_SOFT_UPPER_RATIO = 0.03


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
            if not is_request_only_stage3_op(op):
                ops.append(op)
    if not ops:
        for raw in list(plan.planned_conversions or []):
            op = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw or {})
            if not is_request_only_stage3_op(op):
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


def _h_mode(
    current_h: Optional[float],
    target_h: Optional[float],
    *,
    soft_lower_ratio: float = H_SOFT_LOWER_RATIO,
    soft_upper_ratio: float = H_SOFT_UPPER_RATIO,
) -> str:
    if current_h is None or target_h is None or float(target_h) <= 0:
        return "unconstrained"
    ratio = (float(current_h) - float(target_h)) / float(target_h)
    if float(ratio) < float(soft_lower_ratio) - 1e-9:
        return "raise_h"
    if float(ratio) > float(soft_upper_ratio) + 1e-9:
        return "reduce_h"
    return "balanced"


def _conversion_h_delta(conversion: HistogramConversion) -> float:
    return float(conversion.count) * (
        float(E_SU[int(conversion.to_su), 1].item())
        - float(E_SU[int(conversion.from_su), 1].item())
    )


def _soft_h_direction_allows(
    conversion: HistogramConversion,
    *,
    current_h: Optional[float],
    target_h: Optional[float],
    soft_lower_ratio: float = H_SOFT_LOWER_RATIO,
    soft_upper_ratio: float = H_SOFT_UPPER_RATIO,
) -> bool:
    mode = _h_mode(
        current_h,
        target_h,
        soft_lower_ratio=soft_lower_ratio,
        soft_upper_ratio=soft_upper_ratio,
    )
    if mode == "unconstrained":
        return True
    h_delta = _conversion_h_delta(conversion)
    candidate_h = float(current_h) + float(h_delta)
    current_error = abs(float(current_h) - float(target_h))
    candidate_error = abs(float(candidate_h) - float(target_h))
    if mode == "raise_h":
        return h_delta >= 0.0 and candidate_error <= current_error + 1e-9
    if mode == "reduce_h":
        return h_delta <= 0.0 and candidate_error <= current_error + 1e-9
    candidate_ratio = (float(candidate_h) - float(target_h)) / float(target_h)
    return (
        float(soft_lower_ratio) - 1e-9
        <= float(candidate_ratio)
        <= float(soft_upper_ratio) + 1e-9
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
        donor_types = [12, 13] + donor_types
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


def _rank_gap_candidates(
    candidates: List[HistogramConversion],
    inventory: Dict[int, int],
    demand: Dict[int, int],
    *,
    target: int,
    aromatic_cursor: int,
    current_h: Optional[float],
    target_h: Optional[float],
) -> List[HistogramConversion]:
    mode = _h_mode(current_h, target_h)
    aromatic_order = {
        int(su_type): int(_aromatic_rotation_offset(aromatic_cursor, int(su_type)) or 0)
        for su_type in AROMATIC_PATTERN
    }

    def key(conversion: HistogramConversion) -> Tuple[Any, ...]:
        source = int(conversion.from_su)
        h_delta = _conversion_h_delta(conversion)
        candidate_error = (
            abs(float(current_h) + float(h_delta) - float(target_h))
            if current_h is not None and target_h is not None
            else abs(float(h_delta))
        )
        rotation_source = source in AROMATIC_PATTERN and (
            int(target) == 11 or conversion.via_su is not None and int(conversion.via_su) == 11
        )
        if rotation_source:
            return (0, int(aromatic_order.get(source, 99)), candidate_error, source)
        if int(target) == 11 and mode == "reduce_h":
            return (0, float(h_delta), -_surplus(inventory, demand, source), source)
        return (
            1,
            1 if conversion.via_su is not None else 0,
            candidate_error,
            -_surplus(inventory, demand, source),
            abs(float(h_delta)),
            source,
        )

    return sorted(candidates, key=key)


def _choose_from_gap(
    H: torch.Tensor,
    inventory: Dict[int, int],
    demand: Dict[int, int],
    aromatic_cursor: int,
    current_h: Optional[float] = None,
    target_h: Optional[float] = None,
) -> Tuple[Optional[HistogramConversion], int]:
    gap = {su: int(demand.get(su, 0)) - int(inventory.get(su, 0)) for su in RESOURCE_TYPES}
    targets = [su for su in RESOURCE_TYPES if int(gap.get(su, 0)) > 0]
    if not targets:
        return None, aromatic_cursor
    target = sorted(targets, key=lambda su: (-int(gap[su]), TARGET_PRIORITY[int(su)]))[0]
    candidates = _direct_gap_candidates(int(target), inventory, demand)
    allowed = [
        conversion
        for conversion in candidates
        if _soft_h_direction_allows(
            conversion,
            current_h=current_h,
            target_h=target_h,
        )
    ]
    if not allowed:
        allowed = [
            conversion
            for conversion in _atomic_aromatic_bridge_candidates(int(target), inventory, demand)
            if _soft_h_direction_allows(
                conversion,
                current_h=current_h,
                target_h=target_h,
            )
        ]
    if allowed:
        selected = _rank_gap_candidates(
            allowed,
            inventory,
            demand,
            target=int(target),
            aromatic_cursor=aromatic_cursor,
            current_h=current_h,
            target_h=target_h,
        )[0]
        if int(selected.from_su) in AROMATIC_PATTERN and (
            int(target) == 11 or selected.via_su is not None and int(selected.via_su) == 11
        ):
            aromatic_cursor = advance_su11_aromatic_rotation(
                int(aromatic_cursor),
                int(selected.from_su),
            )
        return selected, aromatic_cursor
    return None, aromatic_cursor


def _positive_gap(demand: Dict[int, int], inventory: Dict[int, int]) -> Dict[int, int]:
    return {
        int(su_type): max(0, int(demand.get(int(su_type), 0)) - int(inventory.get(int(su_type), 0)))
        for su_type in RESOURCE_TYPES
    }


def _does_not_worsen_resource_gap(
    demand: Dict[int, int],
    inventory_before: Dict[int, int],
    inventory_after: Dict[int, int],
) -> bool:
    gap_before = _positive_gap(demand, inventory_before)
    gap_after = _positive_gap(demand, inventory_after)
    return all(
        int(gap_after.get(int(su_type), 0)) <= int(gap_before.get(int(su_type), 0))
        for su_type in RESOURCE_TYPES
    )


def _rank_h_compensation_candidates(
    H: torch.Tensor,
    inventory: Dict[int, int],
    demand: Dict[int, int],
    *,
    target_h: float,
    special_degree_meta: Optional[Dict[str, Any]],
) -> List[Tuple[int, int, torch.Tensor, float]]:
    current_h = _h_total(H, special_degree_meta)
    current_error = abs(float(current_h) - float(target_h))
    reduce_h = float(current_h) > float(target_h)
    ranked: List[Tuple[int, float, int, int, int, torch.Tensor]] = []
    for source in RESOURCE_TYPES:
        surplus = int(inventory.get(int(source), 0)) - int(demand.get(int(source), 0))
        if surplus <= 0:
            continue
        source_h = float(E_SU[int(source), 1].item())
        for target in RESOURCE_TYPES:
            if int(target) == int(source):
                continue
            target_h_value = float(E_SU[int(target), 1].item())
            h_delta = float(target_h_value - source_h)
            if reduce_h and h_delta >= 0:
                continue
            if (not reduce_h) and h_delta <= 0:
                continue
            candidate = _apply_delta(H, _conversion_delta(source, target))
            if candidate is None:
                continue
            candidate_inventory = dict(inventory)
            candidate_inventory[int(source)] = int(candidate_inventory.get(int(source), 0)) - 1
            candidate_inventory[int(target)] = int(candidate_inventory.get(int(target), 0)) + 1
            if not _does_not_worsen_resource_gap(demand, inventory, candidate_inventory):
                continue
            candidate_error = abs(_h_total(candidate, special_degree_meta) - float(target_h))
            if candidate_error + 1e-9 >= current_error:
                continue
            target_deficit = max(0, int(demand.get(int(target), 0)) - int(inventory.get(int(target), 0)))
            ranked.append((
                -int(surplus),
                float(candidate_error),
                -int(target_deficit),
                int(source),
                int(target),
                candidate,
            ))
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return [
        (int(item[3]), int(item[4]), item[5], float(item[1]))
        for item in ranked
    ]


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
    h_compensation_blocked = False
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
        candidate_aromatic_cursor = int(aromatic_cursor)
        current_h = _h_total(H_work, special_degree_meta) if target_h is not None else None
        hard_gap_present = any(
            int(demand.get(int(su_type), 0)) > int(inventory.get(int(su_type), 0))
            for su_type in RESOURCE_TYPES
        )
        if hard_gap_present:
            used_gap_selector = True
            primary, candidate_aromatic_cursor = _choose_from_gap(
                H_work,
                inventory,
                demand,
                aromatic_cursor,
                current_h=current_h,
                target_h=target_h,
            )
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
                primary, candidate_aromatic_cursor = _choose_from_gap(
                    H_work,
                    inventory,
                    preferred_absolute,
                    aromatic_cursor,
                    current_h=current_h,
                    target_h=target_h,
                )
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
                if not _soft_h_direction_allows(
                    candidate,
                    current_h=current_h,
                    target_h=target_h,
                ):
                    soft_direction_blocked = True
                    continue
                primary = candidate
                if int(primary.from_su) in AROMATIC_PATTERN and int(primary.to_su) == 11:
                    candidate_aromatic_cursor = advance_su11_aromatic_rotation(
                        int(aromatic_cursor),
                        int(primary.from_su),
                    )
                elif int(primary.from_su) == 11 and int(primary.to_su) in AROMATIC_PATTERN:
                    candidate_aromatic_cursor = advance_su11_aromatic_rotation(
                        int(aromatic_cursor),
                        int(primary.to_su),
                    )
        if primary is None:
            break
        primary_state = (
            tuple(int(value) for value in H_work.detach().cpu().long().tolist()),
            tuple(sorted((int(key), int(value)) for key, value in inventory.items())),
            tuple(sorted((int(key), int(value)) for key, value in demand.items())),
            int(primary.from_su),
            int(primary.to_su),
            int(primary.count),
            int(primary.via_su) if primary.via_su is not None else -1,
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
        compensation: List[HistogramConversion] = []
        H_final = H_primary
        bundle_inventory = dict(inventory)
        for su_type, value in primary_delta.items():
            bundle_inventory[int(su_type)] = int(bundle_inventory.get(int(su_type), 0)) + int(value)
        h_lower = float(target_h) * (1.0 - float(h_tolerance)) if target_h is not None else None
        h_upper = float(target_h) * (1.0 + float(h_tolerance)) if target_h is not None else None
        h_outside_bounds = bool(
            h_lower is not None
            and h_upper is not None
            and target_h > 0
            and not (float(h_lower) <= _h_total(H_final, special_degree_meta) <= float(h_upper))
        )
        if h_outside_bounds:
            for _ in range(max(0, int(max_h_compensation_ops))):
                ranked_candidates = _rank_h_compensation_candidates(
                    H_final,
                    bundle_inventory,
                    demand,
                    target_h=float(target_h),
                    special_degree_meta=special_degree_meta,
                )
                if not ranked_candidates:
                    break
                source, target, H_final, _ = ranked_candidates[0]
                compensation.append(HistogramConversion(source, target, 1, "bound_h_compensation"))
                bundle_inventory[int(source)] = int(bundle_inventory.get(int(source), 0)) - 1
                bundle_inventory[int(target)] = int(bundle_inventory.get(int(target), 0)) + 1
                current_h = _h_total(H_final, special_degree_meta)
                if float(h_lower) <= float(current_h) <= float(h_upper):
                    break
            final_h = _h_total(H_final, special_degree_meta)
            if not (float(h_lower) <= float(final_h) <= float(h_upper)):
                h_compensation_blocked = True
                rejected_primary_states.add(primary_state)
                continue
        all_ops = [primary] + compensation
        combined_delta = _merge_delta(all_ops)
        if not combined_delta:
            rejected_primary_states.add(primary_state)
            continue
        inventory_after = dict(bundle_inventory)
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
        unresolved_counts=unresolved,
        blocked_reason=(
            "hydrogen_compensation_unavailable_raise_h_tolerance"
            if not groups and h_compensation_blocked
            else "hydrogen_soft_direction_incompatible_with_resource_demand"
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
                "reason": "donor_not_in_remaining_resource_pool",
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
        if not _does_not_worsen_resource_gap(phase_demand, convertible_inventory, candidate_inventory):
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
                candidates = sorted(
                    int(node_id)
                    for node_id, node in nodes_by_id.items()
                    if int(getattr(node, "su_type", -1)) == int(op.from_su)
                )
                node_ids = candidates[:int(op.count)]
            if len(node_ids) != int(op.count):
                rejected_ops.append({
                    "group": group.to_dict(),
                    "reason": "real_donor_node_unavailable",
                    "from_su": int(op.from_su),
                    "required": int(op.count),
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
