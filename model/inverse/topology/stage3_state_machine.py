from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .histogram_adjustment import (
    _h_total,
    build_histogram_adjustment_plan,
    next_su11_aromatic_target,
)
from .iteration_ledger import (
    assign_contract_slots,
    build_iteration_ledger,
    contract_source_status,
    normalize_slot,
    required_counts_from_slots,
    seed_ledger_topology_state,
)
from .sequential_resource import contract_priority
from .stage3_models import (
    AtomicAdjustmentGroup,
    HistogramConversion,
    HistogramAdjustmentPlan,
    Stage3AdjustmentBundle,
    Stage3AdjustmentOp,
    Stage3AdjustmentPlan,
    Stage3PhaseResult,
    SU24ResourceRoleProfile,
    TopologyClosureContract,
    TopologyClosureSlot,
    TopologyResourceTransaction,
    merge_stage3_contracts_by_owner,
)

RESOURCE_TYPES = (11, 22, 23, 24)
EXTERNAL_PHASES = (
    ("external_flex_closure", "flex_chain"),
    ("external_fatty_ring_closure", "fatty_ring"),
    ("external_side_chain_closure", "side_chain"),
)


def _integer_map(values: Dict[Any, Any]) -> Dict[int, int]:
    return {
        int(key): int(value)
        for key, value in dict(values or {}).items()
        if str(key).lstrip("-").isdigit()
    }


def _resource_ids(phase_eval: Any) -> Dict[int, List[int]]:
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    classification = dict(getattr(phase_eval, "classification_view", {}) or {})
    pool = dict(classification.get("resource_pool", {}) or {})
    raw = (
        ledger.get("remaining_ids_by_su")
        or pool.get("remaining_ids_by_su")
        or pool.get("initial_ids_by_su")
        or {}
    )
    result = {
        int(key): sorted({int(value) for value in list(values or [])})
        for key, values in dict(raw or {}).items()
        if str(key).lstrip("-").isdigit()
    }
    donor_raw = ledger.get("su11_donor_ids_by_su") or pool.get("su11_donor_ids_by_su") or {}
    for key, values in dict(donor_raw or {}).items():
        if not str(key).lstrip("-").isdigit():
            continue
        result.setdefault(int(key), [])
        result[int(key)] = sorted(set(result[int(key)]) | {int(value) for value in list(values or [])})
    for su_type in (10, 11, 12, 13, 22, 23, 24):
        result.setdefault(int(su_type), [])
    return result


def _consume(pool: Dict[int, List[int]], required: Dict[int, int]) -> Optional[Dict[int, List[int]]]:
    if any(len(pool.get(int(su), [])) < int(count) for su, count in required.items()):
        return None
    claimed: Dict[int, List[int]] = {}
    for su_type, count in sorted(required.items()):
        selected = list(pool.get(int(su_type), [])[: int(count)])
        claimed[int(su_type)] = selected
        pool[int(su_type)] = list(pool.get(int(su_type), [])[int(count):])
    return claimed


def _deficit(pool: Dict[int, List[int]], required: Dict[int, int]) -> Dict[int, int]:
    return {
        int(su): int(count) - len(pool.get(int(su), []))
        for su, count in required.items()
        if int(count) > len(pool.get(int(su), []))
    }


def _contract_id(contract: Dict[str, Any], fallback: int) -> int:
    value = dict(contract or {}).get("contract_id")
    if value is None:
        raise ValueError(f"canonical topology contract is missing contract_id: {contract}")
    return int(value)


def _closure_contract(contract: Dict[str, Any], phase: str, fallback: int) -> TopologyClosureContract:
    data = dict(contract or {})
    contract_kind = str(data.get("contract_kind", "") or "")
    target_role = str(data.get("target_role", "") or "")
    if not contract_kind or not target_role:
        raise ValueError(f"non-canonical topology contract: {data}")
    mainline_slots = [
        normalize_slot(value)
        for value in list(data.get("mainline_slots", []) or [])
    ]
    branch_slots = [normalize_slot(value) for value in list(data.get("branch_slots", []) or [])]
    required = {
        int(key): int(value)
        for key, value in dict(data.get("required_counts", {}) or {}).items()
        if int(value) > 0
    }
    claimed_source_ids = [
        int(value)
        for value in list(data.get("claimed_source_node_ids", []) or [])
    ]
    return TopologyClosureContract(
        contract_id=_contract_id(data, fallback),
        contract_kind=str(contract_kind),
        phase=str(phase),
        priority=int(contract_priority(data)[0]),
        target_role=str(target_role),
        source_fragment_ids=[int(value) for value in list(data.get("source_fragment_ids", []) or [])],
        source_fragment_classes=[str(value) for value in list(data.get("source_fragment_classes", []) or [])],
        source_node_ids=list(claimed_source_ids),
        claimed_source_node_ids=list(claimed_source_ids),
        context_node_ids=[int(value) for value in list(data.get("context_node_ids", []) or [])],
        mainline_slots=mainline_slots,
        branch_slots=branch_slots,
        slot_requirements=[slot.to_dict() for slot in mainline_slots + branch_slots],
        required_counts={int(key): int(value) for key, value in required.items()},
        required_roles=[],
        edge_plan=[dict(value or {}) for value in list(data.get("edge_plan", []) or [])],
        expected_topology=str(data.get("expected_topology", "") or ""),
        reason=str(data.get("reason", "") or ""),
        blocker_kind=str(data.get("blocker_kind", "") or ""),
        diagnostics=dict(data.get("diagnostics", {}) or {}),
    )


def _external_contracts(phase_eval: Any, nodes: Sequence[Any]) -> List[Dict[str, Any]]:
    del nodes
    contracts = [
        dict(item or {})
        for item in list(getattr(phase_eval, "canonical_contracts", []) or [])
        if dict(item or {})
    ]
    return sorted(merge_stage3_contracts_by_owner(contracts), key=contract_priority)


def build_su24_resource_role_profiles(nodes: Sequence[Any], pool: Dict[int, List[int]]) -> List[SU24ResourceRoleProfile]:
    by_id = {int(getattr(node, "global_id", -1)): node for node in list(nodes or [])}
    profiles: List[SU24ResourceRoleProfile] = []
    for node_id in sorted(pool.get(24, [])):
        node = by_id.get(int(node_id))
        if node is None:
            continue
        hop1_su = _integer_map(dict(getattr(node, "hop1_su", {}) or {}))
        fixed_ids = set(getattr(node, "fixed_hop1_ids", set()) or set())
        hop1_node_ids = sorted(int(value) for value in list(getattr(node, "hop1_ids", set()) or set()))
        degree = int(sum(hop1_su.values()))
        target_degree = int(getattr(node, "target_hop1_degree", 3) or 3)
        remaining_slots = max(0, target_degree - degree)
        roles: List[str] = []
        if fixed_ids == set() and remaining_slots >= 2:
            roles.append("A")
        if remaining_slots >= 1:
            roles.append("B")
        if fixed_ids or int(hop1_su.get(11, 0)) > 0:
            roles.append("C")
        if degree <= target_degree:
            roles.append("D")
        allowed_slots: List[str] = []
        if "A" in roles:
            allowed_slots.extend([
                "side:middle",
                "upper_lower:fixed_anchor",
                "upper_lower:path",
                "fused:left_outer_anchor",
                "fused:right_outer_anchor",
            ])
        if "B" in roles:
            allowed_slots.extend([
                "side:middle",
                "upper_lower:path",
                "fused:left_outer_anchor",
                "fused:right_outer_anchor",
            ])
        if "C" in roles:
            allowed_slots.extend([
                "side:middle",
                "upper_lower:path",
                "fused:left_bridgehead",
                "fused:right_bridgehead",
                "fused:inner_core",
            ])
        if "D" in roles:
            allowed_slots.extend([
                "side:middle",
                "upper_lower:path",
                "fused:inner_core",
            ])
        all_slots = {
            "side:middle", "upper_lower:fixed_anchor", "upper_lower:path",
            "fused:left_outer_anchor", "fused:right_outer_anchor",
            "fused:left_bridgehead", "fused:right_bridgehead", "fused:inner_core",
        }
        profiles.append(SU24ResourceRoleProfile(
            node_id=int(node_id),
            roles=roles,
            available_roles=list(roles),
            hop1_su=hop1_su,
            hop1_node_ids=hop1_node_ids,
            fixed_neighbor_ids=sorted(int(value) for value in fixed_ids),
            fixed_neighbor_count=len(fixed_ids),
            remaining_degree=int(remaining_slots),
            branch_h_count=max(0, int(remaining_slots) - 1),
            requires_su11_cap=bool(remaining_slots > 0 and not fixed_ids),
            allowed_ring_slots=sorted(set(allowed_slots)),
            forbidden_ring_slots=sorted(all_slots - set(allowed_slots)),
        ))
    return profiles


def _su24_ring_contracts(profiles: Sequence[SU24ResourceRoleProfile]) -> List[TopologyClosureContract]:
    available = sorted(profiles, key=lambda item: int(item.node_id))
    contracts: List[TopologyClosureContract] = []
    templates = (
        ("resource_su24_side_ring", "side", ("A", "B"), {11: 2, 22: 2, 23: 4}),
        ("resource_su24_upper_lower_ring", "upper_lower", ("A", "B", "C"), {11: 2, 22: 2, 23: 3}),
        ("resource_su24_fused_ring", "fused", ("A", "B", "C", "D"), {11: 2, 22: 2, 23: 2}),
    )
    for template_index, (name, template_key, role_slots, required) in enumerate(templates):
        for group in combinations(available, len(role_slots)):
            assignments = list(zip(role_slots, group))
            if any(str(role) not in set(item.available_roles or item.roles) for role, item in assignments):
                continue
            contracts.append(TopologyClosureContract(
                contract_id=1_000_000 + len(contracts),
                contract_kind="ring_partner",
                phase="resource_su24_ring_closure",
                priority=int(template_index),
                target_role="fatty_ring",
                source_node_ids=[int(item.node_id) for item in group],
                claimed_source_node_ids=[int(item.node_id) for item in group],
                required_counts=dict(required),
                required_roles=[str(role) for role in role_slots],
                slot_requirements=[
                    {"slot": f"{template_key}:{role}", "node_id": int(item.node_id), "role": str(role)}
                    for role, item in assignments
                ],
                expected_topology=name,
                diagnostics={"ring_template": str(name)},
            ))
    return contracts


def _residual_contracts(
    pool: Dict[int, List[int]],
    chain_23_limit: int,
    max_flex_chains: Optional[int] = None,
) -> List[TopologyClosureContract]:
    contracts: List[TopologyClosureContract] = []
    available = {int(key): len(values) for key, values in pool.items()}
    limit = max(1, int(chain_23_limit))
    contract_id = 2_000_000
    flex_created = 0
    for chain_length in range(int(limit), 0, -1):
        if chain_length <= 0:
            continue
        required = {11: 2, 23: int(chain_length)}
        sequence = [11] + [23] * int(chain_length) + [11]
        while (
            all(int(available.get(su, 0)) >= count for su, count in required.items())
            and (max_flex_chains is None or int(flex_created) < int(max_flex_chains))
        ):
            contracts.append(TopologyClosureContract(
                contract_id=contract_id,
                contract_kind="residual_conversion",
                phase="residual_resource_self_closure",
                priority=0,
                target_role="residual_resource_conversion",
                mainline_slots=[TopologyClosureSlot(
                    slot_id=f"residual:{contract_id}:mainline",
                    kind="RESOURCE_PATH",
                    anchor_node_id=-1,
                    anchor_role="resource_path",
                    required_su_sequence=list(sequence),
                )],
                required_counts=dict(required),
                expected_topology="11-" + "-".join(["23"] * chain_length) + "-11",
                diagnostics={"product_role": "residual_flex_chain"},
            ))
            contract_id += 1
            flex_created += 1
            for su, count in required.items():
                available[su] -= count
    if max_flex_chains is not None and int(flex_created) < int(max_flex_chains):
        return contracts
    for required, topology in (({11: 1, 22: 1, 23: 2}, "11-23-23-22"), ({11: 1, 22: 1, 23: 1}, "11-23-22")):
        sequence = [int(value) for value in topology.split("-")]
        while all(int(available.get(su, 0)) >= count for su, count in required.items()):
            contracts.append(TopologyClosureContract(
                contract_id=contract_id,
                contract_kind="residual_conversion",
                phase="residual_resource_self_closure",
                priority=1,
                target_role="residual_resource_conversion",
                mainline_slots=[TopologyClosureSlot(
                    slot_id=f"residual:{contract_id}:mainline",
                    kind="RESOURCE_PATH",
                    anchor_node_id=-1,
                    anchor_role="resource_path",
                    required_su_sequence=list(sequence),
                )],
                required_counts=dict(required),
                expected_topology=topology,
                diagnostics={"product_role": "residual_side_chain"},
            ))
            contract_id += 1
            for su, count in required.items():
                available[su] -= count
    return contracts


def _apply_conversion_to_histogram(
    H: torch.Tensor,
    conversion: HistogramConversion,
) -> Optional[torch.Tensor]:
    candidate = H.detach().clone().long()
    source = int(conversion.from_su)
    target = int(conversion.to_su)
    count = max(1, int(conversion.count))
    if int(candidate[source].item()) < int(count):
        return None
    candidate[source] -= int(count)
    candidate[target] += int(count)
    return candidate


def _fit_residual_required_counts(
    *,
    pool: Dict[int, List[int]],
    required_counts: Dict[int, int],
    H: torch.Tensor,
    E_target: Any,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int,
    max_conversions: int,
    max_h_compensation_ops: int,
) -> Optional[Dict[str, Any]]:
    deficits = {
        int(su_type): max(0, int(required) - len(pool.get(int(su_type), [])))
        for su_type, required in dict(required_counts or {}).items()
    }
    if not any(deficits.values()):
        return None
    surplus_ids = {
        int(su_type): list(pool.get(int(su_type), []))[max(0, int(required_counts.get(int(su_type), 0))):]
        for su_type in (11, 12, 13, 22, 23, 24)
    }
    if sum(len(values) for values in surplus_ids.values()) < sum(deficits.values()):
        return None

    H_work = H.detach().clone().long()
    conversions: List[HistogramConversion] = []
    rotation_cursor = int(su11_rotation_index)
    target_h = None
    try:
        target_h = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        target_h = None

    target_sequence = [
        int(target)
        for target in (23, 22, 11, 24)
        for _ in range(int(deficits.get(int(target), 0)))
    ]
    for target in target_sequence:
        aromatic_available = [
            int(su_type)
            for su_type in (12, 13)
            if list(surplus_ids.get(int(su_type), []) or [])
        ]
        next_aromatic = None
        next_cursor = int(rotation_cursor)
        if aromatic_available:
            next_aromatic, next_cursor = next_su11_aromatic_target(
                int(rotation_cursor),
                available_targets=aromatic_available,
            )
        candidates: List[tuple[float, int, int, torch.Tensor, HistogramConversion]] = []
        for source in (11, 22, 23, 24, 12, 13):
            source_ids = list(surplus_ids.get(int(source), []) or [])
            if not source_ids or int(source) == int(target):
                continue
            if int(source) in {12, 13} and int(source) != int(next_aromatic or -1):
                continue
            conversion = HistogramConversion(
                from_su=int(source),
                to_su=int(target),
                count=1,
                reason="residual_atomic_template_fit",
                node_ids=[int(source_ids[0])],
                via_su=11 if int(source) in {12, 13} and int(target) != 11 else None,
            )
            H_candidate = _apply_conversion_to_histogram(H_work, conversion)
            if H_candidate is None:
                continue
            h_error = (
                abs(_h_total(H_candidate, special_degree_meta) - float(target_h))
                if target_h is not None
                else 0.0
            )
            candidates.append((
                float(h_error),
                1 if int(source) in {12, 13} else 0,
                int(source_ids[0]),
                H_candidate,
                conversion,
            ))
        if not candidates:
            return None
        _, _, node_id, H_work, selected = min(candidates, key=lambda item: item[:3])
        surplus_ids[int(selected.from_su)].remove(int(node_id))
        conversions.append(selected)
        if int(selected.from_su) in {12, 13}:
            rotation_cursor = int(next_cursor)

    final_h = _h_total(H_work, special_degree_meta)
    compensation_limit = min(
        max(0, int(max_conversions)),
        len(conversions) + max(0, int(max_h_compensation_ops)),
    )
    while target_h is not None and target_h > 0 and len(conversions) < compensation_limit:
        h_lower = float(target_h) * (1.0 - float(h_tolerance))
        h_upper = float(target_h) * (1.0 + float(h_tolerance))
        if float(h_lower) <= float(final_h) <= float(h_upper):
            break
        current_error = abs(float(final_h) - float(target_h))
        aromatic_available = [
            int(su_type)
            for su_type in (12, 13)
            if list(surplus_ids.get(int(su_type), []) or [])
        ]
        next_aromatic_source = None
        next_aromatic_source_cursor = int(rotation_cursor)
        if aromatic_available:
            next_aromatic_source, next_aromatic_source_cursor = next_su11_aromatic_target(
                int(rotation_cursor),
                available_targets=aromatic_available,
            )
        next_aromatic_target, next_aromatic_target_cursor = next_su11_aromatic_target(
            int(rotation_cursor),
        )
        compensation_candidates = []
        for source in (11, 22, 23, 24, 12, 13):
            source_ids = list(surplus_ids.get(int(source), []) or [])
            if not source_ids:
                continue
            if int(source) in {12, 13} and int(source) != int(next_aromatic_source or -1):
                continue
            targets = [11, 22, 23, 24]
            if int(source) == 11 and next_aromatic_target is not None:
                targets.append(int(next_aromatic_target))
            for target in targets:
                if int(source) == int(target):
                    continue
                conversion = HistogramConversion(
                    from_su=int(source),
                    to_su=int(target),
                    count=1,
                    reason="residual_atomic_h_compensation",
                    node_ids=[int(source_ids[0])],
                    via_su=11 if int(source) in {12, 13} and int(target) != 11 else None,
                )
                H_candidate = _apply_conversion_to_histogram(H_work, conversion)
                if H_candidate is None:
                    continue
                candidate_error = abs(
                    _h_total(H_candidate, special_degree_meta) - float(target_h)
                )
                if float(candidate_error) + 1e-9 >= float(current_error):
                    continue
                compensation_candidates.append((
                    float(candidate_error),
                    1 if conversion.via_su is not None else 0,
                    int(source_ids[0]),
                    int(target),
                    H_candidate,
                    conversion,
                ))
        if not compensation_candidates:
            break
        _, _, node_id, _, H_work, selected = min(
            compensation_candidates,
            key=lambda item: item[:4],
        )
        surplus_ids[int(selected.from_su)].remove(int(node_id))
        conversions.append(selected)
        if int(selected.from_su) in {12, 13}:
            rotation_cursor = int(next_aromatic_source_cursor)
        elif int(selected.from_su) == 11 and int(selected.to_su) in {12, 13}:
            rotation_cursor = int(next_aromatic_target_cursor)
        final_h = _h_total(H_work, special_degree_meta)

    if target_h is not None and target_h > 0:
        h_lower = float(target_h) * (1.0 - float(h_tolerance))
        h_upper = float(target_h) * (1.0 + float(h_tolerance))
        if not (float(h_lower) <= float(final_h) <= float(h_upper)):
            return None
    return {
        "required_counts": {int(key): int(value) for key, value in required_counts.items()},
        "conversions": conversions,
        "H_after": H_work,
        "h_after": float(final_h),
        "h_error": (
            abs(float(final_h) - float(target_h))
            if target_h is not None
            else 0.0
        ),
        "rotation_after": int(rotation_cursor),
    }


def _best_residual_template_fit(
    *,
    pool: Dict[int, List[int]],
    H: torch.Tensor,
    E_target: Any,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int,
    remaining_flex_chains: int,
    chain_23_limit: int,
    max_conversions: int,
    max_h_compensation_ops: int,
) -> Optional[Dict[str, Any]]:
    total = sum(len(pool.get(int(su_type), [])) for su_type in (11, 12, 13, 22, 23, 24))
    requirements: List[Tuple[str, int, int, Dict[int, int]]] = []
    if int(remaining_flex_chains) > 0:
        maximum_products = min(int(remaining_flex_chains), int(total // 3))
        for product_count in range(maximum_products, 0, -1):
            for chain_nodes in range(
                int(chain_23_limit * product_count),
                int(product_count) - 1,
                -1,
            ):
                requirements.append((
                    "residual_flex_chain",
                    int(product_count),
                    int(chain_nodes),
                    {11: int(2 * product_count), 23: int(chain_nodes)},
                ))
    else:
        maximum_products = int(total // 3)
        for product_count in range(maximum_products, 0, -1):
            for long_count in range(product_count, -1, -1):
                requirements.append((
                    "residual_side_chain",
                    int(product_count),
                    int(long_count),
                    {11: int(product_count), 22: int(product_count), 23: int(product_count + long_count)},
                ))

    candidates: List[Dict[str, Any]] = []
    for product_role, product_count, long_count, required in requirements:
        consumed_count = int(sum(required.values()))
        if consumed_count > int(total):
            continue
        conversion_count = int(sum(_deficit(pool, required).values()))
        if conversion_count <= 0 or conversion_count > max(0, int(max_conversions)):
            continue
        fitted = _fit_residual_required_counts(
            pool=pool,
            required_counts=required,
            H=H,
            E_target=E_target,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=su11_rotation_index,
            max_conversions=max_conversions,
            max_h_compensation_ops=max_h_compensation_ops,
        )
        if fitted is None:
            continue
        fitted.update({
            "product_role": str(product_role),
            "product_count": int(product_count),
            "long_product_count": int(long_count),
            "consumed_count": int(consumed_count),
            "deficit_conversion_count": int(conversion_count),
            "conversion_count": len(list(fitted["conversions"])),
        })
        candidates.append(fitted)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -int(item["consumed_count"]),
            int(item["conversion_count"]),
            float(item["h_error"]),
            tuple(int(op.node_ids[0]) for op in list(item["conversions"])),
        ),
    )


def _residual_template_fit_plan(
    *,
    phase_eval: Any,
    pool: Dict[int, List[int]],
    H: torch.Tensor,
    E_target: Any,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int,
    remaining_flex_chains: int,
    chain_23_limit: int,
    max_conversions: int,
    max_h_compensation_ops: int,
) -> Optional[Stage3AdjustmentPlan]:
    fitted = _best_residual_template_fit(
        pool=pool,
        H=H,
        E_target=E_target,
        h_tolerance=h_tolerance,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=su11_rotation_index,
        remaining_flex_chains=remaining_flex_chains,
        chain_23_limit=chain_23_limit,
        max_conversions=max_conversions,
        max_h_compensation_ops=max_h_compensation_ops,
    )
    if fitted is None:
        return None
    conversions = list(fitted["conversions"])
    H_after = fitted["H_after"].detach().clone()
    inventory_before = {int(key): len(values) for key, values in pool.items()}
    inventory_after = dict(inventory_before)
    for conversion in conversions:
        inventory_after[int(conversion.from_su)] = int(inventory_after.get(int(conversion.from_su), 0)) - 1
        inventory_after[int(conversion.to_su)] = int(inventory_after.get(int(conversion.to_su), 0)) + 1
    combined_delta: Dict[int, int] = {}
    for conversion in conversions:
        combined_delta[int(conversion.from_su)] = int(combined_delta.get(int(conversion.from_su), 0)) - 1
        combined_delta[int(conversion.to_su)] = int(combined_delta.get(int(conversion.to_su), 0)) + 1
    H_after_primary = _apply_conversion_to_histogram(H, conversions[0])
    if H_after_primary is None:
        return None
    group = AtomicAdjustmentGroup(
        primary_op=conversions[0],
        h_compensation_ops=list(conversions[1:]),
        combined_delta={int(key): int(value) for key, value in combined_delta.items() if int(value) != 0},
        demand_before=dict(fitted["required_counts"]),
        demand_after={},
        h_before=_h_total(H, special_degree_meta),
        h_after_primary=_h_total(H_after_primary, special_degree_meta),
        h_after_final=float(fitted["h_after"]),
    )
    histogram = HistogramAdjustmentPlan(
        phase="residual_resource_self_closure",
        groups=[group],
        primary_adjustment_count=len(conversions),
        physical_conversion_count=sum(2 if op.via_su is not None else 1 for op in conversions),
        H_before=H.detach().clone(),
        H_after=H_after,
        demand_before=dict(fitted["required_counts"]),
        demand_after=dict(fitted["required_counts"]),
        inventory_before=inventory_before,
        inventory_after=inventory_after,
        unresolved_counts={},
        su11_rotation_index_before=int(su11_rotation_index),
        su11_rotation_index_after=int(fitted["rotation_after"]),
    )
    transaction = TopologyResourceTransaction(
        phase="residual_resource_self_closure",
        contract_id=2_750_000,
        target_role="residual_resource_conversion",
        converted_node_ids=[int(op.node_ids[0]) for op in conversions],
        conversion_path=[op.to_dict() for op in conversions],
        h_before=float(group.h_before),
        h_after_primary=float(group.h_after_primary),
        h_after=float(group.h_after_final),
        commit_scope="physical_histogram",
        committed=True,
    )
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="residual_resource_self_closure",
        plan_kind="residual_atomic_template_fit",
        pre_assembly_gate="residual_resource_self_closure",
        active_subphase="residual_resource_self_closure",
        hard_required_counts={str(key): int(value) for key, value in fitted["required_counts"].items()},
        histogram_adjustment_plan=histogram,
        planned_conversions=[
            Stage3AdjustmentOp(
                op="convert",
                from_su=int(conversion.from_su),
                to_su=int(conversion.to_su),
                node_id=int(conversion.node_ids[0]),
                reason=str(conversion.reason),
                via_su=conversion.via_su,
            )
            for conversion in conversions
        ],
        phase_result=Stage3PhaseResult(
            active_phase="residual_resource_self_closure",
            rebuild_required=True,
            primary_adjustment_count=len(conversions),
            h_compensation_count=max(0, len(conversions) - 1),
            transactions=[transaction],
            stop_reason="stage3_rebuild_required",
        ),
        runtime=_runtime_with_pool(phase_eval, pool),
    )
    plan.diagnostics = {
        "stage3_phase": "residual_resource_self_closure",
        "residual_template_fit": {
            key: value
            for key, value in fitted.items()
            if key not in {"conversions", "H_after"}
        },
        "su11_rotation_index_after": int(fitted["rotation_after"]),
    }
    return plan


def _residual_infeasibility_proof(
    *,
    pool: Dict[int, List[int]],
    H: torch.Tensor,
    E_target: Any,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    remaining_flex_chains: int,
    chain_23_limit: int,
    max_conversions: int,
) -> Dict[str, Any]:
    target_h = None
    try:
        target_h = float(E_target.detach().cpu().float()[1].item())
    except Exception:
        target_h = None
    current_h = _h_total(H, special_degree_meta)
    h_bounds = {}
    if target_h is not None and target_h > 0:
        h_bounds = {
            "target": float(target_h),
            "lower": float(target_h) * (1.0 - float(h_tolerance)),
            "upper": float(target_h) * (1.0 + float(h_tolerance)),
            "current": float(current_h),
        }
    return {
        "reason": "no_executable_residual_template_fit",
        "remaining_counts": {
            str(key): len(values)
            for key, values in sorted(pool.items())
            if list(values or [])
        },
        "remaining_flex_target": int(remaining_flex_chains),
        "template_priority": (
            ["11-23...-11"]
            if int(remaining_flex_chains) > 0
            else ["11-23-23-22", "11-23-22"]
        ),
        "allowed_conversion_sources": [11, 12, 13, 22, 23, 24],
        "allowed_template_resources": [11, 22, 23, 24],
        "chain_23_limit": int(chain_23_limit),
        "max_atomic_conversions": int(max_conversions),
        "h_bounds": h_bounds,
        "executable": False,
    }


def _runtime_with_pool(phase_eval: Any, pool: Dict[int, List[int]]) -> Dict[str, Any]:
    assembly = deepcopy(dict(getattr(phase_eval, "assembly_view", {}) or {}))
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    ledger["remaining_ids_by_su"] = {str(key): list(values) for key, values in pool.items()}
    ledger["remaining_counts"] = {str(key): len(values) for key, values in pool.items()}
    ledger["remaining_counts_by_su"] = dict(ledger["remaining_counts"])
    ledger["convertible_counts"] = dict(ledger["remaining_counts"])
    assembly["resource_ledger"] = ledger
    return {
        "phase_evaluation": phase_eval,
        "phase_evaluation_dict": phase_eval.to_dict(),
        "assembly_before_adjustment": assembly,
        "assembly_before_adjustment_dict": assembly,
    }


def _iteration_ledger_for_phase_eval(
    phase_eval: Any,
    nodes: Sequence[Any],
    pool: Dict[int, List[int]],
    contracts: Sequence[Dict[str, Any]],
) -> Any:
    classification = dict(getattr(phase_eval, "classification_view", {}) or {})
    support_ids = [
        int(dict(item or {}).get("node_id"))
        for item in list(classification.get("support_units_only", []) or [])
        if dict(item or {}).get("node_id") is not None
    ]
    claimed_source_ids = [
        int(value)
        for contract in list(contracts or [])
        for value in list(dict(contract or {}).get("claimed_source_node_ids", []) or [])
    ]
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su=pool,
        support_only_node_ids=support_ids,
        claimed_source_ids=claimed_source_ids,
    )
    return seed_ledger_topology_state(
        ledger,
        completed_products=list(assembly.get("products", []) or []),
        pending_contracts=list(contracts or []),
    )


def _ledger_transaction(
    *,
    ledger: Any,
    contract: TopologyClosureContract,
    phase: str,
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
) -> tuple[Any, Optional[TopologyResourceTransaction], Dict[str, Any]]:
    updated, slots, claimed, edge_plan, proof = assign_contract_slots(ledger, contract)
    if updated is None:
        return ledger, None, dict(proof or {})
    product = dict(updated.completed_products[-1] or {})
    h_value = _h_total(H, special_degree_meta)
    transaction = TopologyResourceTransaction(
        phase=str(phase),
        contract_id=int(contract.contract_id),
        target_role=str(contract.target_role),
        source_node_ids=list(contract.claimed_source_node_ids or contract.source_node_ids),
        claimed_resource_ids=dict(claimed),
        product_node_ids=[int(value) for value in list(product.get("source_node_ids", []) or [])],
        slot_assignments=[slot.to_dict() for slot in slots],
        edge_plan=list(edge_plan),
        h_before=float(h_value),
        h_after_primary=float(h_value),
        h_after=float(h_value),
        commit_scope="iteration_ledger",
        committed=True,
    )
    return updated, transaction, {}


def _assembly_product_transactions(
    phase_eval: Any,
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[str, Any]],
) -> List[TopologyResourceTransaction]:
    assembly = dict(getattr(phase_eval, "assembly_view", {}) or {})
    h_value = _h_total(H, special_degree_meta)
    transactions: List[TopologyResourceTransaction] = []
    for index, raw in enumerate(list(assembly.get("products", []) or [])):
        product = dict(raw or {})
        product_kind = str(product.get("product_kind", product.get("product_type", "")) or "")
        phase = (
            "external_fatty_ring_closure"
            if "ring" in product_kind
            else "external_side_chain_closure"
            if "side" in product_kind
            else "external_flex_closure"
        )
        fragment_ids = [int(value) for value in list(product.get("source_fragment_ids", []) or [])]
        product_ids = [
            int(value)
            for value in list(product.get("materialized_node_ids", []) or product.get("source_node_ids", []) or [])
        ]
        transactions.append(TopologyResourceTransaction(
            phase=phase,
            contract_id=int(product.get("contract_id", min(fragment_ids or [index + 1])) or index + 1),
            target_role=product_kind,
            source_node_ids=[int(value) for value in list(product.get("source_node_ids", []) or [])],
            claimed_resource_ids={
                int(key): [int(value) for value in list(values or [])]
                for key, values in dict(product.get("claimed_resource_ids", {}) or {}).items()
                if str(key).lstrip("-").isdigit()
            },
            product_node_ids=sorted(set(product_ids)),
            slot_assignments=[dict(item or {}) for item in list(product.get("slot_assignments", []) or [])],
            edge_plan=[dict(item or {}) for item in list(product.get("edge_plan", []) or [])],
            h_before=float(h_value),
            h_after_primary=float(h_value),
            h_after=float(h_value),
            commit_scope="iteration_ledger",
            committed=True,
        ))
    return transactions


def _bind_real_node_ids(histogram_plan: Any, pool: Dict[int, List[int]]) -> Dict[int, List[int]]:
    work = {int(key): list(values) for key, values in pool.items()}
    for group in list(getattr(histogram_plan, "groups", []) or []):
        for op in [group.primary_op] + list(group.h_compensation_ops or []):
            candidates = list(work.get(int(op.from_su), []))
            selected = candidates[: int(op.count)]
            op.node_ids = selected
            work[int(op.from_su)] = candidates[int(op.count):]
            work.setdefault(int(op.to_su), []).extend(selected)
            work[int(op.to_su)] = sorted(work[int(op.to_su)])
    return work


def _h_compensation_count(groups: Sequence[Any]) -> int:
    return sum(len(list(group.h_compensation_ops or [])) for group in list(groups or []))


def _finalize_phase_plan(
    *,
    phase_eval: Any,
    phase: str,
    H_before: torch.Tensor,
    H_after: torch.Tensor,
    initial_pool: Dict[int, List[int]],
    final_pool: Dict[int, List[int]],
    groups: List[Any],
    ops: List[Dict[str, Any]],
    transactions: List[TopologyResourceTransaction],
    rotation_before: int,
    rotation_after: int,
) -> Stage3AdjustmentPlan:
    histogram = HistogramAdjustmentPlan(
        phase=str(phase),
        groups=list(groups),
        primary_adjustment_count=len(list(groups)),
        physical_conversion_count=sum(
            (2 if group.primary_op.via_su is not None else 1)
            + sum(2 if op.via_su is not None else 1 for op in list(group.h_compensation_ops or []))
            for group in list(groups)
        ),
        H_before=H_before.detach().clone(),
        H_after=H_after.detach().clone(),
        inventory_before={int(key): len(values) for key, values in initial_pool.items()},
        inventory_after={int(key): len(values) for key, values in final_pool.items()},
        demand_before={},
        demand_after={},
        su11_rotation_index_before=int(rotation_before),
        su11_rotation_index_after=int(rotation_after),
    )
    phase_result = Stage3PhaseResult(
        active_phase=str(phase),
        phase_completed=True,
        rebuild_required=True,
        primary_adjustment_count=int(histogram.primary_adjustment_count),
        h_compensation_count=_h_compensation_count(histogram.groups),
        materialized_topology_count=sum(1 for item in transactions if bool(item.committed)),
        transactions=list(transactions),
        stop_reason="stage3_rebuild_required",
    )
    return Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase=str(phase),
        plan_kind="sequential_phase_transactions",
        pre_assembly_gate=str(phase),
        active_subphase=str(phase),
        adjustment_bundles=[Stage3AdjustmentBundle(
            bundle_id=f"{phase}:transactions",
            phase=str(phase),
            bundle_ops=list(ops),
            diagnostics={"sequential_single_topology": True},
        )],
        histogram_adjustment_plan=histogram,
        planned_conversions=[
            Stage3AdjustmentOp(
                op="convert",
                from_su=int(group.primary_op.from_su),
                to_su=int(group.primary_op.to_su),
                node_id=int(list(group.primary_op.node_ids or [-1])[0]),
                reason=str(group.primary_op.reason),
            )
            for group in groups
            if list(group.primary_op.node_ids or [])
        ],
        phase_result=phase_result,
        diagnostics={
            "stage3_phase": str(phase),
            "sequential_single_topology": True,
            "phase_result": phase_result.to_dict(),
            "su11_rotation_index_after": int(rotation_after),
        },
        runtime=_runtime_with_pool(phase_eval, initial_pool),
    )


def _transaction_plan(
    *,
    phase_eval: Any,
    contract: TopologyClosureContract,
    pool: Dict[int, List[int]],
    H: torch.Tensor,
    E_target: Any,
    max_primary_adjustments: int,
    max_h_compensation_ops: int,
    h_tolerance: float,
    special_degree_meta: Optional[Dict[str, Any]],
    su11_rotation_index: int = 0,
    iteration_ledger: Any = None,
    prefix_ops: Optional[List[Dict[str, Any]]] = None,
    prefix_transactions: Optional[List[TopologyResourceTransaction]] = None,
) -> Stage3AdjustmentPlan:
    plan = Stage3AdjustmentPlan(
        ok=False,
        reason="phase_resource_infeasible",
        phase=str(contract.phase),
        plan_kind="single_topology_transaction",
        pre_assembly_gate=str(contract.phase),
        active_subphase=str(contract.phase),
        requirement_contracts=[contract.to_dict()],
        pending_contracts=[contract.to_dict()],
        hard_required_counts={str(key): int(value) for key, value in contract.required_counts.items()},
        minimum_resource_request={str(key): int(value) for key, value in contract.required_counts.items()},
        resource_deficit=_deficit(pool, contract.required_counts),
        runtime=_runtime_with_pool(phase_eval, pool),
    )
    histogram = build_histogram_adjustment_plan(
        plan=plan,
        H=H,
        E_target=E_target,
        max_primary_adjustments=max_primary_adjustments,
        max_h_compensation_ops=max_h_compensation_ops,
        h_tolerance=h_tolerance,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=su11_rotation_index,
    )
    pool_after = _bind_real_node_ids(histogram, pool)
    ledger_after_conversion = deepcopy(iteration_ledger) if iteration_ledger is not None else None
    assigned_slots: List[Any] = []
    edge_plan: List[Dict[str, Any]] = []
    proof: Dict[str, Any] = {}
    claimed: Optional[Dict[int, List[int]]] = None
    if ledger_after_conversion is not None:
        ledger_after_conversion.remaining_resource_ids_by_su = deepcopy(pool_after)
        for group in list(histogram.groups or []):
            for op in [group.primary_op] + list(group.h_compensation_ops or []):
                for node_id in list(op.node_ids or []):
                    owner = ledger_after_conversion.ownership_by_node_id.get(int(node_id))
                    if owner is not None:
                        owner.su_type = int(op.to_su)
                        owner.state = "remaining_resource"
                        owner.owner_contract_id = None
                        owner.owner_product_id = ""
                        owner.owner_slot_id = ""
                        owner.role = "remaining_resource"
        updated_ledger, assigned_slots, claimed_map, edge_plan, proof = assign_contract_slots(
            ledger_after_conversion,
            contract,
        )
        if updated_ledger is not None:
            ledger_after_conversion = updated_ledger
            claimed = claimed_map
    else:
        claimed = _consume(pool_after, contract.required_counts)
    conversions = []
    compensation_ids: List[int] = []
    converted_ids: List[int] = []
    for group in list(histogram.groups or []):
        primary = group.primary_op
        converted_ids.extend(int(value) for value in list(primary.node_ids or []))
        conversions.append(primary.to_dict())
        for compensation in list(group.h_compensation_ops or []):
            compensation_ids.extend(int(value) for value in list(compensation.node_ids or []))
            conversions.append(compensation.to_dict())
    feasible = claimed is not None and not dict(histogram.unresolved_counts or {})
    h_after_primary = _h_total(H, special_degree_meta)
    if list(histogram.groups or []):
        h_after_primary = float(histogram.groups[-1].h_after_primary)
    transaction = TopologyResourceTransaction(
        phase=str(contract.phase),
        contract_id=int(contract.contract_id),
        target_role=str(contract.target_role),
        source_node_ids=list(contract.source_node_ids),
        converted_node_ids=converted_ids,
        conversion_path=conversions,
        compensation_node_ids=compensation_ids,
        claimed_resource_ids=dict(claimed or {}),
        product_node_ids=sorted(set(contract.source_node_ids + [value for values in dict(claimed or {}).values() for value in values])),
        slot_assignments=[slot.to_dict() for slot in list(assigned_slots or [])],
        edge_plan=list(edge_plan),
        h_before=_h_total(H, special_degree_meta),
        h_after_primary=float(h_after_primary),
        h_after=_h_total(getattr(histogram, "H_after", H), special_degree_meta),
        commit_scope="physical_histogram" if feasible else "iteration_ledger",
        committed=bool(feasible),
        rollback_reason="" if feasible else str(histogram.blocked_reason or proof.get("reason", "current_contract_resource_infeasible")),
        infeasibility_proof={} if feasible else dict(proof or {}),
    )
    ops = [dict(value) for value in list(prefix_ops or [])] + [dict(value) for value in conversions]
    plan.ok = bool(feasible and histogram.primary_adjustment_count > 0)
    plan.reason = "stage3_rebuild_required" if plan.ok else "phase_resource_infeasible"
    plan.histogram_adjustment_plan = histogram
    plan.planned_conversions = [
        Stage3AdjustmentOp(
            op="convert",
            from_su=int(value["from_su"]),
            to_su=int(value["to_su"]),
            node_id=int(list(value.get("node_ids", []) or [-1])[0]),
            reason=str(value.get("reason", "")),
        )
        for value in conversions
        if value.get("from_su") is not None and list(value.get("node_ids", []) or [])
    ]
    plan.adjustment_bundles = [Stage3AdjustmentBundle(
        bundle_id=f"{contract.phase}:{contract.contract_id}",
        phase=str(contract.phase),
        bundle_ops=ops,
        diagnostics={"sequential_single_topology": True, "contract_id": int(contract.contract_id)},
    )]
    plan.phase_result = Stage3PhaseResult(
        active_phase=str(contract.phase),
        phase_completed=False,
        rebuild_required=bool(plan.ok),
        primary_adjustment_count=int(histogram.primary_adjustment_count),
        h_compensation_count=_h_compensation_count(histogram.groups),
        materialized_topology_count=len(list(prefix_transactions or [])) + (1 if feasible else 0),
        transactions=list(prefix_transactions or []) + [transaction],
        stop_reason=str(plan.reason),
        infeasibility_proof={} if plan.ok else {
            "contract_id": int(contract.contract_id),
            "required_counts": dict(contract.required_counts),
            "available_counts": {int(key): len(values) for key, values in pool.items()},
            "unresolved_counts": dict(histogram.unresolved_counts or _deficit(pool, contract.required_counts)),
            "slot_assignment_proof": dict(proof or {}),
        },
    )
    plan.diagnostics = {
        "stage3_phase": str(contract.phase),
        "sequential_single_topology": True,
        "current_contract": contract.to_dict(),
        "phase_result": plan.phase_result.to_dict(),
        "histogram_adjustment_plan": histogram.to_dict(),
    }
    plan.unresolved_counts = {str(key): int(value) for key, value in dict(histogram.unresolved_counts or {}).items()}
    plan.requested_counts = {str(key): int(value) for key, value in contract.required_counts.items()}
    if ledger_after_conversion is not None:
        plan.runtime["iteration_topology_ledger"] = ledger_after_conversion
    return plan


def plan_stage3_state_machine(
    *,
    phase_eval: Any,
    H: torch.Tensor,
    nodes: Sequence[Any],
    E_target: Any = None,
    max_primary_adjustments_per_stage: int = 10,
    max_h_compensation_ops_per_adjustment: int = 8,
    h_tolerance: float = 0.06,
    chain_23_limit: int = 6,
    special_degree_meta: Optional[Dict[str, Any]] = None,
    su11_rotation_index: int = 0,
) -> Stage3AdjustmentPlan:
    flex_block = str(getattr(phase_eval, "flex_reconciliation_block_reason", "") or "")
    if flex_block:
        flex_contracts = [
            dict(item or {})
            for item in list(getattr(phase_eval, "flex_reconciliation_contracts", []) or [])
            if dict(item or {})
        ]
        phase_result = Stage3PhaseResult(
            active_phase="flex_range_reconciliation",
            phase_completed=False,
            rebuild_required=False,
            stop_reason="phase_resource_infeasible",
            infeasibility_proof={
                "reason": flex_block,
                "flex_bounds": dict(getattr(phase_eval, "flex_bounds", {}) or {}),
                "flex_counts": dict(getattr(phase_eval, "flex_counts", {}) or {}),
            },
        )
        return Stage3AdjustmentPlan(
            ok=False,
            reason="phase_resource_infeasible",
            blocked_reason=flex_block,
            phase="flex_range_reconciliation",
            plan_kind="flex_range_infeasible",
            pre_assembly_gate="flex_range_reconciliation",
            active_subphase="flex_range_reconciliation",
            requirement_contracts=flex_contracts,
            pending_contracts=flex_contracts,
            phase_result=phase_result,
            diagnostics={
                "stage3_phase": "flex_range_reconciliation",
                "phase_result": phase_result.to_dict(),
            },
            runtime=_runtime_with_pool(phase_eval, _resource_ids(phase_eval)),
        )
    pool = _resource_ids(phase_eval)
    initial_pool = deepcopy(pool)
    if H is not None:
        H_work = H.detach().clone()
    else:
        size = max(40, max((int(getattr(node, "su_type", -1)) for node in nodes), default=-1) + 1)
        H_work = torch.zeros(size, dtype=torch.long)
        for node in nodes:
            su_type = int(getattr(node, "su_type", -1))
            if 0 <= su_type < size:
                H_work[su_type] += 1
    contracts = _external_contracts(phase_eval, nodes)
    iteration_ledger = _iteration_ledger_for_phase_eval(phase_eval, nodes, pool, contracts)
    completed_phases: List[str] = ["flex_range_reconciliation"]
    physical_ops: List[Dict[str, Any]] = []
    direct_transactions: List[TopologyResourceTransaction] = _assembly_product_transactions(
        phase_eval,
        H_work,
        special_degree_meta,
    )
    phase_groups: List[Any] = []
    infeasible_contracts: List[Dict[str, Any]] = []
    stale_identity_contracts: List[Dict[str, Any]] = []
    rotation_cursor = int(su11_rotation_index)
    fallback_id = 1
    for phase, role in EXTERNAL_PHASES:
        phase_adjusted = False
        phase_contracts = [item for item in contracts if str(item.get("target_role", "")) == role]
        for raw in phase_contracts:
            contract = _closure_contract(raw, phase, fallback_id)
            fallback_id += 1
            source_status, _ = contract_source_status(
                iteration_ledger,
                list(contract.claimed_source_node_ids or contract.source_node_ids or []),
            )
            if source_status == "superseded":
                continue
            blocker_reason = str(raw.get("blocker_kind", raw.get("blocked_reason", "")) or "")
            if blocker_reason and not contract.required_counts:
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "source_node_ids": list(contract.source_node_ids),
                    "required_counts": {},
                    "proof": {
                        "contract_id": int(contract.contract_id),
                        "reason": blocker_reason,
                        "source_node_ids": list(contract.source_node_ids),
                    },
                })
                continue
            next_ledger, direct_transaction, direct_proof = _ledger_transaction(
                ledger=iteration_ledger,
                contract=contract,
                phase=phase,
                H=H_work,
                special_degree_meta=special_degree_meta,
            )
            if direct_transaction is not None:
                iteration_ledger = next_ledger
                pool = deepcopy(iteration_ledger.remaining_resource_ids_by_su)
                direct_transactions.append(direct_transaction)
                continue
            if str(direct_proof.get("reason", "") or "") == "contract_identity_conflict":
                stale_proof = {
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "reason": "stale_pending_identity",
                    **dict(direct_proof or {}),
                }
                stale_identity_contracts.append(stale_proof)
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "source_node_ids": list(contract.source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "proof": dict(stale_proof),
                })
                continue
            remaining_limit = max(0, int(max_primary_adjustments_per_stage) - len(phase_groups))
            candidate_plan = _transaction_plan(
                phase_eval=phase_eval,
                contract=contract,
                pool=pool,
                H=H_work,
                E_target=E_target,
                max_primary_adjustments=remaining_limit,
                max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
                h_tolerance=h_tolerance,
                special_degree_meta=special_degree_meta,
                su11_rotation_index=rotation_cursor,
                iteration_ledger=iteration_ledger,
            )
            if not candidate_plan.ok:
                if phase_adjusted:
                    return _finalize_phase_plan(
                        phase_eval=phase_eval,
                        phase=phase,
                        H_before=H,
                        H_after=H_work,
                        initial_pool=initial_pool,
                        final_pool=pool,
                        groups=phase_groups,
                        ops=physical_ops,
                        transactions=direct_transactions,
                        rotation_before=su11_rotation_index,
                        rotation_after=rotation_cursor,
                    )
                failed_transactions = list(candidate_plan.phase_result.transactions or [])
                direct_transactions.extend(failed_transactions)
                infeasible_contracts.append({
                    "phase": str(phase),
                    "contract_id": int(contract.contract_id),
                    "target_role": str(contract.target_role),
                    "source_node_ids": list(contract.source_node_ids),
                    "required_counts": dict(contract.required_counts),
                    "proof": dict(
                        candidate_plan.phase_result.infeasibility_proof
                        or direct_proof
                        or {}
                    ),
                })
                continue
            histogram = candidate_plan.histogram_adjustment_plan
            candidate_ledger = dict(getattr(candidate_plan, "runtime", {}) or {}).get(
                "iteration_topology_ledger"
            )
            if candidate_ledger is not None:
                iteration_ledger = candidate_ledger
                pool = deepcopy(iteration_ledger.remaining_resource_ids_by_su)
            else:
                pool_after = _bind_real_node_ids(histogram, pool)
                claimed = _consume(pool_after, contract.required_counts)
                if claimed is None:
                    return candidate_plan
                pool = pool_after
            H_work = histogram.H_after.detach().clone()
            phase_groups.extend(list(histogram.groups or []))
            rotation_cursor = int(histogram.su11_rotation_index_after)
            direct_transactions.extend(list(candidate_plan.phase_result.transactions or []))
            phase_adjusted = True
            if len(phase_groups) >= int(max_primary_adjustments_per_stage):
                return _finalize_phase_plan(
                    phase_eval=phase_eval,
                    phase=phase,
                    H_before=H,
                    H_after=H_work,
                    initial_pool=initial_pool,
                    final_pool=pool,
                    groups=phase_groups,
                    ops=physical_ops,
                    transactions=direct_transactions,
                    rotation_before=su11_rotation_index,
                    rotation_after=rotation_cursor,
                )
        if phase_adjusted:
            return _finalize_phase_plan(
                phase_eval=phase_eval,
                phase=phase,
                H_before=H,
                H_after=H_work,
                initial_pool=initial_pool,
                final_pool=pool,
                groups=phase_groups,
                ops=physical_ops,
                transactions=direct_transactions,
                rotation_before=su11_rotation_index,
                rotation_after=rotation_cursor,
            )
        completed_phases.append(phase)

    flex_bounds = dict(getattr(phase_eval, "flex_bounds", {}) or {})
    flex_counts = dict(getattr(phase_eval, "flex_counts", {}) or {})
    maximum_flex_capacity = max(
        0,
        int(flex_bounds.get("max", 0) or 0) - int(flex_counts.get("final", 0) or 0),
    )
    assembly_view = dict(getattr(phase_eval, "assembly_view", {}) or {})
    flex_target_plan = dict(assembly_view.get("flex_target_plan", {}) or {})
    requested_flex_gap = max(
        int(flex_target_plan.get("remaining_minimum_gap", 0) or 0),
        int(flex_target_plan.get("remaining_preferred_gap", 0) or 0),
    )
    residual_flex_goal = min(int(maximum_flex_capacity), int(requested_flex_gap))
    direct_residual_contracts = _residual_contracts(pool, chain_23_limit, residual_flex_goal)
    direct_flex_count = 0
    for contract in direct_residual_contracts:
        next_ledger, direct_transaction, _ = _ledger_transaction(
            ledger=iteration_ledger,
            contract=contract,
            phase="residual_resource_self_closure",
            H=H_work,
            special_degree_meta=special_degree_meta,
        )
        if direct_transaction is not None:
            iteration_ledger = next_ledger
            pool = deepcopy(iteration_ledger.remaining_resource_ids_by_su)
            direct_transactions.append(direct_transaction)
            if str(dict(contract.diagnostics or {}).get("product_role", "")) == "residual_flex_chain":
                direct_flex_count += 1
            continue
        return _transaction_plan(
            phase_eval=phase_eval,
            contract=contract,
            pool=pool,
            H=H,
            E_target=E_target,
            max_primary_adjustments=max_primary_adjustments_per_stage,
            max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
            h_tolerance=h_tolerance,
            special_degree_meta=special_degree_meta,
            su11_rotation_index=su11_rotation_index,
            prefix_ops=physical_ops,
            prefix_transactions=direct_transactions,
        )
    remaining_flex_goal = max(0, int(residual_flex_goal) - int(direct_flex_count))
    residual_fit = _residual_template_fit_plan(
        phase_eval=phase_eval,
        pool=pool,
        H=H_work,
        E_target=E_target,
        h_tolerance=h_tolerance,
        special_degree_meta=special_degree_meta,
        su11_rotation_index=rotation_cursor,
        remaining_flex_chains=remaining_flex_goal,
        chain_23_limit=chain_23_limit,
        max_conversions=max_primary_adjustments_per_stage,
        max_h_compensation_ops=max_h_compensation_ops_per_adjustment,
    )
    if residual_fit is not None:
        residual_fit.phase_result.transactions = (
            list(direct_transactions)
            + list(residual_fit.phase_result.transactions or [])
        )
        return residual_fit
    completed_phases.append("residual_resource_self_closure")

    profiles = build_su24_resource_role_profiles(nodes, pool)
    residual_proof = _residual_infeasibility_proof(
        pool=pool,
        H=H_work,
        E_target=E_target,
        h_tolerance=h_tolerance,
        special_degree_meta=special_degree_meta,
        remaining_flex_chains=remaining_flex_goal,
        chain_23_limit=chain_23_limit,
        max_conversions=max_primary_adjustments_per_stage,
    )
    has_remaining_resources = any(list(values or []) for values in pool.values())
    phase_result = Stage3PhaseResult(
        active_phase="complete",
        phase_completed=True,
        rebuild_required=False,
        materialized_topology_count=len(direct_transactions),
        transactions=list(direct_transactions),
        stop_reason="stage3_complete",
        infeasibility_proof=residual_proof if has_remaining_resources else {},
    )
    return Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_complete",
        phase="complete",
        plan_kind="stage3_state_machine_complete",
        pre_assembly_gate="complete",
        active_subphase="complete",
        phase_result=phase_result,
        diagnostics={
            "stage3_phase": "complete",
            "completed_phases": completed_phases,
            "sequential_single_topology": True,
            "su24_resource_profiles": [item.to_dict() for item in profiles],
            "chain_23_limit": int(chain_23_limit),
            "infeasible_contracts": list(infeasible_contracts),
            "stale_identity_contracts": list(stale_identity_contracts),
            "residual_infeasibility_proof": residual_proof if has_remaining_resources else {},
        },
        runtime={
            **_runtime_with_pool(phase_eval, pool),
            "iteration_topology_ledger": iteration_ledger,
        },
    )
