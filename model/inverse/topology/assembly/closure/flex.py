"""Shared external flex materialization for execution and window previews."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .....shared.aromatic_plan import MAX_FLEX_WINDOW_WIDTH
from ...resources.profiles import flex_resource_sequences, select_flex_candidate_groups
from ..state import ownership
from ..state.pending import build_pending_contract, upsert_pending_contract
from . import chain


@dataclass(frozen=True)
class ExternalFlexPhaseState:
    subphases: List[str]
    physical_ok: bool
    count_ok: bool
    window_ok: bool
    pending: bool

    @property
    def ready(self) -> bool:
        return self.physical_ok and self.count_ok and self.window_ok and not self.pending


def _is_pool_contract(contract: Any) -> bool:
    return str(dict(contract or {}).get("diagnostics", {}).get("public_pending_kind", "")) == "flex_pool_supplement"


def available_flex_candidate_groups(
    assembler: Any, *, groups: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    candidate_groups = (
        list(groups) if groups is not None
        else assembler.flex_audit.get("supplemental_flex_candidate_groups", [])
    )
    return [
        dict(group)
        for group in candidate_groups
        if group.get("fragment_ids") and all(
            int(fragment_id) in assembler.view_by_id
            and ownership.fragment_available(assembler, int(fragment_id))
            for fragment_id in group["fragment_ids"]
        )
        and (
            group.get("strategy") != "b_single_closure"
            or chain.single_b_flex_closure_modes(
                assembler, assembler.view_by_id[int(group["fragment_ids"][0])],
            ) is not None
        )
    ]


def _complete_from_resource_pool(assembler: Any) -> None:
    assembler.pending_contracts = [
        contract for contract in assembler.pending_contracts if not _is_pool_contract(contract)
    ]
    missing = max(0, chain.flex_limit(assembler, "min") - chain.current_flex_product_count(assembler))
    sequences = flex_resource_sequences(
        assembler.flex_audit.get("residual_flex_preferred_23", 3),
        assembler.flex_audit.get("residual_flex_min_23", 2),
    )
    physical_failure = False
    while missing > 0 and chain.can_create_more_flex(assembler):
        available = assembler.allocator.claimable_counts()
        if int(available.get("11", 0)) < 2 or int(available.get("23", 0)) < 1:
            break
        reserve_23 = min(missing - 1, max(0, int(available.get("23", 0)) - 1))
        sequence = next(
            sequence for sequence in sequences
            if sequence.count(23) <= int(available.get("23", 0)) - reserve_23
        )
        snapshot = assembler.allocator.snapshot()
        caps = assembler.allocator.claim(11, 2)
        middle = assembler.allocator.claim(23, sequence.count(23))
        if caps is None or middle is None:
            assembler.allocator.rollback(snapshot)
            break
        node_ids = [caps[0]] + list(middle) + [caps[1]]
        product = chain.make_product(
            assembler,
            product_kind="flex_chain",
            reason="external_flex_pool_supplement",
            source_views=[],
            labels=[str(value) for value in sequence],
            node_ids=node_ids,
            branch_map={},
            branch_id_map={},
            diagnostics={"external_flex_pool_supplement": True},
        )
        if not ownership.commit_product(assembler, product, allocator_snapshot=snapshot):
            physical_failure = True
            break
        missing -= 1
    if missing <= 0:
        return
    available = assembler.allocator.claimable_counts()
    required = {"11": 2 * missing, "23": missing}
    contract_id = max(assembler.nodes_by_id, default=0) + 1_000_000
    contract = build_pending_contract(
        unit_id=contract_id,
        pending_kind="flex_pool_supplement",
        target_role="flex_chain",
        contract_kind="structural_blocker" if physical_failure else "resource_capacity",
        subphase="supplemental_flex_closure",
        source_fragment_ids=[],
        source_fragment_classes=[],
        claimed_source_node_ids=[],
        context_node_ids=[],
        mainline_slots=[
            {
                "slot_id": f"flex_pool:{contract_id}:{index}",
                "kind": "RESOURCE_PATH",
                "anchor_node_id": -1,
                "anchor_role": "resource_path",
                "required_su_sequence": [11, 23, 11],
            }
            for index in range(missing)
        ],
        branch_slots=[],
        required_counts={} if physical_failure else required,
        expected_topology="11-23-11",
        reason="external_flex_pool_materialization_failed" if physical_failure else "external_flex_pool_shortage",
        diagnostics={
            "failure_domain": "topology_materialization" if physical_failure else "resource_capacity",
            "missing_flex_count": missing,
            "raw_available_counts": assembler.allocator.available_counts(),
            "claimable_available_counts": available,
            "missing_counts": {
                key: count - int(available.get(key, 0))
                for key, count in required.items() if count > int(available.get(key, 0))
            } if not physical_failure else {},
        },
    )
    upsert_pending_contract(assembler.pending_contracts, contract)


def _physical_ok(assembler: Any) -> bool:
    rendered_ids = [
        int(node_id) for product in assembler.products for node_id in product.rendered_node_ids()
    ]
    return bool(
        not assembler.product_ownership_conflict_count
        and len(rendered_ids) == len(set(rendered_ids))
        and all(product.closed and not getattr(product, "provisional", False) for product in assembler.products)
        and all(
            int(node_id) in assembler.nodes_by_id
            and int(assembler.nodes_by_id[int(node_id)].su_type) == int(su_type)
            for product in assembler.products
            for node_id, su_type in product.expected_su_type_by_node_id.items()
        )
        and not any(
            contract.get("target_role") == "flex_chain"
            and contract.get("failure_domain") == "topology_materialization"
            for contract in assembler.pending_contracts
        )
    )


def run_external_flex_phase(
    assembler: Any, *, complete_quantity: bool = True,
) -> ExternalFlexPhaseState:
    subphases: List[str] = []
    branch_donors = chain.available_branch_donors(assembler)
    completed = set(assembler.completed_phases)
    explicit_pending = chain.has_pending_contracts_for_subphase(assembler, "explicit_branch_closure")
    if "explicit_branch_closure" not in completed or explicit_pending:
        subphases.append("explicit_branch_closure")
        chain.retry_ready_pending_contracts(assembler, branch_donors, subphase="explicit_branch_closure")
        chain.close_explicit_flex_branches_if_needed(assembler, branch_donors)
    explicit_ok = not chain.has_pending_contracts_for_subphase(assembler, "explicit_branch_closure")
    if explicit_ok:
        completed.add("explicit_branch_closure")
    enforced = chain.flex_bounds_enforced(assembler)
    pending = any(contract.get("target_role") == "flex_chain" for contract in assembler.pending_contracts)
    count = chain.current_flex_product_count(assembler)
    needs_flex = pending or "external_flex_closure" not in completed or (enforced and count < chain.flex_limit(assembler, "min"))
    phase_name = "supplemental_flex_closure" if enforced else "unbounded_external_flex_closure"
    if explicit_ok and needs_flex:
        subphases.append(phase_name)
        chain.retry_ready_pending_contracts(assembler, branch_donors, target_role="flex_chain")
        if enforced:
            available_groups = available_flex_candidate_groups(assembler)
            preferred_groups = {
                str(group.get("group_id", "")): index
                for index, group in enumerate(assembler.selected_flex_candidate_groups)
            }
            available_groups.sort(key=lambda group: preferred_groups.get(
                str(group.get("group_id", "")), len(preferred_groups),
            ))
            trials = []
            for prefer_singles in (False, True):
                trial = deepcopy(assembler)
                gap = max(0, chain.flex_limit(trial, "min") - chain.current_flex_product_count(trial))
                groups = select_flex_candidate_groups(available_groups, limit=gap, prefer_single_closures=prefer_singles)
                trial.selected_flex_candidate_groups = groups
                trial.selected_flex_candidate_ids = {
                    int(fragment_id) for group in groups for fragment_id in group["fragment_ids"]
                }
                trial.flex_audit["selected_candidate_groups"] = groups
                trial.flex_audit["selected_candidate_fragment_ids"] = sorted(trial.selected_flex_candidate_ids)
                trial_donors = chain.available_branch_donors(trial)
                chain.assemble_selected_supplemental_flex(trial, trial_donors)
                chain.release_satisfied_flex_selection(trial)
                if complete_quantity and _physical_ok(trial):
                    _complete_from_resource_pool(trial)
                    chain.release_satisfied_flex_selection(trial)
                final_count = chain.current_flex_product_count(trial)
                score = (
                    not _physical_ok(trial),
                    max(0, chain.flex_limit(trial, "min") - final_count),
                    max(0, final_count - chain.flex_limit(trial, "max")),
                    sum(contract.get("target_role") == "flex_chain" for contract in trial.pending_contracts),
                    sum(len(values) for values in trial.allocator.consumed_ids_by_su.values()),
                )
                trials.append((score, trial))
            _, chosen = min(trials, key=lambda item: item[0])
            assembler.__dict__.update(chosen.__dict__)
        else:
            chain.assemble_remaining_flex(assembler, branch_donors)
    count = chain.current_flex_product_count(assembler)
    count_ok = not enforced or chain.flex_limit(assembler, "min") <= count <= chain.flex_limit(assembler, "max")
    window_ok = not enforced or bool(
        assembler.flex_audit.get("flex_window_ok", True)
        and 0 <= int(assembler.flex_audit.get("flex_window_width", 0)) <= MAX_FLEX_WINDOW_WIDTH
    )
    pending = any(contract.get("target_role") == "flex_chain" for contract in assembler.pending_contracts)
    result = ExternalFlexPhaseState(subphases, _physical_ok(assembler) and explicit_ok, count_ok, window_ok, pending)
    if result.ready:
        completed.update({"external_flex_closure", phase_name})
    else:
        completed.discard("external_flex_closure")
        completed.discard(phase_name)
    assembler.completed_phases = completed
    assembler.flex_audit.update({
        "flex_physical_ok": result.physical_ok,
        "flex_count_ok": result.count_ok,
        "flex_window_width_ok": result.window_ok,
    })
    return result
