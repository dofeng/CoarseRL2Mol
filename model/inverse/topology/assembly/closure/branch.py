"""Pure fixed-branch ownership and closure-contract rules."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ...fixed_routes import (
    FixedRouteResolution,
    resolve_fixed_contract_route as resolve_fixed_contract_route_model,
)
from ...resources.profiles import (
    alternative_closure_required_counts,
    closure_sequences,
    minimum_closure_required_counts,
)
from ..state import ownership as ownership_module
from ..core.fragment import (
    AssemblyFragmentView,
    strip_open_markers as _strip_open_markers,
)
from ..core.models import BranchClosureResult, ClosureOperatorSpec
from ..projection.engine import _safe_int
from ..core.rules import BRANCH_CLOSURE_OPERATOR_SPECS, TERMINAL_ENDPOINT_LABELS
from ..core.validation import branch_route_shape_reason as _branch_route_shape_reason


def fixed_contract_required_counts(contract: Mapping[str, Any]) -> Dict[str, int]:
    tail_kind = str(dict(contract or {}).get("tail_closure_kind", "none") or "none")
    if tail_kind != "H1":
        return {}
    return {
        str(key): int(value)
        for key, value in minimum_closure_required_counts("H1").items()
    }


def fixed_contract_nested_fragment_reason(
    contract: Mapping[str, Any],
    owner_view: Optional[Any],
    *,
    node_owner_by_id: Mapping[int, int],
    view_by_id: Mapping[int, Any],
) -> Optional[str]:
    """Reject a complete foreign branch topology nested below a branch slot."""

    if owner_view is None:
        return None
    current_fragment_id = int(owner_view.fragment_id)
    foreign_fragment_ids = {
        int(node_owner_by_id[int(node_id)])
        for node_id in list(dict(contract or {}).get("base_path_node_ids", []) or [])
        if str(node_id).lstrip("-").isdigit()
        and int(node_id) in node_owner_by_id
        and int(node_owner_by_id[int(node_id)]) != current_fragment_id
    }
    for fragment_id in sorted(foreign_fragment_ids):
        foreign_view = view_by_id.get(int(fragment_id))
        if foreign_view is None:
            continue
        contracts = [
            dict(item or {})
            for values in dict(
                getattr(foreign_view, "branch_closure_contracts_by_anchor", {})
                or {}
            ).values()
            for item in list(values or [])
        ]
        stubs = [
            stub
            for values in dict(
                getattr(foreign_view, "real_branch_stubs_by_anchor", {}) or {}
            ).values()
            for stub in list(values or [])
        ]
        if (
            len(contracts) > 1
            or stubs
            or any(
                not bool(item.get("externally_closed", False))
                or str(item.get("tail_closure_kind", "none") or "none") != "none"
                for item in contracts
            )
        ):
            return (
                "nested_branch_topology_forbidden:"
                f"parent={current_fragment_id}:child={fragment_id}"
            )
    return None


__all__ = [
    "claim_branch_from_pool",
    "claim_closure_tail_after_prefix",
    "claim_pool_sequence",
    "consume_branch_placeholder",
    "fixed_contract_flat_route_reason",
    "fixed_contract_nested_fragment_reason",
    "fixed_contract_nested_fragment_reason_for_assembler",
    "fixed_contract_required_counts",
    "materialize_fixed_branch_contract",
    "resolve_branch_closure",
    "resolve_branches",
    "resolve_fixed_branch_contract",
    "resolve_fixed_contract_route",
    "semantic_label_for_closure",
    "synchronize_branch_identity",
]


def claim_pool_sequence(
    assembler,
    *,
    closure_kind: str,
    sequence: Sequence[int],
    branch: bool,
    mode_suffix: str,
) -> Optional[BranchClosureResult]:
    snapshot = assembler.allocator.snapshot()
    pool_ids: Dict[str, List[int]] = {}
    node_ids: List[int] = []
    labels: List[str] = []
    for su_type in list(sequence or []):
        claimed = assembler.allocator.claim(int(su_type), 1)
        if claimed is None:
            assembler.allocator.rollback(snapshot)
            return None
        pool_ids.setdefault(str(int(su_type)), [])
        pool_ids[str(int(su_type))].extend(int(value) for value in claimed)
        node_ids.append(int(claimed[0]))
        labels.append(str(int(su_type)))
    topology = "-".join(labels)
    node_text = "-".join(str(value) for value in node_ids)
    return BranchClosureResult(
        branch_topology=f"-{topology}" if bool(branch) and topology else topology,
        branch_node_ids=f"-{node_text}" if bool(branch) and node_text else node_text,
        branch_kind=str(closure_kind),
        mode=f"pool:{str(closure_kind)}_{str(mode_suffix)}",
        suppressed_support_node_ids=[],
        pool_node_ids_by_su={
            str(key): [int(value) for value in list(values or [])]
            for key, values in pool_ids.items()
            if list(values or [])
        },
        main_labels=list(labels),
        main_node_ids=list(node_ids),
    )


def semantic_label_for_closure(label: str) -> Optional[int]:
    text = str(label or "")
    if text in TERMINAL_ENDPOINT_LABELS:
        return 22
    if text in {"11", "5", "6", "7", "8", "9"}:
        return 11
    if text in {
        "0",
        "2",
        "3",
        "14",
        "15",
        "17",
        "23",
        "24",
        "25",
        "single19d2",
        "single20d2",
        "double19d2",
        "double20d2",
        "single19d3",
        "single20d3",
        "double19d3",
        "double20d3",
        "21d2",
        "21d3",
    }:
        return 23
    if text.isdigit():
        value = int(text)
        if value in {1, 4, 16, 18, 22, 28, 32}:
            return 22
        if value in {0, 2, 3, 14, 15, 17, 19, 20, 21, 23, 24, 25, 27, 29, 31}:
            return 23
    return None


def branch_from_donor_view(
    assembler: Any,
    view: AssemblyFragmentView,
) -> Optional[BranchClosureResult]:
    if str(getattr(view, "class_type", "")) not in {"E", "G"}:
        return None
    # E/G donors are already complete linear branch routes.  A view that
    # owns another branch contract/stub is a topology unit in its own
    # right and cannot be recursively inserted into this branch slot.
    if (
        dict(getattr(view, "branch_closure_contracts_by_anchor", {}) or {})
        or dict(getattr(view, "real_branch_stubs_by_anchor", {}) or {})
    ):
        return None
    render = view.real_render()
    labels = [
        str(token)
        for token in _strip_open_markers(render.topology_notation).split("-")
        if str(token)
    ]
    node_ids = [
        int(token)
        for token in _strip_open_markers(render.node_id_notation).split("-")
        if str(token).isdigit()
    ]
    if not labels or len(labels) != len(node_ids):
        return None
    if bool(getattr(view, "open_right", False)) and not bool(getattr(view, "open_left", False)):
        labels = list(reversed(labels))
        node_ids = list(reversed(node_ids))
    elif not bool(getattr(view, "open_left", False)):
        return None
    semantic = [semantic_label_for_closure(label) for label in labels]
    if any(value is None for value in semantic):
        return None
    normalized = [int(value) for value in semantic if value is not None]
    if normalized not in closure_sequences("H"):
        return None
    topology = "-".join(labels)
    node_text = "-".join(str(node_id) for node_id in node_ids)
    route_ids = set(int(node_id) for node_id in node_ids)
    shape_reason = _branch_route_shape_reason(
        topology,
        node_text,
        assembler.nodes_by_id,
    )
    if shape_reason is not None:
        return None
    owner_map = dict(
        getattr(assembler, "fixed_node_owner_by_id", {}) or {}
    )
    recycled_node_ids = {
        int(value)
        for value in list(getattr(assembler, "recycled_node_ids", set()) or set())
    }
    source_fragment_ids = {
        int(view.fragment_id),
        *(
            int(owner_map[int(node_id)])
            for node_id in route_ids
            if int(node_id) in owner_map
            and int(node_id) not in recycled_node_ids
        ),
    }
    return BranchClosureResult(
        branch_topology=f"-{topology}" if topology else "",
        branch_node_ids=f"-{node_text}" if node_text else "",
        branch_kind="H",
        mode=f"fragment:{str(view.class_type)}_as_H",
        anchor_index=int(view.branch_anchor_index),
        suppressed_support_node_ids=list(render.suppressed_support_node_ids or []),
        source_fragment_ids=sorted(source_fragment_ids),
        source_node_ids=sorted({
            int(value) for value in list(view.source_node_ids or [])
        } | route_ids),
        fixed_source_node_ids=sorted(
            ownership_module.fixed_branch_route_ids(assembler, route_ids)
        ),
        rendered_node_ids=sorted(route_ids),
        pool_node_ids_by_su={},
        main_labels=list(labels),
        main_node_ids=list(node_ids),
    )


def claim_branch_from_pool(
    assembler: Any,
    branch_kind: str,
) -> Optional[BranchClosureResult]:
    for index, sequence in enumerate(closure_sequences(str(branch_kind))):
        result = claim_pool_sequence(assembler, 
            closure_kind=str(branch_kind),
            sequence=sequence,
            branch=True,
            mode_suffix="short" if index == 0 else "long",
        )
        if result is not None:
            return result
    return None


def claim_closure_tail_after_prefix(
    assembler,
    closure_kind: str,
    prefix_sequence: Sequence[int],
) -> Optional[BranchClosureResult]:
    prefix = [int(value) for value in list(prefix_sequence or [])]
    for index, sequence in enumerate(closure_sequences(str(closure_kind))):
        normalized = [int(value) for value in list(sequence or [])]
        if normalized[:len(prefix)] != prefix:
            continue
        suffix = normalized[len(prefix):]
        if not suffix:
            return BranchClosureResult(
                branch_kind=str(closure_kind),
                mode=f"existing_prefix:{str(closure_kind)}",
                main_labels=[],
                main_node_ids=[],
            )
        result = claim_pool_sequence(assembler, 
            closure_kind=str(closure_kind),
            sequence=suffix,
            branch=False,
            mode_suffix=f"tail_{'short' if index == 0 else 'long'}",
        )
        if result is not None:
            return result
    return None


def resolve_branch_closure(
    assembler,
    branch_kind: str,
    branch_donors: List[AssemblyFragmentView],
    *,
    anchor_index: int,
    attach_role: str = "center_branch",
    target_path: str = "center_side_branch",
) -> Optional[BranchClosureResult]:
    operator = BRANCH_CLOSURE_OPERATOR_SPECS.get(str(branch_kind), ClosureOperatorSpec(name=str(branch_kind), scope="branch"))
    if operator.name == "H":
        for donor in list(branch_donors or []):
            if not ownership_module.fragment_available(
                assembler, int(donor.fragment_id)
            ):
                continue
            result = branch_from_donor_view(assembler, donor)
            if result is None:
                continue
            branch_donors[:] = [
                item
                for item in list(branch_donors or [])
                if int(item.fragment_id) != int(donor.fragment_id)
            ]
            result.anchor_index = int(anchor_index)
            result.attach_role = str(attach_role)
            result.target_path = str(target_path)
            return result
    result = claim_branch_from_pool(assembler, branch_kind)
    if result is not None:
        result.anchor_index = int(anchor_index)
        result.attach_role = str(attach_role)
        result.target_path = str(target_path)
        return result
    return None


def consume_branch_placeholder(
    assembler,
    branch_kind: str,
    branch_donors: List[AssemblyFragmentView],
    *,
    anchor_index: int,
    attach_role: str = "center_branch",
    target_path: str = "center_side_branch",
) -> Optional[BranchClosureResult]:
    return resolve_branch_closure(assembler, 
        str(branch_kind),
        branch_donors,
        anchor_index=int(anchor_index),
        attach_role=str(attach_role),
        target_path=str(target_path),
    )


def fixed_contract_nested_fragment_reason_for_assembler(
    assembler,
    contract: Dict[str, Any],
    owner_view: Optional[AssemblyFragmentView],
) -> Optional[str]:
    return fixed_contract_nested_fragment_reason(
        contract,
        owner_view,
        node_owner_by_id=dict(assembler.fixed_node_owner_by_id or {}),
        view_by_id=dict(assembler.view_by_id or {}),
    )


def fixed_contract_flat_route_reason(
    assembler,
    contract: Dict[str, Any],
    owner_view: Optional[AssemblyFragmentView],
) -> Optional[str]:
    resolution = resolve_fixed_contract_route(assembler, contract, owner_view)
    return str(resolution.reason) if not resolution.ok else None


def resolve_fixed_contract_route(
    assembler,
    contract: Dict[str, Any],
    owner_view: Optional[AssemblyFragmentView] = None,
) -> FixedRouteResolution:
    return resolve_fixed_contract_route_model(
        dict(contract or {}),
        nodes_by_id=assembler.nodes_by_id,
        mainline_node_ids=[
            int(value)
            for value in list(getattr(owner_view, "main_node_ids", []) or [])
            if value is not None and _safe_int(value, -1) >= 0
        ],
        node_owner_by_id=dict(
            getattr(assembler, "fixed_node_owner_by_id", {}) or {}
        ),
    )


def materialize_fixed_branch_contract(
    assembler,
    contract: Dict[str, Any],
    *,
    anchor_index: int,
    owner_view: Optional[AssemblyFragmentView] = None,
    tail_result: Optional[BranchClosureResult] = None,
    trailing_open: bool = False,
    route_resolution: Optional[FixedRouteResolution] = None,
) -> BranchClosureResult:
    contract_local = dict(contract or {})
    resolution = route_resolution or resolve_fixed_contract_route(assembler, 
        contract_local,
        owner_view,
    )
    if not resolution.ok:
        raise ValueError(
            "cannot materialize invalid fixed branch contract: "
            f"{resolution.reason}"
        )
    base_labels = list(resolution.labels)
    base_node_ids = list(resolution.node_ids)
    all_labels = list(base_labels)
    all_node_ids = list(base_node_ids)
    pool_ids_by_su: Dict[str, List[int]] = {}
    mode = "fixed_contract:closed"
    if tail_result is not None:
        from .chain import merge_pool_ids

        all_labels.extend(str(label) for label in list(tail_result.main_labels or []) if str(label))
        all_node_ids.extend(int(node_id) for node_id in list(tail_result.main_node_ids or []) if node_id is not None)
        pool_ids_by_su = merge_pool_ids(
            [dict(tail_result.pool_node_ids_by_su or {})]
        )
        mode = f"fixed_contract:{str(contract_local.get('tail_closure_kind', 'tail'))}"
    branch_topology = ""
    branch_node_ids = ""
    rendered_labels = list(base_labels)
    rendered_node_texts = [str(node_id) for node_id in base_node_ids]
    if tail_result is not None:
        rendered_labels.extend(
            str(label) for label in list(tail_result.main_labels or []) if str(label)
        )
        rendered_node_texts.extend(
            str(node_id) for node_id in list(tail_result.main_node_ids or [])
            if node_id is not None
        )
    if rendered_labels:
        branch_topology = "-" + "-".join(str(label) for label in rendered_labels)
    if rendered_node_texts:
        branch_node_ids = "-" + "-".join(str(node_id) for node_id in rendered_node_texts)
    if trailing_open:
        branch_topology += "-"
        branch_node_ids += "-"
        mode = "fixed_contract:open_boundary"
    rendered_route_ids = set(int(value) for value in base_node_ids)
    rendered_route_ids.update(int(value) for value in all_node_ids)
    if tail_result is not None:
        rendered_route_ids.update(
            ownership_module.branch_result_route_ids(assembler, tail_result)
        )
    source_fragment_ids = sorted({
        int(assembler.fragment_owner_by_node_id[int(node_id)])
        for node_id in rendered_route_ids
        if int(node_id) in assembler.fragment_owner_by_node_id
        and int(node_id) not in {
            int(value)
            for value in list(getattr(assembler, "recycled_node_ids", set()) or set())
        }
    })
    return BranchClosureResult(
        branch_topology=str(branch_topology),
        branch_node_ids=str(branch_node_ids),
        branch_kind="actual_fixed_connection_contract",
        mode=str(mode),
        anchor_index=int(anchor_index),
        suppressed_support_node_ids=[],
        source_fragment_ids=source_fragment_ids,
        source_node_ids=sorted(rendered_route_ids),
        fixed_source_node_ids=sorted(
            ownership_module.fixed_branch_route_ids(
                assembler, rendered_route_ids
            )
        ),
        rendered_node_ids=sorted(rendered_route_ids),
        pool_node_ids_by_su=pool_ids_by_su,
        main_labels=list(all_labels),
        main_node_ids=[int(node_id) for node_id in all_node_ids],
    )


def resolve_fixed_branch_contract(
    assembler,
    contract: Dict[str, Any],
    branch_donors: List[AssemblyFragmentView],
    *,
    anchor_index: int,
    owner_view: Optional[AssemblyFragmentView] = None,
    route_resolution: Optional[FixedRouteResolution] = None,
) -> Tuple[Optional[BranchClosureResult], Dict[str, int], str]:
    contract_local = dict(contract or {})
    route_resolution = route_resolution or resolve_fixed_contract_route(assembler, 
        contract_local,
        owner_view,
    )
    if not route_resolution.ok:
        return None, {}, str(route_resolution.reason)
    nested_reason = fixed_contract_nested_fragment_reason_for_assembler(
        assembler,
        contract_local,
        owner_view,
    )
    if nested_reason is not None:
        return None, {}, str(nested_reason)
    role_candidates = {
        str(value)
        for value in list(contract_local.get("role_candidates", []) or [])
        if str(value)
    }
    if "mainline_only" in role_candidates:
        return None, {}, "fixed_branch_role_conflict"
    if bool(contract_local.get("externally_closed", False)):
        return materialize_fixed_branch_contract(assembler, 
            contract_local,
            anchor_index=int(anchor_index),
            owner_view=owner_view,
            route_resolution=route_resolution,
        ), {}, ""
    tail_kind = str(contract_local.get("tail_closure_kind", "none") or "none")
    if tail_kind == "H1":
        required_counts = fixed_contract_required_counts(contract_local)
        if route_resolution.terminal_node_id is not None:
            # A fixed terminal (SU22/28/etc.) is already the branch cap.
            # Appending H1 would create the forbidden route
            # ``...-28-23-22`` and give the terminal two connections.
            return (
                materialize_fixed_branch_contract(assembler, 
                    contract_local,
                    anchor_index=int(anchor_index),
                    owner_view=owner_view,
                    route_resolution=route_resolution,
                ),
                {},
                "",
            )
        tail_result = resolve_branch_closure(assembler, 
            "H1",
            branch_donors,
            anchor_index=int(anchor_index),
            attach_role="fixed_connection_branch",
            target_path="fixed_connection_branch",
        )
        if tail_result is None:
            return None, dict(required_counts or {}), "fixed_branch_H1_missing"
        return (
            materialize_fixed_branch_contract(assembler, 
                contract_local,
                anchor_index=int(anchor_index),
                owner_view=owner_view,
                tail_result=tail_result,
                route_resolution=route_resolution,
            ),
            dict(required_counts or {}),
            "",
        )
    if tail_kind == "H_AFTER_PREFIX":
        prefix_sequence = [
            int(value)
            for value in list(contract_local.get("base_path_effective_sequence", []) or [])
        ]
        tail_result = claim_closure_tail_after_prefix(assembler, "H", prefix_sequence)
        if tail_result is None:
            required_options = []
            for sequence in closure_sequences("H"):
                normalized = [int(value) for value in list(sequence or [])]
                if normalized[:len(prefix_sequence)] != prefix_sequence:
                    continue
                required_options.append(Counter(normalized[len(prefix_sequence):]))
            required_counts = {
                str(key): int(value)
                for key, value in dict(required_options[0] if required_options else {}).items()
                if int(value) > 0
            }
            return None, required_counts, "raw_branch_H_tail_missing"
        return (
            materialize_fixed_branch_contract(assembler, 
                contract_local,
                anchor_index=int(anchor_index),
                owner_view=owner_view,
                tail_result=tail_result,
                route_resolution=route_resolution,
            ),
            {
                str(key): len(list(values or []))
                for key, values in dict(
                    tail_result.pool_node_ids_by_su or {}
                ).items()
                if list(values or [])
            },
            "",
        )
    if tail_kind == "paired_like24":
        partner_spec = dict(contract_local.get("partner_spec", {}) or {})
        if bool(contract_local.get("paired_like24_resolved", False)) or bool(partner_spec.get("resolved", False)):
            return materialize_fixed_branch_contract(assembler, 
                contract_local,
                anchor_index=int(anchor_index),
                owner_view=owner_view,
                route_resolution=route_resolution,
            ), {}, ""
        return None, {}, "fixed_branch_partner_missing"
    if tail_kind == "OPEN_BOUNDARY":
        return materialize_fixed_branch_contract(assembler, 
            contract_local,
            anchor_index=int(anchor_index),
            owner_view=owner_view,
            trailing_open=True,
            route_resolution=route_resolution,
        ), {}, ""
    if tail_kind == "STRUCTURAL_BOUNDARY":
        boundary = dict(contract_local.get("structural_boundary", {}) or {})
        reason = str(boundary.get("reason", "") or "real_branch_structural_boundary")
        return None, {}, str(reason)
    return None, {}, "external_branch_closure_pending"


def resolve_branches(
    assembler,
    view: AssemblyFragmentView,
    branch_donors: List[AssemblyFragmentView],
    *,
    branch_kind_map_override: Optional[Dict[int, List[str]]] = None,
    branch_contract_map_override: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Tuple[Optional[List[BranchClosureResult]], Dict[str, int], Dict[str, Any]]:
    snapshot = assembler.allocator.snapshot()
    branch_stub_map = dict(view.real_branch_stubs_map() or {})
    branch_kind_map = dict(branch_kind_map_override or view.real_branch_kinds_by_anchor() or {})
    branch_contract_map = dict(branch_contract_map_override or view.fixed_branch_contracts_by_anchor() or {})
    branch_closures: List[BranchClosureResult] = []
    required = Counter()
    diagnostics = {
        "branch_modes": [],
        "branch_contract_blockers": {},
        "branch_contract_blocker_details": {},
        "alternative_required_counts": {},
    }
    for anchor in sorted(int(value) for value in branch_contract_map.keys()):
        for contract in list(branch_contract_map.get(int(anchor), []) or []):
            contract_local = dict(contract or {})
            route_resolution = resolve_fixed_contract_route(assembler, 
                contract_local,
                view,
            )
            contract_result, contract_required, blocker = resolve_fixed_branch_contract(assembler, 
                contract_local,
                branch_donors,
                anchor_index=int(anchor),
                owner_view=view,
                route_resolution=route_resolution,
            )
            for key, value in dict(contract_required or {}).items():
                required[str(key)] += int(value)
            if contract_result is None:
                assembler.allocator.rollback(snapshot)
                if str(blocker):
                    diagnostics["branch_contract_blockers"].setdefault(str(anchor), [])
                    diagnostics["branch_contract_blockers"][str(anchor)].append(str(blocker))
                    blocker_detail = {
                        "reason": str(blocker),
                        "failure_domain": (
                            str(route_resolution.failure_domain)
                            if not route_resolution.ok
                            else (
                                "topology_materialization"
                                if (
                                    str(blocker).startswith("nested_branch_topology_forbidden")
                                    or str(blocker).startswith("branch_terminal_not_at_path_end")
                                    or str(blocker).startswith("terminal_has_fixed_continuation")
                                    or str(blocker) in {
                                        "fixed_branch_role_conflict",
                                        "real_branch_structural_boundary",
                                    }
                                )
                                else ""
                            )
                        ),
                        **dict(route_resolution.diagnostics or {}),
                    }
                    structural_boundary = dict(
                        contract_local.get("structural_boundary", {}) or {}
                    )
                    if structural_boundary:
                        blocker_detail["structural_boundary"] = structural_boundary
                    diagnostics["branch_contract_blocker_details"].setdefault(
                        str(anchor), []
                    )
                    diagnostics["branch_contract_blocker_details"][str(anchor)].append(
                        blocker_detail
                    )
                diagnostics["unresolved_branch_contract_count"] = int(sum(
                    len(list(values or []))
                    for values in dict(diagnostics.get("branch_contract_blockers", {}) or {}).values()
                ))
                return None, {str(k): int(v) for k, v in required.items()}, diagnostics
            branch_conflict = ownership_module.preflight_branch_result(
                assembler, contract_result
            )
            if branch_conflict is not None:
                assembler.allocator.rollback(snapshot)
                diagnostics["branch_identity_blockers"] = [branch_conflict]
                diagnostics["unresolved_branch_contract_count"] = 1
                return None, {str(k): int(v) for k, v in required.items()}, diagnostics
            branch_closures.append(contract_result)
            diagnostics["branch_modes"].append(str(contract_result.mode))
    all_anchors = sorted({
        int(anchor)
        for anchor in set(branch_stub_map.keys()) | set(branch_kind_map.keys()) | set(branch_contract_map.keys())
    })
    for anchor in all_anchors:
        for branch_kind in list(branch_kind_map.get(int(anchor), []) or []):
            min_required = minimum_closure_required_counts(str(branch_kind))
            alternatives = alternative_closure_required_counts(str(branch_kind))
            if alternatives:
                diagnostics["alternative_required_counts"][f"{int(anchor)}:{str(branch_kind)}"] = [
                    {str(key): int(value) for key, value in dict(option or {}).items()}
                    for option in alternatives
                ]
            branch_result = consume_branch_placeholder(assembler, 
                str(branch_kind),
                branch_donors,
                anchor_index=int(anchor),
            )
            if branch_result is None:
                for key, value in dict(min_required or {}).items():
                    required[str(key)] += int(value)
                assembler.allocator.rollback(snapshot)
                return None, {str(k): int(v) for k, v in required.items()}, diagnostics
            branch_conflict = ownership_module.preflight_branch_result(
                assembler, branch_result
            )
            if branch_conflict is not None:
                assembler.allocator.rollback(snapshot)
                diagnostics["branch_identity_blockers"] = [branch_conflict]
                diagnostics["unresolved_branch_contract_count"] = 1
                return None, {str(k): int(v) for k, v in required.items()}, diagnostics
            for key, values in dict(branch_result.pool_node_ids_by_su or {}).items():
                if list(values or []):
                    required[str(key)] += len(list(values or []))
            branch_closures.append(branch_result)
            diagnostics["branch_modes"].append(str(branch_result.mode))
        for actual_stub in list(branch_stub_map.get(int(anchor), []) or []):
            actual_result = BranchClosureResult(
                branch_topology=str(actual_stub.topology_notation),
                branch_node_ids=str(actual_stub.node_id_notation),
                branch_kind=str(actual_stub.branch_kind),
                mode=str(actual_stub.mode),
                anchor_index=int(anchor),
                suppressed_support_node_ids=[],
                source_fragment_ids=list(actual_stub.source_fragment_ids or []),
                source_node_ids=list(actual_stub.source_node_ids or []),
                pool_node_ids_by_su={},
                main_labels=[str(actual_stub.topology_notation)],
                main_node_ids=[int(actual_stub.node_id_notation)] if str(actual_stub.node_id_notation).isdigit() else [],
            )
            branch_conflict = ownership_module.preflight_branch_result(
                assembler, actual_result
            )
            if branch_conflict is not None:
                assembler.allocator.rollback(snapshot)
                diagnostics["branch_identity_blockers"] = [branch_conflict]
                diagnostics["unresolved_branch_contract_count"] = 1
                return None, {str(k): int(v) for k, v in required.items()}, diagnostics
            branch_closures.append(actual_result)
    diagnostics["unresolved_branch_contract_count"] = 0
    synchronize_branch_identity(assembler, branch_closures)
    return list(branch_closures), {str(k): int(v) for k, v in required.items()}, diagnostics


def synchronize_branch_identity(
    assembler,
    branch_results: Sequence[BranchClosureResult],
) -> None:
    """Synchronize donor fragments with the actually rendered branch route."""
    owner_map = dict(
        getattr(assembler, "fixed_node_owner_by_id", {}) or {}
    )
    for result in list(branch_results or []):
        route_ids = ownership_module.branch_result_route_ids(assembler, result)
        result.rendered_node_ids = sorted(route_ids)
        result.fixed_source_node_ids = sorted(
            ownership_module.fixed_branch_route_ids(assembler, route_ids)
        )
        result.source_node_ids = sorted({
            int(value) for value in list(result.source_node_ids or [])
        } | route_ids)
        inferred_fragment_ids = {
            int(owner_map[int(node_id)])
            for node_id in route_ids
            if int(node_id) in owner_map
            and int(node_id) not in {
                int(value)
                for value in list(getattr(assembler, "recycled_node_ids", set()) or set())
            }
        }
        result.source_fragment_ids = sorted({
            int(value) for value in list(result.source_fragment_ids or [])
        } | inferred_fragment_ids)
