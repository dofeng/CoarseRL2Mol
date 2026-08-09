from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _format_ids(values: Iterable[Any]) -> str:
    items = []
    for value in list(values or []):
        try:
            items.append(str(int(value)))
        except Exception:
            items.append(str(value))
    return "[" + ", ".join(items) + "]"


def _format_text_list(values: Iterable[Any]) -> str:
    return "[" + ", ".join(str(value) for value in list(values or [])) + "]"


def _normalize_counts(value: Any) -> Dict[str, int]:
    counts = _as_dict(value)
    out: Dict[str, int] = {}
    for key, raw in counts.items():
        try:
            out[str(key)] = int(raw)
        except Exception:
            continue
    return out


def _resource_count_keys(*counts_dicts: Dict[str, int]) -> List[str]:
    keys = {
        str(key)
        for counts in list(counts_dicts or [])
        for key in dict(counts or {}).keys()
    }

    def _sort_key(value: str) -> tuple[int, Any]:
        try:
            return (0, int(value))
        except Exception:
            return (1, str(value))

    return sorted(keys, key=_sort_key)


def _format_count_map(counts: Dict[str, int], keys: Iterable[str] | None = None) -> str:
    normalized = _normalize_counts(counts)
    ordered_keys = list(keys or _resource_count_keys(normalized))
    if not ordered_keys:
        return "{}"
    return "{" + ", ".join(f"{key}: {int(normalized.get(str(key), 0) or 0)}" for key in ordered_keys) + "}"


def _sum_count_map(counts: Dict[str, int]) -> int:
    return int(sum(int(value or 0) for value in dict(counts or {}).values()))


def _group_products(products: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = {
        "Flex Chains": [],
        "Side Chains": [],
        "Fatty Rings": [],
    }
    for item in list(products or []):
        product = _as_dict(item)
        kind = str(product.get("product_kind", ""))
        if kind == "flex_chain":
            groups["Flex Chains"].append(product)
        elif kind == "side_chain":
            groups["Side Chains"].append(product)
        elif kind.startswith("fatty_ring_"):
            groups["Fatty Rings"].append(product)
    return groups


def _product_summary_lines(item: Dict[str, Any], index: int) -> List[str]:
    return [
        f"{index}. [{str(item.get('product_kind', ''))}]",
        f"   actual_su={str(item.get('actual_su', item.get('actual_topology_su', '')))}",
        f"   gid={str(item.get('gid', item.get('actual_topology_gid', '')))}",
        (
            "   source_fragment_ids=%s | source_fragment_classes=%s | source_node_ids=%s"
            % (
                _format_ids(_as_list(item.get("source_fragment_ids"))),
                _format_text_list(_as_list(item.get("source_fragment_classes"))),
                _format_ids(_as_list(item.get("source_node_ids"))),
            )
        ),
    ]


def _pending_summary_lines(item: Dict[str, Any], index: int) -> List[str]:
    return [
        f"{index}. [{str(item.get('pending_kind', ''))}]",
        f"   real={str(item.get('real', item.get('real_topology_partial', '')))}",
        f"   real_gid={str(item.get('real_gid', item.get('real_node_partial', '')))}",
    ]


def _bad_topology_summary_lines(item: Dict[str, Any], index: int) -> List[str]:
    return [
        f"{index}. [{str(item.get('product_kind', ''))}]",
        f"   actual_su={str(item.get('actual_su', item.get('actual_topology_su', '')))}",
        f"   gid={str(item.get('gid', item.get('actual_topology_gid', '')))}",
        f"   reason={str(item.get('reason', ''))}",
    ]


def _unrecognized_summary_lines(item: Dict[str, Any], index: int) -> List[str]:
    return [
        f"{index}. [unrecognized_topology]",
        f"   gid={str(item.get('global_id', item.get('node_id', '')))}",
        f"   su_type={str(item.get('su_type', item.get('subtype', '')))}",
        f"   reason={str(item.get('reason', ''))}",
    ]


def _resource_ledger_summary_lines(classification: Dict[str, Any], assembly: Dict[str, Any]) -> List[str]:
    classification = _as_dict(classification)
    assembly = _as_dict(assembly)
    ledger = _as_dict(assembly.get("resource_ledger"))
    resource_pool = _as_dict(classification.get("resource_pool"))
    initial_counts = _normalize_counts(ledger.get("initial_counts"))
    if not initial_counts:
        initial_counts = _normalize_counts(resource_pool.get("remaining_counts"))
    consumed_counts = _normalize_counts(ledger.get("consumed_counts"))
    remaining_counts = _normalize_counts(ledger.get("remaining_counts"))
    shortage_counts = {
        str(key): int(value)
        for key, value in _normalize_counts(ledger.get("shortage_counts")).items()
        if int(value) > 0
    }
    display_keys = _resource_count_keys(initial_counts, consumed_counts, remaining_counts)
    return [
        "=== Resource Ledger ===",
        "initial_total=%d | consumed_total=%d | remaining_total=%d | shortage_total=%d"
        % (
            _sum_count_map(initial_counts),
            _sum_count_map(consumed_counts),
            _sum_count_map(remaining_counts),
            _sum_count_map(shortage_counts),
        ),
        f"initial_counts={_format_count_map(initial_counts, display_keys)}",
        f"consumed_counts={_format_count_map(consumed_counts, display_keys)}",
        f"remaining_counts={_format_count_map(remaining_counts, display_keys)}",
        f"shortage_counts={_format_count_map(shortage_counts)}",
    ]


def _rebalance_summary_lines(assembly: Dict[str, Any]) -> List[str]:
    diagnostics = _as_dict(_as_dict(assembly).get("diagnostics"))
    breakdown = _as_dict(diagnostics.get("resource_request_breakdown"))
    resource_request_by_closure_tactic = _as_dict(diagnostics.get("resource_request_by_closure_tactic"))
    true_deficit_by_closure_tactic = _as_dict(diagnostics.get("true_deficit_by_closure_tactic"))
    partner_match_requests = _as_dict(diagnostics.get("partner_match_requests"))
    remaining_external_requirements_by_pending_kind = _as_dict(
        diagnostics.get("remaining_external_requirements_by_pending_kind")
    )
    if not breakdown:
        if not any([
            resource_request_by_closure_tactic,
            true_deficit_by_closure_tactic,
            partner_match_requests,
            remaining_external_requirements_by_pending_kind,
        ]):
            return []
    rebalance_trace = [_as_dict(item) for item in _as_list(breakdown.get("resource_rebalance_trace"))]
    post_counts = _normalize_counts(
        breakdown.get("post_rebalance_remaining_counts", breakdown.get("post_conversion_remaining_counts"))
    )
    if not rebalance_trace and not post_counts and not any([
        resource_request_by_closure_tactic,
        true_deficit_by_closure_tactic,
        partner_match_requests,
        remaining_external_requirements_by_pending_kind,
    ]):
        return []
    lines = ["=== Rebalance Summary ==="]
    if post_counts:
        lines.append(f"post_rebalance_remaining_counts={_format_count_map(post_counts)}")
    if rebalance_trace:
        for index, item in enumerate(rebalance_trace, start=1):
            target_su = str(item.get("target_su", ""))
            path_source = str(item.get("path_source", ""))
            path_ops = [str(value) for value in _as_list(item.get("path_ops"))]
            lines.append(
                f"{index}. target={target_su} | source={path_source} | ops={_format_text_list(path_ops)}"
            )
    if resource_request_by_closure_tactic:
        lines.append(f"resource_request_by_closure_tactic={resource_request_by_closure_tactic}")
    if true_deficit_by_closure_tactic:
        lines.append(f"true_deficit_by_closure_tactic={true_deficit_by_closure_tactic}")
    if partner_match_requests:
        lines.append(f"partner_match_requests={partner_match_requests}")
    if remaining_external_requirements_by_pending_kind:
        lines.append(
            "remaining_external_requirements_by_pending_kind="
            f"{remaining_external_requirements_by_pending_kind}"
        )
    return lines


def build_external_assembly_audit_report(
    classification: Dict[str, Any],
    assembly: Dict[str, Any],
    *,
    include_debug_appendix: bool = False,
) -> str:
    del include_debug_appendix

    classification = _as_dict(classification)
    assembly = _as_dict(assembly)
    products = [_as_dict(item) for item in _as_list(assembly.get("products"))]
    pending_units = [_as_dict(item) for item in _as_list(assembly.get("pending_units"))]
    diagnostics = _as_dict(assembly.get("diagnostics"))
    bad_topologies = [_as_dict(item) for item in _as_list(diagnostics.get("bad_topologies"))]
    classification_rejects = [_as_dict(item) for item in _as_list(classification.get("classification_rejects"))]
    unclassified_non_resource = [
        _as_dict(item)
        for item in _as_list(classification.get("unclassified_nodes"))
        if not bool(_as_dict(item).get("resource_pool_member", False))
    ]
    unrecognized_topologies = classification_rejects + unclassified_non_resource
    groups = _group_products(products)

    lines: List[str] = [
        "=== External Topology Assembly Audit ===",
        "Products=%d | Pending=%d | ok=%s | reason=%s"
        % (
            len(products),
            len(pending_units),
            str(bool(assembly.get("ok", False))),
            str(assembly.get("reason", "")),
        ),
    ]
    lines.append("")
    lines.extend(_resource_ledger_summary_lines(classification, assembly))
    rebalance_lines = _rebalance_summary_lines(assembly)
    if rebalance_lines:
        lines.append("")
        lines.extend(rebalance_lines)

    for section_name in ("Flex Chains", "Side Chains", "Fatty Rings"):
        lines.append("")
        lines.append(f"=== Complete {section_name} ===")
        items = list(groups.get(section_name, []))
        if not items:
            lines.append("(none)")
            continue
        for index, item in enumerate(items, start=1):
            lines.extend(_product_summary_lines(item, index))

    lines.append("")
    lines.append("=== Pending Topologies ===")
    if not pending_units:
        lines.append("(none)")
    else:
        for index, item in enumerate(pending_units, start=1):
            lines.extend(_pending_summary_lines(item, index))

    lines.append("")
    lines.append("=== Unrecognized Topologies ===")
    if not unrecognized_topologies:
        lines.append("(none)")
    else:
        for index, item in enumerate(unrecognized_topologies, start=1):
            lines.extend(_unrecognized_summary_lines(item, index))

    if bad_topologies:
        lines.append("")
        lines.append("=== Bad Topologies ===")
        for index, item in enumerate(bad_topologies, start=1):
            lines.extend(_bad_topology_summary_lines(item, index))

    return "\n".join(lines)
