"""Export the inverse graph as an immutable placement contract.

The inverse pipeline owns node identity and topology.  This adapter only
serializes that already-decided graph into placement units and cross-unit
contracts; it does not allocate resources or perform substitution.
"""

from __future__ import annotations

from collections import deque
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


AROMATIC_SU_TYPES = frozenset(range(5, 14))


def _dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_public_dict"):
        try:
            return dict(value.to_public_dict() or {})
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict() or {})
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _node_id(row: Any) -> int:
    data = _dict(row)
    return _int(data.get("node_id", data.get("global_id", -1)))


def _path_node_ids(value: Any) -> List[int]:
    result: List[int] = []
    for row in list(value or []):
        data = _dict(row)
        if str(data.get("kind", "node")) != "node":
            continue
        node_id = _node_id(data)
        if node_id >= 0 and node_id not in result:
            result.append(node_id)
    return result


def _product_node_ids(product: Mapping[str, Any]) -> List[int]:
    result: List[int] = []
    for key in ("main_path", "actual_main_path"):
        for node_id in _path_node_ids(product.get(key, [])):
            if node_id not in result:
                result.append(node_id)
    for branch in list(product.get("branches", []) or []):
        for node_id in _path_node_ids(branch):
            if node_id not in result:
                result.append(node_id)
    for key in ("source_node_ids", "fixed_source_node_ids", "resource_node_ids"):
        for value in list(product.get(key, []) or []):
            node_id = _int(value)
            if node_id >= 0 and node_id not in result:
                result.append(node_id)
    if not result:
        for key in ("actual_topology_gid", "actual_main_gid", "node_id_notation"):
            for value in re.findall(r"#?(\d+)", str(product.get(key, "") or "")):
                node_id = _int(value)
                if node_id >= 0 and node_id not in result:
                    result.append(node_id)
    for values in dict(product.get("claimed_resource_ids_by_su", {}) or {}).values():
        for value in list(values or []):
            node_id = _int(value)
            if node_id >= 0 and node_id not in result:
                result.append(node_id)
    return result


def _node_rows(nodes: Sequence[Any]) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for raw in list(nodes or []):
        data = _dict(raw)
        node_id = _node_id(data)
        if node_id < 0:
            continue
        result[node_id] = {
            "node_id": node_id,
            "su_type": _int(data.get("su_type", -1)),
            "hop1_ids": sorted({_int(value) for value in list(data.get("hop1_ids", []) or []) if _int(value) >= 0}),
            "fixed_hop1_ids": sorted({_int(value) for value in list(data.get("fixed_hop1_ids", []) or []) if _int(value) >= 0}),
            "role": str(data.get("role", data.get("su_name", "")) or ""),
        }
    return result


def _aromatic_components(node_rows: Mapping[int, Mapping[str, Any]]) -> List[List[int]]:
    aromatic = {
        int(node_id)
        for node_id, row in node_rows.items()
        if int(row.get("su_type", -1)) in AROMATIC_SU_TYPES
    }
    adjacency = {
        node_id: sorted(
            int(neighbor)
            for neighbor in list(node_rows[node_id].get("hop1_ids", []) or [])
            if int(neighbor) in aromatic
        )
        for node_id in aromatic
    }
    components: List[List[int]] = []
    visited: Set[int] = set()
    for root in sorted(aromatic):
        if root in visited:
            continue
        queue = deque([root])
        visited.add(root)
        component: List[int] = []
        while queue:
            node_id = queue.popleft()
            component.append(node_id)
            for neighbor in adjacency.get(node_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _linear_axials(node_ids: Iterable[int]) -> Dict[int, List[int]]:
    """Assign stable, collision-free axial coordinates within one unit.

    The inverse graph remains the authority for edges.  Local coordinates are
    only a rigid embedding seed for MCTS, so a deterministic axial row is
    preferable to inventing a new topology or changing node identity.
    """
    return {int(node_id): [index, 0] for index, node_id in enumerate(node_ids)}


def _phase(product_type: str) -> str:
    kind = str(product_type or "").lower()
    if "flex" in kind or "ring" in kind:
        return "flex"
    if "side" in kind:
        return "side"
    return "side"


def _path_edges(product: Mapping[str, Any], allowed: Set[int]) -> List[List[int]]:
    paths: List[List[int]] = []
    main = [node_id for node_id in _path_node_ids(product.get("main_path", [])) if node_id in allowed]
    if len(main) > 1:
        paths.append(main)
    for branch in list(product.get("branches", []) or []):
        path = [node_id for node_id in _path_node_ids(branch) if node_id in allowed]
        if len(path) > 1:
            paths.append(path)
    return paths


def build_rl_mtcs_payload(
    nodes: Sequence[Any],
    topology_result: Any,
    *,
    include_unclassified: bool = False,
) -> Dict[str, Any]:
    """Create the complete inverse-to-placement JSON contract.

    ``topology_result`` is the dictionary returned by
    :func:`audit_stage3_topology` or an equivalent serializable audit.
    """
    node_rows = _node_rows(nodes)
    audit = _dict(topology_result)
    classification = _dict(audit.get("classification", {}))
    products = list(audit.get("products", []) or [])
    if isinstance(audit.get("products"), Mapping):
        products = list(dict(audit.get("products", {}) or {}).values())
    if not products:
        nested_products = _dict(audit.get("external_assembly", {})).get("products", [])
        products = list(nested_products.values()) if isinstance(nested_products, Mapping) else list(nested_products or [])

    components = _aromatic_components(node_rows)
    aromatic_owner: Dict[int, str] = {}
    aromatic_units: List[Dict[str, Any]] = []
    for index, component in enumerate(components):
        unit_id = f"aromatic:{index}"
        for node_id in component:
            aromatic_owner[node_id] = unit_id
        coords = _linear_axials(component)
        edges = [
            [node_id, neighbor]
            for node_id in component
            for neighbor in list(node_rows[node_id].get("hop1_ids", []) or [])
            if neighbor in set(component) and node_id < neighbor
        ]
        aromatic_units.append({
            "unit_id": unit_id,
            "kind": "aromatic_cluster",
            "product_type": "aromatic_cluster",
            "nodes": [
                {
                    "node_id": node_id,
                    "su_type": int(node_rows[node_id]["su_type"]),
                    "local_axial": coords[node_id],
                    "role": str(node_rows[node_id].get("role", "aromatic")),
                }
                for node_id in component
            ],
            "internal_edges": edges,
            "ports": [],
            "closed": True,
        })

    owner: Dict[int, str] = dict(aromatic_owner)
    topology_units: List[Dict[str, Any]] = []
    product_owner: Dict[int, str] = {}
    for index, raw in enumerate(products):
        product = _dict(raw)
        product_id = str(product.get("product_id", index))
        unit_id = f"topology:{product_id}"
        node_ids = [node_id for node_id in _product_node_ids(product) if node_id in node_rows and node_id not in aromatic_owner]
        if not node_ids:
            continue
        if any(node_id in product_owner for node_id in node_ids):
            overlap = sorted(node_id for node_id in node_ids if node_id in product_owner)
            raise ValueError(f"inverse products overlap on node IDs: {overlap[:12]}")
        product_owner.update({node_id: unit_id for node_id in node_ids})
        owner.update({node_id: unit_id for node_id in node_ids})
        coords = _linear_axials(node_ids)
        edges: List[List[int]] = []
        seen_edges: Set[Tuple[int, int]] = set()
        for path in _path_edges(product, set(node_ids)):
            for left, right in zip(path, path[1:]):
                key = (min(left, right), max(left, right))
                if key not in seen_edges:
                    edges.append([left, right])
                    seen_edges.add(key)
        topology_units.append({
            "unit_id": unit_id,
            "product_type": str(product.get("product_type", product.get("product_kind", "side_chain")) or "side_chain"),
            "nodes": [
                {
                    "node_id": node_id,
                    "su_type": int(node_rows[node_id]["su_type"]),
                    "local_axial": coords[node_id],
                    "role": str(node_rows[node_id].get("role", "")),
                }
                for node_id in node_ids
            ],
            "internal_edges": edges,
            "ports": [],
            "source_fragment_ids": [int(value) for value in list(product.get("source_fragment_ids", []) or [])],
            "topology_notation": str(product.get("actual_topology_su", product.get("topology_notation", "")) or ""),
            "node_id_notation": str(product.get("actual_topology_gid", product.get("node_id_notation", "")) or ""),
            "closed": bool(product.get("closed", True)),
        })

    # Add support units only when explicitly requested.  Silent ownership
    # completion would hide an incomplete inverse result from RL_init.
    missing = sorted(set(node_rows) - set(owner))
    if missing and include_unclassified:
        for node_id in missing:
            unit_id = f"support:{node_id}"
            owner[node_id] = unit_id
            topology_units.append({
                "unit_id": unit_id,
                "product_type": "side_support",
                "nodes": [{"node_id": node_id, "su_type": int(node_rows[node_id]["su_type"]), "local_axial": [0, 0]}],
                "internal_edges": [],
                "ports": [],
                "closed": True,
            })

    units_by_id = {unit["unit_id"]: unit for unit in aromatic_units + topology_units}
    port_index: Dict[Tuple[str, int, int], str] = {}
    contracts: List[Dict[str, Any]] = []
    seen_contracts: Set[Tuple[int, int]] = set()

    # Graph hop1 edges are authoritative for cross-unit connections.  Product
    # paths are included as a fallback because some inverse exports omit the
    # adjacency cache for materialized products.
    graph_edges: Set[Tuple[int, int]] = set()
    for node_id, row in node_rows.items():
        for neighbor in list(row.get("hop1_ids", []) or []):
            if neighbor in node_rows and node_id != neighbor:
                graph_edges.add((min(node_id, neighbor), max(node_id, neighbor)))
    for product in products:
        path_values = [list(product.get("main_path", []) or [])] + list(product.get("branches", []) or [])
        for path in path_values:
            ids = _path_node_ids(path)
            graph_edges.update((min(left, right), max(left, right)) for left, right in zip(ids, ids[1:]))

    def ensure_port(unit_id: str, node_id: int, neighbor_id: int) -> str:
        key = (unit_id, int(node_id), int(neighbor_id))
        if key in port_index:
            return port_index[key]
        port_id = f"{unit_id}:port:{node_id}:{neighbor_id}"
        units_by_id[unit_id].setdefault("ports", []).append({
            "port_id": port_id,
            "node_id": int(node_id),
            "role": "external",
            "phase": "",
        })
        port_index[key] = port_id
        return port_id

    for left, right in sorted(graph_edges):
        left_unit, right_unit = owner.get(left, ""), owner.get(right, "")
        if not left_unit or not right_unit or left_unit == right_unit:
            continue
        pair = (left, right)
        if pair in seen_contracts:
            continue
        seen_contracts.add(pair)
        left_phase = "aromatic" if left_unit.startswith("aromatic:") else _phase(units_by_id[left_unit].get("product_type", ""))
        right_phase = "aromatic" if right_unit.startswith("aromatic:") else _phase(units_by_id[right_unit].get("product_type", ""))
        kind = "rigid" if left_phase == right_phase == "aromatic" else "flex" if "flex" in {left_phase, right_phase} else "side"
        contracts.append({
            "contract_id": f"edge:{left}-{right}",
            "source_unit_id": left_unit,
            "source_node_id": left,
            "source_port_id": ensure_port(left_unit, left, right),
            "target_unit_id": right_unit,
            "target_node_id": right,
            "target_port_id": ensure_port(right_unit, right, left),
            "kind": kind,
            "mandatory": True,
        })

    # Keep formal ownership/audit information beside the immutable units.
    ownership_ledger = _dict(audit.get("ownership_ledger", {}))
    ownership_states = _dict(ownership_ledger.get("states", {}))
    support_context_ids = sorted(
        {
            _int(value)
            for value in list(ownership_states.get("support_context", []) or [])
            if _int(value) >= 0
        }
    )
    if support_context_ids:
        ownership_ledger["support_context_node_ids"] = support_context_ids
    ownership_ledger["unit_by_node_id"] = {
        str(node_id): str(unit_id)
        for node_id, unit_id in sorted(owner.items())
    }
    return {
        "aromatic_units": aromatic_units,
        "topology_products": topology_units,
        "topology_units": topology_units,
        "planned_topology": topology_units,
        "external_contracts": contracts,
        "ownership_ledger": ownership_ledger,
        "nodes": list(node_rows.values()),
        "pending_units": list(audit.get("pending_contracts", audit.get("pending_units", [])) or []),
        "classification": classification,
        "audit": audit,
        "ok": bool(audit.get("ok", False)),
        "topology_assembly_ok": bool(audit.get("ok", False)),
        "diagnostics": {
            "exporter": "inverse_graph_to_rl_mtcs",
            "aromatic_unit_count": len(aromatic_units),
            "topology_unit_count": len(topology_units),
            "external_contract_count": len(contracts),
            "unowned_node_ids": missing,
        },
    }


__all__ = ["build_rl_mtcs_payload"]
