"""CSV initialization inventories and export reports."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from RL_MTCS.io.input import initialize, load_node_rows, load_topology_rows
from RL_MTCS.mdp.types import TopologyUnit, ExternalContract, InverseInputError
from RL_MTCS.io.output import contract_rows
from RL_MTCS.geometry import HEX_DIRECTIONS


def _metadata(unit: TopologyUnit) -> Dict[str, str]:
    return {str(key): str(value) for key, value in unit.metadata}


def _count_rows(values: Iterable[Any], field_name: str) -> List[Dict[str, Any]]:
    counts = Counter(str(value) for value in values)
    return [
        {field_name: key, "count": int(value)}
        for key, value in sorted(counts.items(), key=lambda item: item[0])
    ]


def _su_distribution(node_su: Mapping[int, int], names: Mapping[int, str]) -> List[Dict[str, Any]]:
    counts = Counter(int(value) for value in node_su.values())
    return [
        {
            "su_type": int(su_type),
            "su_name": str(names.get(int(su_type), "")),
            "count": int(count),
        }
        for su_type, count in sorted(counts.items())
    ]


def _unit_su_counts(unit: TopologyUnit) -> Dict[str, int]:
    return {
        str(su_type): int(count)
        for su_type, count in sorted(Counter(int(node.su_type) for node in unit.nodes).items())
    }


def _bbox(unit: TopologyUnit) -> Dict[str, int]:
    q_values = [int(node.local_axial[0]) for node in unit.nodes]
    r_values = [int(node.local_axial[1]) for node in unit.nodes]
    return {
        "q_min": min(q_values),
        "q_max": max(q_values),
        "r_min": min(r_values),
        "r_max": max(r_values),
    }


def _external_boundaries(
    unit_id: str,
    contracts: Sequence[ExternalContract],
) -> List[int]:
    result: List[int] = []
    for contract in contracts:
        if contract.source_unit_id == unit_id:
            result.append(int(contract.target_node_id))
        elif contract.target_unit_id == unit_id:
            result.append(int(contract.source_node_id))
    return sorted(set(result))


def _aromatic_rows(
    units: Sequence[TopologyUnit],
    contracts: Sequence[ExternalContract],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: item.unit_id):
        port_counts = Counter(str(port.phase or "unspecified") for port in unit.ports)
        rows.append({
            "unit_id": str(unit.unit_id),
            "cluster_kind": str(unit.product_type),
            "ring_count": int(len(unit.local_rings)),
            "node_count": int(len(unit.nodes)),
            "internal_edge_count": int(len(unit.internal_edges)),
            "port_count": int(len(unit.ports)),
            "rigid_port_count": int(port_counts.get("rigid", 0)),
            "flex_port_count": int(port_counts.get("flex", 0)),
            "side_port_count": int(port_counts.get("side", 0)),
            "node_ids": [int(node.node_id) for node in unit.nodes],
            "external_boundary_ids": _external_boundaries(unit.unit_id, contracts),
            "su_distribution": _unit_su_counts(unit),
            "local_bbox": _bbox(unit),
        })
    return rows


def _topology_rows(
    units: Sequence[TopologyUnit],
    contracts: Sequence[ExternalContract],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: int(_metadata(item).get("csv_row", -1))):
        meta = _metadata(unit)
        related = [
            contract for contract in contracts
            if unit.unit_id in {contract.source_unit_id, contract.target_unit_id}
        ]
        rows.append({
            "csv_row": int(meta.get("csv_row", -1)),
            "unit_id": str(unit.unit_id),
            "source_product_type": str(meta.get("source_product_type", "")),
            "assembly_phase": str(unit.phase),
            "su_notation": str(unit.topology_notation),
            "gid_notation": str(unit.node_id_notation),
            "node_count": int(len(unit.nodes)),
            "internal_edge_count": int(len(unit.internal_edges)),
            "boundary_port_count": int(len(unit.ports)),
            "boundary_node_ids": _external_boundaries(unit.unit_id, contracts),
            "contract_ids": [str(contract.contract_id) for contract in related],
            "source_fragment_ids": [int(value) for value in unit.source_fragment_ids],
            "assembly_unavailable_reason": meta.get("assembly_unavailable_reason", ""),
            "node_ids": [int(node.node_id) for node in unit.nodes],
            "su_distribution": _unit_su_counts(unit),
            "local_bbox": _bbox(unit),
        })
    return rows


def _su25_rows(
    units: Sequence[TopologyUnit],
    contracts: Sequence[ExternalContract],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in units:
        node_map = unit.node_map()
        for node in unit.nodes:
            if int(node.su_type) != 25:
                continue
            neighbors: List[Dict[str, Any]] = []
            directions = []
            for edge in unit.internal_edges:
                if int(node.node_id) not in {int(edge.source_node_id), int(edge.target_node_id)}:
                    continue
                other_id = (
                    int(edge.target_node_id)
                    if int(edge.source_node_id) == int(node.node_id)
                    else int(edge.source_node_id)
                )
                other = node_map[other_id]
                direction = (
                    int(other.local_axial[0]) - int(node.local_axial[0]),
                    int(other.local_axial[1]) - int(node.local_axial[1]),
                )
                directions.append(direction)
                neighbors.append({
                    "node_id": other_id,
                    "su_type": int(other.su_type),
                    "geometry_role": ('fixed_su22_center' if (node.node_id, other_id)
                                      in unit.su25_center_branches else 'honeycomb_vertex'),
                    "direction": [int(direction[0]), int(direction[1])],
                    "kind": "internal",
                })
            external = [
                contract for contract in contracts
                if (
                    contract.source_unit_id == unit.unit_id
                    and int(contract.source_node_id) == int(node.node_id)
                ) or (
                    contract.target_unit_id == unit.unit_id
                    and int(contract.target_node_id) == int(node.node_id)
                )
            ]
            rows.append({
                "unit_id": str(unit.unit_id),
                "node_id": int(node.node_id),
                "fixed_center_branch_id": next((child for parent, child in unit.su25_center_branches
                                               if parent == node.node_id), None),
                "local_axial": [int(node.local_axial[0]), int(node.local_axial[1])],
                "degree": int(len(neighbors) + len(external)),
                "distinct_internal_direction_count": int(len(set(directions))),
                "neighbors": neighbors,
                "external_contract_ids": [str(item.contract_id) for item in external],
                "assembly_unavailable_reason": _metadata(unit).get('assembly_unavailable_reason', ''),
            })
    return rows


def _validate_units(
    units: Sequence[TopologyUnit],
    source_node_ids: set[int],
    ownership: Mapping[int, str],
    su25_rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    errors: List[str] = []
    if set(int(value) for value in ownership) != source_node_ids:
        missing = sorted(source_node_ids - set(int(value) for value in ownership))
        unexpected = sorted(set(int(value) for value in ownership) - source_node_ids)
        errors.append(f"ownership_mismatch:missing={missing[:12]}:unexpected={unexpected[:12]}")
    for unit in units:
        coords = {int(node.node_id): tuple(node.local_axial) for node in unit.nodes}
        if _metadata(unit).get('assembly_unavailable_reason'):
            continue
        if len(set(coords.values())) != len(coords):
            errors.append(f"duplicate_local_axial:{unit.unit_id}")
        for edge in unit.internal_edges:
            left = coords[int(edge.source_node_id)]
            right = coords[int(edge.target_node_id)]
            delta = (int(right[0]) - int(left[0]), int(right[1]) - int(left[1]))
            if delta not in HEX_DIRECTIONS:
                errors.append(
                    f"non_hex_internal_edge:{unit.unit_id}:{edge.source_node_id}-{edge.target_node_id}"
                )
    for row in su25_rows:
        if int(row.get("degree", 0)) != 4:
            errors.append(f"su25_degree_not_four:{row.get('node_id')}:{row.get('degree')}")
    return errors


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_cell(row.get(key, "")) for key in fieldnames})


def inspect_csv_initialization(
    nodes_csv: str | Path,
    topology_specs_csv: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Initialize the CSV pair, validate it, and return a serializable report."""
    nodes_path = Path(nodes_csv).resolve()
    topology_path = Path(topology_specs_csv).resolve()
    directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else nodes_path.parent / "rl_mtcs_initialization"
    )

    node_rows = load_node_rows(nodes_path)
    source_topologies = load_topology_rows(topology_path)
    try:
        initialized = initialize(nodes_path, topology_path)
    except InverseInputError as exc:
        report = {"ok": False, "status": "input_geometry_invalid", "failure_scope": "input",
                  "validation": {"ok": False, "errors": [str(exc)]},
                  "diagnostics": exc.diagnostics}
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "initialization_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    problem = initialized["problem"]
    state = initialized["state"]
    aromatic_units = list(initialized["aromatic_units"])
    topology_units = list(initialized["topology_units"])
    contracts = list(problem.contracts)
    names = {int(row.su_type): str(row.su_name) for row in node_rows.values()}
    problem_diagnostics = {
        str(key): str(value) for key, value in problem.diagnostics
    }
    excluded_node_ids = sorted(
        int(value)
        for value in problem_diagnostics.get(
            "excluded_surplus_aromatic_node_ids", ""
        ).split(",")
        if value.strip()
    )

    aromatic_rows = _aromatic_rows(aromatic_units, contracts)
    topology_rows = _topology_rows(topology_units, contracts)
    contract_records = contract_rows(contracts)
    su25_rows = _su25_rows(topology_units, contracts)
    validation_errors = _validate_units(
        list(problem.units), set(problem.node_map()), problem.ownership_map(), su25_rows
    )
    validation_warnings: List[str] = []
    initial_audit = state.audit()
    input_errors = [json.loads(issue) for issue in problem.input_issues]
    if not initial_audit['legal']:
        validation_errors.append('initial_state_audit_failed')
    if initial_audit.get("duplicate_axials"):
        validation_errors.append("initial_aromatic_staging_has_duplicate_axials")
    if initial_audit.get("non_hex_edges"):
        validation_errors.append("initial_aromatic_staging_has_non_hex_edges")
    placed_initial_su = {
        int(node_id): int(state.current_su_types[node_id])
        for node_id in state.node_axial
    }
    if set(placed_initial_su.values()) - {12, 13}:
        validation_errors.append(
            "initial_aromatic_nodes_are_not_only_su12_su13"
        )
    for unit in aromatic_units:
        ring_coordinates = {
            tuple(coord) for ring in unit.local_rings for coord in ring
        }
        outside_ring = sorted(
            int(node.node_id)
            for node in unit.nodes
            if tuple(node.local_axial) not in ring_coordinates
        )
        if outside_ring:
            validation_errors.append(
                f"aromatic_nodes_outside_ring_sequences:{unit.unit_id}:{outside_ring}"
            )

    report: Dict[str, Any] = {
        "ok": not validation_errors,
        "input_valid": not input_errors,
        "input_errors": input_errors,
        "inputs": {
            "nodes_csv": str(nodes_path),
            "topology_specs_csv": str(topology_path),
            "node_csv_row_count": len(node_rows),
            "topology_csv_row_count": len(source_topologies),
        },
        "summary": {
            "node_count": len(problem.node_su_types),
            "source_node_count": len(node_rows),
            "assembly_node_count": len(problem.node_su_types),
            "excluded_surplus_aromatic_node_ids": excluded_node_ids,
            "unit_count": len(problem.units),
            "aromatic_unit_count": len(aromatic_units),
            "topology_unit_count": len(topology_units),
            "contract_count": len(contracts),
            "source_rigid_pair_count": len(problem.source_rigid_contracts),
            "rigid_port_count": sum(su == 10 for _node_id, su in problem.node_su_types),
            "initially_placed_unit_count": len(state.placements),
            "initially_placed_node_count": len(state.node_axial),
            "initial_component_count": state.component_count(),
            "anchor_unit_id": str(initialized["anchor_unit_id"]),
            "su25_count": len(su25_rows),
        },
        "distributions": {
            "su_types": _su_distribution(state.current_node_map(), names),
            "planned_su_types": _su_distribution(problem.node_map(), names),
            "initially_placed_aromatic_su_types": _su_distribution(
                placed_initial_su, names
            ),
            "aromatic_kinds": _count_rows(
                (unit.product_type for unit in aromatic_units), "cluster_kind"
            ),
            "topology_phases": _count_rows(
                (unit.phase for unit in topology_units), "assembly_phase"
            ),
            "ring_geometry_modes": _count_rows(
                (_metadata(unit).get("ring_geometry_mode", "tree")
                 for unit in topology_units
                 if _metadata(unit).get("source_product_type", "").lower()
                 in {"aliphatic_ring", "fatty_ring", "fatty_ring_fused", "fatty_ring_side", "fatty_ring_upper_lower"}),
                "ring_geometry_mode",
            ),
            "source_product_types": _count_rows(
                (_metadata(unit).get("source_product_type", "") for unit in topology_units),
                "source_product_type",
            ),
            "topology_node_counts": _count_rows(
                (len(unit.nodes) for unit in topology_units), "node_count"
            ),
            "topology_boundary_counts": _count_rows(
                (len(unit.ports) for unit in topology_units), "boundary_port_count"
            ),
            "contract_kinds": _count_rows(
                (contract.kind for contract in contracts), "kind"
            ),
        },
        "aromatic_clusters": aromatic_rows,
        "topology_units": topology_rows,
        "contracts": contract_records,
        "source_rigid_pairs": contract_rows(problem.source_rigid_contracts),
        "su25_sites": su25_rows,
        "initial_state_audit": initial_audit,
        "validation": {
            "ok": not validation_errors,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "owned_node_count": len(problem.ownership),
            "source_node_count": len(node_rows),
            "assembly_node_count": len(problem.node_su_types),
            "excluded_surplus_aromatic_node_ids": excluded_node_ids,
        },
        "outputs": {
            "directory": str(directory),
            "report_json": str(directory / "initialization_report.json"),
            "su_distribution_csv": str(directory / "su_distribution.csv"),
            "aromatic_clusters_csv": str(directory / "aromatic_clusters.csv"),
            "topology_units_csv": str(directory / "topology_units.csv"),
            "contracts_csv": str(directory / "contracts.csv"),
            "su25_sites_csv": str(directory / "su25_sites.csv"),
        },
    }

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "initialization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(directory / "su_distribution.csv", report["distributions"]["su_types"])
    _write_csv(directory / "aromatic_clusters.csv", aromatic_rows)
    _write_csv(directory / "topology_units.csv", topology_rows)
    _write_csv(directory / "contracts.csv", contract_records)
    _write_csv(directory / "su25_sites.csv", su25_rows)
    return report


def print_initialization_report(report: Mapping[str, Any]) -> None:
    """Print the high-signal distributions; full rows remain in CSV/JSON."""
    summary = dict(report.get("summary", {}))
    print("[RL_MTCS initialization]")
    print(f"ok={bool(report.get('ok', False))}")
    for key in (
        "source_node_count", "assembly_node_count",
        "excluded_surplus_aromatic_node_ids",
        "node_count", "unit_count", "aromatic_unit_count", "topology_unit_count",
        "contract_count", "initially_placed_unit_count", "initially_placed_node_count",
        "initial_component_count", "anchor_unit_id", "su25_count",
    ):
        print(f"{key}={summary.get(key)}")
    for name, rows in dict(report.get("distributions", {})).items():
        print(f"\n[{name}]")
        for row in rows:
            print("  " + ", ".join(f"{key}={value}" for key, value in row.items()))
    validation = dict(report.get("validation", {}))
    print("\n[validation]")
    print(f"ok={validation.get('ok')}")
    for error in validation.get("errors", []):
        print(f"  ERROR: {error}")
    print(f"\noutputs={dict(report.get('outputs', {})).get('directory', '')}")
