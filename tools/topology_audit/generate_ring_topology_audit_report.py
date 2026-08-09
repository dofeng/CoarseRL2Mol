from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.inverse.topology.assembly_external import ExternalTopologyAssembler
from model.inverse.topology.classification.classifier import classify_topology_units
from model.inverse.topology.ring_topology import audit_ring_topology_contract, cycle_edges
from model.inverse.topology.stage3_models import RingTopologyContract
from tests.inverse.topology.helpers import connect, make_node

def _raw_su24_ring_fixture(
    *,
    a_count: int,
    c_count: int,
    terminal_22_count: int,
) -> List[Any]:
    nodes: List[Any] = []
    next_center_id = 100
    next_cap_id = 10
    next_open_id = 200
    for _ in range(int(a_count)):
        center = make_node(next_center_id, 24)
        cap = make_node(next_cap_id, 11)
        opens = [make_node(next_open_id + offset, 23) for offset in range(2)]
        for neighbor in [cap] + opens:
            connect(center, neighbor)
        nodes.extend([center, cap] + opens)
        next_center_id += 1
        next_cap_id += 1
        next_open_id += 2
    for _ in range(int(c_count)):
        center = make_node(next_center_id, 24)
        opens = [make_node(next_open_id + offset, 23) for offset in range(3)]
        for neighbor in opens:
            connect(center, neighbor)
        nodes.extend([center] + opens)
        next_center_id += 1
        next_open_id += 3
    nodes.extend(
        make_node(300 + offset, 22)
        for offset in range(int(terminal_22_count))
    )
    return nodes


def _normalized_edges(cycle: Sequence[int]) -> List[List[int]]:
    return [list(edge) for edge in sorted(cycle_edges(cycle))]


def _claimed_ids(product: Any) -> List[int]:
    return sorted(
        int(node_id)
        for values in dict(product.claimed_resource_ids_by_su or {}).values()
        for node_id in list(values or [])
    )


def _shape_evidence(contract: RingTopologyContract) -> Dict[str, Any]:
    cycles = [[int(value) for value in cycle] for cycle in contract.cycles]
    evidence: Dict[str, Any] = {
        "cycle_count": len(cycles),
        "cycle_lengths": [len(cycle) for cycle in cycles],
        "cycle_unique_node_counts": [len(set(cycle)) for cycle in cycles],
        "cycle_edges": [_normalized_edges(cycle) for cycle in cycles],
        "all_ring_node_ids": sorted({node_id for cycle in cycles for node_id in cycle}),
    }
    if str(contract.ring_kind) == "fused" and len(cycles) == 2:
        shared_nodes = set(cycles[0]) & set(cycles[1])
        shared_edges = cycle_edges(cycles[0]) & cycle_edges(cycles[1])
        evidence.update(
            {
                "shared_bridgehead_ids": sorted(shared_nodes),
                "shared_edges": [list(edge) for edge in sorted(shared_edges)],
                "total_unique_ring_nodes": len(set(cycles[0]) | set(cycles[1])),
            }
        )
    if str(contract.ring_kind) == "upper_lower" and cycles:
        fixed_ids = [
            int(value)
            for value in dict(contract.role_node_ids or {}).get("fixed_anchor", [])
        ]
        external_ids = sorted(
            {
                int(value)
                for edge in list(contract.external_attachments or [])
                for value in list(edge or [])
                if int(value) not in set(cycles[0])
            }
        )
        evidence.update(
            {
                "fixed_anchor_ids": fixed_ids,
                "cycle_returns_to_fixed_anchor": bool(fixed_ids and cycles[0][0] == fixed_ids[0]),
                "external_attachment_ids": external_ids,
                "external_attachment_outside_cycle": not bool(set(external_ids) & set(cycles[0])),
            }
        )
    return evidence


def _assemble_case(
    *,
    name: str,
    a_count: int,
    c_count: int,
    terminal_22_count: int,
    preferred_nonfused: Optional[str] = None,
) -> Dict[str, Any]:
    nodes = _raw_su24_ring_fixture(
        a_count=a_count,
        c_count=c_count,
        terminal_22_count=terminal_22_count,
    )
    node_lookup = {int(node.global_id): node for node in nodes}
    hop1_before = {
        int(node.global_id): [int(value) for value in list(node.hop1_ids or [])]
        for node in nodes
    }
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    result = assembler.assemble_pool_su24_rings(
        preferred_nonfused=preferred_nonfused,
    )
    if len(assembler.products) != 1:
        raise RuntimeError(f"{name}: expected one product, got {len(assembler.products)}")
    product = assembler.products[0]
    contract = product.ring_contract
    if contract is None:
        raise RuntimeError(f"{name}: materialized product has no ring contract")
    claimed_ids = _claimed_ids(product)
    audit_reason = audit_ring_topology_contract(
        contract,
        nodes=nodes,
        owned_node_ids=product.source_node_ids,
    )
    hop1_after = {
        int(node.global_id): [int(value) for value in list(node.hop1_ids or [])]
        for node in nodes
    }
    duplicate_claims = sorted(
        node_id for node_id in set(claimed_ids) if claimed_ids.count(node_id) > 1
    )
    node_su_rows = [
        {
            "global_id": node_id,
            "actual_su": int(node_lookup[node_id].su_type),
            "contract_su": int(contract.node_su_by_id[node_id]),
            "profile": str(contract.node_profiles_by_id.get(node_id, "")),
            "roles": sorted(
                role
                for role, values in dict(contract.role_node_ids or {}).items()
                if node_id in {int(value) for value in list(values or [])}
            ),
        }
        for node_id in contract.declared_node_ids()
    ]
    checks = {
        "graph_audit_passed": audit_reason is None,
        "contract_complete": bool(contract.complete),
        "closed_only_after_audit": bool(product.closed and contract.complete and audit_reason is None),
        "declared_ids_equal_owned_ids": contract.declared_node_ids()
        == sorted(int(value) for value in product.source_node_ids),
        "all_declared_ids_are_real": set(contract.declared_node_ids()) <= set(node_lookup),
        "declared_su_matches_real_nodes": all(
            int(row["actual_su"]) == int(row["contract_su"])
            for row in node_su_rows
        ),
        "no_duplicate_resource_claims": not duplicate_claims,
        "hop1_unchanged": hop1_before == hop1_after,
        "no_public_fake_slots": not product.mainline_slots and not product.branch_slots,
    }
    return {
        "case": name,
        "fixture": {
            "su24_A": a_count,
            "su24_C": c_count,
            "terminal_SU22": terminal_22_count,
        },
        "product_kind": product.product_kind,
        "reason": product.reason,
        "closed": bool(product.closed),
        "topology_notation": contract.topology_notation,
        "node_id_notation": contract.node_id_notation,
        "candidate_signature": contract.candidate_signature,
        "cycles": contract.cycles,
        "shape_evidence": _shape_evidence(contract),
        "role_node_ids": contract.role_node_ids,
        "external_attachments": contract.external_attachments,
        "like24_node_ids": contract.like24_node_ids,
        "node_rows": node_su_rows,
        "branch_paths": contract.branch_paths,
        "branch_requirements": contract.branch_requirements,
        "required_resource_counts": contract.required_resource_counts,
        "claimed_resource_ids_by_su": product.claimed_resource_ids_by_su,
        "owned_node_ids": sorted(int(value) for value in product.source_node_ids),
        "remaining_ids_by_su": result["remaining_ids_by_su"],
        "audit_reason": audit_reason,
        "checks": checks,
    }


def _known_invalid_case() -> Dict[str, Any]:
    su_by_id = {
        214: 11,
        584: 24,
        572: 24,
        571: 24,
        42: 4,
        463: 23,
        427: 22,
        464: 23,
        502: 23,
    }
    nodes = [make_node(node_id, su_type) for node_id, su_type in su_by_id.items()]
    contract = RingTopologyContract(
        ring_kind="upper_lower",
        cycles=[[584, 572, 571, 464, 502]],
        node_labels_by_id={node_id: str(su_type) for node_id, su_type in su_by_id.items()},
        node_su_by_id=dict(su_by_id),
        node_profiles_by_id={584: "A", 572: "C", 571: "C"},
        role_node_ids={"fixed_anchor": [584]},
        external_attachments=[[214, 584]],
        branch_paths=[
            {"anchor_node_id": 572, "node_ids": [463, 427], "kind": "H", "state": "FILLED"},
            {
                "anchor_node_id": 571,
                "node_ids": [42],
                "kind": "FIXED_TERMINAL",
                "state": "FILLED",
            },
        ],
        branch_requirements=[
            {"anchor_node_id": 572, "node_ids": [463, 427], "kind": "H", "state": "FILLED"},
            {
                "anchor_node_id": 571,
                "node_ids": [42],
                "kind": "FIXED_TERMINAL",
                "state": "FILLED",
            },
        ],
        topology_notation="11-24-24(-23-22)-24(-4)-23-23",
        node_id_notation="214-584-572(-463-427)-571(-42)-464-502",
        like24_node_ids=[584, 572, 571],
    )
    reason = audit_ring_topology_contract(contract, nodes=nodes)
    return {
        "topology_notation": contract.topology_notation,
        "node_id_notation": contract.node_id_notation,
        "declared_cycle": contract.cycles[0],
        "declared_cycle_length": len(contract.cycles[0]),
        "audit_reason": reason,
        "correctly_rejected": reason == "ring_cycle_node_count:upper_lower:0:5",
    }


def _all_true(checks: Iterable[Dict[str, bool]]) -> bool:
    return all(bool(value) for group in checks for value in group.values())


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Stage3 Aliphatic Ring Topology Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"Overall fixture verdict: **{'PASS' if report['overall_passed'] else 'FAIL'}**",
        "",
    ]
    for case in report["successful_materializations"]:
        shape = case["shape_evidence"]
        lines.extend(
            [
                f"## {case['case']}",
                "",
                f"- Product: `{case['product_kind']}` / `{case['reason']}`",
                f"- Topology: `{case['topology_notation']}`",
                f"- Global IDs: `{case['node_id_notation']}`",
                f"- Cycles: `{json.dumps(case['cycles'], ensure_ascii=False)}`",
                f"- Cycle edges: `{json.dumps(shape['cycle_edges'], ensure_ascii=False)}`",
                f"- Roles: `{json.dumps(case['role_node_ids'], ensure_ascii=False, sort_keys=True)}`",
                f"- External attachments: `{json.dumps(case['external_attachments'], ensure_ascii=False)}`",
                f"- Resource claims: `{json.dumps(case['claimed_resource_ids_by_su'], ensure_ascii=False, sort_keys=True)}`",
                f"- Audit reason: `{case['audit_reason']}`",
                f"- Checks: `{json.dumps(case['checks'], ensure_ascii=False, sort_keys=True)}`",
            ]
        )
        if case["product_kind"] == "fatty_ring_fused":
            lines.extend(
                [
                    f"- Shared bridgeheads: `{shape['shared_bridgehead_ids']}`",
                    f"- Shared edge: `{shape['shared_edges']}`",
                    f"- Unique ring nodes: `{shape['total_unique_ring_nodes']}`",
                ]
            )
        lines.extend(
            [
                "",
                "Node binding:",
                "",
                "| global_id | actual SU | contract SU | profile | roles |",
                "|---:|---:|---:|:---:|:---|",
            ]
        )
        for row in case["node_rows"]:
            lines.append(
                f"| {row['global_id']} | {row['actual_su']} | {row['contract_su']} | "
                f"{row['profile']} | {', '.join(row['roles'])} |"
            )
        lines.extend(
            [
                "",
                "Branch requirements:",
                "",
                "```json",
                json.dumps(case["branch_requirements"], ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    invalid = report["known_invalid_product"]
    lines.extend(
        [
            "## Known Invalid Product",
            "",
            f"- Topology: `{invalid['topology_notation']}`",
            f"- Global IDs: `{invalid['node_id_notation']}`",
            f"- Declared cycle: `{invalid['declared_cycle']}`",
            f"- Declared cycle length: `{invalid['declared_cycle_length']}`",
            f"- Audit reason: `{invalid['audit_reason']}`",
            f"- Correctly rejected: `{invalid['correctly_rejected']}`",
            "",
            "## Scope Limitation",
            "",
            "These successful materializations exercise the residual-SU24 ring assembler. "
            "The external-fragment ring path has classification, scheduling and rejection tests, "
            "but still needs successful side/upper/fused materialization fixtures with the same report.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a node-level Stage3 ring audit report.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "test_results" / "ring_topology_audit",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [
        _assemble_case(
            name="Side ring",
            a_count=2,
            c_count=1,
            terminal_22_count=3,
        ),
        _assemble_case(
            name="Upper/lower ring",
            a_count=2,
            c_count=1,
            terminal_22_count=1,
            preferred_nonfused="upper_lower",
        ),
        _assemble_case(
            name="Fused ring",
            a_count=2,
            c_count=4,
            terminal_22_count=4,
        ),
    ]
    invalid = _known_invalid_case()
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "successful_materializations": successful,
        "known_invalid_product": invalid,
        "overall_passed": _all_true(case["checks"] for case in successful)
        and bool(invalid["correctly_rejected"]),
    }
    json_path = output_dir / "ring_topology_audit.json"
    markdown_path = output_dir / "ring_topology_audit.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(markdown_path)
    print(json_path)
    return 0 if bool(report["overall_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
