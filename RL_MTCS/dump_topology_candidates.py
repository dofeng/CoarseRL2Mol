"""Dump topology assembly candidates and allocation balance for final_nodes CSVs.

This utility is intentionally diagnostic-only.  It reads a final node CSV,
re-runs the allocator/topology assembler, and prints the closed topology
structures, pending candidates, residual plans, and optional node coverage.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .RL_allocator import FlexAllocator
    from .allocator_models import _chain_required_resource_counts, chain_23_count, chain_actual_topology_su
except ImportError:  # Allow direct execution: python RL_MTCS/dump_topology_candidates.py ...
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from RL_MTCS.RL_allocator import FlexAllocator
    from RL_MTCS.allocator_models import _chain_required_resource_counts, chain_23_count, chain_actual_topology_su


RESOURCE_KEYS = ("11", "22", "23", "24", "25")


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _parse_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return tuple(int(k) for k, v in parsed.items() for _ in range(int(v)))
        if isinstance(parsed, (list, tuple, set)):
            return tuple(int(x) for x in parsed)
        return (int(parsed),)
    except Exception:
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        out: List[int] = []
        for part in parts:
            try:
                out.append(int(float(part)))
            except Exception:
                continue
        return tuple(out)


def _read_nodes(csv_path: Path) -> List[SimpleNamespace]:
    with csv_path.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))
    nodes: List[SimpleNamespace] = []
    for idx, row in enumerate(rows):
        su_col = next((c for c in ("su_type", "center_su_idx", "su_idx", "type") if c in row), "")
        if not su_col:
            raise ValueError(f"Cannot find SU type column in {csv_path}")
        gid = _parse_optional_int(row.get("global_id"))
        if gid is None:
            gid = idx
        node = SimpleNamespace(
            global_id=int(gid),
            su_type=int(float(row[su_col])),
            hop1_ids=_parse_int_tuple(row.get("hop1_ids")),
            hop1_su=Counter(_parse_int_tuple(row.get("hop1_su", row.get("hop1", "")))),
            hop2_su=Counter(_parse_int_tuple(row.get("hop2_su", row.get("hop2", "")))),
            target_degree=_parse_optional_int(row.get("target_degree", row.get("target_hop1_degree"))),
            target_hop1_degree=_parse_optional_int(row.get("target_hop1_degree", row.get("target_degree"))),
            target_fixed_anchor_count=_parse_optional_int(row.get("target_fixed_anchor_count")) or 0,
            special_anchor_partition=str(row.get("special_anchor_partition", row.get("anchor_partition", "")) or ""),
            anchor_partition=str(row.get("anchor_partition", row.get("special_anchor_partition", "")) or ""),
            special_anchor_mode=str(row.get("special_anchor_mode", row.get("anchor_mode", "")) or ""),
            anchor_mode=str(row.get("anchor_mode", row.get("special_anchor_mode", "")) or ""),
        )
        nodes.append(node)
    return nodes


def _allocate_nodes(nodes: Sequence[Any], show_allocator_log: bool) -> Any:
    alloc = FlexAllocator(nodes=list(nodes))
    if show_allocator_log:
        return alloc.allocate()
    with redirect_stdout(io.StringIO()):
        return alloc.allocate()


def _fmt_path(values: Iterable[int]) -> str:
    return "-".join(str(int(x)) for x in list(values or []))


def _chain_display(chain: Any) -> str:
    meta = dict(getattr(chain, "metadata", {}) or {})
    resolved = meta.get("resolved_topology") or meta.get("display_topology") or meta.get("open_topology")
    if resolved:
        return str(resolved)
    return _fmt_path(getattr(chain, "composition", []) or [])


def _chain_row(chain: Any, idx: int, section: str) -> Dict[str, Any]:
    meta = dict(getattr(chain, "metadata", {}) or {})
    return {
        "idx": int(idx),
        "section": str(section),
        "chain_type": str(getattr(chain, "chain_type", "")),
        "origin_type": str(getattr(chain, "origin_type", "")),
        "display": _chain_display(chain),
        "composition": [int(x) for x in list(getattr(chain, "composition", []) or [])],
        "actual_topology_su": [int(x) for x in chain_actual_topology_su(chain)],
        "count_23": int(chain_23_count(chain)),
        "resources": _chain_required_resource_counts(chain),
        "source_ids": [int(x) for x in list(getattr(chain, "source_ids", []) or []) if int(x) >= 0],
        "candidate_group_id": str(meta.get("candidate_group_id", meta.get("source_group_id", "")) or ""),
        "topology_state": str(meta.get("topology_state", "")),
    }


def _closed_rows(result: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for section, chains in (
        ("bridge", getattr(result, "bridge_chains", []) or []),
        ("side", getattr(result, "side_chains", []) or []),
        ("branch", getattr(result, "branch_chains", []) or []),
    ):
        for chain in list(chains or []):
            rows.append(_chain_row(chain, len(rows), section))
    return rows


def _covered_source_ids(result: Any) -> Dict[int, List[str]]:
    covered: Dict[int, List[str]] = defaultdict(list)
    for row in _closed_rows(result):
        label = f"{row['section']}[{row['idx']}]:{row['display']}"
        for gid in row["source_ids"]:
            covered[int(gid)].append(label)
    for chain in list(getattr(result, "pending_closure_topologies", []) or []):
        meta = dict(getattr(chain, "metadata", {}) or {})
        label = f"pending:{meta.get('candidate_group_id', getattr(chain, 'origin_type', ''))}"
        for gid in list(getattr(chain, "source_ids", []) or []):
            if int(gid) >= 0:
                covered[int(gid)].append(label)
    return covered


def _print_json_block(title: str, value: Any) -> None:
    print(f"\n## {title}")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _print_table(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n## {title} ({len(rows)})")
    if not rows:
        return
    for row in rows:
        resources = row.get("resources", {})
        print(
            f"[{row.get('idx')}] {row.get('section')} {row.get('chain_type')} "
            f"origin={row.get('origin_type')} 23={row.get('count_23')} "
            f"res={resources} src={row.get('source_ids')} :: {row.get('display')}"
        )


def _print_topology_balance(result: Any) -> None:
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    print("\n## Topology Balance")
    print(f"plan_feasible={bool(diag.get('topology_plan_feasible', False))}")
    print(f"materialized_ok={bool(diag.get('topology_materialized_ok', diag.get('ok', False)))}")
    print(f"materialization_required={bool(diag.get('materialization_required', False))}")
    print(f"pending_total={int(getattr(result, 'pending_closure_total_count', 0) or 0)}")
    print(f"unallocated_branch={int(getattr(result, 'unallocated_branch', 0) or 0)}")
    print(f"remaining={{'11': {result.remaining_11}, '22': {result.remaining_22}, '23': {result.remaining_23}, '24': {result.remaining_24}, '25': {result.remaining_25}}}")
    if diag.get("planned_conversions"):
        _print_json_block("Planned Conversions", diag.get("planned_conversions"))
    if diag.get("residual_growth_actions"):
        _print_json_block("Residual Growth Actions", diag.get("residual_growth_actions"))
    if diag.get("residual_chain_actions"):
        _print_json_block("Residual Chain Actions", diag.get("residual_chain_actions"))
    if diag.get("chain_23_limit_violations"):
        _print_json_block("23-Chain Limit Violations", diag.get("chain_23_limit_violations"))


def _print_node_coverage(nodes: Sequence[Any], result: Any) -> None:
    covered = _covered_source_ids(result)
    print("\n## Node Coverage")
    for node in sorted(list(nodes or []), key=lambda n: int(getattr(n, "global_id", -1))):
        gid = int(getattr(node, "global_id", -1))
        labels = covered.get(gid, [])
        label_text = "; ".join(labels) if labels else "UNASSIGNED"
        print(f"gid={gid:>5} su={int(getattr(node, 'su_type', -1)):>2} {label_text}")


def _print_debug_details(result: Any) -> None:
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    _print_json_block("Topology Diagnostics", diag)
    _print_json_block("Selected Options", list(getattr(result, "topology_selected_options", []) or []))
    _print_json_block("Unclosed Candidates", list(getattr(result, "topology_unclosed_candidates", []) or []))
    _print_json_block("Recommended Adjustments", list(diag.get("recommended_adjustments", []) or []))


def _print_dump_sections(
    nodes: Sequence[Any],
    result: Any,
    *,
    show_debug_details: bool,
    show_node_coverage: bool,
) -> None:
    print(f"nodes={len(list(nodes or []))}")
    _print_topology_balance(result)
    _print_table("Closed Topologies", _closed_rows(result))
    pending_rows = [
        _chain_row(chain, idx, "pending")
        for idx, chain in enumerate(list(getattr(result, "pending_closure_topologies", []) or []))
    ]
    _print_table("Pending / Unclosed Topologies", pending_rows)
    if show_debug_details:
        _print_debug_details(result)
    if show_node_coverage:
        _print_node_coverage(nodes, result)


def _run_block_c_adjustment(
    nodes: Sequence[Any],
    *,
    phase: str,
    max_steps: int,
    show_block_c_log: bool,
    units_per_step: int,
    apply_recommendation_count: bool,
) -> tuple[Sequence[Any], Any, List[Dict[str, Any]], Dict[str, Any]]:
    from model.inverse.layer4.inverse_layer4 import Layer4Adjuster

    import torch

    adjuster = Layer4Adjuster(device=torch.device("cpu"))
    work_nodes = [SimpleNamespace(**vars(n)) for n in list(nodes or [])]
    hist, moves, meta = adjuster.run_block_c_topology_closure_loop(
        nodes=work_nodes,
        E_target=None,
        S_target=None,
        max_steps=max(1, int(max_steps)),
        phase=str(phase),
        rebuild_callback=None,
        show_log=bool(show_block_c_log),
        units_per_step=max(1, int(units_per_step)),
        apply_recommendation_count=bool(apply_recommendation_count),
    )
    result = _allocate_nodes(work_nodes, show_allocator_log=False)
    print("\n## Block C Adjustment Summary")
    print(f"hist={hist.tolist() if hasattr(hist, 'tolist') else hist}")
    print(f"moves={len(list(moves or []))}")
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    return work_nodes, result, list(moves or []), dict(meta or {})


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Dump closed topology structures and SU balance from a final_nodes CSV.")
    parser.add_argument("csv_path", help="Path to final_nodes_by_global_id.csv")
    parser.add_argument("--show-allocator-log", action="store_true", help="Print FlexAllocator's internal summary before this dump.")
    parser.add_argument("--show-node-coverage", action="store_true", help="Print one line per input node with recognized assignments.")
    parser.add_argument("--show-debug-details", action="store_true", help="Print allocator internals, resource ledgers, pending candidates, and node-level diagnostics.")
    parser.add_argument("--run-block-c-adjustment", action="store_true", help="Run Layer4 block_c skeleton adjustment in memory, then dump the adjusted topology allocation.")
    parser.add_argument("--block-c-phase", choices=("branch", "extra", "full"), default="branch", help="Which block_c skeleton adjustment phase to run when --run-block-c-adjustment is enabled.")
    parser.add_argument("--block-c-backend", choices=("auto", "layer4", "fallback"), default="auto", help="Accepted for compatibility; this restored utility uses Layer4 directly.")
    parser.add_argument("--block-c-template-lib", default="", help="Accepted for compatibility with the original script.")
    parser.add_argument("--block-c-max-steps", type=int, default=50, help="Maximum block_c skeleton adjustment iterations.")
    parser.add_argument("--block-c-units-per-step", type=int, default=1, help="Maximum topology recommendation units to apply in one block_c step.")
    parser.add_argument("--block-c-apply-recommendation-count", action="store_true", help="Apply each topology recommendation's full count in one block_c step.")
    parser.add_argument("--show-block-c-log", action="store_true", help="Print block_c adjustment logs instead of suppressing them.")
    parser.add_argument("--no-block-c-step-trace", action="store_true", help="Accepted for compatibility with the original script.")
    args = parser.parse_args(argv)

    csv_path = Path(str(args.csv_path)).expanduser()
    nodes = _read_nodes(csv_path)
    print(f"input_csv: {csv_path}")

    print("========== Before Block C Adjustment ==========")
    result = _allocate_nodes(nodes, show_allocator_log=bool(args.show_allocator_log))
    _print_dump_sections(
        nodes,
        result,
        show_debug_details=bool(args.show_debug_details),
        show_node_coverage=bool(args.show_node_coverage),
    )

    if bool(args.run_block_c_adjustment):
        adjusted_nodes, adjusted_result, _moves, _meta = _run_block_c_adjustment(
            nodes,
            phase=str(args.block_c_phase),
            max_steps=int(args.block_c_max_steps),
            show_block_c_log=bool(args.show_block_c_log),
            units_per_step=max(1, int(args.block_c_units_per_step)),
            apply_recommendation_count=bool(args.block_c_apply_recommendation_count),
        )
        print("========== After Block C Adjustment ==========")
        _print_dump_sections(
            adjusted_nodes,
            adjusted_result,
            show_debug_details=bool(args.show_debug_details),
            show_node_coverage=bool(args.show_node_coverage),
        )


if __name__ == "__main__":
    main()
