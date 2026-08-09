from __future__ import annotations

from pathlib import Path

from RL_MTCS.RL_init import initialize


def test_ym8_initialization_contains_only_real_su12_su13_ring_sites():
    root = Path(__file__).resolve().parents[2]
    outputs = root / "test_results" / "YM-8" / "final_outputs"
    initialized = initialize(
        outputs / "final_nodes_by_global_id.csv",
        outputs / "final_topology_specs.csv",
    )
    problem = initialized["problem"]
    state = initialized["state"]
    diagnostics = dict(problem.diagnostics)

    assert len(problem.node_su_types) == 615
    assert diagnostics["source_node_count"] == "616"
    assert diagnostics["excluded_surplus_aromatic_node_ids"] == "274"
    assert 274 not in problem.node_map()
    assert 274 not in problem.ownership_map()
    assert {
        state.current_node_map()[node_id] for node_id in state.node_axial
    } == {12, 13}

    for unit in initialized["aromatic_units"]:
        ring_coordinates = {
            tuple(coord) for ring in unit.local_rings for coord in ring
        }
        assert {tuple(node.local_axial) for node in unit.nodes} == ring_coordinates
