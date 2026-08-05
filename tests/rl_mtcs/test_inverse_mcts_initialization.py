from __future__ import annotations

import pytest

from RL_MTCS.RL_init import initialize
from RL_MTCS.RL_mcts import MCTSConfig, MCTSRunner
from RL_MTCS.inverse_contract import InverseInputError


def _complete_payload():
    return {
        "aromatic_units": [
            {
                "unit_id": "a0",
                "kind": "benzene",
                "nodes": [{"node_id": 1, "su_type": 13, "local_axial": [0, 0]}],
                "ports": [{"port_id": "rigid", "node_id": 1, "direction": [1, 0]}],
            },
            {
                "unit_id": "a1",
                "kind": "benzene",
                "nodes": [{"node_id": 2, "su_type": 13, "local_axial": [0, 0]}],
                "ports": [
                    {"port_id": "rigid", "node_id": 2, "direction": [-1, 0]},
                    {"port_id": "flex", "node_id": 2, "direction": [0, 1]},
                ],
            },
        ],
        "topology_products": [
            {
                "unit_id": "flex0",
                "product_type": "flex_chain",
                "source_node_ids": [3, 5],
                "nodes": [
                    {"node_id": 3, "su_type": 23, "local_axial": [0, 0]},
                    {"node_id": 5, "su_type": 23, "local_axial": [1, 0]},
                ],
                "edges": [[3, 5]],
                "ports": [
                    {"port_id": "left", "node_id": 3, "direction": [0, -1]},
                    {"port_id": "right", "node_id": 5, "direction": [1, 0]},
                ],
            },
            {
                "unit_id": "side0",
                "product_type": "side_chain",
                "source_node_ids": [4],
                "nodes": [{"node_id": 4, "su_type": 22, "local_axial": [0, 0]}],
                "ports": [{"port_id": "left", "node_id": 4, "direction": [-1, 0]}],
            },
        ],
        "nodes": [
            {"node_id": 1, "su_type": 13},
            {"node_id": 2, "su_type": 13},
            {"node_id": 3, "su_type": 23},
            {"node_id": 4, "su_type": 22},
            {"node_id": 5, "su_type": 23},
        ],
        "external_contracts": [
            {
                "contract_id": "c-rigid",
                "source_unit_id": "a0",
                "source_node_id": 1,
                "source_port_id": "rigid",
                "target_unit_id": "a1",
                "target_node_id": 2,
                "target_port_id": "rigid",
                "kind": "rigid",
            },
            {
                "contract_id": "c-flex",
                "source_unit_id": "a1",
                "source_node_id": 2,
                "source_port_id": "flex",
                "target_unit_id": "flex0",
                "target_node_id": 3,
                "target_port_id": "left",
                "kind": "flex",
            },
            {
                "contract_id": "c-side",
                "source_unit_id": "flex0",
                "source_node_id": 5,
                "source_port_id": "right",
                "target_unit_id": "side0",
                "target_node_id": 4,
                "target_port_id": "left",
                "kind": "side",
            },
        ],
    }


def test_initializer_rejects_legacy_csv_path():
    with pytest.raises(InverseInputError, match="CSV/SU-histogram"):
        initialize("legacy_nodes.csv")


def test_initializer_preserves_inverse_units_and_node_identity():
    data = initialize(_complete_payload())
    assert data["diagnostics"]["allocation"] is False
    assert data["diagnostics"]["substitution"] is False
    assert {unit.unit_id for unit in data["aromatic_units"]} == {"a0", "a1"}
    assert data["node_su_types"] == {1: 13, 2: 13, 3: 23, 4: 22, 5: 23}
    assert data["state"].placements


def test_mcts_places_complete_units_without_substitution():
    result = MCTSRunner(
        _complete_payload(),
        MCTSConfig(iterations=12, flex_iterations=12, side_iterations=12, candidate_k=12, seed=7),
    ).run_all_stages()
    assert result["status"] == "complete"
    assert result["audit"]["missing_node_ids"] == []
    assert result["audit"]["pending_contract_ids"] == []
    assert result["audit"]["component_count"] == 1
    assert "substitution" not in result
    su_by_id = {int(row["node_id"]): int(row["su_type"]) for row in result["molecule"]["nodes"]}
    assert su_by_id == {1: 13, 2: 13, 3: 23, 4: 22, 5: 23}

