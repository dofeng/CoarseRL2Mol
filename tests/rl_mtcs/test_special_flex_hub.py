from __future__ import annotations

from itertools import combinations
from pathlib import Path

from RL_MTCS.RL_init import initialize
from RL_MTCS.csv_contract import _layout_topology_tree, parse_topology_notation
from RL_MTCS.flex_assembly import (
    _coordinate_aspect,
    _flex_port_groups,
    _select_backbone_topologies,
)
from RL_MTCS.hex_assembly_state import HexAssemblyState, axial_distance
from RL_MTCS.inverse_contract import (
    HEX_DIRECTIONS,
    InverseAssemblyInput,
    valid_hex_directions,
)
from RL_MTCS.inverse_mcts import TopologyPlacementStage


SPECIAL_GID = "121-0-414(-309-308-420)-415(-504-116)-503-115"
SPECIAL_SU = {
    121: 9,
    0: 0,
    414: 20,
    309: 15,
    308: 15,
    420: 22,
    415: 20,
    504: 27,
    116: 6,
    503: 27,
    115: 6,
}


def test_three_aromatic_hub_edges_are_hex_neighbors():
    parsed = parse_topology_notation(SPECIAL_GID)
    coordinates = _layout_topology_tree(parsed, SPECIAL_SU)
    assert coordinates == {
        121: (2, 0),
        0: (1, 0),
        414: (0, -1),
        309: (0, -2),
        308: (-1, -3),
        420: (-1, -4),
        415: (-1, -1),
        504: (-1, 0),
        116: (0, 1),
        503: (-2, -2),
        115: (-3, -2),
    }
    for left, right in parsed.edges:
        delta = (
            coordinates[right][0] - coordinates[left][0],
            coordinates[right][1] - coordinates[left][1],
        )
        assert delta in HEX_DIRECTIONS
        if (
            SPECIAL_SU[left] <= 25
            and SPECIAL_SU[right] <= 25
            and len([edge for edge in parsed.edges if left in edge]) <= 2
            and len([edge for edge in parsed.edges if right in edge]) <= 2
        ):
            assert delta in valid_hex_directions(coordinates[left])


def test_three_aromatic_hub_has_no_nonbonded_hex_neighbors():
    parsed = parse_topology_notation(SPECIAL_GID)
    coordinates = _layout_topology_tree(parsed, SPECIAL_SU)
    edge_keys = {frozenset(edge) for edge in parsed.edges}
    projected_nonedge_neighbors = {
        frozenset((left, right))
        for left, right in combinations(parsed.nodes, 2)
        if frozenset((left, right)) not in edge_keys
        and axial_distance(coordinates[left], coordinates[right]) == 1
    }
    assert projected_nonedge_neighbors == set()


def test_three_aromatic_hub_has_no_straight_bond_triplets():
    parsed = parse_topology_notation(SPECIAL_GID)
    coordinates = _layout_topology_tree(parsed, SPECIAL_SU)
    adjacency = {node_id: set() for node_id in parsed.nodes}
    for left, right in parsed.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    for center, neighbors in adjacency.items():
        for left, right in combinations(sorted(neighbors), 2):
            left_direction = (
                coordinates[left][0] - coordinates[center][0],
                coordinates[left][1] - coordinates[center][1],
            )
            right_direction = (
                coordinates[right][0] - coordinates[center][0],
                coordinates[right][1] - coordinates[center][1],
            )
            assert left_direction != (
                -right_direction[0], -right_direction[1]
            ), f"straight bonded triplet: {left}-{center}-{right}"


def test_ym8_three_aromatic_hub_is_the_first_backbone_topology():
    root = Path(__file__).resolve().parents[2]
    outputs = root / "test_results" / "YM-8" / "final_outputs"
    initialized = initialize(
        outputs / "final_nodes_by_global_id.csv",
        outputs / "final_topology_specs.csv",
    )
    state = HexAssemblyState(initialized["problem"])

    topology = state.unit("topology:6")
    assert {
        int(node.node_id): tuple(node.local_axial)
        for node in topology.nodes
    } == {
        0: (0, 0),
        414: (-1, -1),
        309: (-1, -2),
        308: (-2, -3),
        420: (-2, -4),
        415: (-2, -1),
        504: (-2, 0),
        503: (-3, -2),
    }
    assert {
        int(port.node_id): tuple(port.direction)
        for port in topology.ports
    } == {
        0: (1, 0),
        504: (1, 1),
        503: (-1, 0),
    }
    assert {
        node_id: state.problem.node_map()[node_id]
        for node_id in (121, 116, 115)
    } == {121: 9, 116: 6, 115: 6}

    groups = _flex_port_groups(state, "topology:6")
    assert [
        [record[2] for record in group]
        for group in groups
    ] == [[115], [116], [121]]

    backbone_ids, _interior_ids = _select_backbone_topologies(state, 20)
    assert backbone_ids[0] == "topology:6"


def _ym8_three_way_subproblem() -> InverseAssemblyInput:
    """Keep the real YM-8 units and contracts involved in topology:6."""
    root = Path(__file__).resolve().parents[2]
    outputs = root / "test_results" / "YM-8" / "final_outputs"
    problem = initialize(
        outputs / "final_nodes_by_global_id.csv",
        outputs / "final_topology_specs.csv",
    )["problem"]
    hub_contracts = tuple(
        contract
        for contract in problem.contracts
        if contract.source_unit_id == "topology:6"
        or contract.target_unit_id == "topology:6"
    )
    unit_ids = {"topology:6"} | {
        unit_id
        for contract in hub_contracts
        for unit_id in (contract.source_unit_id, contract.target_unit_id)
        if unit_id != "topology:6"
    }
    units = tuple(unit for unit in problem.units if unit.unit_id in unit_ids)
    node_ids = {
        int(node.node_id)
        for unit in units
        for node in unit.nodes
    }
    contracts = tuple(
        contract
        for contract in problem.contracts
        if contract in hub_contracts
    )
    return InverseAssemblyInput(
        units=units,
        contracts=contracts,
        node_su_types=tuple(
            (node_id, su_type)
            for node_id, su_type in problem.node_su_types
            if int(node_id) in node_ids
        ),
        ownership=tuple(
            (node_id, unit_id)
            for node_id, unit_id in problem.ownership
            if int(node_id) in node_ids
        ),
        diagnostics=problem.diagnostics,
    )


def test_ym8_production_units_complete_the_real_three_way_connection():
    """The production CSV path fixes one hub pose and attaches both free arms."""
    state = HexAssemblyState(_ym8_three_way_subproblem())
    aromatic_unit_ids = sorted(
        unit.unit_id for unit in state.problem.units
        if unit.phase == "aromatic"
    )
    for unit_id, translation in zip(
        aromatic_unit_ids, ((2, 0), (12, 0), (24, 0))
    ):
        assert state.place_unit(unit_id, 0, translation)
    assert {
        node_id: state.current_node_map()[node_id]
        for node_id in (121, 115, 116)
    } == {121: 13, 115: 13, 116: 13}

    contract_order = (
        "flex:6:121-0",
        "flex:6:115-503",
        "flex:6:116-504",
    )
    stage = TopologyPlacementStage(state, "flex", contract_order=contract_order)
    anchor_action = next(
        action
        for action in stage.get_candidates(32)
        if int(action["source_rotation"]) == 0
        and tuple(action["source_translation"]) == (0, 0)
    )
    assert stage.step(anchor_action)
    fixed_hub_pose = state.placements["topology:6"]

    contracts_by_id = {
        contract.contract_id: contract
        for contract in state.problem.contracts
    }
    ownership = state.problem.ownership_map()
    for contract_id in contract_order[1:]:
        contract = contracts_by_id[contract_id]
        aromatic_node_id = (
            contract.source_node_id
            if state.unit(contract.source_unit_id).phase == "aromatic"
            else contract.target_node_id
        )
        aromatic_unit_id = ownership[int(aromatic_node_id)]
        action = next(
            candidate
            for candidate in stage.get_candidates(32)
            if candidate.get("move_component_unit_id") == aromatic_unit_id
        )
        assert stage.step(action)
        assert state.placements["topology:6"] == fixed_hub_pose

    assert stage.is_done()
    assert state.satisfied_contracts == set(contract_order)
    assert state.component_count() == 1
    assert {
        node_id: state.node_axial[node_id]
        for node_id in (121, 0, 115, 503, 116, 504)
    } == {
        121: (1, 0),
        0: (0, 0),
        115: (-4, -2),
        503: (-3, -2),
        116: (-1, 1),
        504: (-2, 0),
    }
    assert {
        node_id: state.current_node_map()[node_id]
        for node_id in (121, 115, 116)
    } == {121: 9, 115: 6, 116: 6}
    audit = state.audit()
    assert audit["complete"] is True
    assert audit["geometry_valid"] is True
    assert audit["topology_conformation_valid"] is True
    assert audit["port_directions_valid"] is True
    assert audit["nonbonded_contacts_valid"] is True
    assert audit["geometry_fallback_contract_ids"] == []


def test_coordinate_aspect_uses_strict_q_span_over_r_span_bounds():
    root = Path(__file__).resolve().parents[2]
    outputs = root / "test_results" / "YM-8" / "final_outputs"
    state = HexAssemblyState(initialize(
        outputs / "final_nodes_by_global_id.csv",
        outputs / "final_topology_specs.csv",
    )["problem"])
    state.occupied_axial = {(q, r): index for index, (q, r) in enumerate(
        ((0, 0), (8, 0), (0, 5)), start=1
    )}

    shape = _coordinate_aspect(state)
    assert shape["q_span"] == 8
    assert shape["r_span"] == 5
    assert shape["q_over_r"] == 1.6
    assert shape["valid"] is True

    state.occupied_axial = {(0, 0): 1, (5, 1): 2}
    assert _coordinate_aspect(state)["valid"] is False
