from itertools import combinations
from pathlib import Path

from RL_MTCS.RL_mcts import MCTSConfig, MCTSRunner
from RL_MTCS.flex_assembly import (
    build_flex_assembly_plan,
    run_planned_flex_search,
)
from RL_MTCS.hex_assembly_state import HexAssemblyState, axial_add
from RL_MTCS.inverse_mcts import TopologyPlacementStage
from RL_MTCS.inverse_contract import (
    HEX_VERTEX_OFFSETS,
    ExternalContract,
    InverseAssemblyInput,
    TopologyUnit,
    UnitEdge,
    UnitNode,
    UnitPort,
)


def _three_component_flex_problem() -> InverseAssemblyInput:
    aromatic_units = []
    contracts = []
    for index, node_id in enumerate((1, 2, 3)):
        unit_id = f"aromatic:{index}"
        aromatic_port = f"a{index}"
        topology_port = f"t{index}"
        aromatic_units.append(TopologyUnit(
            unit_id=unit_id,
            product_type="aromatic_unit",
            nodes=(UnitNode(node_id, 13, (0, 0)),),
            ports=(UnitPort(aromatic_port, node_id, phase="flex"),),
        ))
        contracts.append(ExternalContract(
            contract_id=f"flex:{index}",
            source_unit_id="topology:0",
            source_node_id=10,
            target_unit_id=unit_id,
            target_node_id=node_id,
            kind="flex",
            source_port_id=topology_port,
            target_port_id=aromatic_port,
        ))
    topology = TopologyUnit(
        unit_id="topology:0",
        product_type="flex_chain",
        nodes=(UnitNode(10, 25, (0, 0)),),
        ports=tuple(
            UnitPort(f"t{index}", 10, phase="flex")
            for index in range(3)
        ),
    )
    units = tuple(aromatic_units) + (topology,)
    return InverseAssemblyInput(
        units=units,
        contracts=tuple(contracts),
        node_su_types=((1, 13), (2, 13), (3, 13), (10, 25)),
        ownership=((1, "aromatic:0"), (2, "aromatic:1"),
                   (3, "aromatic:2"), (10, "topology:0")),
    )


def test_multiport_flex_unit_merges_every_outside_rigid_component():
    state = HexAssemblyState(_three_component_flex_problem())
    assert state.place_unit("aromatic:0", 0, (0, 0))
    assert state.place_unit("aromatic:1", 0, (12, 0))
    assert state.place_unit("aromatic:2", 0, (24, 0))

    plan = build_flex_assembly_plan(state)
    assert len(plan.backbone_steps) == 1
    assert len(plan.backbone_steps[0].merge_contract_ids) == 2
    assert len(plan.backbone_contract_ids) == 3
    assert not plan.interior_contract_ids

    result, _plan = run_planned_flex_search(
        state,
        iterations=2,
        candidate_k=8,
        max_steps=3,
        seed=7,
        strict=True,
    )
    assert result.state.component_count() == 1
    assert result.info["result"]["contracts_done"] == 3
    assert result.info["result"]["contracts_total"] == 3
    assert result.info["result"]["done"] is True
    assert result.state.audit()["geometry_valid"] is True


def _benzene_unit(
    index: int,
    first_node_id: int,
    port_node_id: int,
    port_coordinate: tuple[int, int],
    port_direction: tuple[int, int],
    port_name: str,
) -> TopologyUnit:
    port_index = HEX_VERTEX_OFFSETS.index(port_coordinate)
    node_ids = list(first_node_id + offset for offset in range(6))
    node_ids[port_index] = int(port_node_id)
    return TopologyUnit(
        unit_id=f"aromatic:{index}",
        product_type="benzene",
        nodes=tuple(
            UnitNode(node_id, 13, coordinate, role="aromatic_site")
            for node_id, coordinate in zip(node_ids, HEX_VERTEX_OFFSETS)
        ),
        internal_edges=tuple(
            UnitEdge(node_ids[offset], node_ids[(offset + 1) % 6], "aromatic")
            for offset in range(6)
        ),
        ports=(UnitPort(
            f"aromatic:{port_name}",
            node_ids[port_index],
            phase="flex",
            direction=port_direction,
        ),),
        local_rings=(tuple(HEX_VERTEX_OFFSETS),),
    )


def _anchored_three_way_flex_problem() -> InverseAssemblyInput:
    """Three complete benzene clusters around the real 9/6/6 hub."""
    topology_directions = ((1, 0), (-1, 0), (1, 1))
    aromatic_specs = (
        (0, 1000, 121, (-1, 0), (-1, 0), "anchor"),
        (1, 1100, 115, (1, 0), (1, 0), "right"),
        (2, 1200, 116, (-1, -1), (-1, -1), "left"),
    )
    contract_names = ("anchor", "right", "left")
    topology_port_nodes = (0, 503, 504)
    aromatic_units = []
    contracts = []
    for spec, topology_node_id in zip(aromatic_specs, topology_port_nodes):
        index, _first_node_id, port_node_id, _port_coordinate, _port_direction, name = spec
        aromatic = _benzene_unit(*spec)
        aromatic_units.append(aromatic)
        topology_port = f"topology:{name}"
        contracts.append(ExternalContract(
            contract_id=f"flex:{name}",
            source_unit_id="topology:0",
            source_node_id=topology_node_id,
            target_unit_id=f"aromatic:{index}",
            target_node_id=port_node_id,
            kind="flex",
            source_port_id=topology_port,
            target_port_id=f"aromatic:{name}",
        ))

    # Rebased at GID 0.  Both SU27 exits turn before reaching the SU6 ring
    # sites; neither continues the incoming SU20-SU27 direction.
    topology_coordinates = {
        0: (0, 0),
        414: (-1, -1),
        415: (-2, -1),
        503: (-3, -2),
        309: (-1, -2),
        308: (-2, -3),
        420: (-2, -4),
        504: (-2, 0),
    }
    topology_su = {
        0: 0,
        414: 20,
        415: 20,
        503: 27,
        309: 15,
        308: 15,
        420: 22,
        504: 27,
    }
    topology = TopologyUnit(
        unit_id="topology:0",
        product_type="flex_chain",
        nodes=tuple(
            UnitNode(node_id, topology_su[node_id], coordinate)
            for node_id, coordinate in topology_coordinates.items()
        ),
        internal_edges=tuple(
            UnitEdge(left, right, "flex")
            for left, right in (
                (0, 414),
                (414, 415),
                (415, 503),
                (414, 309),
                (309, 308),
                (308, 420),
                (415, 504),
            )
        ),
        ports=tuple(
            UnitPort(
                f"topology:{name}",
                node_id,
                phase="flex",
                direction=direction,
            )
            for name, node_id, direction in zip(
                contract_names, topology_port_nodes, topology_directions
            )
        ),
        topology_notation=(
            "9-0-single20d3(-15-15-22)-double20d3(-27-6)-27-6"
        ),
    )
    units = tuple(aromatic_units) + (topology,)
    ownership = tuple(
        (int(node.node_id), str(unit.unit_id))
        for unit in units
        for node in unit.nodes
    )
    planned_aromatic_su = {121: 9, 115: 6, 116: 6}
    return InverseAssemblyInput(
        units=units,
        contracts=tuple(contracts),
        node_su_types=tuple(
            (
                int(node.node_id),
                int(planned_aromatic_su.get(int(node.node_id), node.su_type)),
            )
            for unit in units
            for node in unit.nodes
        ),
        ownership=ownership,
    )


def _assert_no_straight_bond_triplets(state: HexAssemblyState) -> None:
    adjacency = {int(node_id): set() for node_id in state.node_axial}
    for edge in state.edges:
        left = int(edge.source_node_id)
        right = int(edge.target_node_id)
        adjacency[left].add(right)
        adjacency[right].add(left)

    for center, neighbors in adjacency.items():
        center_axial = state.node_axial[center]
        for left, right in combinations(sorted(neighbors), 2):
            left_direction = (
                state.node_axial[left][0] - center_axial[0],
                state.node_axial[left][1] - center_axial[1],
            )
            right_direction = (
                state.node_axial[right][0] - center_axial[0],
                state.node_axial[right][1] - center_axial[1],
            )
            assert left_direction != (
                -right_direction[0], -right_direction[1]
            ), (
                f"straight bonded triplet {left}-{center}-{right}: "
                f"{left_direction}, {right_direction}"
            )


def test_three_way_flex_is_fixed_after_the_first_aromatic_attachment():
    """The first attachment fixes the hub pose and both remaining sites."""
    state = HexAssemblyState(_anchored_three_way_flex_problem())
    assert state.current_node_map()[121] == 13
    assert state.current_node_map()[115] == 13
    assert state.current_node_map()[116] == 13
    assert state.problem.node_map()[121] == 9
    assert state.problem.node_map()[115] == 6
    assert state.problem.node_map()[116] == 6

    assert state.place_unit("aromatic:0", 0, (2, 0))
    assert state.place_unit("aromatic:1", 0, (12, 0))
    assert state.place_unit("aromatic:2", 0, (24, 0))
    stage = TopologyPlacementStage(
        state,
        "flex",
        contract_order=("flex:anchor", "flex:right", "flex:left"),
    )

    anchor_action = next(
        action
        for action in stage.get_candidates(32)
        if int(action["source_rotation"]) == 0
        and tuple(action["source_translation"]) == (0, 0)
    )
    assert stage.step(anchor_action)
    assert state.node_axial[121] == (1, 0)
    assert state.node_axial[0] == (0, 0)
    assert state.current_node_map()[121] == 9
    assert state.current_node_map()[115] == 13
    assert state.current_node_map()[116] == 13
    fixed_topology_pose = state.placements["topology:0"]

    expected_sites = {
        "aromatic:1": axial_add(state.node_axial[503], (-1, 0)),
        "aromatic:2": axial_add(state.node_axial[504], (1, 1)),
    }
    for unit_id, node_id in (("aromatic:1", 115), ("aromatic:2", 116)):
        action = next(
            candidate
            for candidate in stage.get_candidates(32)
            if candidate.get("move_component_unit_id") == unit_id
        )
        assert stage.step(action)
        assert state.placements["topology:0"] == fixed_topology_pose
        assert state.node_axial[node_id] == expected_sites[unit_id]

    assert state.placements["aromatic:0"].translation == (2, 0)
    assert state.placements["aromatic:1"].translation == (-5, -2)
    assert state.placements["aromatic:2"].translation == (0, 2)
    assert {
        node_id: state.node_axial[node_id]
        for node_id in (121, 115, 116, 0, 503, 504)
    } == {
        121: (1, 0),
        115: (-4, -2),
        116: (-1, 1),
        0: (0, 0),
        503: (-3, -2),
        504: (-2, 0),
    }
    assert state.current_node_map()[121] == 9
    assert state.current_node_map()[115] == 6
    assert state.current_node_map()[116] == 6
    assert state.satisfied_contracts == {
        "flex:anchor", "flex:right", "flex:left"
    }
    assert state.component_count() == 1
    _assert_no_straight_bond_triplets(state)
    audit = state.audit()
    assert audit["geometry_valid"] is True
    assert audit["topology_conformation_valid"] is True
    assert audit["port_directions_valid"] is True
    assert audit["nonbonded_contacts_valid"] is True


def test_ym8_csv_search_consumes_every_fixed_flex_and_side_topology():
    root = Path(__file__).resolve().parents[2]
    outputs = root / "test_results" / "YM-8" / "final_outputs"
    runner = MCTSRunner(
        outputs / "final_nodes_by_global_id.csv",
        config=MCTSConfig(
            iterations=16,
            flex_iterations=4,
            candidate_k=1,
            beam_width=1,
            seed=7,
        ),
        topology_specs_csv=outputs / "final_topology_specs.csv",
    )
    rigid = runner.run_rigid_stage()
    flex = runner.run_flex_stage(rigid)
    result = flex["result"]
    audit = flex["state"].audit()

    assert result["done"] is True
    assert result["backbone_contracts_done"] == 37
    assert result["interior_contracts_done"] == 14
    assert result["contracts_done"] == result["contracts_total"] == 51
    assert result["coordinate_aspect"]["valid"] is True
    assert flex["best_candidate"].action_sequence[0][
        "topology_unit_id"
    ] == "topology:6"
    assert audit["component_count"] == 1
    assert audit["geometry_valid"] is True

    side = runner.run_side_stage(flex)
    side_result = side["result"]
    side_audit = side["state"].audit()
    assert side_result["done"] is True
    assert side_result["contracts_done"] == side_result["contracts_total"] == 119
    assert side_result["resource_allocation_used"] is False
    assert side_result["force_completion_used"] is False
    assert side_result["coordinate_aspect"]["valid"] is True
    assert len(side["best_candidate"].action_sequence) == 119
    assert side_audit["complete"] is True
    assert side_audit["placed_node_count"] == side_audit["expected_node_count"] == 615
    assert side_audit["component_count"] == 1
    assert side_audit["nonbonded_contacts_valid"] is True
    assert side_audit["geometry_valid"] is True
    assert not side_audit["geometry_fallback_contract_ids"]
