from __future__ import annotations

from RL_MTCS.csv_contract import _layout_topology_tree, parse_topology_notation
from RL_MTCS.hex_assembly_state import (
    HexAssemblyState,
    axial_add,
    axial_distance,
    transform_axial,
)
from RL_MTCS.inverse_contract import (
    HEX_DIRECTIONS,
    InverseAssemblyInput,
    TopologyUnit,
    UnitEdge,
    UnitNode,
)


def test_repository_hex_distance_accepts_exactly_the_six_neighbor_offsets():
    origin = (7, -4)
    for direction in HEX_DIRECTIONS:
        neighbor = axial_add(origin, direction)
        assert axial_distance(origin, neighbor) == 1
        for rotation in range(6):
            transformed_origin = transform_axial(origin, rotation)
            transformed_neighbor = transform_axial(neighbor, rotation)
            assert axial_distance(transformed_origin, transformed_neighbor) == 1

    assert axial_distance(origin, axial_add(origin, (1, -1))) == 2


def test_canonical_horizontal_chain_uses_right_up_right_down_steps():
    parsed = parse_topology_notation("1-2-3-4-5")
    coordinates = _layout_topology_tree(
        parsed,
        {1: 11, 2: 23, 3: 23, 4: 23, 5: 11},
    )
    assert coordinates == {
        1: (1, 1),
        2: (2, 2),
        3: (3, 2),
        4: (4, 3),
        5: (5, 3),
    }


def test_su25_uses_three_honeycomb_sites_plus_the_opposite_fourth_site():
    parsed = parse_topology_notation("1-2-3(-4)(-5-6)-7-8")
    coordinates = _layout_topology_tree(
        parsed,
        {1: 11, 2: 23, 3: 25, 4: 22, 5: 15, 6: 16, 7: 23, 8: 11},
    )

    assert coordinates[3] == (3, 2)
    assert coordinates[5] == (3, 1)
    assert coordinates[6] == (4, 1)
    assert coordinates[4] == (3, 3)
    first_step = (
        coordinates[5][0] - coordinates[3][0],
        coordinates[5][1] - coordinates[3][1],
    )
    second_step = (
        coordinates[6][0] - coordinates[5][0],
        coordinates[6][1] - coordinates[5][1],
    )
    assert first_step != second_step
    directions = {
        (
            coordinates[node_id][0] - coordinates[3][0],
            coordinates[node_id][1] - coordinates[3][1],
        )
        for node_id in (2, 4, 5, 7)
    }
    assert directions == {(-1, 0), (0, -1), (0, 1), (1, 1)}


def _three_node_problem() -> InverseAssemblyInput:
    unit = TopologyUnit(
        unit_id="topology:0",
        product_type="flex_chain",
        nodes=(
            UnitNode(1, 23, (0, 0)),
            UnitNode(2, 23, (1, 1)),
            UnitNode(3, 23, (2, 1)),
        ),
        internal_edges=(UnitEdge(1, 2, "flex"), UnitEdge(2, 3, "flex")),
    )
    return InverseAssemblyInput(
        units=(unit,),
        contracts=(),
        node_su_types=((1, 23), (2, 23), (3, 23)),
        ownership=((1, unit.unit_id), (2, unit.unit_id), (3, unit.unit_id)),
    )


def test_reflected_placement_is_reconstructed_from_immutable_local_geometry():
    state = HexAssemblyState(_three_node_problem())
    assert state.place_unit("topology:0", 4, (9, -3), reflection=True)
    audit = state.audit()
    assert audit["topology_conformation_valid"] is True
    assert audit["geometry_valid"] is True

    state.node_axial[3] = axial_add(state.node_axial[3], (1, 0))
    audit = state.audit()
    assert audit["topology_conformation_valid"] is False
    assert audit["geometry_valid"] is False
    assert audit["topology_conformation_errors"][0]["node_id"] == 3


def test_reembedded_marker_always_invalidates_topology_conformation():
    state = HexAssemblyState(_three_node_problem())
    assert state.place_unit("topology:0", 0, (0, 0))
    state.reembedded_units.add("topology:0")
    audit = state.audit()
    assert audit["topology_conformation_valid"] is False
    assert audit["geometry_valid"] is False
