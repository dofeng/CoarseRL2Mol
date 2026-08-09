from __future__ import annotations

import pytest

from RL_MTCS.hex_assembly_state import HexAssemblyState
from RL_MTCS.inverse_contract import (
    ExternalContract,
    InverseAssemblyInput,
    TopologyUnit,
    UnitNode,
    UnitPort,
)


def _two_unit_problem(kind: str, planned_aromatic_su: int) -> tuple[InverseAssemblyInput, ExternalContract]:
    aromatic = TopologyUnit(
        unit_id="aromatic:0",
        product_type="benzene",
        nodes=(UnitNode(1, 13, (0, 0), role="aromatic_site"),),
        ports=(UnitPort("aromatic-port", 1, phase=kind),),
    )
    other_type = "benzene" if kind == "rigid" else f"{kind}_chain"
    other_su = 13 if kind == "rigid" else 23
    other = TopologyUnit(
        unit_id="aromatic:1" if kind == "rigid" else "topology:0",
        product_type=other_type,
        nodes=(UnitNode(2, other_su, (0, 0)),),
        ports=(UnitPort("other-port", 2, phase=kind),),
    )
    contract = ExternalContract(
        contract_id=f"{kind}:0",
        source_unit_id=aromatic.unit_id,
        source_node_id=1,
        target_unit_id=other.unit_id,
        target_node_id=2,
        kind=kind,
        source_port_id="aromatic-port",
        target_port_id="other-port",
    )
    problem = InverseAssemblyInput(
        units=(aromatic, other),
        contracts=(contract,),
        node_su_types=((1, planned_aromatic_su), (2, 10 if kind == "rigid" else other_su)),
        ownership=((1, aromatic.unit_id), (2, other.unit_id)),
    )
    return problem, contract


@pytest.mark.parametrize(
    ("kind", "planned_su"),
    (("rigid", 10), ("flex", 5), ("side", 11)),
)
def test_successful_contract_activates_only_its_aromatic_endpoints(kind: str, planned_su: int):
    problem, contract = _two_unit_problem(kind, planned_su)
    state = HexAssemblyState(problem)
    assert state.current_node_map() == {1: 13, 2: 13 if kind == "rigid" else 23}
    assert state.place_unit(contract.source_unit_id, 0, (0, 0))

    ok, reason = state.connect({
        "kind": kind,
        "contract_id": contract.contract_id,
        "source_unit_id": contract.source_unit_id,
        "source_node_id": contract.source_node_id,
        "source_port_id": contract.source_port_id,
        "target_unit_id": contract.target_unit_id,
        "target_node_id": contract.target_node_id,
        "target_port_id": contract.target_port_id,
        "target_rotation": 0,
        "target_translation": (1, 0),
    })

    assert (ok, reason) == (True, "")
    assert state.current_node_map()[1] == planned_su
    assert state.current_node_map()[2] == (10 if kind == "rigid" else 23)
    rows = {row["node_id"]: row for row in state.to_dict()["nodes"]}
    assert rows[1]["su_type"] == planned_su
    assert rows[1]["planned_su_type"] == planned_su


def test_force_connection_uses_the_same_aromatic_activation_semantics():
    problem, contract = _two_unit_problem("flex", 5)
    state = HexAssemblyState(problem)
    assert state.place_unit(contract.source_unit_id, 0, (0, 0))
    assert state.place_unit(
        contract.target_unit_id,
        0,
        (1, 0),
        allowed_contact=(contract.source_node_id, contract.target_node_id),
    )

    assert state.force_connect_contract(contract, "test")
    assert state.current_node_map() == {1: 5, 2: 23}
