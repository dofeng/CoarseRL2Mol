from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RL_MTCS import AssemblyConfig
from test_rl_mtcs_sites import synthetic_hsq_fixture
from RL_MTCS.conformers import topology_conformer_info
from RL_MTCS.flex import (
    _append_site_contract,
    secondary_flex_displacement_candidates,
)
from RL_MTCS.search import domain_conflict_check
from RL_MTCS.geometry import HEX_DIRECTIONS
from RL_MTCS.rigid import TopologyPlacementStage, search_phase
from RL_MTCS.output import export_rigid_views
from RL_MTCS.state import relabel_aromatic_sites
from RL_MTCS.side import _aromatic_component_map
from RL_MTCS.models import InverseInputError
from RL_MTCS.input import NodeRow, _rigid_pairs


HSQ_OUTPUTS = (
    Path(__file__).resolve().parents[1]
    / "test_results"
    / "HSQ-1"
    / "outer_cycle_1"
    / "final_outputs"
)


def test_hsq_inverse_counts_and_contract_phases():
    from RL_MTCS.input import build_csv_inverse_input
    problem = build_csv_inverse_input(HSQ_OUTPUTS / "final_nodes_by_global_id.csv",
                                      HSQ_OUTPUTS / "final_topology_specs.csv")
    data = {"units": problem.units, "contracts": problem.contracts, "problem": problem,
            "aromatic_units": [u for u in problem.units if u.phase == "aromatic"]}
    units = data["units"]
    contracts = data["contracts"]
    assert sum(unit.phase == "aromatic" for unit in units) == 17
    assert sum(unit.product_type == "flex_chain" for unit in units) == 9
    assert sum(unit.product_type == "aliphatic_ring" for unit in units) == 13
    assert sum(unit.product_type == "side_chain" for unit in units) == 34
    assert len(data["problem"].source_rigid_contracts) == 14
    assert sum(contract.kind == "rigid" for contract in contracts) == 0
    assert sum(port.phase == "rigid" for unit in data["aromatic_units"]
               for port in unit.ports) == 28
    assert sum(contract.kind == "flex" for contract in contracts) == 44
    assert sum(contract.kind == "side" for contract in contracts) == 34


def test_su10_ports_do_not_require_source_hop1_pairing():
    nodes = {
        10: NodeRow(10, 10, "SU10", (), ""),
        11: NodeRow(11, 10, "SU10", (), ""),
    }
    assert _rigid_pairs(nodes) == []


def test_su10_pairing_is_selected_at_rigid_placement_time():
    data = synthetic_hsq_fixture()
    state = data["state"]
    assert not state.phase_done("rigid")
    source = state.problem.source_rigid_contracts[0]
    left = state.unit(source.source_unit_id)
    right = state.unit(source.target_unit_id)
    left_port = next(port for port in left.ports
                     if port.phase == "rigid" and port.node_id == source.source_node_id)
    right_port = next(port for port in right.ports
                      if port.phase == "rigid" and port.node_id == source.target_node_id)
    actions = TopologyPlacementStage(state, "rigid")._candidate_for_pair(
        left, left_port, right, right_port
    )
    assert actions
    action = actions[0]
    action["contract_id"] = (
        f"rigid:dynamic:{min(source.source_node_id, source.target_node_id)}-"
        f"{max(source.source_node_id, source.target_node_id)}"
    )
    ok, reason = state.connect(action)
    assert ok, reason
    assert action["contract_id"] in state.satisfied_contracts
    assert len([contract for contract in state.problem.contracts
                if contract.kind == "rigid"]) == 1
    assert state.connect(action)[1] == "port_already_used"
    assert not state.phase_done("rigid")
    component_map = _aromatic_component_map(state)
    assert component_map[left.unit_id] == component_map[right.unit_id]


def test_dynamic_rigid_contract_survives_aromatic_relabel():
    data = synthetic_hsq_fixture()
    state = data["state"]
    source = state.problem.source_rigid_contracts[0]
    left = state.unit(source.source_unit_id)
    right = state.unit(source.target_unit_id)
    left_port = next(port for port in left.ports if port.phase == "rigid"
                     and port.node_id == source.source_node_id)
    right_port = next(port for port in right.ports if port.phase == "rigid"
                      and port.node_id == source.target_node_id)
    action = TopologyPlacementStage(state, "rigid")._candidate_for_pair(
        left, left_port, right, right_port
    )[0]
    action["contract_id"] = (
        f"rigid:dynamic:{min(source.source_node_id, source.target_node_id)}-"
        f"{max(source.source_node_id, source.target_node_id)}"
    )
    ok, reason = state.connect(action)
    assert ok, reason
    relabelled = relabel_aromatic_sites(state, {}, [], stage="flex")
    assert action["contract_id"] in relabelled.satisfied_contracts
    assert any(edge.contract_id == action["contract_id"] for edge in relabelled.edges)
    assert not relabelled.audit_port_direction_errors()


def test_initial_aromatic_views_are_individual_and_exact(tmp_path):
    data = synthetic_hsq_fixture()
    result = export_rigid_views(data["state"], tmp_path, initial=True)
    assert result["count"] == 17
    assert len(list(tmp_path.glob("aromatic_*.png"))) == 17
    assert len(list(tmp_path.glob("aromatic_*.json"))) == 17


def test_rigid_search_respects_tree_node_limit():
    data = synthetic_hsq_fixture()
    result = search_phase(
        TopologyPlacementStage(data["state"], "rigid"),
        iterations=1, candidate_k=1, beam_width=1, max_steps=128,
        max_tree_nodes=16,
    )
    diagnostics = result.info["result"]
    assert diagnostics["pairing_search"]["tree_nodes"] <= 16
    assert all(row["tree_nodes"] <= 16
               for row in diagnostics.get("mcts_searches", []))
    if not diagnostics["done"]:
        assert diagnostics["status"] in {
            "budget_exhausted", "candidate_space_exhausted"
        }
        assert diagnostics["candidate_space_complete"] is False


def test_bounded_domain_conflict_is_soft_until_complete():
    domains = {"a": [(1,)], "b": [(1,)]}
    bounded = domain_conflict_check(domains, complete=False)
    complete = domain_conflict_check(domains, complete=True)
    assert bounded["bounded"] is True
    assert bounded["hard"] is False
    assert complete["hard"] is True


def test_config_caps_memory_parameters():
    config = AssemblyConfig(
        search_budget=10**9,
        max_tree_nodes=10**9,
        domain_check_limit=10**9,
        domain_pose_limit=10**9,
    )
    assert config.search_budget <= 20000
    assert config.max_tree_nodes <= 256
    assert config.domain_check_limit <= 512
    assert config.domain_pose_limit <= 64


def test_conformer_identity_is_preserved():
    data = synthetic_hsq_fixture()
    ring = next(
        unit for unit in data["topology_units"]
        if unit.product_type == "aliphatic_ring"
    )
    variants, _complete = topology_conformer_info(ring, 4)
    original_nodes = {node.node_id for node in ring.nodes}
    original_edges = {
        frozenset((edge.source_node_id, edge.target_node_id))
        for edge in ring.internal_edges
    }
    for variant in variants:
        assert {node.node_id for node in variant.nodes} == original_nodes
        assert {
            frozenset((edge.source_node_id, edge.target_node_id))
            for edge in variant.internal_edges
        } == original_edges


def test_secondary_flex_displacement_rules_cover_horizontal_and_vertical_paths():
    horizontal = secondary_flex_displacement_candidates(
        (0, 0), (4, 2), (1, 1), (-1, 0)
    )
    assert [(row["direction"], row["chain_length"])
            for row in horizontal] == [("horizontal", 3)]
    vertical = secondary_flex_displacement_candidates(
        (0, 0), (1, 3), (0, 1), (0, -1)
    )
    assert [(row["direction"], row["chain_length"])
            for row in vertical] == [("vertical", 2)]
    assert secondary_flex_displacement_candidates(
        (0, 0), (2, 2), (1, 1), (-1, -1)
    ) == []


def test_flex_chain_does_not_generate_independent_bent_ports():
    data = synthetic_hsq_fixture()
    chain = next(
        unit for unit in data["topology_units"]
        if unit.product_type == "flex_chain"
    )
    variants, complete = topology_conformer_info(chain, 12)
    assert complete is True
    assert len(variants) == 1


def test_flex_site_commit_uses_inverse_endpoint_su_and_is_single_use():
    data = synthetic_hsq_fixture()
    state = data["state"]
    contract = next(
        item for item in state.problem.contracts
        if item.contract_id == "flex:5:29-482"
    )
    topology = state.unit("topology:5")

    # Find one legal staging pose against a physical SU13 site.  The selected
    # site intentionally need not be the original CSV boundary GID: Flex may
    # remap a chain endpoint onto any free aromatic perimeter slot.
    placed = None
    for site_id, site_axial in sorted(state.node_axial.items()):
        if state.current_su_types.get(int(site_id)) != 13:
            continue
        for direction in HEX_DIRECTIONS:
            trial = state.clone()
            target = (site_axial[0] + direction[0], site_axial[1] + direction[1])
            if not trial.place_unit(
                topology.unit_id,
                0,
                target,
                allowed_contacts=[(482, int(site_id))],
            ):
                continue
            placed = (trial, int(site_id))
            break
        if placed is not None:
            break
    assert placed is not None
    trial, site_id = placed

    _append_site_contract(trial, contract, 482, site_id)
    aromatic_boundary_id = contract.target_node_id
    expected_su_type = state.problem.node_map()[aromatic_boundary_id]
    assert expected_su_type == 5
    assert trial.current_su_types[site_id] == expected_su_type
    assert f"aromatic_site:{site_id}" in trial.used_ports
    assert contract.contract_id in trial.satisfied_contracts
    assert trial.audit()["flex_endpoint_su_valid"] is True

    try:
        _append_site_contract(trial, contract, 482, site_id)
    except InverseInputError:
        pass
    else:
        raise AssertionError("a consumed flex aromatic site was reused")
