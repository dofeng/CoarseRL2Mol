from itertools import islice
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RL_MTCS import AssemblyConfig
from test_rl_mtcs_sites import synthetic_hsq_fixture
from RL_MTCS.actions.conformers import iter_topology_conformers
from RL_MTCS.actions.flex import (
    _append_site_contract,
)
from RL_MTCS.geometry import HEX_DIRECTIONS
from RL_MTCS.actions.rigid_geometry import RigidPoseGenerator
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.io.output import export_rigid_views
from RL_MTCS.mdp.transition import relabel_aromatic_sites
from RL_MTCS.mdp.types import InverseInputError
from RL_MTCS.io.input import NodeRow, _rigid_pairs


HSQ_OUTPUTS = (
    Path(__file__).resolve().parents[1]
    / "test_results"
    / "HSQ-1"
    / "outer_cycle_1"
    / "final_outputs"
)


def test_hsq_inverse_counts_and_contract_phases():
    from RL_MTCS.io.input import build_csv_inverse_input
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
    assert sum(contract.kind == "flex" for contract in contracts) == 18
    assert sum(contract.kind == "ring" for contract in contracts) == 26
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
    actions = RigidPoseGenerator(state)._candidate_for_pair(
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
    assert right.unit_id in state.component_unit_ids(left.unit_id)


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
    action = RigidPoseGenerator(state)._candidate_for_pair(
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


def test_rigid_search_selects_from_limited_result_candidates():
    from test_rl_mtcs_sites import aromatic_fixture
    state, info = StageController(AssemblyConfig(
        connection_candidate_limit=2,forward_max_rollbacks=0,verbose=False)).run(aromatic_fixture(3), 'rigid')
    attempt=info['retry_selection']['attempts'][0]
    assert attempt['search']['result_candidate_count'] == 2
    assert attempt['search']['stop_reason'] == 'result_candidates_ready'
    assert not attempt['search']['search_exhausted']
    assert state.audit()['geometry_valid']




def test_config_has_only_legal_candidate_cap_and_checkpoint_retry_count():
    from dataclasses import fields
    config=AssemblyConfig(connection_candidate_limit=11)
    assert config.connection_candidate_limit==11
    assert config.forward_checkpoint_interval==4 and config.clearance_radius==3
    names={f.name for f in fields(config)}
    assert not names & {'search_budget','max_tree_nodes','max_steps','domain_check_limit',
                        'nmr_model_call_limit','strict_geometry','nmr_weight','nmr_stage_weights'}
    assert AssemblyConfig(nmr_batch_size=4).nmr_batch_size == 4


def test_conformer_identity_is_preserved():
    data = synthetic_hsq_fixture()
    ring = next(
        unit for unit in data["topology_units"]
        if unit.product_type == "aliphatic_ring"
    )
    variants = list(islice(iter_topology_conformers(ring), 4))
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




def test_generated_two_endpoint_chain_enumerates_valid_conformers():
    from RL_MTCS.actions.conformers import validate_conformer, _embedding_phases
    data = synthetic_hsq_fixture()
    chain = next(
        unit for unit in data["topology_units"]
        if unit.product_type == "flex_chain"
    )
    variants = list(iter_topology_conformers(chain))
    assert len(variants) > 1
    for variant in variants:
        validate_conformer(chain, variant)
        assert _embedding_phases(variant, {n.node_id: n.local_axial for n in variant.nodes},
                                 [p.direction for p in variant.ports])


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
                attachment_edges=[(482, int(site_id))],
            ):
                continue
            try:
                _append_site_contract(trial.clone(), contract, 482, int(site_id))
            except InverseInputError:
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
