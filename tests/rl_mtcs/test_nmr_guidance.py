from __future__ import annotations

from model.forward.g2s.rl_mtcs_utils import NMRGuidanceScorer
from RL_MTCS.models import InverseAssemblyInput, TopologyUnit, UnitEdge, UnitNode
from RL_MTCS.state import HexAssemblyState, relabel_aromatic_sites


def _planned_hetero_aromatic_problem() -> InverseAssemblyInput:
    unit = TopologyUnit(
        unit_id="aromatic:0",
        product_type="benzene",
        nodes=(
            UnitNode(1, 12, (0, 0)),
            UnitNode(2, 13, (1, 0)),
            UnitNode(3, 13, (1, 1)),
        ),
        internal_edges=(
            UnitEdge(1, 2, "aromatic"),
            UnitEdge(2, 3, "aromatic"),
        ),
    )
    return InverseAssemblyInput(
        units=(unit,),
        contracts=(),
        node_su_types=((1, 26), (2, 11), (3, 13)),
        ownership=((1, unit.unit_id), (2, unit.unit_id), (3, unit.unit_id)),
    )


def test_nmr_gid_permutation_uses_planned_label_for_fixed_aromatic_slot():
    state = HexAssemblyState(_planned_hetero_aromatic_problem())
    effective = NMRGuidanceScorer._effective_su_map(
        state,
        boundary_site_ids={2: 3},
        protected_aromatic_ids=(),
    )

    assert effective == {1: 26, 2: 13, 3: 11}


def test_complete_side_relabel_commits_all_planned_su_labels():
    state = HexAssemblyState(_planned_hetero_aromatic_problem())
    assert state.place_unit("aromatic:0", 0, (0, 0))

    final_state = relabel_aromatic_sites(
        state,
        boundary_site_ids={},
        actions=[],
        replay_contract_kinds=(),
        stage="side",
    )

    assert final_state.is_complete()
    assert final_state.current_node_map() == final_state.problem.node_map()
    assert final_state.current_node_map()[1] == 26
