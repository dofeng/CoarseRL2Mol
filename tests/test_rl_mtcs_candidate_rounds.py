"""A round selects from finite result candidates, not an exhausted product tree."""
from types import SimpleNamespace

import pytest

from RL_MTCS.policy.mcts import MCTS, SearchGuidance
from RL_MTCS.config import AssemblyConfig
from RL_MTCS.policy.checkpoints import StageController
from test_rl_mtcs_sites import aromatic_fixture


def test_ten_results_do_not_enumerate_the_thirteen_level_product_tree():
    endpoints = []
    records = {}

    def evaluate(state):
        endpoints.append(state)
        return sum(state) / 117

    tree = MCTS((), expand=lambda s: (s + (i,) for i in range(10)),
        accept=lambda s: len(s) == 13, signature=lambda s: s, evaluate=evaluate,
        result_limit=10, seed=7,
        guidance=SearchGuidance(record=lambda s, children, value:
                               records.update({s: (children, value)})))
    chosen = tree.run()
    info = tree.diagnostics()
    assert len(endpoints) == len(set(endpoints)) == info['result_candidate_count'] == 10
    assert chosen == max(endpoints, key=sum)
    assert len(chosen) == 13
    assert info['visited_states'] <= 1 + 10 * 13
    assert not info['search_exhausted']
    assert info['stop_reason'] == 'result_candidates_ready'
    assert info['root_legal_action_count'] == info['root_evaluated_action_count'] == 10
    assert info['root_actions_covered']
    assert info['retained_nodes'] == 1 and info['transition_bytes'] == 0
    assert len(records) == info['visited_states']
    assert records[()][0] and any(not c['closed'] for c in records[()][0])
    count = len(endpoints)
    assert tree.run() == chosen  # Finished rounds cannot resume discarded trees.
    assert len(endpoints) == count


def test_last_admitted_root_action_is_evaluated_and_can_win():
    endpoints = []
    tree = MCTS((), expand=lambda s: (s+(i,) for i in range(6)) if len(s)<3 else iter(()),
        accept=lambda s: len(s)==3, signature=lambda s:s,
        evaluate=lambda s: endpoints.append(s) or s[0]/5, result_limit=6)
    chosen = tree.run()
    assert {s[0] for s in endpoints} == set(range(6))
    assert chosen[0] == 5
    assert tree.diagnostics()['root_actions_covered']


def test_formal_expansion_uses_value_to_choose_deep_continuations():
    tree = MCTS((), expand=lambda s: (s+(i,) for i in range(4)) if len(s)<3 else iter(()),
        accept=lambda s: len(s)==3, signature=lambda s:s,
        evaluate=lambda s: sum(s)/9, result_limit=4, order_by_value=True)
    chosen = tree.run()
    assert chosen==(3,3,3)
    assert tree.diagnostics()['root_evaluated_action_count']==4


def test_duplicate_and_rejected_whole_results_do_not_consume_candidate_slots():
    # Distinct paths reach an equal result. Only the result identity counts.
    paths = {(): [(1,), (2,), (3,), (4,)],
             (1,): [(1, 9)], (2,): [(2, 9)], (3,): [(3, 8)], (4,): [(4, 7)]}
    tree = MCTS((), expand=lambda s: iter(paths.get(s, ())),
        accept=lambda s: False, signature=lambda s: ('result', s[-1]) if len(s) == 2 else s,
        evaluate=lambda s: s[-1] / 10 if s else 0, excluded={('result', 8)},
        result_limit=2)
    assert tree.run()[-1] == 9
    assert tree.diagnostics()['result_candidate_count'] == 2
    assert tree.result_signatures == {('result', 9), ('result', 7)}


@pytest.mark.parametrize('excluded,expected,count', [((), (1,), 1), ({(1,)}, None, 0)])
def test_fewer_than_ten_results_and_legal_partial_terminal(excluded, expected, count):
    tree = MCTS((), expand=lambda s: iter([(1,)] if not s else []),
        accept=lambda s: False, signature=lambda s: s, evaluate=lambda s: .4,
        result_limit=10, excluded=excluded)
    assert tree.run() == expected
    assert tree.diagnostics()['result_candidate_count'] == count
    assert tree.diagnostics()['stop_reason'] == 'tree_exhausted'
    assert tree.diagnostics()['search_exhausted']


def test_result_limit_is_not_a_connection_depth_budget():
    tree = MCTS(0, expand=lambda s: iter([s + 1]) if s < 150 else iter(()),
        accept=lambda s: False, signature=lambda s: s, evaluate=lambda s: s / 150,
        result_limit=1)
    assert tree.run() == 150
    assert tree.diagnostics()['max_depth'] == 150
    assert tree.diagnostics()['result_candidate_count'] == 1


def test_selected_rigid_result_gets_one_forward_after_candidate_selection(monkeypatch):
    import RL_MTCS.policy.checkpoints as module
    controller = StageController(AssemblyConfig(connection_candidate_limit=2,
        forward_max_rollbacks=0, verbose=False))
    calls = []

    def forward(state):
        calls.append(state.state_signature())
        assert state.audit()['geometry_valid']
        return SimpleNamespace(to_dict=lambda: {
            'forward_completion': 1., 'audit': {'legal': True}})

    monkeypatch.setattr(module, 'fast_rollout_remaining', forward)
    chosen, info = controller.checkpoint(aromatic_fixture(3), 'rigid')
    search = info['retry_selection']['attempts'][0]['search']
    assert search['result_candidate_count'] == 2
    assert search['selected_reward'] == max(search['result_candidate_rewards'])
    assert search['stop_reason'] == 'result_candidates_ready'
    assert calls == [chosen.state_signature()]


def test_verbose_search_reports_candidate_selection(capsys):
    controller = StageController(AssemblyConfig(connection_candidate_limit=2,
        forward_max_rollbacks=0, verbose=True))
    _, info = controller.search(aromatic_fixture(3), 'rigid')
    output = capsys.readouterr().out
    assert 'candidates=2/2' in output
    assert 'stop=result_candidates_ready' in output
    assert info['result_candidate_count'] == 2
