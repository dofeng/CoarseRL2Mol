from __future__ import annotations

from collections import Counter, defaultdict
import inspect

import numpy as np
import pytest
import torch

from model.inverse.layer1.hop1_adjuster import Hop1Adjuster, PeakRegion
from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.layer1.inverse_layer1 import Layer1Assigner, Layer1GreedyCompleter
from model.inverse.layer1.layer1_validator import Layer1GraphValidator
from model.shared.coarse_graph import NUM_SU_TYPES, PPM_AXIS
from model.shared.inverse_common import _NodeV3, lorentzian_spectrum


def _template(center_su: int, hop1: tuple[int, ...], hop2: tuple[int, ...], mu: float, weight: int):
    return (
        (center_su, hop1, hop2),
        {
            'center_mu': mu,
            'mu_min': mu - 1.0,
            'mu_max': mu + 1.0,
            'sample_count': weight,
            'samples': {'mu': torch.tensor([mu])},
        },
    )


def test_runtime_config_is_restricted_to_the_new_contract() -> None:
    pipeline = object.__new__(InversePipelineV3)
    layer1, layer3 = pipeline._build_layer_runtime_configs()
    assert layer1 == {
        'neg_peak_threshold': -0.8,
        'pos_peak_threshold': 0.8,
        'max_candidate_trials_per_region': 40,
        'max_moves_per_region': 4,
    }
    assert 'hwhm' not in layer3
    with pytest.raises(ValueError, match='unsupported keys'):
        pipeline._build_layer_runtime_configs(layer1_config={'search_profile': 'fast'})
    with pytest.raises(ValueError, match='unsupported keys'):
        pipeline._build_layer_runtime_configs(layer3_config={'hwhm': 1.0})


def test_inverse_constructors_do_not_expose_obsolete_intensity_controls() -> None:
    assert 'nmr_intensity_scale' not in inspect.signature(InversePipelineV3).parameters
    assert 'unit_peak_intensity' not in inspect.signature(InversePipelineV3).parameters
    assert 'intensity_scale' not in inspect.signature(Layer1Assigner).parameters
    assert 'unit_peak_intensity' not in inspect.signature(Hop1Adjuster).parameters


def test_active_library_supplies_weighted_median_unit_peak(tmp_path) -> None:
    templates = dict([
        _template(22, (23,), (), 10.0, 1),
        _template(22, (23,), (24,), 20.0, 5),
    ])
    library = tmp_path / 'library.pt'
    torch.save({'templates': templates}, library)
    adjuster = Hop1Adjuster(
        template_lib_path=str(library),
        carbon_su_set={22},
        exact_hop1_required_centers={22},
    )
    node = _NodeV3(global_id=0, su_type=22, target_hop1_degree=1)
    node.hop1_su = Counter({23: 1})
    snapshot = adjuster.build_eval_snapshot([node], torch.zeros_like(PPM_AXIS))
    assignment = snapshot['assignments'][0]
    assert assignment['mu'] == 20.0
    assert assignment['pi'] == 1.0
    expected = lorentzian_spectrum(
        torch.tensor([20.0]),
        torch.ones(1),
        PPM_AXIS,
        hwhm=1.0,
    )
    assert torch.allclose(snapshot['S_recon'].cpu(), expected, atol=1e-7)


def test_direct_median_hit_excludes_range_only_fallback(tmp_path) -> None:
    direct_key, direct = _template(22, (24,), (), 50.0, 1)
    fallback_key, fallback = _template(22, (25,), (), 40.0, 1)
    fallback['mu_min'] = 49.0
    fallback['mu_max'] = 52.0
    library = tmp_path / 'library.pt'
    torch.save({'templates': {direct_key: direct, fallback_key: fallback}}, library)
    adjuster = Hop1Adjuster(template_lib_path=str(library), carbon_su_set={22})
    candidates = adjuster.find_shift_guided_hop1_targets(
        center_su=22,
        current_hop1=(23,),
        target_mu_range=(49.0, 51.0),
    )
    assert [candidate['hop1_tuple'] for candidate in candidates] == [(24,)]


def test_region_order_is_fixed() -> None:
    assert [name for name, _low, _high in Hop1Adjuster.REGION_ORDER] == [
        '0-90',
        '90-160',
        '160-240',
    ]


def test_strict_swap_cannot_remove_a_fixed_edge() -> None:
    nodes = [
        _NodeV3(global_id=index, su_type=su_type, target_hop1_degree=1)
        for index, su_type in enumerate((22, 23, 24, 25))
    ]
    for left, right in ((0, 1), (2, 3)):
        nodes[left].hop1_ids.append(right)
        nodes[right].hop1_ids.append(left)
        nodes[left].hop1_su[nodes[right].su_type] += 1
        nodes[right].hop1_su[nodes[left].su_type] += 1
    nodes[0].fixed_hop1_ids.add(1)
    nodes[1].fixed_hop1_ids.add(0)
    adjuster = Hop1Adjuster(enforce_external_requirements=False)
    result = adjuster.execute_hop1_replacement(nodes, 0, (24,))
    assert result['success'] is False
    assert nodes[0].hop1_ids == [1]
    assert nodes[1].hop1_ids == [0]


def test_layer1_rejects_odd_entry_ports_without_balancing(monkeypatch) -> None:
    assigner = Layer1Assigner(device='cpu')
    node = _NodeV3(global_id=0, su_type=22, target_hop1_degree=1)
    monkeypatch.setattr(
        assigner,
        '_configure_exact_hop1_shift_constraints',
        lambda _path: None,
    )
    monkeypatch.setattr(assigner, '_initialize_node_pool', lambda _histogram: [node])

    with pytest.raises(RuntimeError, match='odd entry open-port count'):
        assigner.build_rule_valid_graph(
            H_init=torch.zeros(NUM_SU_TYPES),
            E_target=torch.zeros(6),
            eval_lib_path=None,
        )


class _CompletionAssigner:
    def __init__(self, nodes, allowed):
        self._exact_hop1_shift_constraints_enabled = False
        self._carbon_su_set = set()
        self._available_ids_by_su = defaultdict(set)
        for node in nodes:
            self._available_ids_by_su[int(node.su_type)].add(int(node.global_id))
        self.allowed = {tuple(sorted(edge)) for edge in allowed}

    @staticmethod
    def _node_target_degree(node):
        return int(node.target_hop1_degree)

    @staticmethod
    def _node_exact_hop1_shift_state_ok(_node, **_kwargs):
        return True

    def _can_add_hop1_connection(self, _nodes, left, right):
        return tuple(sorted((left.global_id, right.global_id))) in self.allowed

    def _add_bidirectional_hop1(self, nodes, left_id, right_id):
        left, right = nodes[left_id], nodes[right_id]
        if tuple(sorted((left_id, right_id))) not in self.allowed:
            return False
        if right_id in left.hop1_ids:
            return False
        left.hop1_ids.append(right_id)
        right.hop1_ids.append(left_id)
        left.hop1_su[right.su_type] += 1
        right.hop1_su[left.su_type] += 1
        return True

    @staticmethod
    def _remove_bidirectional_hop1_with_force(nodes, left_id, right_id, force=False):
        left, right = nodes[left_id], nodes[right_id]
        left.hop1_ids.remove(right_id)
        right.hop1_ids.remove(left_id)
        left.hop1_su[right.su_type] -= 1
        right.hop1_su[left.su_type] -= 1
        return True


def test_completion_selects_the_node_with_fewest_legal_partners() -> None:
    nodes = [
        _NodeV3(global_id=index, su_type=10 + index, target_hop1_degree=1)
        for index in range(4)
    ]
    assigner = _CompletionAssigner(nodes, {(0, 2), (0, 3), (1, 2)})
    summary = Layer1GreedyCompleter(assigner, nodes).solve()
    assert summary['final_open_ports'] == 0
    assert nodes[1].hop1_ids == [2]
    assert nodes[0].hop1_ids == [3]


def test_completion_partner_order_prefers_the_less_used_projected_motif() -> None:
    nodes = [
        _NodeV3(global_id=index, su_type=su_type, target_hop1_degree=1)
        for index, su_type in enumerate((10, 11, 12, 10, 11))
    ]
    nodes[3].hop1_ids = [4]
    nodes[3].hop1_su = Counter({11: 1})
    nodes[4].hop1_ids = [3]
    nodes[4].hop1_su = Counter({10: 1})
    assigner = _CompletionAssigner(nodes, {(0, 1), (0, 2)})
    assigner._exact_hop1_shift_constraints_enabled = True
    assigner._carbon_su_set = {10}
    assigner._exact_hop1_signatures_by_center_degree = {
        (10, 1): ((11,), (12,)),
    }

    completer = Layer1GreedyCompleter(assigner, nodes)
    assert completer._order_partners(nodes[0], [1, 2]) == [2, 1]


def test_completion_reselects_after_rewind_opens_a_new_partner() -> None:
    nodes = [
        _NodeV3(global_id=index, su_type=10 + index, target_hop1_degree=1)
        for index in range(4)
    ]
    assigner = _CompletionAssigner(nodes, {(0, 1), (0, 2), (1, 3)})
    completer = Layer1GreedyCompleter(assigner, nodes)

    assert completer._add_edge(0, 1)
    completer._decision_stack.append({
        'center_id': 0,
        'chosen_id': 1,
        'alternatives': [1],
        'next_index': 1,
    })

    summary = completer.solve()

    assert summary['final_open_ports'] == 0
    assert set(nodes[0].hop1_ids) == {2}
    assert set(nodes[1].hop1_ids) == {3}
    assert completer.repair_rewinds == 1


def test_completion_failed_repair_restores_graph_and_reports_fresh_domain() -> None:
    nodes = [
        _NodeV3(global_id=index, su_type=10 + index, target_hop1_degree=1)
        for index in range(4)
    ]
    assigner = _CompletionAssigner(nodes, {(0, 1)})
    completer = Layer1GreedyCompleter(assigner, nodes)

    assert completer._add_edge(0, 1)
    completer._decision_stack.append({
        'center_id': 0,
        'chosen_id': 1,
        'alternatives': [1],
        'next_index': 1,
    })

    with pytest.raises(RuntimeError, match=r"legal_partner_types': \[\]"):
        completer.solve()

    assert nodes[0].hop1_ids == [1]
    assert nodes[1].hop1_ids == [0]


class _ValidatorAssigner:
    _carbon_su_set = set()

    @staticmethod
    def _histogram_from_nodes(nodes):
        histogram = torch.zeros(NUM_SU_TYPES, dtype=torch.long)
        for node in nodes:
            histogram[int(node.su_type)] += 1
        return histogram

    @staticmethod
    def _node_target_degree(node):
        return int(node.target_hop1_degree)

    @staticmethod
    def _su11_fixed_external_errors(_nodes):
        return []

    @staticmethod
    def _current_neighbor_nodes(nodes, node):
        return [nodes[index] for index in node.hop1_ids]

    @staticmethod
    def _port_sets_for_node(_node):
        return None

    @staticmethod
    def _violates_terminal_neighbor_limit(_nodes, _node):
        return False

    @staticmethod
    def _forbidden_p_pair_rows(_nodes):
        return []

    @staticmethod
    def _node_anchor_partition(_node):
        return None

    @staticmethod
    def _required_external_candidates_for_node(_node):
        return []

    @staticmethod
    def _node_fixed_anchor_target(_node):
        return 0

    @staticmethod
    def _has_required_external_for_node(_node, _nodes):
        return True

    @staticmethod
    def _fixed_anchor_invariant_errors(_nodes, **_kwargs):
        return []

    @staticmethod
    def _node_edge_semantics_ok(_nodes, _left, _right, E_target=None):
        return True


def test_validator_does_not_apply_global_hydrogen_tolerance() -> None:
    node = _NodeV3(global_id=0, su_type=22, target_hop1_degree=0)
    histogram = torch.zeros(NUM_SU_TYPES, dtype=torch.long)
    histogram[22] = 1
    valid, errors = Layer1GraphValidator(_ValidatorAssigner()).validate(
        [node],
        histogram,
        E_target=torch.tensor([1.0, 999999.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert valid is True
    assert errors == []


def test_validator_requires_the_su9_carbonyl_edge_to_be_bidirectionally_fixed() -> None:
    nodes = [
        _NodeV3(global_id=0, su_type=9, target_hop1_degree=1),
        _NodeV3(global_id=1, su_type=0, target_hop1_degree=1),
    ]
    nodes[0].hop1_ids = [1]
    nodes[0].hop1_su = Counter({0: 1})
    nodes[1].hop1_ids = [0]
    nodes[1].hop1_su = Counter({9: 1})
    histogram = torch.zeros(NUM_SU_TYPES, dtype=torch.long)
    histogram[9] = 1
    histogram[0] = 1
    validator = Layer1GraphValidator(_ValidatorAssigner())

    valid, errors = validator.validate(nodes, histogram)
    assert valid is False
    assert any('未双向锁定' in error for error in errors)

    nodes[0].fixed_hop1_ids.add(1)
    nodes[1].fixed_hop1_ids.add(0)
    valid, errors = validator.validate(nodes, histogram)
    assert valid is True
    assert errors == []


@pytest.mark.parametrize(
    ('final_r2', 'expected_hop1', 'expected_commit'),
    ((0.6, (3,), True), (0.4, (1,), False)),
)
def test_region_batch_allows_an_intermediate_drop_and_rolls_back_at_batch_end(
    monkeypatch,
    final_r2,
    expected_hop1,
    expected_commit,
) -> None:
    adjuster = Hop1Adjuster(
        carbon_su_set={22},
        enforce_external_requirements=False,
    )
    node = _NodeV3(global_id=0, su_type=22, target_hop1_degree=1)
    node.hop1_su = Counter({1: 1})
    nodes = [node]
    target = torch.ones(4)
    ppm = np.asarray([10.0, 20.0, 100.0, 180.0])
    state_r2 = {1: 0.5, 2: 0.4, 3: float(final_r2)}
    attempted_states = []

    def _state() -> int:
        return int(next(iter(node.hop1_su.elements())))

    def _snapshot(_nodes, _target):
        state = _state()
        recon_by_state = {
            1: torch.tensor([2.0, 0.0, 1.0, 1.0]),
            2: torch.tensor([3.0, -1.0, 1.0, 1.0]),
            3: torch.tensor([1.1, 0.9, 1.0, 1.0]),
        }
        recon = recon_by_state[state]
        return {
            'ppm': ppm,
            'S_target': target,
            'S_recon': recon,
            'diff': (target - recon).numpy(),
            'r2': state_r2[state],
            'matched_cnt': 1,
            'carbon_cnt': 1,
            'assignments': {0: {'mu': 10.0}},
        }

    negative = PeakRegion(0.0, 90.0, -1.0, 10.0, area=1.0)
    positive = PeakRegion(0.0, 90.0, 1.0, 20.0, area=1.0)
    monkeypatch.setattr(adjuster, 'build_eval_snapshot', _snapshot)
    monkeypatch.setattr(
        adjuster,
        'build_assignments',
        lambda _nodes: {0: {'mu': 10.0}},
    )
    monkeypatch.setattr(
        adjuster,
        'analyze_difference_spectrum',
        lambda *_args, **_kwargs: (
            ([negative], [positive]) if _state() < 3 else ([], [])
        ),
    )
    monkeypatch.setattr(
        adjuster,
        '_positive_regions_for_su',
        lambda *_args, **_kwargs: [positive],
    )
    monkeypatch.setattr(
        adjuster,
        'find_shift_guided_hop1_targets',
        lambda *_args, **_kwargs: [{
            'hop1_tuple': (_state() + 1,),
            'motif_usage': 0,
            'distance': 0.0,
        }],
    )
    monkeypatch.setattr(
        adjuster,
        'replaceable_resource_shortage',
        lambda *_args, **_kwargs: Counter(),
    )

    def _execute(**kwargs):
        undo_state = adjuster._capture_topology_state(nodes)
        new_state = int(tuple(kwargs['new_hop1_tuple'])[0])
        node.hop1_su = Counter({new_state: 1})
        attempted_states.append(new_state)
        return {
            'success': True,
            'affected_nodes': [0],
            '_undo_state': undo_state,
        }

    monkeypatch.setattr(adjuster, 'execute_hop1_replacement', _execute)
    _nodes, summary = adjuster.adjust_regions(
        nodes,
        target,
        E_target=None,
        validator=lambda _candidate: (True, []),
        max_candidate_trials_per_region=4,
        max_moves_per_region=2,
    )

    assert state_r2[2] < state_r2[1]
    assert attempted_states[:2] == [2, 3]
    assert tuple(node.hop1_su.elements()) == expected_hop1
    assert summary['regions'][0]['committed'] is expected_commit
    assert summary['total_adjustments'] == (2 if expected_commit else 0)
