import copy
from collections import Counter
from types import SimpleNamespace
from types import MethodType

import pytest
import torch

from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.layer1.inverse_layer1 import Layer1Assigner
from model.inverse.layer1.inverse_layer1 import Layer1GreedyCompleter
from model.shared.coarse_graph import NUM_SU_TYPES


class _Node:
    def __init__(
        self,
        global_id: int,
        su_type: int,
        remaining_ports: int,
        hop1_su: Counter | None = None,
    ) -> None:
        self.global_id = int(global_id)
        self.su_type = int(su_type)
        self._remaining_ports = int(remaining_ports)
        self.hop1_su = Counter(hop1_su or {})
        self.hop1_ids = []

    def remaining_hop1_slots(self) -> int:
        return int(self._remaining_ports)


def _completer(nodes, template_costs=None):
    template_costs = dict(template_costs or {})
    completer = object.__new__(Layer1GreedyCompleter)
    completer.nodes = nodes
    completer.assigner = SimpleNamespace(
        _exact_hop1_shift_constraints_enabled=False,
        _carbon_su_set={int(node.su_type) for node in nodes},
        _node_target_degree=lambda node: int(
            node.remaining_hop1_slots() + sum(node.hop1_su.values())
        )
    )
    completer._component = lambda _start_id: set()
    completer._candidate_template_concentration_costs = (
        lambda _center, partner_ids: {
            int(partner_id): float(template_costs.get(int(partner_id), 0.0))
            for partner_id in partner_ids
        }
    )
    return completer


def test_template_diversity_and_resource_heuristics_share_primary_score():
    center = _Node(0, 11, 3, Counter({22: 1}))
    repeated_su_high_ports = _Node(1, 22, 3)
    new_su_low_ports = _Node(2, 23, 1)
    balanced_new_su = _Node(3, 24, 2)
    nodes = [center, repeated_su_high_ports, new_su_low_ports, balanced_new_su]

    completer = _completer(nodes, template_costs={1: 10, 2: 0, 3: 100})
    ordered = completer._order_partners(center, [1, 2, 3])

    assert completer._joint_primary_scores(center, [1, 2, 3]) == {
        1: pytest.approx(0.50),
        2: pytest.approx(0.25),
        3: pytest.approx(0.625),
    }
    assert ordered == [2, 1, 3]


def test_template_diversity_can_outweigh_partner_port_advantage():
    center = _Node(0, 11, 2)
    high_port_crowded = _Node(1, 22, 3)
    low_port_diverse = _Node(2, 23, 1)
    nodes = [center, high_port_crowded, low_port_diverse]
    completer = _completer(nodes, template_costs={1: 10, 2: 0})

    ordered = completer._order_partners(center, [1, 2])

    assert ordered == [2, 1]


def test_partial_nodes_contribute_soft_exact_template_usage():
    center = _Node(0, 11, 2)
    partial_11 = _Node(1, 11, 1, Counter({22: 1}))
    partner_22 = _Node(2, 22, 1)
    partner_24 = _Node(3, 24, 1)
    nodes = [center, partial_11, partner_22, partner_24]
    completer = object.__new__(Layer1GreedyCompleter)
    completer.nodes = nodes
    completer.assigner = SimpleNamespace(
        _exact_hop1_shift_constraints_enabled=True,
        _carbon_su_set={11, 22, 24},
        _node_target_degree=lambda node: int(
            node.remaining_hop1_slots() + sum(node.hop1_su.values())
        ),
        _exact_hop1_signatures_by_center_degree={
            (11, 2): ((22, 22), (24, 24)),
            (22, 1): ((11,),),
            (24, 1): ((11,),),
        },
    )
    completer._exact_options_cache = {}
    completer._component = lambda _start_id: set()

    costs = completer._candidate_template_concentration_costs(
        center,
        [2, 3],
    )

    assert costs[3] < costs[2]
    assert completer._order_partners(center, [2, 3]) == [3, 2]


def test_partner_without_two_sided_exact_template_is_excluded():
    center = _Node(0, 11, 1)
    legal = _Node(1, 22, 1)
    illegal = _Node(2, 23, 1)
    nodes = [center, legal, illegal]
    completer = object.__new__(Layer1GreedyCompleter)
    completer.nodes = nodes
    completer.assigner = SimpleNamespace(
        _exact_hop1_shift_constraints_enabled=True,
        _carbon_su_set={11, 22, 23},
        _node_target_degree=lambda node: 1,
        _exact_hop1_signatures_by_center_degree={
            (11, 1): ((22,), (23,)),
            (22, 1): ((11,),),
            (23, 1): ((12,),),
        },
    )
    completer._exact_options_cache = {}
    completer._component = lambda _start_id: set()

    assert completer._order_partners(center, [1, 2]) == [1]


def test_primary_scoring_is_invariant_to_su_number_renaming():
    def _ordered_ids(center_su, repeated_su, other_su, final_su):
        center = _Node(0, center_su, 3, Counter({repeated_su: 1}))
        nodes = [
            center,
            _Node(1, repeated_su, 3),
            _Node(2, other_su, 1),
            _Node(3, final_su, 2),
        ]
        completer = _completer(
            nodes,
            template_costs={1: 10, 2: 0, 3: 100},
        )
        return completer._order_partners(center, [1, 2, 3])

    assert _ordered_ids(11, 22, 23, 24) == _ordered_ids(4, 7, 8, 9)


def test_forced_branch_requires_same_primary_frontier_and_prefix():
    nodes = [_Node(0, 11, 2), _Node(1, 22, 2), _Node(2, 23, 2)]
    completer = _completer(nodes)
    completer.phase = 'non_aromatic'
    completer._decision_stack = []
    completer._edge_state_signature = lambda: ((7, 8),)
    completer._joint_primary_frontier = lambda _center, _ordered: [1]
    completer._add_edge = lambda _center_id, _partner_id: True
    completer.decisions = 0

    completer.forced_decision = {
        'phase': 'non_aromatic',
        'stack_index': 0,
        'center_id': 0,
        'chosen_partner_id': 2,
        'prefix_edge_signature': [(7, 8)],
    }
    with pytest.raises(RuntimeError, match='partner_not_in_primary_frontier'):
        completer._commit_first_candidate(nodes[0], [1, 2], 'ordinary')

    completer.forced_decision = {
        'phase': 'non_aromatic',
        'stack_index': 0,
        'center_id': 0,
        'chosen_partner_id': 1,
        'prefix_edge_signature': [(6, 8)],
    }
    with pytest.raises(RuntimeError, match='replay_prefix_mismatch'):
        completer._commit_first_candidate(nodes[0], [1, 2], 'ordinary')


def test_graph_variants_are_independent_and_deduplicated():
    assigner = object.__new__(Layer1Assigner)
    seed_nodes = [SimpleNamespace(marker=0)]
    seed_markers = []

    def fake_build(
        self,
        H_init,
        E_target,
        eval_lib_path,
        identity_seed_nodes=None,
        verbose=False,
        forced_decision=None,
    ):
        del self, H_init, E_target, eval_lib_path, verbose
        seed_markers.append(identity_seed_nodes[0].marker)
        identity_seed_nodes[0].marker = 99
        if forced_decision and int(forced_decision['chosen_partner_id']) == 12:
            raise RuntimeError('synthetic branch failure')
        edge_marker = (
            1
            if not forced_decision
            else 2
        )
        nodes = [
            SimpleNamespace(
                global_id=0,
                su_type=11,
                hop1_ids=[edge_marker],
            )
        ]
        completion = {
            'forced_decision_applied': bool(forced_decision),
            'branch_points': [] if forced_decision else [{
                'phase': 'non_aromatic',
                'stack_index': 4,
                'center_id': 0,
                'chosen_partner_id': 10,
                'prefix_edge_signature': [(0, 4)],
                'primary_frontier': [10, 11, 12, 13],
                'alternative_partner_ids': [11, 12, 13],
            }],
        }
        return nodes, completion

    assigner.build_rule_valid_graph = MethodType(fake_build, assigner)
    variants, diagnostics = assigner.build_rule_valid_graph_variants(
        H_init=torch.zeros(NUM_SU_TYPES),
        E_target=torch.zeros(6),
        eval_lib_path='unused.pt',
        identity_seed_nodes=seed_nodes,
        candidate_count=4,
        branch_limit=4,
    )

    assert seed_nodes[0].marker == 0
    assert seed_markers == [0, 0, 0, 0]
    assert len(variants) == 2
    assert diagnostics['failed_branch_count'] == 1
    assert len({row['graph_signature'] for row in variants}) == 2
    assert variants[1]['branch']['chosen_partner_id'] == 11
    assert variants[1]['branch']['prefix_edge_signature'] == [(0, 4)]


def test_graph_variant_sampling_covers_both_completion_phases():
    assigner = object.__new__(Layer1Assigner)
    forced_phases = []

    def branch_point(phase, stack_index, center_id, chosen_id, alternative_id):
        return {
            'phase': phase,
            'stack_index': stack_index,
            'center_id': center_id,
            'chosen_partner_id': chosen_id,
            'prefix_edge_signature': [],
            'primary_frontier': [chosen_id, alternative_id],
            'alternative_partner_ids': [alternative_id],
        }

    points = [
        branch_point('non_aromatic', 0, 0, 10, 11),
        branch_point('non_aromatic', 1, 1, 12, 13),
        branch_point('aromatic', 0, 2, 14, 15),
        branch_point('aromatic', 1, 3, 16, 17),
    ]

    def fake_build(
        self,
        H_init,
        E_target,
        eval_lib_path,
        identity_seed_nodes=None,
        verbose=False,
        forced_decision=None,
    ):
        del self, H_init, E_target, eval_lib_path, identity_seed_nodes, verbose
        if forced_decision:
            forced_phases.append(forced_decision['phase'])
            edge_marker = int(forced_decision['chosen_partner_id'])
        else:
            edge_marker = 10
        return [
            SimpleNamespace(global_id=0, su_type=11, hop1_ids=[edge_marker])
        ], {
            'branch_points': [] if forced_decision else points,
            'forced_decision_applied': bool(forced_decision),
        }

    assigner.build_rule_valid_graph = MethodType(fake_build, assigner)
    variants, diagnostics = assigner.build_rule_valid_graph_variants(
        H_init=torch.zeros(NUM_SU_TYPES),
        E_target=torch.zeros(6),
        eval_lib_path='unused.pt',
        candidate_count=3,
        branch_limit=2,
    )

    assert len(variants) == 3
    assert forced_phases == ['non_aromatic', 'aromatic']
    assert diagnostics['considered_branch_point_count_by_phase'] == {
        'non_aromatic': 1,
        'aromatic': 1,
    }


class _FakeHop1Adjuster:
    def build_eval_snapshot(self, nodes, _target):
        node = nodes[0]
        r2 = (
            node.final_layer1_r2
            if node.stage == 'refined'
            else node.raw_layer1_r2
        )
        return {
            'r2': float(r2),
            'matched_cnt': 1,
            'carbon_cnt': 1,
            'assignments': {0: {'mu': 10.0}},
        }

    @staticmethod
    def apply_assignments_to_nodes(nodes, assignments):
        for node in nodes:
            node.mu = float(assignments[int(node.global_id)]['mu'])
            node.pi = 1.0

    def evaluate_with_library(self, nodes, S_target, output_dir=None):
        del S_target, output_dir
        return {'r2': float(nodes[0].final_layer1_r2)}


class _FakeLayer1Assigner:
    def __init__(self):
        self._hop1_adjuster = _FakeHop1Adjuster()
        self._carbon_su_set = {11}
        self._last_hop1_adjust_diagnostics = {}
        self._last_shift_coverage_audit = {}
        self._last_completion_diagnostics = {}
        self.refined_tags = []
        self.variants = [
            self._variant('A', 0.4, 0.5),
            self._variant('B', 0.3, 0.4),
            self._variant('C', 0.2, 0.9),
        ]

    @staticmethod
    def _variant(tag, raw_layer1_r2, final_layer1_r2):
        node = SimpleNamespace(
            global_id=0,
            su_type=11,
            hop1_ids=[],
            hop1_su=Counter({22: 1}),
            mu=0.0,
            pi=0.0,
            tag=tag,
            stage='raw',
            raw_layer1_r2=float(raw_layer1_r2),
            final_layer1_r2=float(final_layer1_r2),
        )
        return {
            'variant_id': tag,
            'nodes': [node],
            'construction': {'tag': tag},
            'branch': None,
            'graph_signature': f'raw-{tag}',
        }

    def build_rule_valid_graph_variants(self, **_kwargs):
        return self.variants, {'generated_candidate_count': len(self.variants)}

    @staticmethod
    def audit_layer1_shift_coverage(_nodes, _path):
        return {'ok': True, 'matched_count': 1, 'carbon_count': 1}

    def layer1_assign(self, prebuilt_nodes, **_kwargs):
        nodes = copy.deepcopy(prebuilt_nodes)
        nodes[0].stage = 'refined'
        self.refined_tags.append(nodes[0].tag)
        self._last_hop1_adjust_diagnostics = {
            'initial_r2': nodes[0].raw_layer1_r2,
            'total_adjustments': 1,
            'moves': [],
        }
        self._last_shift_coverage_audit = {'ok': True, 'tag': nodes[0].tag}
        return nodes

    @staticmethod
    def validate_graph_consistency(**_kwargs):
        return True, []

    @staticmethod
    def _graph_signature(nodes):
        return f"final-{nodes[0].tag}"


def test_multistart_prefilters_then_selects_by_final_layer2_r2():
    pipeline = object.__new__(InversePipelineV3)
    pipeline.device = 'cpu'
    pipeline.layer4_adjuster = SimpleNamespace(fixed_partition_meta={})
    pipeline.layer1_assigner = _FakeLayer1Assigner()
    pipeline.layer2_estimator = SimpleNamespace(stats={})
    raw_layer2 = {'A': 0.9, 'B': 0.8, 'C': 0.7}
    final_layer2 = {'A': 0.1, 'B': 0.6, 'C': 0.99}

    def fake_layer2(
        self,
        nodes,
        S_target,
        E_target,
        lib_path=None,
        output_dir=None,
        verbose=False,
    ):
        del S_target, E_target, lib_path, output_dir, verbose
        node = nodes[0]
        score = (
            final_layer2[node.tag]
            if node.stage == 'refined'
            else raw_layer2[node.tag]
        )
        node.layer2_r2 = float(score)
        node.mu = 100.0 + float(score)
        self.layer2_estimator.stats = {'tag': node.tag, 'score': score}
        return nodes

    def fake_difference(self, nodes, S_target):
        del self, S_target
        return {'r2': float(nodes[0].layer2_r2), 'ppm': [], 'diff': []}

    pipeline.layer2_assign = MethodType(fake_layer2, pipeline)
    pipeline._compute_difference_spectrum_from_nodes_mu = MethodType(
        fake_difference,
        pipeline,
    )

    nodes, _histogram, difference, r2 = pipeline._run_layer12_cycle(
        H_seed=torch.zeros(NUM_SU_TYPES),
        S_target=torch.zeros(8),
        E_target=torch.zeros(6),
        eval_lib_path='unused.pt',
        layer1_runtime_cfg={
            'layer1_multistart_count': 3,
            'layer1_multistart_refine_count': 2,
            'layer1_multistart_branch_limit': 4,
            'verbose': False,
        },
    )

    assert pipeline.layer1_assigner.refined_tags == ['A', 'B']
    assert nodes[0].tag == 'B'
    assert r2 == pytest.approx(0.6)
    multistart = difference['layer1_build_diagnostics']['multistart']
    assert multistart['selected_variant_id'] == 'B'
    rows = {row['variant_id']: row for row in multistart['candidates']}
    assert rows['A']['selected_for_refinement'] is True
    assert rows['B']['selected_for_refinement'] is True
    assert rows['C']['selected_for_refinement'] is False
    assert rows['B']['selected'] is True
    assert rows['A']['selected'] is False
    assert rows['C']['status'] == 'raw_scored'


def test_layer1_multistart_config_defaults_and_validation():
    pipeline = object.__new__(InversePipelineV3)
    layer1_config, _layer3_config = pipeline._build_layer_runtime_configs()
    assert layer1_config['layer1_multistart_count'] == 6
    assert layer1_config['layer1_multistart_refine_count'] == 2
    assert layer1_config['layer1_multistart_branch_limit'] == 12

    with pytest.raises(ValueError, match='layer1_multistart_count must be >= 1'):
        pipeline._build_layer_runtime_configs({
            'layer1_multistart_count': 0,
        })
    with pytest.raises(
        ValueError,
        match='layer1_multistart_branch_limit must be >= 0',
    ):
        pipeline._build_layer_runtime_configs({
            'layer1_multistart_branch_limit': -1,
        })
