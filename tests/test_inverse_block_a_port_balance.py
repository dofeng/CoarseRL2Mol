"""Regression coverage for converted histograms and Block A transactions."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.layer0.inverse_layer0 import Layer0Estimator
from model.inverse.layer1.port_role_balance import (
    apply_layer1_port_balance,
    audit_layer1_port_roles,
    select_layer1_port_balance,
)
from model.inverse.layer4 import inverse_layer4_block_a as block_a
from model.inverse.layer4.inverse_layer4 import Layer4Adjuster
from model.shared.inverse_common import E_SU, get_effective_hist_element_vector


def histogram(counts):
    result = torch.zeros(33, dtype=torch.long)
    for su, count in counts.items():
        result[su] = count
    return result


def empty_special_meta():
    degrees = {19: {1: 0, 2: 0, 3: 0}, 20: {1: 0, 2: 0, 3: 0}, 21: {2: 0, 3: 0}}
    return {
        'special_degree_meta': degrees,
        'special_partition_meta': {19: {'ether': dict(degrees[19]), 'thio': dict(degrees[19])}},
        'special_anchor_mode_meta': {
            19: {name: dict(degrees[19]) for name in ('ether_single', 'ether_double', 'thio_single', 'thio_double')},
            20: {'single': dict(degrees[20]), 'double': {2: 0, 3: 0}},
            21: {'single': dict(degrees[21]), 'double': dict(degrees[21])},
        },
    }


def make_adjuster(meta=None, target=None):
    layer0 = Layer0Estimator(None, E_SU)
    adjuster = Layer4Adjuster(layer0_estimator=layer0)
    adjuster.fixed_partition_meta = copy.deepcopy(meta or empty_special_meta())
    layer0.fixed_partition_meta = copy.deepcopy(adjuster.fixed_partition_meta)
    if target is not None:
        adjuster.E_target = target
    return adjuster, layer0


@pytest.fixture
def log_case():
    # First accepted histogram in the user's LY-1 log.
    H = histogram({
        0: 1, 1: 1, 2: 3, 3: 14, 5: 2, 6: 5, 7: 2, 9: 16, 10: 38,
        11: 60, 12: 142, 13: 237, 14: 5, 15: 14, 16: 1, 17: 3, 18: 3,
        19: 7, 20: 2, 22: 4, 23: 31, 24: 8, 25: 1, 26: 3, 27: 3,
        28: 2, 29: 2, 30: 2, 31: 2,
    })
    meta = empty_special_meta()
    meta.update({
        'o_base_19': 5, 'o_fixed_edges_19': 7,
        's_reserved_19': 2, 's_fixed_edges_19': 2,
        'special_degree_meta': {19: {1: 1, 2: 4, 3: 2}, 20: {1: 1, 2: 0, 3: 1}, 21: {2: 0, 3: 0}},
        'special_partition_meta': {19: {'ether': {1: 1, 2: 3, 3: 1}, 'thio': {1: 0, 2: 1, 3: 1}}},
        'special_anchor_mode_meta': {
            19: {
                'ether_single': {1: 1, 2: 1, 3: 1}, 'ether_double': {2: 2, 3: 0},
                'thio_single': {1: 0, 2: 1, 3: 1}, 'thio_double': {2: 0, 3: 0},
            },
            20: {'single': {1: 1, 2: 0, 3: 1}, 'double': {2: 0, 3: 0}},
            21: {'single': {2: 0, 3: 0}, 'double': {2: 0, 3: 0}},
        },
    })
    target = torch.tensor([600, 350, 27, 7, 4, 0], dtype=torch.float)
    return H, meta, target


def test_converted_odd_histogram_balanced_with_actual_metadata(log_case):
    H, meta, E = log_case
    # Layer0's balanced target is not the same as the conversion-based result.
    target = H.clone()
    for su, count in {1: 0, 2: 0, 3: 18, 5: 4, 9: 18, 19: 8, 28: 4, 29: 4}.items():
        target[su] = count
    target_meta = copy.deepcopy(meta)
    target_meta['special_degree_meta'][19] = {1: 1, 2: 4, 3: 3}
    target_meta['special_partition_meta'][19]['ether'] = {1: 1, 2: 3, 3: 2}
    target_meta['special_anchor_mode_meta'][19]['ether_single'][3] = 2
    target_meta.update(o_base_19=6, o_fixed_edges_19=8)
    actual = target.clone()
    actual[11] -= 4
    actual[24] -= 1
    before = audit_layer1_port_roles(actual, target_meta['special_degree_meta'])
    assert before['total_ports'] == 1505
    assert not before['ok']
    layer0 = Layer0Estimator(None, E_SU)
    original = actual.clone()
    out, final_meta = layer0._balance_existing_histogram_port_roles(actual, E, target_meta)
    after = audit_layer1_port_roles(out, final_meta['special_degree_meta'])
    assert after['total_ports'] == 1506
    assert after['ok']
    assert final_meta['layer1_port_role_balance']['after'] == after
    assert torch.equal(actual, original)
    assert out[11] < target[11]  # No substitution of the additive Layer0 target.
    elements = get_effective_hist_element_vector(out, final_meta['special_degree_meta'])
    assert torch.equal(elements[[0, 2, 3, 4, 5]], E[[0, 2, 3, 4, 5]])


def test_block_a_log_moves_rebalance_actual_output_before_returning(log_case, monkeypatch):
    H, meta, E = log_case
    adjuster, layer0 = make_adjuster(meta, E)
    balanced_target = H.clone()
    for su, count in {1: 0, 2: 0, 3: 18, 5: 4, 9: 18, 19: 8, 28: 4, 29: 4}.items():
        balanced_target[su] = count
    target_meta = copy.deepcopy(meta)
    target_meta['special_degree_meta'][19] = {1: 1, 2: 4, 3: 3}
    target_meta['special_partition_meta'][19]['ether'] = {1: 1, 2: 3, 3: 2}
    target_meta['special_anchor_mode_meta'][19]['ether_single'][3] = 2
    target_meta.update(o_base_19=6, o_fixed_edges_19=8)

    def propose(*args, **kwargs):
        candidate = H.clone()
        candidate[1] -= 1
        candidate[2] -= 3
        candidate[3] += 4
        return candidate, [{'from': 2, 'to': 3}] * 3 + [{'from': 1, 'to': 3}], {}

    monkeypatch.setattr(block_a, 'adjust_carbonyl_by_difference_impl', propose)
    monkeypatch.setattr(
        layer0, '_rebalance_oxygen_linked_units_after_carbonyl_adjust',
        lambda *args: (balanced_target.clone(), {'fixed_partition_meta': copy.deepcopy(target_meta)}),
    )
    out, moves, diag = adjuster.adjust_block_a_carbonyl_anchor(H, None, None, max_moves=4, carbonyl_max_moves=4)
    assert len([mv for mv in moves if mv['block'] == 'A_center']) == 4
    assert diag['post_oxygen_rebalance']['converted_port_role_audit']['total_ports'] == 1505
    assert diag['final_port_role_audit']['total_ports'] == 1506
    assert diag['final_port_role_audit']['ok']
    assert adjuster.fixed_partition_meta['layer1_port_role_balance']['after'] == diag['final_port_role_audit']
    assert layer0.fixed_partition_meta == adjuster.fixed_partition_meta
    assert adjuster._evaluate_required_hist_constraints(out, E)['ok']


@pytest.mark.parametrize('with_target', [False, True])
def test_histogram_precheck_rejects_odd_ports_even_with_stale_pass_record(with_target):
    H = histogram({13: 6, 22: 1})
    meta = empty_special_meta()
    meta['layer1_port_role_balance'] = {'after': {'ok': True, 'total_ports': 12}}
    adjuster, _ = make_adjuster(meta)
    E = get_effective_hist_element_vector(H) if with_target else None
    diagnostic = adjuster._evaluate_required_hist_constraints(H, E)
    assert not diagnostic['ok']
    assert diagnostic['port_role_audit']['total_ports'] == 13
    assert 'layer1_port_roles_unbalanced' in diagnostic['reason']


def test_histogram_precheck_rejects_even_total_with_odd_internal_roles():
    H = histogram({11: 1, 12: 1, 13: 5, 22: 2})
    adjuster, _ = make_adjuster()
    audit = adjuster._evaluate_required_hist_constraints(H, None)
    assert audit['port_role_audit']['total_even']
    assert not audit['ok']


def test_role_balance_chooses_direction_within_su12_cap():
    H = histogram({11: 1, 12: 3, 13: 4, 22: 2})
    audit = audit_layer1_port_roles(H)
    assert audit['role_balance_required']
    choice = select_layer1_port_balance(H, audit)
    assert (choice['from_su'], choice['to_su']) == (12, 11)
    out, diagnostic = apply_layer1_port_balance(H, choice)
    assert diagnostic['after']['ok']
    assert out[12] == 2


def test_role_balance_rejects_forced_conversion_above_cap():
    H = histogram({11: 1, 12: 3, 13: 4, 22: 2})
    with pytest.raises(RuntimeError, match='su12_cap_exceeded'):
        apply_layer1_port_balance(H, {'required': True, 'possible': True, 'from_su': 11, 'to_su': 12, 'count': 1})


def test_parity_fallback_updates_all_special_degree_views():
    H = histogram({13: 6, 19: 1, 29: 1})
    meta = empty_special_meta()
    meta.update(o_base_19=1, o_fixed_edges_19=1, s_reserved_19=0, s_fixed_edges_19=0)
    meta['special_degree_meta'][19][3] = 1
    meta['special_partition_meta'][19]['ether'][3] = 1
    meta['special_anchor_mode_meta'][19]['ether_single'][3] = 1
    E = get_effective_hist_element_vector(H, meta['special_degree_meta'])
    out, final = Layer0Estimator(None, E_SU)._balance_existing_histogram_port_roles(H, E, meta)
    assert torch.equal(out, H)
    assert final['special_degree_meta'][19] == {1: 0, 2: 1, 3: 0}
    assert final['special_partition_meta'][19]['ether'][2] == 1
    assert final['special_anchor_mode_meta'][19]['ether_single'][2] == 1
    assert final['layer1_port_role_balance']['after']['ok']


def test_inconsistent_mode_metadata_rejected_without_publishing_state():
    H = histogram({13: 6, 19: 1, 28: 1})
    meta = empty_special_meta()
    meta['special_degree_meta'][19][3] = 1
    layer0 = Layer0Estimator(None, E_SU)
    before = copy.deepcopy(layer0.fixed_partition_meta)
    with pytest.raises(ValueError, match='special_anchor_mode_meta_mismatch'):
        layer0._balance_existing_histogram_port_roles(H, get_effective_hist_element_vector(H), meta)
    assert layer0.fixed_partition_meta == before


def test_pipeline_rollback_restores_every_metadata_view():
    pipe = InversePipelineV3.__new__(InversePipelineV3)
    pipe.layer4_adjuster = SimpleNamespace()
    pipe.layer0_estimator = SimpleNamespace()
    fields = {
        'special_degree_meta': 'special_degree_meta',
        'special_partition_meta': 'special_partition_meta',
        'special_anchor_mode_meta': 'special_anchor_mode_meta',
        'degree_parity_adjustment': 'degree_parity_adjustment',
        'port_role_balance_adjustment': 'layer1_port_role_balance',
    }
    saved = {key: {'accepted': True} for key in fields.values()}
    for attribute in fields:
        setattr(pipe.layer0_estimator, attribute, {'rejected': True})
    pipe._set_fixed_partition_meta(saved)
    for attribute, key in fields.items():
        assert getattr(pipe.layer0_estimator, attribute) == saved[key]
    saved['special_degree_meta']['accepted'] = False
    assert pipe.layer0_estimator.special_degree_meta['accepted']
    pipe._set_fixed_partition_meta({})
    assert all(getattr(pipe.layer0_estimator, attribute) == {} for attribute in fields)


def test_node_sync_recomputes_port_audit_and_fixed_edge_counts():
    pipe = InversePipelineV3.__new__(InversePipelineV3)
    pipe.layer0_estimator = SimpleNamespace()
    pipe.layer4_adjuster = SimpleNamespace()
    meta = empty_special_meta()
    meta['layer1_port_role_balance'] = {'after': {'ok': True, 'total_ports': 999}}
    meta['degree_parity_adjustment'] = {'target_degree_sum_after': 999}
    meta['o_fixed_edges_19'] = 999
    pipe._set_fixed_partition_meta(meta)
    nodes = [SimpleNamespace(su_type=13) for _ in range(6)]
    nodes += [
        SimpleNamespace(su_type=19, target_hop1_degree=2, target_fixed_anchor_count=2, special_anchor_partition='ether'),
        SimpleNamespace(su_type=29),
    ]
    pipe._sync_fixed_partition_meta_from_nodes(nodes)
    actual = pipe.layer4_adjuster.fixed_partition_meta
    assert actual['layer1_port_role_balance']['after']['total_ports'] == 16
    assert actual['layer1_port_role_balance']['after']['ok']
    assert actual['degree_parity_adjustment']['target_degree_sum_after'] == 16
    assert actual['o_fixed_edges_19'] == 2
    assert actual['s_fixed_edges_19'] == 0
    assert actual['n19_total'] == 1
    assert pipe.layer0_estimator.special_anchor_mode_meta[19]['ether_double'][2] == 1


def test_pipeline_candidates_restore_start_and_skip_duplicate_results():
    H = histogram({1: 3, 3: 1, 13: 6})
    pipe = InversePipelineV3.__new__(InversePipelineV3)
    pipe.layer0_estimator = SimpleNamespace()
    pipe.layer4_adjuster = SimpleNamespace()
    saved = empty_special_meta()
    pipe._set_fixed_partition_meta(saved)
    calls = []

    def adjust(**kwargs):
        assert pipe.layer4_adjuster.fixed_partition_meta == saved
        assert pipe.layer0_estimator.special_anchor_mode_meta == saved['special_anchor_mode_meta']
        assert torch.equal(kwargs['H'], H)
        length = kwargs['carbonyl_max_moves']
        calls.append(length)
        # Two long proposals collapse to the same legal one-move result.
        count = min(length, 1)
        candidate = H.clone()
        candidate[1] -= count
        candidate[3] += count
        return candidate, [{'from': 1, 'to': 3}] * count, {}

    pipe.layer4_adjuster.adjust_by_stage = adjust
    iterator = pipe._iter_block_a_candidates(H, [], {}, torch.zeros(6), torch.zeros(2400), {'max_moves': 3, 'carbonyl_max_moves': 3}, 'test')
    candidate, _, _ = next(iterator)
    assert candidate[1] == 2
    pipe.layer0_estimator.special_anchor_mode_meta = {'rejected': True}
    pipe.layer4_adjuster.fixed_partition_meta = {'rejected': True}
    assert list(iterator) == []
    assert calls == [3, 2, 1, 0]
    assert pipe.layer4_adjuster.fixed_partition_meta == saved


def test_pipeline_accepted_candidate_keeps_committed_metadata_on_close():
    H = histogram({1: 3, 3: 1})
    pipe = InversePipelineV3.__new__(InversePipelineV3)
    pipe.layer0_estimator = SimpleNamespace()
    pipe.layer4_adjuster = SimpleNamespace()
    pipe._set_fixed_partition_meta(empty_special_meta())

    def adjust(**kwargs):
        candidate = H.clone()
        candidate[1] -= 1
        candidate[3] += 1
        return candidate, [{'from': 1, 'to': 3}], {}

    pipe.layer4_adjuster.adjust_by_stage = adjust
    iterator = pipe._iter_block_a_candidates(H, [], {}, torch.zeros(6), torch.zeros(2400), {}, 'test')
    next(iterator)
    pipe._set_fixed_partition_meta({'accepted': True})
    iterator.close()
    assert pipe.layer4_adjuster.fixed_partition_meta == {'accepted': True}


def spectral_adjuster():
    adjuster, _ = make_adjuster()
    adjuster._su_common_stats_cache = {
        su: {'mu_median': mu, 'mu_common_min': mu - 1, 'mu_common_max': mu + 1}
        for su, mu in {0: 166, 1: 175, 2: 169, 3: 196}.items()
    }
    return adjuster


def test_carbonyl_does_not_consume_deficient_donors():
    adjuster = spectral_adjuster()
    H = histogram({1: 1, 2: 3, 3: 14})
    ppm = np.linspace(150, 240, 1801)
    residual = sum(2.0 / (1.0 + (ppm - mu) ** 2) for mu in (169, 175, 196))
    out, moves, meta = block_a.adjust_carbonyl_by_difference_impl(
        adjuster, H, ppm, residual, score_rel_threshold=0.02, max_moves=4, min_keep=0,
    )
    assert meta['needs'] == {1: 1, 2: 1, 3: 1}
    assert not moves
    assert torch.equal(out, H)


def test_carbonyl_updates_residual_instead_of_repeating_satisfied_move():
    adjuster = spectral_adjuster()
    H = histogram({1: 3, 3: 1})
    ppm = np.linspace(150, 240, 1801)
    residual = 1.0 / (1.0 + (ppm - 196) ** 2) - 1.0 / (1.0 + (ppm - 175) ** 2)
    out, moves, meta = block_a.adjust_carbonyl_by_difference_impl(
        adjuster, H, ppm, residual, score_rel_threshold=0.02, max_moves=4, min_keep=0,
    )
    assert len(moves) == 1
    assert (moves[0]['from'], moves[0]['to']) == (1, 3)
    assert meta['remaining_needs'] == {1: 0, 2: 0, 3: 0}
    assert out[1] == 2 and out[3] == 2


def test_rejected_carbonyl_bundle_tries_shorter_prefix_and_restores_metadata(monkeypatch):
    H = histogram({1: 2, 3: 1, 11: 2, 13: 6, 22: 2})
    E = get_effective_hist_element_vector(H)
    adjuster, layer0 = make_adjuster(target=E)
    saved = copy.deepcopy(adjuster.fixed_partition_meta)
    calls = []

    def propose(*args, **kwargs):
        out = H.clone()
        out[1] -= 2
        out[3] += 2
        return out, [{'from': 1, 'to': 3}] * 2, {}

    def rebalance(candidate, target):
        calls.append(int(candidate[1]))
        if candidate[1] == 0:
            layer0.special_anchor_mode_meta = {'rejected': True}
            raise RuntimeError('long_bundle_infeasible')
        assert layer0.special_anchor_mode_meta != {'rejected': True}
        # Return a balanced oxygen target for a one-move synthetic candidate.
        target_hist = H.clone()
        target_hist[1] -= 1
        target_hist[3] += 1
        target_hist[28] += 1
        target_hist[11] -= 1
        target_hist[5] += 1
        return target_hist, {'fixed_partition_meta': copy.deepcopy(saved)}

    monkeypatch.setattr(block_a, 'adjust_carbonyl_by_difference_impl', propose)
    monkeypatch.setattr(layer0, '_rebalance_oxygen_linked_units_after_carbonyl_adjust', rebalance)
    out, moves, diag = adjuster.adjust_block_a_carbonyl_anchor(H, None, None, max_moves=2, carbonyl_max_moves=2)
    assert calls == [0, 1]
    assert len([mv for mv in moves if mv['block'] == 'A_center']) == 1
    assert out[1] == 1 and out[3] == 2
    assert diag['final_port_role_audit']['ok']
    assert any(row['reason'] == 'oxygen_rebalance_exception' for row in diag['rejected_bundles'])
    assert layer0.special_anchor_mode_meta != {'rejected': True}


def test_all_rejected_bundles_preserve_histogram_and_metadata(monkeypatch):
    H = histogram({1: 2, 3: 1, 13: 6, 22: 2})
    E = get_effective_hist_element_vector(H)
    adjuster, layer0 = make_adjuster(target=E)
    saved = copy.deepcopy(adjuster.fixed_partition_meta)
    layer0.special_anchor_mode_meta = copy.deepcopy(saved['special_anchor_mode_meta'])

    def propose(*args, **kwargs):
        return H.clone(), [{'from': 1, 'to': 3}] * 2, {}

    def fail(*args):
        layer0.special_anchor_mode_meta = {'rejected': True}
        raise RuntimeError('no_feasible_bundle')

    monkeypatch.setattr(block_a, 'adjust_carbonyl_by_difference_impl', propose)
    monkeypatch.setattr(layer0, '_rebalance_oxygen_linked_units_after_carbonyl_adjust', fail)
    out, moves, _ = adjuster.adjust_block_a_carbonyl_anchor(H, None, None, max_moves=2, carbonyl_max_moves=2)
    assert torch.equal(out, H)
    assert not moves
    assert adjuster.fixed_partition_meta == saved
    assert layer0.special_anchor_mode_meta == saved['special_anchor_mode_meta']
