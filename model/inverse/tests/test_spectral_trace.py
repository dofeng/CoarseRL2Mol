"""Regression tests for opt-in observers, with no checkpoints or GPU required."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.layer3.inverse_layer3 import Layer3Estimator
from model.inverse.spectral_trace import SpectralTraceWriter
from model.shared.coarse_graph import E_SU, PPM_AXIS
from model.shared.inverse_common import _NodeV3, lorentzian_spectrum


KEY = (22, (), ())
ELEMENTS = torch.tensor([1., 4., 0., 0., 0., 0.])


def node_at(mu):
    node = _NodeV3(0, 22, mu=mu, z_vec=torch.zeros(16))
    node.template_key = KEY
    return node


def peak(mu):
    return lorentzian_spectrum(torch.tensor([float(mu)]), torch.ones(1), PPM_AXIS)


def reconstruct(nodes):
    carbon = [n for n in nodes if float(E_SU[n.su_type, 0]) > 0]
    return lorentzian_spectrum(torch.tensor([float(n.mu) for n in carbon]),
                               torch.ones(len(carbon)), PPM_AXIS)


def estimator(sample_mu=(17., 22.)):
    result = Layer3Estimator(torch.device('cpu'))
    result._template_cache = {'templates': {KEY: {
        'center_mu': sample_mu[0], 'center_z': torch.zeros(16),
        'samples': {'z': torch.stack([torch.zeros(16), torch.ones(16)]),
                    'mu': torch.tensor(sample_mu)},
    }}}
    return result


def test_checkpoint_roundtrip_parent_counts_and_observer_is_read_only(tmp_path):
    nodes = [node_at(17.)]
    initial_z = nodes[0].z_vec.clone()
    rng = torch.get_rng_state().clone()
    writer = SpectralTraceWriter(output_root=tmp_path, run_id='repeat', target=peak(22),
                                 ppm_axis=PPM_AXIS, reconstruct=reconstruct)
    parent = writer.checkpoint(stage='initial', nodes=nodes)
    assert nodes[0].mu == 17. and torch.equal(nodes[0].z_vec, initial_z)
    assert torch.equal(torch.get_rng_state(), rng)
    changed = [node_at(22.)]
    changed[0].su_type = 23
    child = writer.checkpoint(stage='block_a_accepted', nodes=changed,
                              parent_checkpoint_id=parent, action={'op': '22_to_23'})
    writer.finish(status='complete')
    assert not writer.errors
    data = pd.read_csv(writer.directory / child / 'spectrum.csv')
    np.testing.assert_allclose(data.target - data.reconstructed, data.residual, atol=1e-15)
    np.testing.assert_allclose(data.reconstructed, peak(22).numpy(), atol=1e-15)
    np.testing.assert_allclose(data.reconstructed_display, data.reconstructed / peak(22).max().item())
    counts = pd.read_csv(writer.directory / child / 'su_counts.csv').set_index('su_type')
    assert counts.loc[22, 'delta_from_parent'] == -1
    assert counts.loc[23, 'delta_from_parent'] == 1
    changes = pd.read_csv(writer.directory / child / 'node_changes.csv')
    assert set(changes.field) == {'su_type', 'mu'}
    duplicate = SpectralTraceWriter(output_root=tmp_path, run_id='repeat', target=peak(22),
                                    ppm_axis=PPM_AXIS, reconstruct=reconstruct)
    assert duplicate.directory != writer.directory
    assert json.loads((writer.directory / 'manifest.json').read_text())['checkpoint_count'] == 2


def test_bad_export_warns_without_mutating_input(tmp_path):
    nodes = [node_at(17.)]
    writer = SpectralTraceWriter(output_root=tmp_path, run_id='test', target=peak(22),
                                 ppm_axis=PPM_AXIS, reconstruct=reconstruct)
    with pytest.warns(RuntimeWarning, match='unknown spectral checkpoint parent'):
        assert writer.checkpoint(stage='bad', nodes=nodes, parent_checkpoint_id='missing') is None
    assert nodes[0].mu == 17.
    writer.finish(status='complete')
    assert json.loads((writer.directory / 'manifest.json').read_text())['export_error_count'] == 1


@pytest.mark.parametrize('rollback', [False, True])
def test_real_layer3_same_results_with_and_without_detailed_observer(tmp_path, rollback):
    target = peak(17. if rollback else 22.)
    samples = (22., 26.) if rollback else (17., 22.)
    iterations = 0 if rollback else 5
    baseline = estimator(samples)
    observed = estimator(samples)
    input_nodes = [node_at(17.)]
    simple_nodes, simple_r2 = baseline.layer3_adjust_templates(
        nodes=deepcopy(input_nodes), S_target=target, E_target=ELEMENTS,
        max_iters=iterations, verbose=False)
    writer = SpectralTraceWriter(output_root=tmp_path, run_id='layer3', target=target,
                                 ppm_axis=PPM_AXIS, reconstruct=reconstruct)
    writer.checkpoint(stage='layer3_input', nodes=input_nodes)
    detailed_nodes, detailed_r2 = observed.layer3_adjust_templates(
        nodes=deepcopy(input_nodes), S_target=target, E_target=ELEMENTS,
        max_iters=iterations, verbose=False, output_dir=str(tmp_path / 'layer3_csv'),
        checkpoint_callback=lambda **payload: writer.checkpoint(
            **payload, decision='provisional', spectral_contract='template_sample_mu_layer3'))
    assert simple_r2 == detailed_r2
    assert simple_nodes[0].mu == detailed_nodes[0].mu
    assert torch.equal(simple_nodes[0].z_vec, detailed_nodes[0].z_vec)
    assert baseline.last_diagnostics == observed.last_diagnostics
    assert baseline.last_diagnostics['accepted'] is (not rollback)
    records = pd.read_csv(writer.directory / 'checkpoints.csv')
    assert (records.loc[records.stage.str.startswith('layer3_sample_'), 'decision'] == 'provisional').all()
    assert not writer.errors
    if not rollback:
        moves = pd.read_csv(tmp_path / 'layer3_csv/layer3_z_moves.csv')
        assert len(moves) == 1 and moves.iloc[0].new_mu == 22.
        assert (records.stage == 'layer3_sample_move').sum() == 1


def small_pipeline(monkeypatch):
    """Exercise the real infer coordinator with deterministic candidate services.

    Inventory/topology search is stubbed; the real Layer3 optimizer and all new
    trace paths run, including two outer cycles and selection of an earlier best.
    """
    import model.inverse.inverse_pipeline as module
    monkeypatch.setattr(module, 'visualize_su_distribution', lambda *a, **k: None)
    pipeline = InversePipelineV3.__new__(InversePipelineV3)
    pipeline.device = 'cpu'
    pipeline.E_SU = E_SU
    pipeline.default_template_lib_path = 'unused-test-library.pt'
    pipeline.layer0_estimator = SimpleNamespace(fixed_partition_meta={})
    pipeline.layer1_assigner = SimpleNamespace(
        _hop1_adjuster=None,
        precheck_fixed_assignment_feasibility=lambda *a: (True, 'ok'))
    pipeline.layer2_estimator = SimpleNamespace()
    pipeline.layer3_estimator = estimator((21., 22.))
    pipeline.layer4_adjuster = SimpleNamespace(
        fixed_partition_meta={},
        _evaluate_required_hist_constraints=lambda *a: {'ok': True},
        set_spectrum_hop1_template_options=lambda *a: None,
        _summarize_hist_changes=lambda *a: [],
        adjust_by_stage=lambda **k: (
            k['H'], [{'op': 'test_hetero_adjustment'}] if k['block_b_substage'] == 'ether_2829' else [], {}))
    pipeline._build_layer_runtime_configs = lambda **k: ({}, {})
    pipeline.estimate_su_histogram = lambda *a: pipeline._histogram_from_nodes([node_at(17.)])
    pipeline._sync_fixed_partition_meta_from_nodes = lambda *a: None
    pipeline._validate_layer12_candidate = lambda *a: {'ok': True, 'reason': 'ok'}
    pipeline._audit_final_su_identity = lambda *a: {'ok': True}
    for method in ['_print_histogram_and_elements', '_print_effective_resource_summary',
                   '_print_final_topology_specs']:
        setattr(pipeline, method, lambda *a, **k: None)
    calls = iter([17., 19., 20., 19., 20.])

    def layer12(**kwargs):
        mu = next(calls)
        nodes = [node_at(mu)]
        callback = kwargs.get('checkpoint_callback')
        if callback is not None:
            for stage, shift in [('initial_layer1_raw', 16.), ('initial_layer2_raw', 16.5),
                                 ('initial_layer1_refined', 17.)]:
                callback(stage=stage, nodes=[node_at(shift)], reconstruction=peak(shift))
        difference = pipeline._compute_difference_spectrum_from_nodes_mu(nodes, kwargs['S_target'])
        difference.update(source_stage='layer2', spectral_contract='template_center_mu')
        return nodes, pipeline._histogram_from_nodes(nodes), difference, difference['r2']

    pipeline._run_layer12_cycle = layer12
    pipeline._iter_block_a_candidates = lambda H, *a: [(H, [], {'op': 'test_carbonyl_adjustment'})]

    def stage3(**kwargs):
        nodes = [node_at(21.)]
        diff = pipeline._compute_difference_spectrum_from_nodes_mu(nodes, kwargs['S_target'])
        diff.update(source_stage='layer2', spectral_contract='template_center_mu')
        pipeline._final_stage3_completion_diagnostics = {'stage3_complete': True}
        pipeline._final_stage3_snapshot_authoritative = True
        pipeline._final_stage3_snapshot_node_ids = [0]
        callback = kwargs.get('checkpoint_callback')
        if callback is not None:
            callback(stage='stage3_cycle_decision', spectrum_available=False,
                     iteration=1, action={'accepted': True, 'rebuild_valid': False})
            callback(stage='stage3_rebuild_accepted', nodes=nodes, difference=diff,
                     iteration=1, action={'op': 'test_resource_adjustment'})
        return nodes, pipeline._histogram_from_nodes(nodes), diff, {
            'stop_reason': 'stage3_complete', 'accepted_cycles': 1,
            'diagnostics': {'total_cycle_count': 1}}

    pipeline._run_topology_stage3_resource_cycles = stage3

    def layer3(**kwargs):
        kwargs.pop('lib_path')
        return pipeline.layer3_estimator.layer3_adjust_templates(**kwargs, verbose=False)

    pipeline.layer3_adjust_templates = layer3

    def final_export(**kwargs):
        directory = Path(kwargs['output_dir'])
        directory.mkdir(parents=True)
        spectrum = pipeline.reconstruct_spectrum(kwargs['nodes'])
        pd.DataFrame({'ppm': PPM_AXIS.numpy(), 'target': kwargs['S_target'].numpy(),
                      'reconstructed': spectrum.numpy()}).to_csv(directory / 'final_nmr_comparison.csv', index=False)

    pipeline._save_final_nmr_comparison = final_export
    return pipeline


def test_real_infer_coordinator_modes_and_cycle_seed_parents(tmp_path, monkeypatch):
    target = peak(22.)
    simple = small_pipeline(monkeypatch)
    detailed = small_pipeline(monkeypatch)
    simple_nodes, simple_hist = simple.infer(
        target, ELEMENTS, output_dir=str(tmp_path / 'simple'),
        save_intermediates=False, outer_max_cycles=2)
    detailed_nodes, detailed_hist = detailed.infer(
        target, ELEMENTS, output_dir=str(tmp_path / 'detailed'), output_mode='detailed',
        save_intermediates=False, outer_max_cycles=2)
    assert torch.equal(simple_hist, detailed_hist)
    assert simple_nodes[0].mu == detailed_nodes[0].mu == 22.
    assert torch.equal(simple_nodes[0].z_vec, detailed_nodes[0].z_vec)
    assert not (tmp_path / 'simple/spectral_trace').exists()
    root = next((tmp_path / 'detailed/spectral_trace').iterdir())
    records = pd.read_csv(root / 'checkpoints.csv', keep_default_na=False)
    assert {'block_a_accepted', 'block_b_accepted', 'stage3_rebuild_accepted',
            'stage3_complete', 'layer3_sample_move', 'final_selected'} <= set(records.stage)
    seed2 = records.loc[(records.stage == 'outer_cycle_input') & (records.outer_cycle == 2)].iloc[0]
    parent = records.loc[records.checkpoint_id == seed2.parent_checkpoint_id].iloc[0]
    assert parent.stage == 'stage3_complete' and parent.outer_cycle == 1
    selected = records.loc[records.stage == 'final_selected'].iloc[0]
    assert selected.outer_cycle == 1  # second cycle ties; first remains selected
    selected_parent = records.loc[records.checkpoint_id == selected.parent_checkpoint_id].iloc[0]
    assert selected_parent.stage == 'layer3_accepted' and selected_parent.outer_cycle == 1
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['export_error_count'] == 0
    assert (tmp_path / 'detailed/final_outputs/final_stage3_resource_cycles.csv').exists()
    for folder in records.checkpoint_directory:
        spectrum = pd.read_csv(root / folder / 'spectrum.csv')
        np.testing.assert_allclose(spectrum.target - spectrum.reconstructed, spectrum.residual, atol=2e-7)
        assert len(spectrum) == 2400
