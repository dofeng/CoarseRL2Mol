from types import SimpleNamespace

import pytest
import torch

from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.layer1.hop1_adjuster import Hop1Adjuster


def test_layer1_regions_are_overlapping_sixty_ppm_windows() -> None:
    assert Hop1Adjuster.REGION_ORDER == (
        ('0-60', 0.0, 60.0),
        ('30-90', 30.0, 90.0),
        ('60-120', 60.0, 120.0),
        ('90-150', 90.0, 150.0),
        ('120-180', 120.0, 180.0),
        ('150-210', 150.0, 210.0),
        ('180-240', 180.0, 240.0),
    )

    for below, above in (
        (59.0, 61.0),
        (89.0, 91.0),
        (119.0, 121.0),
        (149.0, 151.0),
        (179.0, 181.0),
        (209.0, 211.0),
    ):
        assert any(
            float(lo) <= below <= float(hi)
            and float(lo) <= above <= float(hi)
            for _name, lo, hi in Hop1Adjuster.REGION_ORDER
        )


def _cycle_test_adjuster(monkeypatch, committed_moves_by_cycle):
    adjuster = object.__new__(Hop1Adjuster)
    state = {'r2': 0.1, 'call': 0}

    def build_snapshot(_nodes, _target):
        return {
            'r2': float(state['r2']),
            'assignments': {0: {'mu': 100.0}},
        }

    def adjust_once(**kwargs):
        cycle_index = int(state['call'])
        state['call'] += 1
        move_count = int(committed_moves_by_cycle[cycle_index])
        before = float(state['r2'])
        state['r2'] = before + 0.1 * move_count
        return kwargs['nodes'], {
            'moves': [
                {'region': '60-120', 'move': index + 1}
                for index in range(move_count)
            ],
            'regions': [{
                'region': '60-120',
                'committed_moves': move_count,
                'committed': bool(move_count),
            }],
            'final_r2': float(state['r2']),
        }

    monkeypatch.setattr(adjuster, 'build_eval_snapshot', build_snapshot)
    monkeypatch.setattr(adjuster, '_adjust_regions_once', adjust_once)
    monkeypatch.setattr(
        adjuster,
        'apply_assignments_to_nodes',
        lambda _nodes, _assignments: None,
    )
    return adjuster, state


def test_region_adjustment_aggregates_two_cycles(monkeypatch) -> None:
    adjuster, state = _cycle_test_adjuster(monkeypatch, [1, 2])
    nodes = [SimpleNamespace(global_id=0)]

    result_nodes, summary = adjuster.adjust_regions(
        nodes=nodes,
        S_target=torch.zeros(1),
        E_target=None,
        validator=lambda _nodes: (True, []),
        max_moves_per_region=6,
        region_adjustment_cycles=2,
    )

    assert result_nodes is nodes
    assert state['call'] == 2
    assert summary['configured_cycles'] == 2
    assert summary['completed_cycles'] == 2
    assert summary['total_adjustments'] == 3
    assert [row['cycle'] for row in summary['moves']] == [1, 2, 2]
    assert summary['stop_reason'] == 'cycle_limit'
    assert summary['initial_r2'] == pytest.approx(0.1)
    assert summary['final_r2'] == pytest.approx(0.4)


def test_region_adjustment_stops_after_cycle_without_commits(monkeypatch) -> None:
    adjuster, state = _cycle_test_adjuster(monkeypatch, [0, 1, 1])

    _nodes, summary = adjuster.adjust_regions(
        nodes=[SimpleNamespace(global_id=0)],
        S_target=torch.zeros(1),
        E_target=None,
        validator=lambda _nodes: (True, []),
        region_adjustment_cycles=3,
    )

    assert state['call'] == 1
    assert summary['configured_cycles'] == 3
    assert summary['completed_cycles'] == 1
    assert summary['converged'] is True
    assert summary['stop_reason'] == 'no_committed_moves'


def test_region_adjustment_runtime_defaults_and_validation() -> None:
    pipeline = object.__new__(InversePipelineV3)
    layer1_config, _layer3_config = pipeline._build_layer_runtime_configs()

    assert layer1_config['max_moves_per_region'] == 6
    assert layer1_config['region_adjustment_cycles'] == 2

    with pytest.raises(ValueError, match='region_adjustment_cycles must be >= 1'):
        pipeline._build_layer_runtime_configs({
            'region_adjustment_cycles': 0,
        })
