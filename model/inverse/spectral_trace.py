"""Read-only, opt-in spectral checkpoints for figure source data.

This observer never changes nodes, tensors, random state or acceptance decisions.
Only detailed inference creates a writer; normal inference does no trace I/O.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Optional, Sequence
import warnings

import numpy as np
import pandas as pd
import torch

from ..shared.coarse_graph import E_SU, NUM_SU_TYPES, SU_DEFS


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(_json_value(payload), ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class SpectralTraceWriter:
    """Write one durable directory per observed state, plus an ordered index.

    Parent IDs describe the actual seed/branch, not just file order. Internal
    Layer3 moves remain provisional until its candidate is accepted as a whole.
    A failed export warns and is recorded; it cannot change inference results.
    """

    INDEX_COLUMNS = [
        'sequence', 'checkpoint_id', 'parent_checkpoint_id', 'stage', 'outer_cycle',
        'iteration', 'decision', 'spectral_contract', 'r2', 'recorded_r2', 'rmse',
        'mae', 'max_absolute_residual', 'node_count', 'carbon_count',
        'changed_node_count', 'elapsed_seconds', 'checkpoint_directory',
    ]

    def __init__(self, *, output_root: Path, run_id: str, target: Any,
                 ppm_axis: Any, reconstruct: Callable, metadata: Optional[dict] = None):
        self.target = _array(target).copy()
        self.ppm = _array(ppm_axis).copy()
        if self.target.shape != self.ppm.shape or not np.isfinite(self.target).all():
            raise ValueError('spectral_trace_requires_complete_finite_target')
        self.reconstruct = reconstruct
        self.started = time.perf_counter()
        self.sequence = 0
        self.last_checkpoint_id = ''
        self.errors: list[dict] = []
        self.export_seconds = 0.0
        self._directories: dict[str, Path] = {}
        self._stages: dict[str, str] = {}
        safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(run_id)) or 'run'
        trace_root = Path(output_root) / 'spectral_trace'
        trace_root.mkdir(parents=True, exist_ok=True)
        self.directory = trace_root / safe_id
        suffix = 1
        while self.directory.exists():
            self.directory = trace_root / f'{safe_id}-{suffix}'
            suffix += 1
        self.directory.mkdir()
        pd.DataFrame({'ppm': self.ppm, 'target': self.target}).to_csv(
            self.directory / 'target.csv', index=False)
        self.divisor = float(np.max(np.abs(self.target))) or 1.0
        self.manifest = {
            'format': 'graphnmr_spectral_trace', 'version': 1, 'run_id': str(run_id),
            'output_mode': 'detailed', 'status': 'running', 'point_count': len(self.ppm),
            'display_divisor': self.divisor, 'residual_definition': 'target - reconstructed',
            'peak_hwhm_ppm': 1.0, 'independent_rescaling': False,
            'checkpoint_count': 0, 'export_error_count': 0,
            'stage_order_warning': 'Use parent_checkpoint_id and spectral_contract; outer cycles restart from the saved Layer2 state, not the preceding Layer3 spectrum.',
            'layer3_moves': 'Provisional fixed-template sample moves; use layer3_decision events for whole-candidate acceptance or rollback.',
            'metadata': metadata or {},
        }
        _write_json(self.directory / 'manifest.json', self.manifest)

    def _error(self, stage: str, exc: Exception) -> None:
        error = {'stage': stage, 'error_type': type(exc).__name__, 'error': str(exc)}
        self.errors.append(error)
        warnings.warn(f'Spectral trace export failed at {stage}: {exc}', RuntimeWarning)
        try:
            with (self.directory / 'export_errors.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(error, ensure_ascii=False) + '\n')
        except OSError:
            pass

    def event(self, kind: str, **payload: Any) -> None:
        try:
            row = {'event': kind, 'elapsed_seconds': time.perf_counter() - self.started,
                   **payload}
            with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False) + '\n')
        except Exception as exc:
            self._error(kind, exc)

    def checkpoint(self, *, stage: str, nodes: Sequence[Any],
                   difference: Optional[dict] = None, reconstruction: Any = None,
                   outer_cycle: int = 0, iteration: Any = None,
                   decision: str = 'accepted', spectral_contract: str = 'template_center_mu',
                   parent_checkpoint_id: Optional[str] = None,
                   action: Any = None, metadata: Optional[dict] = None) -> Optional[str]:
        started = time.perf_counter()
        try:
            return self._checkpoint(stage=stage, nodes=nodes, difference=difference,
                                    reconstruction=reconstruction, outer_cycle=outer_cycle,
                                    iteration=iteration, decision=decision,
                                    spectral_contract=spectral_contract,
                                    parent_checkpoint_id=parent_checkpoint_id,
                                    action=action, metadata=metadata)
        except Exception as exc:
            self._error(stage, exc)
            return None
        finally:
            self.export_seconds += time.perf_counter() - started

    def _checkpoint(self, *, stage, nodes, difference, reconstruction, outer_cycle,
                    iteration, decision, spectral_contract, parent_checkpoint_id,
                    action, metadata):
        parent = self.last_checkpoint_id if parent_checkpoint_id is None else parent_checkpoint_id
        if parent and parent not in self._directories:
            raise ValueError(f'unknown spectral checkpoint parent: {parent}')
        recorded_r2 = (metadata or {}).get('r2')
        if reconstruction is not None:
            predicted = _array(reconstruction)
        elif difference is not None:
            np.testing.assert_allclose(_array(difference['ppm']), self.ppm, rtol=0, atol=2e-5)
            predicted = self.target - _array(difference['diff'])
            recorded_r2 = difference.get('r2')
        else:
            with torch.no_grad():
                predicted = _array(self.reconstruct(nodes))
        if predicted.shape != self.target.shape or not np.isfinite(predicted).all():
            raise ValueError('spectral_checkpoint_requires_complete_finite_spectrum')
        residual = self.target - predicted
        denominator = float(np.sum((self.target - self.target.mean()) ** 2))
        r2 = float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else None
        rows = []
        ordered_nodes = sorted(nodes, key=lambda n: int(n.global_id))
        z_vectors = [getattr(node, 'z_vec', None) for node in ordered_nodes]
        # Copy latent vectors in one batch rather than synchronizing the GPU
        # once per node at every checkpoint.
        batched_z = None
        if z_vectors and all(isinstance(z, torch.Tensor) for z in z_vectors):
            if len({(z.device, z.numel()) for z in z_vectors}) == 1:
                batched_z = torch.stack([z.detach().reshape(-1) for z in z_vectors]).cpu().tolist()
        for node_idx, node in enumerate(ordered_nodes):
            row = {key: getattr(node, key, None) for key in [
                'global_id', 'su_type', 'mu', 'pi', 'template_key',
                'target_hop1_degree', 'target_fixed_anchor_count',
                'special_anchor_partition', 'special_anchor_mode',
            ]}
            row['template_key'] = str(row['template_key'])
            for key in ['hop1_ids', 'hop2_ids', 'fixed_hop1_ids']:
                row[key] = json.dumps(sorted(int(v) for v in (getattr(node, key, None) or [])))
            for key in ['hop1_su', 'hop2_su', 'score_components']:
                row[key] = json.dumps(_json_value(dict(getattr(node, key, None) or {})), ensure_ascii=False)
            z = getattr(node, 'z_vec', None)
            row['z_vec'] = json.dumps(batched_z[node_idx] if batched_z is not None else
                                      _array(z).tolist() if z is not None else [])
            rows.append(_json_value(row))
        frame = pd.DataFrame(rows, columns=[
            'global_id', 'su_type', 'mu', 'pi', 'template_key', 'target_hop1_degree',
            'target_fixed_anchor_count', 'special_anchor_partition', 'special_anchor_mode',
            'hop1_ids', 'hop2_ids', 'fixed_hop1_ids', 'hop1_su', 'hop2_su',
            'score_components', 'z_vec'])
        if frame.global_id.duplicated().any():
            raise ValueError('spectral_checkpoint_duplicate_global_node_ids')
        counts = np.bincount(frame.su_type.to_numpy(dtype=int), minlength=NUM_SU_TYPES)
        if len(counts) != NUM_SU_TYPES:
            raise ValueError('spectral_checkpoint_invalid_su_type')
        previous_counts = np.zeros(NUM_SU_TYPES, dtype=int)
        changes = []
        if parent:
            prior = pd.read_csv(self._directories[parent] / 'nodes.csv', keep_default_na=False)
            previous_counts = np.bincount(prior.su_type.to_numpy(dtype=int), minlength=NUM_SU_TYPES)
            old = {int(row['global_id']): row for row in prior.to_dict('records')}
            new = {int(row['global_id']): row for row in frame.to_dict('records')}
            for node_id in sorted(set(old) | set(new)):
                for key in ['su_type', 'mu', 'template_key', 'hop1_ids', 'hop2_ids', 'z_vec']:
                    before, after = old.get(node_id, {}).get(key), new.get(node_id, {}).get(key)
                    equal_number = (isinstance(before, (int, float)) and isinstance(after, (int, float))
                                    and np.isclose(before, after, rtol=0, atol=1e-10))
                    if before != after and not equal_number and not (before == '' and after is None):
                        changes.append({'global_id': node_id, 'field': key, 'before': before, 'after': after})
        self.sequence += 1
        safe_stage = re.sub(r'[^A-Za-z0-9_.-]+', '_', stage)
        checkpoint_id = f'{self.sequence:06d}_{safe_stage}'
        final = self.directory / checkpoint_id
        temporary = self.directory / f'.{checkpoint_id}.tmp'
        temporary.mkdir()
        pd.DataFrame({'ppm': self.ppm, 'target': self.target, 'reconstructed': predicted,
                      'residual': residual, 'target_display': self.target / self.divisor,
                      'reconstructed_display': predicted / self.divisor,
                      'residual_display': residual / self.divisor}).to_csv(temporary / 'spectrum.csv', index=False)
        frame.to_csv(temporary / 'nodes.csv', index=False)
        pd.DataFrame({'su_type': np.arange(NUM_SU_TYPES),
                      'name': [definition[0] for definition in SU_DEFS],
                      'count': counts, 'parent_count': previous_counts,
                      'delta_from_parent': counts - previous_counts if parent else np.zeros(NUM_SU_TYPES, dtype=int),
                      'has_parent': bool(parent)}).to_csv(temporary / 'su_counts.csv', index=False)
        pd.DataFrame(changes, columns=['global_id', 'field', 'before', 'after']).to_csv(
            temporary / 'node_changes.csv', index=False)
        carbon = E_SU.detach().cpu().numpy()[:, 0]
        record = {'sequence': self.sequence, 'checkpoint_id': checkpoint_id,
                  'parent_checkpoint_id': parent, 'stage': stage, 'outer_cycle': outer_cycle,
                  'iteration': iteration, 'decision': decision, 'spectral_contract': spectral_contract,
                  'r2': r2, 'recorded_r2': recorded_r2,
                  'rmse': float(np.sqrt(np.mean(residual ** 2))), 'mae': float(np.mean(abs(residual))),
                  'max_absolute_residual': float(np.max(abs(residual))),
                  'node_count': len(nodes), 'carbon_count': float(np.dot(counts, carbon)),
                  'changed_node_count': len({row['global_id'] for row in changes}),
                  'elapsed_seconds': time.perf_counter() - self.started,
                  'checkpoint_directory': checkpoint_id}
        segment_scores = {}
        for label, lo, hi in [('aliphatic', 0, 90), ('aromatic', 90, 160), ('carbonyl', 160, 240)]:
            mask = (self.ppm >= lo) & (self.ppm <= hi)
            total = float(np.sum((self.target[mask] - self.target[mask].mean()) ** 2)) if mask.any() else 0
            segment_scores[label] = float(1 - np.sum(residual[mask] ** 2) / total) if total > 0 else None
        _write_json(temporary / 'metadata.json', {**record, 'action': action,
                    'metadata': metadata or {}, 'segment_r2': segment_scores})
        temporary.replace(final)
        index_path = self.directory / 'checkpoints.csv'
        write_header = not index_path.exists()
        with index_path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.INDEX_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(_json_value(record))
        self._directories[checkpoint_id] = final
        self._stages[checkpoint_id] = stage
        self.last_checkpoint_id = checkpoint_id
        self.event('checkpoint', **record)
        return checkpoint_id

    def latest(self, stage: str) -> str:
        return next((key for key in reversed(self._stages) if self._stages[key] == stage), '')

    def finish(self, *, status: str, **summary: Any) -> None:
        try:
            self.manifest.update(status=status, checkpoint_count=len(self._directories),
                                 export_error_count=len(self.errors), summary=summary,
                                 elapsed_seconds=time.perf_counter() - self.started,
                                 checkpoint_export_seconds=self.export_seconds)
            _write_json(self.directory / 'manifest.json', self.manifest)
            self.event('run_finished', status=status, **summary)
        except Exception as exc:
            self._error('finish', exc)
