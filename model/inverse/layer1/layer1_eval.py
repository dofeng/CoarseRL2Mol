from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...paths import Z_LIBRARY_DIR
from ...shared.coarse_graph import PPM_AXIS, SU_DEFS
from ..hop1_adjuster import hop1_counter_to_multiset, multiset_l1_distance
from ...shared.inverse_common import (
    _NodeV3,
    evaluate_mu_pi_assignments,
    resolve_eval_inputs,
    save_node_peak_rows,
    save_spectrum_comparison,
    save_spectrum_figure,
)


class Layer1NmrEvaluator:
    def __init__(self,
                 device: str | torch.device,
                 E_SU_tensor: torch.Tensor,
                 intensity_scale: float = 1.0,
                 unit_peak_intensity: bool = True):
        self.device = device
        self.E_SU = E_SU_tensor
        self.intensity_scale = float(intensity_scale)
        self.unit_peak_intensity = bool(unit_peak_intensity)
        self._layer1_lib_index_cache: Optional[Dict[str, object]] = None
        self._carbon_su_set = {
            int(i)
            for i in range(int(self.E_SU.shape[0]))
            if float(self.E_SU[int(i), 0].detach().cpu().item()) > 0.0
        }

    def _is_carbon_su(self, center_su: int) -> bool:
        return int(center_su) in self._carbon_su_set

    @staticmethod
    def _empty_assignment_meta(hop1_ms: Tuple[int, ...],
                               approx_used: bool = False,
                               n_templates: int = 0,
                               w_sum: float = 0.0,
                               mu_min: float = 0.0,
                               mu_max: float = 0.0) -> Dict[str, object]:
        return {
            'matched': False,
            'approx_used': bool(approx_used),
            'chosen_hop1_ms': tuple(hop1_ms),
            'n_templates': int(n_templates),
            'w_sum': float(w_sum),
            'mu_min': float(mu_min),
            'mu_max': float(mu_max),
        }

    def resolve_inputs(self, S_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return resolve_eval_inputs(S_target, PPM_AXIS, self.device)

    def get_library_index(self, lib_path: Optional[str]) -> Dict[str, object]:
        if lib_path is None:
            lib_path = str(Z_LIBRARY_DIR / 'subgraph_library.pt')

        cache = self._layer1_lib_index_cache
        if isinstance(cache, dict) and cache.get('path') == lib_path:
            return cache

        lib = torch.load(lib_path, map_location='cpu', weights_only=False)
        templates = lib.get('templates', {}) if isinstance(lib, dict) else {}

        agg: Dict[Tuple[int, Tuple[int, ...]], Dict[str, object]] = {}
        center_to_hop1 = defaultdict(list)

        for key, tpl in templates.items():
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            center_su, hop1_ms, _ = key
            try:
                center_i = int(center_su)
                hop1_tuple = tuple(int(x) for x in tuple(hop1_ms))
            except Exception:
                continue

            mu_s = tpl.get('samples', {}).get('mu', None)
            pi_s = tpl.get('samples', {}).get('pi', None)
            if mu_s is None or pi_s is None:
                continue
            if not torch.is_tensor(mu_s) or not torch.is_tensor(pi_s):
                continue
            if int(mu_s.numel()) <= 0 or int(pi_s.numel()) <= 0:
                continue

            sample_count = int(tpl.get('sample_count', 0))
            center_mu = float(tpl.get('center_mu', 0.0))
            if sample_count <= 0:
                sample_count = int(mu_s.numel()) if int(mu_s.numel()) > 0 else 1
            if center_mu == 0.0:
                continue

            agg_key = (center_i, hop1_tuple)
            if agg_key not in agg:
                agg[agg_key] = {
                    'mu_values': [],
                    'pi_values': [],
                    'weights': [],
                    'mu_min': float(tpl.get('mu_min', center_mu)),
                    'mu_max': float(tpl.get('mu_max', center_mu)),
                    'n_templates': 0,
                }
                center_to_hop1[center_i].append(hop1_tuple)

            entry = agg[agg_key]
            entry['mu_values'].append(float(center_mu))
            try:
                center_pi = float(tpl.get('center_pi', 0.0))
            except Exception:
                center_pi = 0.0
            if center_pi <= 0.0:
                try:
                    center_pi = float(torch.median(pi_s.detach().float()).item())
                except Exception:
                    center_pi = 1.0
            entry['pi_values'].append(float(max(1e-6, center_pi)))
            entry['weights'].append(float(sample_count))
            entry['mu_min'] = float(min(float(entry['mu_min']), float(tpl.get('mu_min', center_mu))))
            entry['mu_max'] = float(max(float(entry['mu_max']), float(tpl.get('mu_max', center_mu))))
            entry['n_templates'] = int(entry['n_templates']) + 1

        for entry in agg.values():
            mu_values = np.asarray(entry.get('mu_values', []), dtype=np.float64)
            pi_values = np.asarray(entry.get('pi_values', []), dtype=np.float64)
            weights = np.asarray(entry.get('weights', []), dtype=np.float64)
            if mu_values.size == 0 or pi_values.size == 0 or weights.size == 0:
                continue
            order = np.argsort(mu_values)
            mu_sorted = mu_values[order]
            pi_sorted = pi_values[order]
            w_sorted = np.maximum(weights[order], 1e-8)
            entry['mu_sorted'] = mu_sorted
            entry['pi_sorted'] = pi_sorted
            entry['w_sorted'] = w_sorted
            entry['cdf'] = np.cumsum(w_sorted) / float(np.sum(w_sorted))

        cache = {
            'path': lib_path,
            'agg': agg,
            'center_to_hop1': dict(center_to_hop1),
            'approx_hop1_cache': {},
            'n_templates_total': len(templates),
        }
        self._layer1_lib_index_cache = cache
        return cache

    def build_grouped_assignments(self,
                                  nodes: List[_NodeV3],
                                  lib_path: Optional[str],
                                  allow_approx: bool) -> Dict[int, Dict[str, object]]:
        lib_index = self.get_library_index(lib_path)
        grouped = defaultdict(list)
        for node in nodes:
            center_su = int(node.su_type)
            if not self._is_carbon_su(center_su):
                continue
            hop1_ms = hop1_counter_to_multiset(node.hop1_su)
            grouped[(center_su, hop1_ms)].append(node)

        assignments: Dict[int, Dict[str, object]] = {}
        agg = lib_index.get('agg', {})
        center_to_hop1 = lib_index.get('center_to_hop1', {})
        approx_hop1_cache = lib_index.setdefault('approx_hop1_cache', {})

        for (center_su, hop1_ms), group_nodes in grouped.items():
            info = agg.get((int(center_su), tuple(hop1_ms)))
            chosen_hop1 = tuple(hop1_ms)
            approx_used = False

            if info is None and allow_approx:
                approx_key = (int(center_su), tuple(hop1_ms))
                best = approx_hop1_cache.get(approx_key)
                if best is None:
                    hop1_keys = center_to_hop1.get(int(center_su), [])
                    if hop1_keys:
                        best = min(
                            hop1_keys,
                            key=lambda ms: (multiset_l1_distance(ms, hop1_ms), abs(len(ms) - len(hop1_ms))),
                        )
                        approx_hop1_cache[approx_key] = tuple(best)
                if best is not None:
                    info = agg.get((int(center_su), tuple(best)))
                    chosen_hop1 = tuple(best)
                    approx_used = True

            if info is None:
                for node in group_nodes:
                    assignments[int(node.global_id)] = {
                        'mu': None,
                        'pi': None,
                        'meta': self._empty_assignment_meta(chosen_hop1),
                    }
                continue

            mu_sorted = np.asarray(info.get('mu_sorted', []), dtype=np.float64)
            pi_sorted = np.asarray(info.get('pi_sorted', []), dtype=np.float64)
            w_sorted = np.asarray(info.get('w_sorted', []), dtype=np.float64)
            cdf = np.asarray(info.get('cdf', []), dtype=np.float64)
            if mu_sorted.size == 0 or pi_sorted.size == 0 or w_sorted.size == 0 or cdf.size == 0:
                for node in group_nodes:
                    assignments[int(node.global_id)] = {
                        'mu': None,
                        'pi': None,
                        'meta': self._empty_assignment_meta(
                            chosen_hop1,
                            approx_used=approx_used,
                            n_templates=int(info.get('n_templates', 0)),
                            mu_min=float(info.get('mu_min', 0.0)),
                            mu_max=float(info.get('mu_max', 0.0)),
                        ),
                    }
                continue

            ordered_nodes = sorted(group_nodes, key=lambda n: int(getattr(n, 'global_id', 0)))
            n_nodes = len(ordered_nodes)
            for idx, node in enumerate(ordered_nodes):
                q = (idx + 0.5) / max(1.0, float(n_nodes))
                pick = int(np.searchsorted(cdf, q, side='left'))
                pick = min(max(pick, 0), len(mu_sorted) - 1)
                assignments[int(node.global_id)] = {
                    'mu': float(mu_sorted[pick]),
                    'pi': float(max(1e-6, pi_sorted[pick])),
                    'meta': {
                        'matched': True,
                        'approx_used': approx_used,
                        'chosen_hop1_ms': chosen_hop1,
                        'n_templates': int(info.get('n_templates', 0)),
                        'w_sum': float(np.sum(w_sorted)),
                        'mu_min': float(info.get('mu_min', 0.0)),
                        'mu_max': float(info.get('mu_max', 0.0)),
                    },
                }
        return assignments

    def build_peak_rows(self,
                        nodes: List[_NodeV3],
                        assignments: Dict[int, Dict[str, object]]) -> Dict[str, object]:
        su_names = [name for name, _ in SU_DEFS]
        mus: List[float] = []
        pis: List[float] = []
        rows: List[Dict[str, object]] = []
        matched_cnt = 0
        carbon_cnt = 0

        for node in nodes:
            center_su = int(node.su_type)
            hop1_ms = hop1_counter_to_multiset(node.hop1_su)
            is_carbon = self._is_carbon_su(center_su)
            assigned = assignments.get(int(node.global_id))

            if is_carbon:
                carbon_cnt += 1
                if assigned is None:
                    mu = None
                    pi = None
                    meta = self._empty_assignment_meta(hop1_ms)
                else:
                    mu = assigned.get('mu')
                    pi = assigned.get('pi')
                    meta = dict(assigned.get('meta', {}) or {})
                if bool(meta.get('matched', False)):
                    matched_cnt += 1
                if mu is not None and pi is not None:
                    mus.append(float(mu))
                    if bool(self.unit_peak_intensity):
                        pis.append(1.0)
                    else:
                        pis.append(float(max(0.0, pi)))
            else:
                mu = 0.0
                pi = 0.0
                meta = self._empty_assignment_meta(hop1_ms)

            rows.append({
                'global_id': int(node.global_id),
                'center_su_idx': center_su,
                'center_su': su_names[center_su] if 0 <= center_su < len(su_names) else str(center_su),
                'hop1_ms': '[' + ' '.join(str(x) for x in hop1_ms) + ']',
                'hop1_tuple': tuple(hop1_ms),
                'matched': bool(meta.get('matched', False)),
                'approx_used': bool(meta.get('approx_used', False)),
                'chosen_hop1_ms': '[' + ' '.join(str(x) for x in meta.get('chosen_hop1_ms', ())) + ']',
                'n_templates': int(meta.get('n_templates', 0)),
                'sample_weight_sum': float(meta.get('w_sum', 0.0)),
                'mu': float(mu) if mu is not None else np.nan,
                'pi': float(pi) if pi is not None else np.nan,
                'mu_min': float(meta.get('mu_min', 0.0)),
                'mu_max': float(meta.get('mu_max', 0.0)),
            })

        return {
            'rows': rows,
            'mus': mus,
            'pis': pis,
            'matched_cnt': int(matched_cnt),
            'carbon_cnt': int(carbon_cnt),
        }

    def build_eval_snapshot(self,
                            nodes: List[_NodeV3],
                            S_target: torch.Tensor,
                            lib_path: Optional[str],
                            hwhm: float,
                            allow_approx: bool) -> Dict[str, object]:
        S_eval, ppm_axis = self.resolve_inputs(S_target)
        assignments = self.build_grouped_assignments(nodes, lib_path, allow_approx)
        peak_data = self.build_peak_rows(nodes, assignments)

        matched_cnt = int(peak_data['matched_cnt'])
        carbon_cnt = int(peak_data['carbon_cnt'])
        matched_ratio = float(matched_cnt) / max(1.0, float(carbon_cnt))

        snapshot: Dict[str, object] = {
            'ppm': ppm_axis.detach().cpu().numpy(),
            'rows': list(peak_data['rows']),
            'matched_cnt': matched_cnt,
            'carbon_cnt': carbon_cnt,
            'matched_ratio': matched_ratio,
            'n_peaks': int(len(peak_data['mus'])),
        }
        snapshot.update(
            evaluate_mu_pi_assignments(
                S_target=S_eval,
                ppm_axis=ppm_axis,
                mus=peak_data['mus'],
                pis=peak_data['pis'],
                hwhm=float(hwhm),
                intensity_scale=float(self.intensity_scale),
                device=self.device,
                unit_peak_intensity=bool(self.unit_peak_intensity),
            )
        )
        return snapshot

    def compute_difference_spectrum(self,
                                    nodes: List[_NodeV3],
                                    S_target: torch.Tensor,
                                    lib_path: Optional[str],
                                    hwhm: float,
                                    allow_approx: bool) -> Dict[str, object]:
        snapshot = self.build_eval_snapshot(
            nodes=nodes,
            S_target=S_target,
            lib_path=lib_path,
            hwhm=hwhm,
            allow_approx=allow_approx,
        )
        return {
            'ppm': np.asarray(snapshot.get('ppm', []), dtype=np.float64),
            'diff': np.asarray(snapshot.get('diff', []), dtype=np.float64),
            'r2': float(snapshot.get('r2', 0.0)),
            'alpha': float(snapshot.get('alpha', 0.0)),
            'n_peaks': int(snapshot.get('n_peaks', 0)),
        }

    def evaluate_with_library(self,
                              nodes: List[_NodeV3],
                              S_target: torch.Tensor,
                              lib_path: Optional[str],
                              output_dir: Optional[str],
                              hwhm: float,
                              allow_approx: bool) -> Dict[str, float]:
        snapshot = self.build_eval_snapshot(
            nodes=nodes,
            S_target=S_target,
            lib_path=lib_path,
            hwhm=hwhm,
            allow_approx=allow_approx,
        )

        matched_cnt = int(snapshot.get('matched_cnt', 0))
        carbon_cnt = int(snapshot.get('carbon_cnt', 0))
        matched_ratio = float(snapshot.get('matched_ratio', 0.0))
        if output_dir:
            save_node_peak_rows(snapshot.get('rows', []), output_dir)

        if int(snapshot.get('n_peaks', 0)) <= 0:
            print("[Layer1-NMR-Eval] 未找到可用模板峰，跳过谱图重构")
            return {
                'r2': 0.0,
                'r2_carbonyl': 0.0,
                'r2_aromatic': 0.0,
                'r2_aliphatic': 0.0,
                'matched_ratio': matched_ratio,
            }

        ppm = np.asarray(snapshot.get('ppm', []), dtype=np.float64)
        if output_dir:
            save_spectrum_comparison(
                S_target=snapshot['S_target'],
                ppm=ppm,
                S_recon_raw=snapshot['S_recon_raw'].detach().cpu().numpy(),
                S_fit=snapshot['S_fit'].detach().cpu().numpy(),
                diff=np.asarray(snapshot.get('diff', []), dtype=np.float64),
                output_dir=output_dir,
                prefix='layer1_library',
            )
            try:
                save_spectrum_figure(
                    S_target=snapshot['S_target'],
                    S_fit=snapshot['S_fit'],
                    ppm=ppm,
                    output_dir=output_dir,
                    layer_name='Layer1-Library',
                )
            except Exception as e:
                print(f"[Layer1-NMR-Eval] 绘图失败: {e}")

        r2 = float(snapshot.get('r2', 0.0))
        r2_carb = float(snapshot.get('r2_carbonyl', 0.0))
        r2_aro = float(snapshot.get('r2_aromatic', 0.0))
        r2_ali = float(snapshot.get('r2_aliphatic', 0.0))
        alpha = float(snapshot.get('alpha', 1.0))
        print(f"[Layer1-NMR-Eval] R2={r2:.4f} (carbonyl={r2_carb:.4f}, aromatic={r2_aro:.4f}, aliphatic={r2_ali:.4f}), alpha={alpha:.4f}")
        return {
            'r2': r2,
            'r2_carbonyl': r2_carb,
            'r2_aromatic': r2_aro,
            'r2_aliphatic': r2_ali,
            'matched_ratio': matched_ratio,
        }
