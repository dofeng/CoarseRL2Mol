import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path

from ...paths import Z_LIBRARY_DIR
from ...shared.coarse_graph import SU_DEFS, E_SU, NUM_SU_TYPES, PPM_AXIS
from ...shared.inverse_common import (
    _NodeV3, compute_segment_r2, reconstruct_from_mu_pi,
    visualize_spectrum_comparison, evaluate_spectrum_reconstruction,
    multiset_from_counter, multiset_l1_distance, multiset_diff_nodes,
    reconstruction_peak_scale,
)

# ============================================================================
# Layer2 Estimator
# ============================================================================

class Layer2Estimator:
    """
    Layer2: 2-hop 推导和模板检索
    
    功能：
    1. 从1-hop邻居推导2-hop邻居
    2. 模板精确匹配和近似匹配
    3. z向量初始化
    4. mu/pi重建和NMR评估
    """
    
    def __init__(self, device: torch.device = None,
                 lib_path: Optional[str] = None):
        """
        初始化 Layer2 估计器
        
        Args:
            device: 计算设备
            lib_path: 子图库路径
        """
        self.device = device or torch.device('cpu')
        # Effective aromatic response used only in reconstructed spectra.
        self.aromatic_response_scale: float = 1.0
        self.E_SU = E_SU.to(self.device)
        self._carbon_su_set = {
            int(i)
            for i in range(NUM_SU_TYPES)
            if float(E_SU[int(i), 0].detach().cpu().item()) > 0.0
        }
        
        # 路径配置
        base_dir = Z_LIBRARY_DIR
        self.lib_path = lib_path or str(base_dir / 'subgraph_library.pt')
        
        
        # 缓存
        self._template_cache = None
        self._layer1_mu_prior_cache = None
        
        # 统计
        self.stats = {
            'exact_match': 0,
            'approx_match': 0,
            'missing': 0,
        }

    def _reset_node_template_state(self,
                                   node: _NodeV3,
                                   device: torch.device,
                                   template_key: Optional[Tuple] = None,
                                   match_mode: str = 'missing',
                                   mu_prior: Optional[float] = None) -> None:
        """Clear stale Layer2 template/spectral state before assigning new values."""
        node.template_key = template_key
        try:
            if isinstance(getattr(node, 'z_vec', None), torch.Tensor):
                node.z_vec = torch.zeros_like(node.z_vec, device=device)
            else:
                node.z_vec = torch.zeros(16, dtype=torch.float, device=device)
        except Exception:
            node.z_vec = torch.zeros(16, dtype=torch.float, device=device)
        node.mu = 0.0
        node.pi = 0.0

        score_components = getattr(node, 'score_components', None)
        if isinstance(score_components, dict):
            score_components['layer2_match_mode'] = str(match_mode)
            score_components['layer2_mu_prior'] = mu_prior
            score_components['layer2_template_mu'] = None
            score_components['layer2_template_sample_count'] = 0
            score_components['layer2_sample_idx'] = None
            score_components['layer2_spectral_source'] = 'unassigned'
            score_components['spectral_source'] = 'unassigned'
            score_components['layer2_candidate_count'] = 0
            score_components['layer2_residual_sse_before'] = None
            score_components['layer2_residual_sse_after'] = None
            score_components['layer2_residual_sse_gain'] = None
            score_components['layer2_negative_overshoot'] = None

    def _valid_mu(self, mu: Optional[float]) -> bool:
        try:
            mu_f = float(mu)
            return bool(np.isfinite(mu_f) and 0.0 < mu_f <= 260.0)
        except Exception:
            return False

    def _build_layer1_mu_prior_index(self, lib: Dict) -> Dict[Tuple[int, Tuple[int, ...]], float]:
        """
        从完整模板库聚合出 (center_su, hop1_ms) -> center_mu 加权中位数，
        作为 Layer2 在近似 2-hop 模板选择时的 1-hop 化学位移先验。
        """
        cache = getattr(self, '_layer1_mu_prior_cache', None)
        cache_key = id(lib)
        if isinstance(cache, dict) and cache.get('cache_key') == int(cache_key):
            return cache.get('priors', {}) or {}

        templates = lib.get('templates', {}) if isinstance(lib, dict) else {}
        agg: Dict[Tuple[int, Tuple[int, ...]], Dict[str, List[float]]] = {}
        for kt, tpl in templates.items():
            if not (isinstance(kt, tuple) and len(kt) == 3 and isinstance(tpl, dict)):
                continue
            try:
                center_su = int(kt[0])
                hop1_ms = tuple(sorted(int(x) for x in tuple(kt[1])))
                center_mu = float(tpl.get('center_mu'))
            except Exception:
                continue
            if not np.isfinite(center_mu):
                continue
            try:
                weight = float(int(tpl.get('sample_count', 0)))
            except Exception:
                weight = 0.0
            if float(weight) <= 0.0:
                weight = 1.0
            key = (int(center_su), tuple(hop1_ms))
            entry = agg.setdefault(key, {'mu': [], 'w': []})
            entry['mu'].append(float(center_mu))
            entry['w'].append(float(weight))

        priors: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        for key, entry in agg.items():
            mu_vals = np.asarray(entry.get('mu', []), dtype=np.float64)
            weights = np.asarray(entry.get('w', []), dtype=np.float64)
            if mu_vals.size == 0 or weights.size == 0:
                continue
            order = np.argsort(mu_vals)
            mu_sorted = mu_vals[order]
            w_sorted = np.maximum(weights[order], 1e-8)
            cdf = np.cumsum(w_sorted) / float(np.sum(w_sorted))
            pick = int(np.searchsorted(cdf, 0.5, side='left'))
            pick = min(max(pick, 0), len(mu_sorted) - 1)
            priors[key] = float(mu_sorted[pick])

        self._layer1_mu_prior_cache = {
            'cache_key': int(cache_key),
            'priors': priors,
        }
        return priors

    def _get_layer1_mu_prior(self,
                             center_su: int,
                             hop1_ms: Tuple[int, ...],
                             lib: Dict) -> Optional[float]:
        priors = self._build_layer1_mu_prior_index(lib)
        val = priors.get((int(center_su), tuple(hop1_ms)))
        return float(val) if val is not None and np.isfinite(float(val)) else None

    def _resolve_layer1_mu_prior(self,
                                 node: _NodeV3,
                                 center_su: int,
                                 hop1_ms: Tuple[int, ...],
                                 lib: Dict) -> Optional[float]:
        """Use the node's accepted Layer1 chemical shift before library fallback."""
        try:
            node_mu = float(getattr(node, 'mu', 0.0))
            if self._valid_mu(node_mu):
                return float(node_mu)
        except Exception:
            pass
        return self._get_layer1_mu_prior(center_su, hop1_ms, lib)
    
    # ========================================================================
    # 主方法
    # ========================================================================
    
    def layer2_assign(self, nodes: List[_NodeV3],
                      S_target: torch.Tensor,
                      E_target: torch.Tensor,
                      output_dir: Optional[str] = None,
                      verbose: bool = False,
                      emit_summary: bool = True) -> List[_NodeV3]:
        """
        Layer2: 为每个节点分配2-hop邻居和z向量
        
        Args:
            nodes: Layer1分配后的节点列表
            S_target: 目标谱图
            E_target: 目标元素组成
            output_dir: 输出目录
        
        Returns:
            nodes: 更新后的节点列表
        """
        device = self.device
        S_target = S_target.to(device).flatten()
        E_target = E_target.to(device).flatten()
        
        if verbose:
            print("\n" + "=" * 60)
            print("Layer2: 2-hop推导 & 模板匹配")
            print("=" * 60)

        self.stats = {
            'exact_match': 0,
            'approx_match': 0,
            'missing': 0,
            'forced_approx': 0,
            'ambiguous_approx': 0,
            'approx_residual_fill': 0,
            'approx_least_damage': 0,
            'inventory_filtered_candidates': 0,
            'inventory_relaxed_assignments': 0,
        }
        
        # 2-hop推导
        self._derive_hop2(nodes)
        
        # 加载模板库
        lib = self._get_template_library()
        if lib is None:
            raise RuntimeError(f"Layer2 template library unavailable: {self.lib_path}")
        coverage_errors: List[str] = []
        fixed_assignments: List[Dict[str, object]] = []
        ambiguous_assignments: List[Dict[str, object]] = []
        available_su_types = {int(node.su_type) for node in nodes}
        inventory_filtered_su_types = Counter()

        for n in nodes:
            center_su = int(n.su_type)
            if center_su not in self._carbon_su_set:
                self._reset_node_template_state(
                    n,
                    device=device,
                    template_key=None,
                    match_mode='non_carbon',
                    mu_prior=None,
                )
                continue
            hop1_ms = multiset_from_counter(n.hop1_su)
            hop2_ms = multiset_from_counter(n.hop2_su)

            mu_prior_val = None
            try:
                mu_prior_tmp = self._resolve_layer1_mu_prior(
                    n,
                    center_su,
                    hop1_ms,
                    lib,
                )
                if mu_prior_tmp is not None and float(mu_prior_tmp) > 1e-6:
                    mu_prior_val = float(mu_prior_tmp)
            except Exception as e:
                import logging
                logging.debug(f"Failed to get mu_prior for node {n.global_id}: {e}")
            
            inventory_filter_diag: Dict[str, object] = {}
            mode, candidates = self._minimum_hop2_distance_candidates(
                center_su,
                hop1_ms,
                hop2_ms,
                mu_prior_val,
                lib,
                available_su_types=available_su_types,
                filter_diagnostics=inventory_filter_diag,
            )
            self.stats['inventory_filtered_candidates'] += int(
                inventory_filter_diag.get('filtered_candidates', 0)
            )
            inventory_filtered_su_types.update(
                inventory_filter_diag.get('filtered_su_types', {})
            )
            if any(bool(candidate.get('inventory_relaxed', False)) for candidate in candidates):
                self.stats['inventory_relaxed_assignments'] += 1
            if mode == 'exact':
                self.stats['exact_match'] += 1
                fixed_assignments.append({
                    'node': n,
                    'candidate': candidates[0],
                    'mode': 'exact',
                    'mu_prior': mu_prior_val,
                    'candidate_count': 1,
                })
            elif candidates:
                self.stats['approx_match'] += 1
                assignment = {
                    'node': n,
                    'candidates': candidates,
                    'mu_prior': mu_prior_val,
                    'candidate_count': len(candidates),
                }
                if len(candidates) == 1:
                    self.stats['forced_approx'] += 1
                    fixed_assignments.append({
                        'node': n,
                        'candidate': candidates[0],
                        'mode': 'forced_approx',
                        'mu_prior': mu_prior_val,
                        'candidate_count': 1,
                    })
                else:
                    self.stats['ambiguous_approx'] += 1
                    ambiguous_assignments.append(assignment)
                    self._reset_node_template_state(
                        n,
                        device=device,
                        template_key=None,
                        match_mode='approx_pending',
                        mu_prior=mu_prior_val,
                    )
                    n.score_components['layer2_candidate_count'] = len(candidates)
            else:
                self.stats['missing'] += 1
                self._reset_node_template_state(
                    n,
                    device=device,
                    template_key=None,
                    match_mode='missing',
                    mu_prior=mu_prior_val,
                )
                if mode == 'missing_inventory':
                    coverage_errors.append(
                        f"node={int(n.global_id)} SU{center_su} has no inventory-compatible "
                        f"approximate template; absent_su_types="
                        f"{sorted(int(value) for value in inventory_filter_diag.get('filtered_su_types', {}))}"
                    )
                else:
                    coverage_errors.append(
                        f"node={int(n.global_id)} SU{center_su} has no template with exact "
                        f"hop1={list(hop1_ms)} for derived hop2={list(hop2_ms)}"
                    )

        self.stats['inventory_filtered_su_types'] = dict(
            sorted((int(key), int(value)) for key, value in inventory_filtered_su_types.items())
        )

        for assignment in fixed_assignments:
            error = self._assign_template_candidate(
                node=assignment['node'],
                candidate=assignment['candidate'],
                match_mode=str(assignment['mode']),
                mu_prior=assignment['mu_prior'],
                candidate_count=int(assignment['candidate_count']),
            )
            if error:
                coverage_errors.append(error)

        fixed_recon = self.reconstruct_spectrum(nodes).to(device=device, dtype=torch.float64)
        target_eval = S_target.to(device=device, dtype=torch.float64).flatten()
        eval_size = min(int(target_eval.numel()), int(fixed_recon.numel()))
        target_eval = target_eval[:eval_size]
        current_recon = fixed_recon
        current_residual = target_eval - current_recon[:eval_size]
        self.stats['fixed_reconstruction_sse'] = float(
            torch.sum(current_residual * current_residual).item()
        )

        ambiguous_assignments.sort(key=lambda item: (
            int(item['candidates'][0]['hop2_match_tier']),
            int(item['candidate_count']),
            int(item['node'].global_id),
        ))
        ppm_axis = PPM_AXIS.to(device=device, dtype=torch.float64).flatten()
        for assignment in ambiguous_assignments:
            node = assignment['node']
            mu_prior_val = assignment['mu_prior']
            candidates = assignment['candidates']
            current_sse = float(torch.sum(current_residual * current_residual).item())
            scored_candidates: List[Tuple[Tuple[object, ...], Dict[str, object], torch.Tensor, float, float]] = []

            for candidate in candidates:
                validation_error = self._validate_template_candidate(node, candidate)
                if validation_error:
                    continue
                template_mu = float(candidate['template_mu'])
                peak = self._unit_peak(
                    template_mu,
                    ppm_axis,
                    peak_scale=reconstruction_peak_scale(
                        int(node.su_type),
                        self.aromatic_response_scale,
                    ),
                )
                candidate_residual = current_residual - peak[:eval_size]
                candidate_sse = float(torch.sum(candidate_residual * candidate_residual).item())
                negative_overshoot = float(
                    torch.clamp(-candidate_residual, min=0.0).max().item()
                ) if eval_size > 0 else 0.0
                rank = (
                    candidate_sse,
                    negative_overshoot,
                    *self._candidate_prior_rank(candidate, mu_prior_val),
                )
                scored_candidates.append((
                    rank,
                    candidate,
                    peak,
                    candidate_sse,
                    negative_overshoot,
                ))

            if not scored_candidates:
                coverage_errors.append(
                    f"node={int(node.global_id)} has no valid template in minimum "
                    f"hop2-distance tier"
                )
                continue

            _, chosen, chosen_peak, chosen_sse, negative_overshoot = min(
                scored_candidates,
                key=lambda item: item[0],
            )
            improves_residual = bool(chosen_sse < current_sse - 1e-12)
            match_mode = 'approx_residual_fill' if improves_residual else 'approx_least_damage'
            self.stats[match_mode] += 1
            error = self._assign_template_candidate(
                node=node,
                candidate=chosen,
                match_mode=match_mode,
                mu_prior=mu_prior_val,
                candidate_count=int(assignment['candidate_count']),
                residual_meta={
                    'sse_before': current_sse,
                    'sse_after': chosen_sse,
                    'negative_overshoot': negative_overshoot,
                },
            )
            if error:
                coverage_errors.append(error)
                continue
            current_recon = current_recon + chosen_peak
            current_residual = target_eval - current_recon[:eval_size]

        if coverage_errors:
            preview = " | ".join(coverage_errors[:8])
            if len(coverage_errors) > 8:
                preview += f" | ... {len(coverage_errors) - 8} more"
            raise RuntimeError(f"Layer2 complete template coverage failed: {preview}")

        self.stats['selected_spectral_source'] = 'template'
        self.stats['spectral_contract'] = 'template_center_mu'
        
        # NMR评估
        S_recon = self.reconstruct_spectrum(nodes)
        eval_info = evaluate_spectrum_reconstruction(
            S_target,
            S_recon,
            ppm_axis=PPM_AXIS.to(device),
        )
        r2 = float(eval_info.get('r2', 0.0))
        if verbose:
            print(f"  匹配: 精确={self.stats['exact_match']}, 近似={self.stats['approx_match']}, 未匹配={self.stats['missing']}")
            print(
                f"  近似选择: 唯一={self.stats['forced_approx']}, "
                f"补差谱={self.stats['approx_residual_fill']}, "
                f"最小损伤={self.stats['approx_least_damage']}, "
                f"库存过滤={self.stats['inventory_filtered_candidates']}, "
                f"库存妥协={self.stats['inventory_relaxed_assignments']}"
            )
            print(f"  R²={r2:.4f}")
        elif emit_summary:
            print(
                f"[Layer2] exact={self.stats['exact_match']} "
                f"approx={self.stats['approx_match']} "
                f"(forced={self.stats['forced_approx']} "
                f"fill={self.stats['approx_residual_fill']} "
                f"damage={self.stats['approx_least_damage']}) "
                f"filtered={self.stats['inventory_filtered_candidates']} "
                f"relaxed={self.stats['inventory_relaxed_assignments']} "
                f"missing={self.stats['missing']} r2={r2:.4f}",
                flush=True,
            )
        
        # 保存结果
        if output_dir:
            self._save_results(nodes, S_target, S_recon, r2, output_dir)
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                match_detail_rows = self._build_match_detail_rows(nodes)
                pd.DataFrame(match_detail_rows).to_csv(str(Path(output_dir) / 'layer2_template_match_detail.csv'), index=False)
            except Exception as e:
                print(f"  [Layer2] 保存模板匹配明细失败: {e}")
        
        if verbose:
            print("Layer2 完成\n")
        
        return nodes
    
    # ========================================================================
    # 2-hop 推导
    # ========================================================================
    
    def _derive_hop2(self, nodes: List[_NodeV3]):
        """从1-hop邻居推导2-hop邻居"""
        node_map: Dict[int, _NodeV3] = {}
        for n in nodes:
            try:
                node_map[int(n.global_id)] = n
            except Exception as e:
                import logging
                logging.debug(f"Failed to add node {getattr(n, 'global_id', '?')} to node_map: {e}")
                continue
        for n in nodes:
            c_id = int(n.global_id)
            hop2 = Counter()
            hop2_ids = set()
            
            for nb_id in n.hop1_ids:
                try:
                    nb = node_map.get(int(nb_id), None)
                except:
                    continue
                if nb is None:
                    continue
                try:
                    if c_id not in {int(node_id) for node_id in list(nb.hop1_ids or [])}:
                        continue
                except Exception:
                    continue
                
                for nb2_id in nb.hop1_ids:
                    nb2_id_i = int(nb2_id)
                    if nb2_id_i == c_id:
                        continue
                    if nb2_id_i in hop2_ids:
                        continue
                    
                    try:
                        nb2 = node_map.get(nb2_id_i, None)
                        if nb2 is None:
                            continue
                        if int(nb.global_id) not in {
                            int(node_id) for node_id in list(nb2.hop1_ids or [])
                        }:
                            continue
                        su2 = int(nb2.su_type)
                    except:
                        continue
                    
                    hop2[su2] += 1
                    hop2_ids.add(nb2_id_i)
            
            n.hop2_su = hop2
            n.hop2_ids = sorted(int(node_id) for node_id in hop2_ids)
    
    # ========================================================================
    # 模板检索
    # ========================================================================
    
    def _get_template_library(self) -> Optional[Dict]:
        """获取模板库（带缓存）"""
        if self._template_cache is not None:
            return self._template_cache
        
        lib_path = self.lib_path
        if not Path(lib_path).exists():
            return None
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    lib = torch.load(lib_path, map_location='cpu', weights_only=False)
                except TypeError:
                    lib = torch.load(lib_path, map_location='cpu')
            templates = lib.get('templates', {}) if isinstance(lib, dict) else {}
            center_index = lib.get('center_index', {}) if isinstance(lib, dict) else {}

            center_hop1_index = defaultdict(list)
            for kt, tpl in templates.items():
                if not (isinstance(kt, tuple) and len(kt) == 3 and isinstance(tpl, dict)):
                    continue
                try:
                    center_su = int(kt[0])
                    h1_t = tuple(sorted(int(x) for x in tuple(kt[1])))
                    h2_t = tuple(sorted(int(x) for x in tuple(kt[2])))
                except Exception:
                    continue
                try:
                    sample_count = int(tpl.get('sample_count', 0))
                except Exception:
                    sample_count = 0
                mu_tpl = None
                try:
                    if tpl.get('center_mu') is not None:
                        mu_tpl = float(tpl.get('center_mu'))
                except Exception:
                    mu_tpl = None
                item = (kt, h1_t, h2_t, int(sample_count), mu_tpl)
                center_hop1_index[(int(center_su), tuple(h1_t))].append(item)
            
            self._template_cache = {
                'templates': templates,
                'center_index': center_index,
                'center_hop1_index': dict(center_hop1_index),
            }
            return self._template_cache
        except Exception as e:
            print(f"  加载模板库失败: {e}")
            return None
    
    def _select_template_key(self, center_su: int,
                              hop1_ms: Tuple[int, ...],
                              hop2_ms: Tuple[int, ...],
                              mu_prior: Optional[float],
                              lib: Dict) -> Tuple[Optional[Tuple], str, Dict[str, object]]:
        """Backward-compatible single-template lookup used by diagnostics/tests."""
        mode, candidates = self._minimum_hop2_distance_candidates(
            center_su,
            hop1_ms,
            hop2_ms,
            mu_prior,
            lib,
        )
        if not candidates:
            return None, 'missing', self._missing_match_meta()
        if mode == 'exact':
            chosen = candidates[0]
            return chosen['key'], 'exact', self._candidate_match_meta(chosen)

        chosen = min(
            candidates,
            key=lambda candidate: self._candidate_prior_rank(candidate, mu_prior),
        )
        return chosen['key'], 'approx_h2', self._candidate_match_meta(chosen)

    @staticmethod
    def _missing_match_meta() -> Dict[str, object]:
        return {
            'chosen_hop1': (),
            'chosen_hop2': (),
            'hop2_diff_nodes': '',
            'hop2_match_tier': '',
            'template_mu': '',
            'mu_diff': '',
        }

    @staticmethod
    def _template_key_rank(key: Tuple) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
        return (
            int(key[0]),
            tuple(int(value) for value in tuple(key[1])),
            tuple(int(value) for value in tuple(key[2])),
        )

    def _candidate_prior_rank(self,
                              candidate: Dict[str, object],
                              mu_prior: Optional[float]) -> Tuple[object, ...]:
        template_mu = candidate.get('template_mu')
        has_mu_prior = bool(mu_prior is not None and self._valid_mu(template_mu))
        mu_diff = (
            abs(float(template_mu) - float(mu_prior))
            if has_mu_prior
            else float('inf')
        )
        return (
            int(not has_mu_prior),
            float(mu_diff),
            -int(candidate.get('sample_count', 0)),
            self._template_key_rank(candidate['key']),
        )

    @staticmethod
    def _candidate_match_meta(candidate: Dict[str, object]) -> Dict[str, object]:
        missing_su_types = tuple(int(v) for v in tuple(candidate.get('inventory_missing_su_types', ()) or ()))
        return {
            'chosen_hop1': tuple(candidate.get('chosen_hop1', ())),
            'chosen_hop2': tuple(candidate.get('chosen_hop2', ())),
            'hop2_diff_nodes': int(candidate.get('hop2_diff_nodes', 0)),
            'hop2_match_tier': int(candidate.get('hop2_match_tier', 0)),
            'template_mu': candidate.get('template_mu', ''),
            'mu_diff': candidate.get('mu_diff', ''),
            'inventory_relaxed': bool(candidate.get('inventory_relaxed', False)),
            'inventory_missing_su_types': list(missing_su_types),
        }

    def _minimum_hop2_distance_candidates(
            self,
            center_su: int,
            hop1_ms: Tuple[int, ...],
            hop2_ms: Tuple[int, ...],
            mu_prior: Optional[float],
            lib: Dict,
            available_su_types: Optional[Set[int]] = None,
            filter_diagnostics: Optional[Dict[str, object]] = None,
    ) -> Tuple[str, List[Dict[str, object]]]:
        """Return all templates in the minimum 2-hop-distance tier.

        Hop1 remains a hard constraint. If the exact hop1 bucket exists but all
        inventory-compatible candidates are filtered out, fall back to the same
        hop1 bucket in a relaxed mode instead of widening hop1.
        """
        templates = lib.get('templates', {})
        key_exact = (int(center_su), tuple(hop1_ms), tuple(hop2_ms))
        if key_exact in templates:
            tpl = templates.get(key_exact, {})
            template_mu = tpl.get('center_mu') if isinstance(tpl, dict) else None
            return 'exact', [{
                'key': key_exact,
                'template': tpl,
                'chosen_hop1': tuple(hop1_ms),
                'chosen_hop2': tuple(hop2_ms),
                'hop2_diff_nodes': 0,
                'hop2_match_tier': 0,
                'template_mu': float(template_mu) if self._valid_mu(template_mu) else template_mu,
                'mu_diff': 0.0,
                'sample_count': int(tpl.get('sample_count', 0)) if isinstance(tpl, dict) else 0,
            }]

        same_h1 = list(
            lib.get('center_hop1_index', {}).get(
                (int(center_su), tuple(hop1_ms)),
                [],
            ) or []
        )
        strict_candidates: List[Dict[str, object]] = []
        relaxed_candidates: List[Dict[str, object]] = []
        seen_keys = set()
        filtered_candidates = 0
        filtered_su_types = Counter()
        for kt, h1_t, h2_t, sample_count, indexed_mu in same_h1:
            if kt in seen_keys:
                continue
            seen_keys.add(kt)
            tpl = templates.get(kt)
            if not isinstance(tpl, dict):
                continue
            candidate_hop2 = tuple(int(value) for value in tuple(h2_t))
            template_mu = tpl.get('center_mu', indexed_mu)
            hop2_diff = int(multiset_diff_nodes(tuple(hop2_ms), candidate_hop2))
            mu_diff = (
                abs(float(template_mu) - float(mu_prior))
                if mu_prior is not None and self._valid_mu(template_mu)
                else float('inf')
            )
            candidate = {
                'key': kt,
                'template': tpl,
                'chosen_hop1': tuple(h1_t),
                'chosen_hop2': candidate_hop2,
                'hop2_diff_nodes': hop2_diff,
                'hop2_match_tier': hop2_diff,
                'template_mu': float(template_mu) if self._valid_mu(template_mu) else template_mu,
                'mu_diff': float(mu_diff) if np.isfinite(mu_diff) else '',
                'sample_count': int(tpl.get('sample_count', sample_count)),
                'inventory_relaxed': False,
                'inventory_missing_su_types': (),
            }
            relaxed_candidates.append(candidate)
            if available_su_types is not None:
                absent_su_types = sorted(
                    set(candidate_hop2) - set(available_su_types)
                )
                if absent_su_types:
                    filtered_candidates += 1
                    filtered_su_types.update(absent_su_types)
                    candidate = dict(candidate)
                    candidate['inventory_relaxed'] = True
                    candidate['inventory_missing_su_types'] = tuple(absent_su_types)
                    relaxed_candidates[-1] = candidate
                    continue
            strict_candidates.append(candidate)

        chosen_pool = strict_candidates if strict_candidates else relaxed_candidates

        if filter_diagnostics is not None:
            filter_diagnostics['filtered_candidates'] = int(filtered_candidates)
            filter_diagnostics['filtered_su_types'] = dict(filtered_su_types)
            filter_diagnostics['strict_candidates'] = int(len(strict_candidates))
            filter_diagnostics['relaxed_candidates'] = int(len(relaxed_candidates))
            filter_diagnostics['relaxed_fallback_used'] = bool(
                not strict_candidates and bool(relaxed_candidates) and filtered_candidates > 0
            )
        if not chosen_pool:
            return ('missing_inventory' if filtered_candidates else 'missing'), []
        minimum_tier = min(int(candidate['hop2_match_tier']) for candidate in chosen_pool)
        minimum_candidates = [
            candidate
            for candidate in chosen_pool
            if int(candidate['hop2_match_tier']) == minimum_tier
        ]
        minimum_candidates.sort(
            key=lambda candidate: self._candidate_prior_rank(candidate, mu_prior)
        )
        if strict_candidates:
            return 'approx_h2_tier', minimum_candidates
        return 'approx_h2_tier_relaxed_inventory', minimum_candidates

    def _validate_template_candidate(self,
                                     node: _NodeV3,
                                     candidate: Dict[str, object]) -> Optional[str]:
        template_key = candidate.get('key')
        tpl = candidate.get('template')
        if not isinstance(tpl, dict):
            return f"node={int(node.global_id)} template={template_key} is invalid"
        center_z = tpl.get('center_z')
        if not torch.is_tensor(center_z) or int(center_z.numel()) <= 0:
            return f"node={int(node.global_id)} template={template_key} has no center_z"
        template_mu = tpl.get('center_mu')
        if not self._valid_mu(template_mu):
            return (
                f"node={int(node.global_id)} template={template_key} "
                f"has invalid center_mu={template_mu}"
            )
        return None

    def _assign_template_candidate(
            self,
            node: _NodeV3,
            candidate: Dict[str, object],
            match_mode: str,
            mu_prior: Optional[float],
            candidate_count: int,
            residual_meta: Optional[Dict[str, float]] = None) -> Optional[str]:
        validation_error = self._validate_template_candidate(node, candidate)
        if validation_error:
            return validation_error

        template_key = candidate['key']
        tpl = candidate['template']
        self._reset_node_template_state(
            node,
            device=self.device,
            template_key=template_key,
            match_mode=match_mode,
            mu_prior=mu_prior,
        )
        template_mu = float(tpl['center_mu'])
        node.z_vec = tpl['center_z'].detach().clone().to(self.device)
        node.mu = template_mu
        node.pi = 1.0

        score_components = node.score_components
        score_components['layer2_template_mu'] = template_mu
        score_components['layer2_template_sample_count'] = int(tpl.get('sample_count', 0))
        score_components['layer2_hop2_diff_nodes'] = int(candidate['hop2_diff_nodes'])
        score_components['layer2_hop2_match_tier'] = int(candidate['hop2_match_tier'])
        score_components['layer2_mu_diff'] = candidate.get('mu_diff', '')
        score_components['layer2_candidate_count'] = int(candidate_count)
        score_components['layer2_inventory_relaxed'] = bool(candidate.get('inventory_relaxed', False))
        missing_su_types = tuple(int(v) for v in tuple(candidate.get('inventory_missing_su_types', ()) or ()))
        score_components['layer2_inventory_missing_su_types'] = list(missing_su_types)
        score_components['layer2_library_mu'] = template_mu
        score_components['layer2_library_source'] = 'template_center_mu'
        score_components['layer2_sample_idx'] = int(tpl.get('center_sample_idx', -1))
        score_components['layer2_spectral_source'] = 'layer2_template_center_mu'
        score_components['spectral_source'] = 'layer2_template_center_mu'
        if residual_meta is not None:
            sse_before = float(residual_meta['sse_before'])
            sse_after = float(residual_meta['sse_after'])
            score_components['layer2_residual_sse_before'] = sse_before
            score_components['layer2_residual_sse_after'] = sse_after
            score_components['layer2_residual_sse_gain'] = sse_before - sse_after
            score_components['layer2_negative_overshoot'] = float(
                residual_meta['negative_overshoot']
            )
        return None

    @staticmethod
    def _unit_peak(
        mu: float,
        ppm_axis: torch.Tensor,
        hwhm: float = 1.0,
        *,
        peak_scale: float = 1.0,
    ) -> torch.Tensor:
        hwhm_sq = float(hwhm) ** 2
        peak = hwhm_sq / ((ppm_axis - float(mu)) ** 2 + hwhm_sq)
        return peak * float(peak_scale)
    
    # ========================================================================
    # 谱图重建
    # ========================================================================
    
    def reconstruct_spectrum(self, nodes: List[_NodeV3]) -> torch.Tensor:
        """从节点重建NMR谱图"""
        device = self.device
        ppm_axis = PPM_AXIS.to(device)
        
        mus = []
        pis = []
        peak_scales = []
        
        for n in nodes:
            # 只处理含碳的SU
            if int(n.su_type) not in self._carbon_su_set:
                continue
            
            mu = float(getattr(n, 'mu', 0.0))
            if mu == 0.0:
                continue
            mus.append(mu)
            pis.append(1.0)
            peak_scales.append(
                reconstruction_peak_scale(
                    int(n.su_type),
                    self.aromatic_response_scale,
                )
            )
        
        return reconstruct_from_mu_pi(
            mus=mus,
            pis=pis,
            ppm_axis=ppm_axis,
            hwhm=1.0,
            intensity_scale=1.0,
            device=device,
            unit_peak_intensity=True,
            peak_scales=peak_scales,
        )

    def _build_match_detail_rows(self, nodes: List[_NodeV3]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for n in nodes:
            try:
                center_su = int(n.su_type)
                hop1_ms = multiset_from_counter(n.hop1_su)
                hop2_ms = multiset_from_counter(n.hop2_su)
            except Exception:
                continue

            tpl_key = getattr(n, 'template_key', None)
            chosen_hop1: Tuple[int, ...] = ()
            chosen_hop2: Tuple[int, ...] = ()
            if isinstance(tpl_key, tuple) and len(tpl_key) == 3:
                try:
                    chosen_hop1 = tuple(int(x) for x in tuple(tpl_key[1]))
                    chosen_hop2 = tuple(int(x) for x in tuple(tpl_key[2]))
                except Exception:
                    chosen_hop1 = ()
                    chosen_hop2 = ()

            sc = getattr(n, 'score_components', {}) or {}
            rows.append({
                'global_id': int(getattr(n, 'global_id', -1)),
                'center_su': int(center_su),
                'observed_hop1': str(tuple(hop1_ms)),
                'observed_hop2': str(tuple(hop2_ms)),
                'chosen_hop1': str(tuple(chosen_hop1)),
                'chosen_hop2': str(tuple(chosen_hop2)),
                'hop2_diff_nodes': sc.get('layer2_hop2_diff_nodes', ''),
                'layer1_mu_prior': sc.get('layer2_mu_prior', ''),
                'template_mu': sc.get('layer2_template_mu', ''),
                'mu_diff': sc.get('layer2_mu_diff', ''),
                'match_mode': sc.get('layer2_match_mode', ''),
                'candidate_count': sc.get('layer2_candidate_count', ''),
                'residual_sse_before': sc.get('layer2_residual_sse_before', ''),
                'residual_sse_after': sc.get('layer2_residual_sse_after', ''),
                'residual_sse_gain': sc.get('layer2_residual_sse_gain', ''),
                'negative_overshoot': sc.get('layer2_negative_overshoot', ''),
                'selected_z_source': 'template_z' if tpl_key is not None else '',
                'selected_spectral_source': sc.get('layer2_spectral_source', ''),
                'final_mu': float(getattr(n, 'mu', 0.0) or 0.0),
                'final_pi': float(getattr(n, 'pi', 0.0) or 0.0),
                'sample_idx': sc.get('layer2_sample_idx', ''),
                'template_key': str(tpl_key) if tpl_key is not None else '',
            })
        return rows
    
    # ========================================================================
    # 结果保存
    # ========================================================================
    
    def _save_results(self, nodes: List[_NodeV3],
                      S_target: torch.Tensor,
                      S_recon: torch.Tensor,
                      r2: float,
                      output_dir: str):
        """保存Layer2结果"""
        try:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            ppm_axis = PPM_AXIS.cpu().numpy()
            
            # 保存谱图对比
            df = pd.DataFrame({
                'ppm': ppm_axis,
                'target': S_target.cpu().numpy(),
                'reconstructed': S_recon.cpu().numpy(),
                'difference': (S_target - S_recon).cpu().numpy(),
            })
            df.to_csv(str(out_dir / 'layer2_spectrum_comparison.csv'), index=False)

            try:
                visualize_spectrum_comparison(
                    S_target=S_target.detach().cpu(),
                    S_recon=S_recon.detach().cpu(),
                    ppm_axis=PPM_AXIS.detach().cpu(),
                    layer_name='Layer2',
                    save_dir=str(out_dir),
                )
                visualize_spectrum_comparison(
                    S_target=S_target.detach().cpu(),
                    S_recon=S_recon.detach().cpu(),
                    ppm_axis=PPM_AXIS.detach().cpu(),
                    layer_name='Layer2-Library',
                    save_dir=str(out_dir),
                )
            except Exception:
                pass

            try:
                ppm_axis_t = PPM_AXIS.to(S_target.device).flatten()
                carbon_count = sum(
                    1 for node in nodes if int(node.su_type) in self._carbon_su_set
                )
                metrics = {
                    'r2': float(r2),
                    'r2_carbonyl': float(compute_segment_r2(S_target, S_recon, ppm_axis_t, 'carbonyl')),
                    'r2_aromatic': float(compute_segment_r2(S_target, S_recon, ppm_axis_t, 'aromatic')),
                    'r2_aliphatic': float(compute_segment_r2(S_target, S_recon, ppm_axis_t, 'aliphatic')),
                    'exact_match': int(self.stats.get('exact_match', 0)),
                    'approx_match': int(self.stats.get('approx_match', 0)),
                    'missing': int(self.stats.get('missing', 0)),
                    'forced_approx': int(self.stats.get('forced_approx', 0)),
                    'ambiguous_approx': int(self.stats.get('ambiguous_approx', 0)),
                    'approx_residual_fill': int(self.stats.get('approx_residual_fill', 0)),
                    'approx_least_damage': int(self.stats.get('approx_least_damage', 0)),
                    'inventory_filtered_candidates': int(
                        self.stats.get('inventory_filtered_candidates', 0)
                    ),
                    'inventory_filtered_su_types': str(
                        self.stats.get('inventory_filtered_su_types', {})
                    ),
                    'matched_ratio': float(
                        (int(self.stats.get('exact_match', 0)) + int(self.stats.get('approx_match', 0))) /
                        max(1, int(carbon_count))
                    ),
                    'n_nodes': int(len(nodes)),
                    'n_carbon_nodes': int(carbon_count),
                    'selected_spectral_source': self.stats.get('selected_spectral_source'),
                }
                pd.DataFrame([metrics]).to_csv(str(out_dir / 'layer2_eval_metrics.csv'), index=False)
            except Exception:
                pass
            
            # 保存节点详情
            su_names = [name for name, _ in SU_DEFS]
            lib = self._get_template_library()
            templates = lib.get('templates', {}) if isinstance(lib, dict) else {}

            node_data = []
            peaks_data = []
            for n in nodes:
                center_su = int(n.su_type)
                hop1_ms = multiset_from_counter(n.hop1_su)
                hop2_ms = multiset_from_counter(n.hop2_su)

                key_obs = (center_su, tuple(hop1_ms), tuple(hop2_ms))
                tpl_key = getattr(n, 'template_key', None)
                tpl_h1 = None
                tpl_h2 = None
                if isinstance(tpl_key, tuple) and len(tpl_key) == 3:
                    tpl_h1 = tuple(tpl_key[1])
                    tpl_h2 = tuple(tpl_key[2])

                matched = tpl_key is not None
                approx_used = bool(matched and tpl_key != key_obs)

                d_h1 = multiset_l1_distance(tuple(hop1_ms), tuple(tpl_h1)) if tpl_h1 is not None else None
                d_h2 = multiset_l1_distance(tuple(hop2_ms), tuple(tpl_h2)) if tpl_h2 is not None else None
                d_total = (int(d_h1) + int(d_h2)) if d_h1 is not None and d_h2 is not None else None

                tpl = templates.get(tpl_key, {}) if matched else {}
                sample_count = int(tpl.get('sample_count', 0)) if isinstance(tpl, dict) else 0

                z_vec = getattr(n, 'z_vec', None)
                z_norm = None
                z_head = None
                if isinstance(z_vec, torch.Tensor) and z_vec.numel() > 0:
                    try:
                        z_norm = float(z_vec.detach().float().norm().item())
                        z_head = ",".join([f"{float(v):.4f}" for v in z_vec.detach().flatten()[:4].cpu().tolist()])
                    except Exception:
                        pass

                node_data.append({
                    'global_id': n.global_id,
                    'su_type': n.su_type,
                    'su_name': su_names[int(n.su_type)] if int(n.su_type) < len(su_names) else str(n.su_type),
                    'mu': n.mu,
                    'pi': n.pi,
                    'hop1_degree': sum(n.hop1_su.values()),
                    'hop2_degree': sum(n.hop2_su.values()),
                    'hop1_ms': str(list(hop1_ms)),
                    'hop2_ms': str(list(hop2_ms)),
                    'matched': bool(matched),
                    'approx_used': bool(approx_used),
                    'distance_h1': d_h1,
                    'distance_h2': d_h2,
                    'distance_total': d_total,
                    'hop1_diff_nodes': multiset_diff_nodes(tuple(hop1_ms), tuple(tpl_h1)) if tpl_h1 is not None else None,
                    'hop2_diff_nodes': multiset_diff_nodes(tuple(hop2_ms), tuple(tpl_h2)) if tpl_h2 is not None else None,
                    'mu_prior': (getattr(n, 'score_components', {}) or {}).get('layer2_mu_prior', None),
                    'mu_tpl': (getattr(n, 'score_components', {}) or {}).get('layer2_template_mu', None),
                    'match_mode': (getattr(n, 'score_components', {}) or {}).get('layer2_match_mode', None),
                    'candidate_count': (getattr(n, 'score_components', {}) or {}).get('layer2_candidate_count', None),
                    'residual_sse_gain': (getattr(n, 'score_components', {}) or {}).get('layer2_residual_sse_gain', None),
                    'spectral_source': (getattr(n, 'score_components', {}) or {}).get('layer2_spectral_source', None),
                    'sample_count': sample_count,
                    'z_norm': z_norm,
                    'z_head4': z_head,
                    'template_key': str(n.template_key),
                })

                peaks_data.append({
                    'global_id': n.global_id,
                    'center_su_idx': int(n.su_type),
                    'center_su': su_names[int(n.su_type)] if int(n.su_type) < len(su_names) else str(n.su_type),
                    'hop1_ms': str(list(hop1_ms)),
                    'hop2_ms': str(list(hop2_ms)),
                    'matched': bool(matched),
                    'approx_used': bool(approx_used),
                    'match_mode': (getattr(n, 'score_components', {}) or {}).get('layer2_match_mode', None),
                    'candidate_count': (getattr(n, 'score_components', {}) or {}).get('layer2_candidate_count', None),
                    'residual_sse_gain': (getattr(n, 'score_components', {}) or {}).get('layer2_residual_sse_gain', None),
                    'chosen_template_key': str(tpl_key),
                    'chosen_hop1_ms': str(list(tpl_h1)) if tpl_h1 is not None else '',
                    'chosen_hop2_ms': str(list(tpl_h2)) if tpl_h2 is not None else '',
                    'distance_h1': d_h1 if d_h1 is not None else '',
                    'distance_h2': d_h2 if d_h2 is not None else '',
                    'distance_total': d_total if d_total is not None else '',
                    'hop1_diff_nodes': multiset_diff_nodes(tuple(hop1_ms), tuple(tpl_h1)) if tpl_h1 is not None else '',
                    'hop2_diff_nodes': multiset_diff_nodes(tuple(hop2_ms), tuple(tpl_h2)) if tpl_h2 is not None else '',
                    'mu_prior': (getattr(n, 'score_components', {}) or {}).get('layer2_mu_prior', None),
                    'mu_tpl': (getattr(n, 'score_components', {}) or {}).get('layer2_template_mu', None),
                    'spectral_source': (getattr(n, 'score_components', {}) or {}).get('layer2_spectral_source', None),
                    'sample_count': sample_count,
                    'mu': float(getattr(n, 'mu', 0.0)),
                    'pi': float(getattr(n, 'pi', 0.0)),
                })
            
            pd.DataFrame(node_data).to_csv(str(out_dir / 'layer2_nodes_detail.csv'), index=False)
            pd.DataFrame(peaks_data).to_csv(str(out_dir / 'layer2_node_peaks.csv'), index=False)
            
            print(f"  结果已保存至 {output_dir}")
        except Exception as e:
            print(f"  保存失败: {e}")


# ============================================================================
# 独立运行接口
# ============================================================================

def run_layer2_estimation(nodes: List[_NodeV3],
                          S_target: torch.Tensor,
                          E_target: torch.Tensor,
                          lib_path: Optional[str] = None,
                          device: str = 'cpu',
                          output_dir: Optional[str] = None) -> List[_NodeV3]:
    """
    独立运行 Layer2 估计
    
    Args:
        nodes: Layer1分配后的节点列表
        S_target: 目标谱图
        E_target: 目标元素组成
        lib_path: 模板库路径
        device: 计算设备
        output_dir: 输出目录
    
    Returns:
        nodes: 更新后的节点列表
    """
    estimator = Layer2Estimator(
        device=torch.device(device),
        lib_path=lib_path
    )
    return estimator.layer2_assign(
        nodes=nodes,
        S_target=S_target,
        E_target=E_target,
        output_dir=output_dir
    )
