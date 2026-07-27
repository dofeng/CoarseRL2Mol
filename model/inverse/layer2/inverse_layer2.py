import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from ...paths import Z_LIBRARY_DIR
from ...shared.coarse_graph import SU_DEFS, E_SU, NUM_SU_TYPES, PPM_AXIS
from ...shared.inverse_common import (
    _NodeV3, compute_segment_r2, reconstruct_from_mu_pi,
    visualize_spectrum_comparison, evaluate_spectrum_reconstruction,
    multiset_from_counter, multiset_l1_distance, multiset_diff_nodes,
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
        self.E_SU = E_SU.to(self.device)
        self._carbon_su_set = {
            int(i)
            for i in range(NUM_SU_TYPES)
            if float(E_SU[int(i), 0].detach().cpu().item()) > 0.0
        }
        
        # 路径配置
        base_dir = Z_LIBRARY_DIR
        self.lib_path = lib_path or str(base_dir / 'subgraph_library.pt')
        
        self.intensity_scale = 1.0
        self.unit_peak_intensity = True
        
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
                      hwhm: float = 1.0) -> List[_NodeV3]:
        """
        Layer2: 为每个节点分配2-hop邻居和z向量
        
        Args:
            nodes: Layer1分配后的节点列表
            S_target: 目标谱图
            E_target: 目标元素组成
            output_dir: 输出目录
            hwhm: 谱峰半高宽
        
        Returns:
            nodes: 更新后的节点列表
        """
        device = self.device
        S_target = S_target.to(device).flatten()
        E_target = E_target.to(device).flatten()
        
        print("\n" + "=" * 60)
        print("Layer2: 2-hop推导 & 模板匹配")
        print("=" * 60)

        self.stats = {
            'exact_match': 0,
            'approx_match': 0,
            'missing': 0,
        }
        
        # 2-hop推导
        self._derive_hop2(nodes)
        
        # 加载模板库
        lib = self._get_template_library()
        if lib is None:
            raise RuntimeError(f"Layer2 template library unavailable: {self.lib_path}")
        templates = lib.get('templates', {})
        coverage_errors: List[str] = []

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
            
            # 查找最佳模板
            tpl_key, mode, match_meta = self._select_template_key(
                center_su,
                hop1_ms,
                hop2_ms,
                mu_prior_val,
                lib,
            )
            tpl = templates.get(tpl_key, None) if tpl_key else None
            if tpl_key is not None and not isinstance(tpl, dict):
                tpl_key = None
                mode = 'missing'
            self._reset_node_template_state(
                n,
                device=device,
                template_key=tpl_key,
                match_mode=mode,
                mu_prior=mu_prior_val,
            )

            if mode == 'exact':
                self.stats['exact_match'] += 1
            elif mode.startswith('approx'):
                self.stats['approx_match'] += 1
            else:
                self.stats['missing'] += 1
                coverage_errors.append(
                    f"node={int(n.global_id)} SU{center_su} has no template with exact "
                    f"hop1={list(hop1_ms)} for derived hop2={list(hop2_ms)}"
                )
                continue
            
            try:
                if isinstance(getattr(n, 'score_components', None), dict) and isinstance(tpl, dict):
                    n.score_components['layer2_template_mu'] = float(tpl.get('center_mu')) if tpl.get('center_mu') is not None else None
                    n.score_components['layer2_template_sample_count'] = int(tpl.get('sample_count', 0))
                    n.score_components['layer2_hop2_diff_nodes'] = match_meta.get('hop2_diff_nodes')
                    n.score_components['layer2_hop2_match_tier'] = match_meta.get('hop2_match_tier')
                    n.score_components['layer2_mu_diff'] = match_meta.get('mu_diff')
            except Exception as e:
                import logging
                logging.debug(f"Failed to save template info for node {n.global_id}: {e}")

            center_z = tpl.get('center_z') if isinstance(tpl, dict) else None
            if not torch.is_tensor(center_z) or int(center_z.numel()) <= 0:
                coverage_errors.append(
                    f"node={int(n.global_id)} template={tpl_key} has no center_z"
                )
                continue
            n.z_vec = center_z.detach().clone().to(device)
            template_mu = tpl.get('center_mu') if isinstance(tpl, dict) else None
            if not self._valid_mu(template_mu):
                coverage_errors.append(
                    f"node={int(n.global_id)} template={tpl_key} has invalid center_mu={template_mu}"
                )
                continue
            n.mu = float(template_mu)
            n.pi = 1.0
            score_components = getattr(n, 'score_components', None)
            if isinstance(score_components, dict):
                score_components['layer2_library_mu'] = float(template_mu)
                score_components['layer2_library_source'] = 'template_center_mu'
                score_components['layer2_sample_idx'] = int(tpl.get('center_sample_idx', -1))
                score_components['layer2_spectral_source'] = 'layer2_template_center_mu'
                score_components['spectral_source'] = 'layer2_template_center_mu'

        if coverage_errors:
            preview = " | ".join(coverage_errors[:8])
            if len(coverage_errors) > 8:
                preview += f" | ... {len(coverage_errors) - 8} more"
            raise RuntimeError(f"Layer2 complete template coverage failed: {preview}")

        self.stats['selected_spectral_source'] = 'template'
        self.stats['spectral_contract'] = 'template_center_mu'
        
        print(f"  匹配: 精确={self.stats['exact_match']}, 近似={self.stats['approx_match']}, 未匹配={self.stats['missing']}")
        
        # NMR评估
        S_recon = self.reconstruct_spectrum(nodes, hwhm=hwhm)
        
        # 计算最优缩放因子
        eval_info = evaluate_spectrum_reconstruction(
            S_target,
            S_recon,
            ppm_axis=PPM_AXIS.to(device),
            fit_scale=True,
            nonnegative_alpha=True,
        )
        S_fit = eval_info['S_fit']
        alpha = float(eval_info.get('alpha', 1.0))
        r2 = float(eval_info.get('r2', 0.0))
        print(f"  R²={r2:.4f}, α={alpha:.4f}")
        
        # 保存结果
        if output_dir:
            self._save_results(nodes, S_target, S_fit, r2, alpha, output_dir, S_recon_raw=S_recon)
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                match_detail_rows = self._build_match_detail_rows(nodes)
                pd.DataFrame(match_detail_rows).to_csv(str(Path(output_dir) / 'layer2_template_match_detail.csv'), index=False)
            except Exception as e:
                print(f"  [Layer2] 保存模板匹配明细失败: {e}")
        
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
        """选择最佳模板键"""
        templates = lib.get('templates', {})
        
        # 精确匹配
        key_exact = (int(center_su), tuple(hop1_ms), tuple(hop2_ms))
        if key_exact in templates:
            tpl = templates.get(key_exact, {})
            return key_exact, 'exact', {
                'chosen_hop1': tuple(hop1_ms),
                'chosen_hop2': tuple(hop2_ms),
                'hop2_diff_nodes': 0,
                'hop2_match_tier': 0,
                'template_mu': float(tpl.get('center_mu')) if isinstance(tpl, dict) and tpl.get('center_mu') is not None else None,
                'mu_diff': 0.0,
            }
        
        # 近似匹配
        center_hop1_index = lib.get('center_hop1_index', {})
        # 1-hop is fixed by Layer1 and must remain exact. Approximation is
        # allowed only for the derived 2-hop multiset.
        same_h1 = list(center_hop1_index.get((int(center_su), tuple(hop1_ms)), []) or [])
        if not same_h1:
            return None, 'missing', {
                'chosen_hop1': (),
                'chosen_hop2': (),
                'hop2_diff_nodes': '',
                'template_mu': '',
                'mu_diff': '',
            }
        
        best = None
        best_meta: Dict[str, object] = {}
        for kt, h1_t, h2_t, sc, mu_tpl in same_h1:
            hop2_diff = multiset_diff_nodes(tuple(hop2_ms), tuple(h2_t))

            mu_diff = abs(float(mu_tpl) - float(mu_prior)) if mu_prior is not None and mu_tpl is not None else float('inf')
            mu_missing = int(not (mu_prior is not None and mu_tpl is not None))

            cand = (
                int(hop2_diff),
                int(mu_missing),
                float(mu_diff),
                -int(sc),
                kt,
            )
            if best is None or cand < best:
                best = cand
                best_meta = {
                    'chosen_hop1': tuple(h1_t),
                    'chosen_hop2': tuple(h2_t),
                    'hop2_diff_nodes': int(hop2_diff),
                    'hop2_match_tier': int(hop2_diff),
                    'template_mu': float(mu_tpl) if mu_tpl is not None else '',
                    'mu_diff': float(mu_diff) if np.isfinite(float(mu_diff)) else '',
                }

        if best is None:
            return None, 'missing', {
                'chosen_hop1': (),
                'chosen_hop2': (),
                'hop2_diff_nodes': '',
                'template_mu': '',
                'mu_diff': '',
            }
        return best[-1], 'approx_h2', best_meta
    
    # ========================================================================
    # 谱图重建
    # ========================================================================
    
    def reconstruct_spectrum(self, nodes: List[_NodeV3],
                             hwhm: float = 1.0) -> torch.Tensor:
        """从节点重建NMR谱图"""
        device = self.device
        ppm_axis = PPM_AXIS.to(device)
        
        mus = []
        pis = []
        
        for n in nodes:
            # 只处理含碳的SU
            if int(n.su_type) not in self._carbon_su_set:
                continue
            
            mu = float(getattr(n, 'mu', 0.0))
            pi = float(getattr(n, 'pi', 0.0))
            
            if pi <= 0.0 or mu == 0.0:
                continue
            
            mus.append(mu)
            if bool(getattr(self, 'unit_peak_intensity', False)):
                pis.append(1.0)
            else:
                pis.append(pi)
        
        try:
            s = float(getattr(self, 'intensity_scale', 1.0))
        except Exception as e:
            import logging
            logging.debug(f"Failed to get intensity_scale: {e}")
            s = 1.0
        
        return reconstruct_from_mu_pi(
            mus=mus,
            pis=pis,
            ppm_axis=ppm_axis,
            hwhm=float(hwhm),
            intensity_scale=float(s),
            device=device,
            unit_peak_intensity=bool(getattr(self, 'unit_peak_intensity', False)),
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
                      S_fit: torch.Tensor,
                      r2: float,
                      alpha: float,
                      output_dir: str,
                      S_recon_raw: Optional[torch.Tensor] = None):
        """保存Layer2结果"""
        try:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            ppm_axis = PPM_AXIS.cpu().numpy()
            
            # 保存谱图对比
            df = pd.DataFrame({
                'ppm': ppm_axis,
                'target': S_target.cpu().numpy(),
                'reconstructed_raw': S_recon_raw.cpu().numpy() if S_recon_raw is not None else S_fit.cpu().numpy(),
                'reconstructed': S_fit.cpu().numpy(),
                'difference': (S_target - S_fit).cpu().numpy(),
            })
            df.to_csv(str(out_dir / 'layer2_spectrum_comparison.csv'), index=False)

            try:
                visualize_spectrum_comparison(
                    S_target=S_target.detach().cpu(),
                    S_recon=S_fit.detach().cpu(),
                    ppm_axis=PPM_AXIS.detach().cpu(),
                    layer_name='Layer2',
                    save_dir=str(out_dir),
                )
                visualize_spectrum_comparison(
                    S_target=S_target.detach().cpu(),
                    S_recon=S_fit.detach().cpu(),
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
                    'r2_carbonyl': float(compute_segment_r2(S_target, S_fit, ppm_axis_t, 'carbonyl')),
                    'r2_aromatic': float(compute_segment_r2(S_target, S_fit, ppm_axis_t, 'aromatic')),
                    'r2_aliphatic': float(compute_segment_r2(S_target, S_fit, ppm_axis_t, 'aliphatic')),
                    'alpha': float(alpha),
                    'exact_match': int(self.stats.get('exact_match', 0)),
                    'approx_match': int(self.stats.get('approx_match', 0)),
                    'missing': int(self.stats.get('missing', 0)),
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
