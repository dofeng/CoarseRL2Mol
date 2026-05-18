import torch
import torch.nn.functional as F
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
                 lib_path: Optional[str] = None,
                 vae_model = None):
        """
        初始化 Layer2 估计器
        
        Args:
            device: 计算设备
            lib_path: 子图库路径
            vae_model: VAE模型（可选，用于z解码）
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
        
        # VAE模型
        self.vae = vae_model

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

    def _assign_diverse_template_samples(
        self,
        nodes: List[_NodeV3],
        templates: Dict,
        g_embed: torch.Tensor,
    ) -> None:
        """
        对共享同一 template_key 的节点做样本展开，避免所有节点坍缩到同一个 center_z/center_mu。
        """
        grouped: Dict[Tuple, List[_NodeV3]] = defaultdict(list)
        for node in nodes:
            tpl_key = getattr(node, 'template_key', None)
            if tpl_key is None:
                continue
            grouped[tpl_key].append(node)

        for tpl_key, group_nodes in grouped.items():
            tpl = templates.get(tpl_key, None)
            if not isinstance(tpl, dict):
                continue
            samples = tpl.get('samples', {}) or {}
            z_samples = samples.get('z', None)
            mu_samples = samples.get('mu', None)
            pi_samples = samples.get('pi', None)

            if z_samples is None or mu_samples is None or pi_samples is None:
                continue
            if not torch.is_tensor(mu_samples) or int(mu_samples.numel()) <= 0:
                continue

            sorted_idx = tpl.get('sorted_idx_by_mu', None)
            if torch.is_tensor(sorted_idx) and int(sorted_idx.numel()) > 0:
                sorted_ids = [int(x) for x in sorted_idx.detach().cpu().tolist()]
            else:
                sorted_ids = list(range(int(mu_samples.numel())))

            if not sorted_ids:
                continue

            def _node_order_key(node: _NodeV3):
                try:
                    prior = float((getattr(node, 'score_components', {}) or {}).get('layer2_mu_prior', 0.0) or 0.0)
                except Exception:
                    prior = 0.0
                if prior <= 0.0:
                    try:
                        prior = float(getattr(node, 'mu', 0.0) or 0.0)
                    except Exception:
                        prior = 0.0
                return (float(prior), int(getattr(node, 'global_id', 0)))

            ordered_nodes = sorted(group_nodes, key=_node_order_key)
            n_nodes = len(ordered_nodes)
            n_samples = len(sorted_ids)

            chosen_ids: List[int] = []
            base_count = int(min(n_nodes, n_samples))
            if base_count <= 0:
                continue

            if base_count == 1:
                desired_positions = [0.5 * float(n_samples - 1)]
            else:
                desired_positions = np.linspace(0, n_samples - 1, num=base_count)

            used_positions = set()
            spread_ids: List[int] = []
            for pos in desired_positions:
                ranked_positions = sorted(
                    range(n_samples),
                    key=lambda i: (abs(float(i) - float(pos)), int(i)),
                )
                chosen_pos = next((i for i in ranked_positions if i not in used_positions), ranked_positions[0])
                used_positions.add(int(chosen_pos))
                spread_ids.append(int(sorted_ids[int(chosen_pos)]))

            cycle_idx = 0
            while len(chosen_ids) < n_nodes:
                if len(spread_ids) == 1:
                    chosen_ids.append(int(spread_ids[0]))
                    continue
                rot = int(cycle_idx) % len(spread_ids)
                block = spread_ids[rot:] + spread_ids[:rot]
                if int(cycle_idx) % 2 == 1:
                    block = list(reversed(block))
                chosen_ids.extend(int(x) for x in block)
                cycle_idx += 1
            chosen_ids = chosen_ids[:n_nodes]

            pending_assignments = []
            for node, sample_idx in zip(ordered_nodes, chosen_ids):
                try:
                    z_cand = z_samples[int(sample_idx)].detach().clone().to(self.device)
                    mu_lib = float(mu_samples[int(sample_idx)].detach().cpu().item())
                    pi_lib = float(pi_samples[int(sample_idx)].detach().cpu().item())
                except Exception:
                    continue
                pending_assignments.append((node, int(sample_idx), z_cand, float(mu_lib), float(pi_lib)))

            decoded_batch = self._decode_mu_pi_batch(
                int(tpl_key[0]),
                [item[2] for item in pending_assignments],
                g_embed,
            )
            for idx, (node, sample_idx, z_cand, mu_lib, pi_lib) in enumerate(pending_assignments):
                node.z_vec = z_cand
                decoded = decoded_batch[idx] if decoded_batch is not None and idx < len(decoded_batch) else None
                if decoded is not None:
                    node.mu, node.pi = decoded
                else:
                    node.mu = float(mu_lib)
                    node.pi = float(max(1e-6, pi_lib))

                try:
                    if isinstance(getattr(node, 'score_components', None), dict):
                        node.score_components['layer2_sample_idx'] = int(sample_idx)
                except Exception:
                    pass

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
                hop1_ms = tuple(int(x) for x in tuple(kt[1]))
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
            print("  ⚠ 无法加载模板库")
            for n in nodes:
                self._reset_node_template_state(
                    n,
                    device=device,
                    template_key=None,
                    match_mode='missing',
                    mu_prior=None,
                )
            return nodes
        
        templates = lib.get('templates', {})
        g_embed = self._global_embed_from_elements(E_target)
        template_use_counter: Counter = Counter()
        
        for n in nodes:
            center_su = int(n.su_type)
            hop1_ms = multiset_from_counter(n.hop1_su)
            hop2_ms = multiset_from_counter(n.hop2_su)

            mu_prior_val = None
            try:
                mu_prior_tmp = self._get_layer1_mu_prior(center_su, hop1_ms, lib)
                if mu_prior_tmp is None:
                    mu_prior_tmp = float(getattr(n, 'mu', 0.0))
                if mu_prior_tmp is not None and float(mu_prior_tmp) > 1e-6:
                    mu_prior_val = float(mu_prior_tmp)
            except Exception as e:
                import logging
                logging.debug(f"Failed to get mu_prior for node {n.global_id}: {e}")
            
            # 查找最佳模板
            tpl_key, mode = self._select_template_key(
                center_su,
                hop1_ms,
                hop2_ms,
                mu_prior_val,
                lib,
                template_use_counter=template_use_counter,
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
            if tpl_key is not None:
                template_use_counter[tpl_key] += 1

            if mode == 'exact':
                self.stats['exact_match'] += 1
            elif mode.startswith('approx'):
                self.stats['approx_match'] += 1
            else:
                self.stats['missing'] += 1
            
            try:
                if isinstance(getattr(n, 'score_components', None), dict) and isinstance(tpl, dict):
                    n.score_components['layer2_template_mu'] = float(tpl.get('center_mu')) if tpl.get('center_mu') is not None else None
                    n.score_components['layer2_template_sample_count'] = int(tpl.get('sample_count', 0))
            except Exception as e:
                import logging
                logging.debug(f"Failed to save template info for node {n.global_id}: {e}")

            has_center_z = False
            if isinstance(tpl, dict) and tpl.get('center_z') is not None:
                try:
                    n.z_vec = tpl['center_z'].detach().clone().to(device)
                    has_center_z = True
                except Exception:
                    has_center_z = False

            has_sample_bank = False
            if isinstance(tpl, dict):
                samples = tpl.get('samples', {}) or {}
                z_samples = samples.get('z', None)
                mu_samples = samples.get('mu', None)
                pi_samples = samples.get('pi', None)
                has_sample_bank = (
                    torch.is_tensor(z_samples)
                    and torch.is_tensor(mu_samples)
                    and torch.is_tensor(pi_samples)
                    and int(mu_samples.numel()) > 0
                )

            decoded = (
                self._decode_mu_pi_from_z(center_su, n.z_vec, g_embed)
                if has_center_z and not has_sample_bank
                else None
            )
            if decoded is not None:
                n.mu, n.pi = decoded
            elif isinstance(tpl, dict):
                try:
                    n.mu = float(tpl.get('center_mu', 0.0) or 0.0)
                except Exception:
                    n.mu = 0.0
                try:
                    n.pi = float(tpl.get('center_pi', 0.0) or 0.0)
                except Exception:
                    n.pi = 0.0

        self._assign_diverse_template_samples(nodes, templates, g_embed)
        
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
            
            for nb_id in n.hop1_ids:
                try:
                    nb = node_map.get(int(nb_id), None)
                except:
                    continue
                if nb is None:
                    continue
                
                for nb2_id in nb.hop1_ids:
                    nb2_id_i = int(nb2_id)
                    if nb2_id_i == c_id:
                        continue
                    
                    try:
                        nb2 = node_map.get(nb2_id_i, None)
                        if nb2 is None:
                            continue
                        su2 = int(nb2.su_type)
                    except:
                        continue
                    
                    hop2[su2] += 1
            
            n.hop2_su = hop2
    
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
            center_degree_index = defaultdict(list)
            for kt, tpl in templates.items():
                if not (isinstance(kt, tuple) and len(kt) == 3 and isinstance(tpl, dict)):
                    continue
                try:
                    center_su = int(kt[0])
                    h1_t = tuple(int(x) for x in tuple(kt[1]))
                    h2_t = tuple(int(x) for x in tuple(kt[2]))
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
                center_degree_index[(int(center_su), int(len(h1_t)))].append(item)
            
            self._template_cache = {
                'templates': templates,
                'center_index': center_index,
                'center_hop1_index': dict(center_hop1_index),
                'center_degree_index': dict(center_degree_index),
            }
            return self._template_cache
        except Exception as e:
            print(f"  加载模板库失败: {e}")
            return None
    
    def _select_template_key(self, center_su: int,
                              hop1_ms: Tuple[int, ...],
                              hop2_ms: Tuple[int, ...],
                              mu_prior: Optional[float],
                              lib: Dict,
                              template_use_counter: Optional[Counter] = None) -> Tuple[Optional[Tuple], str]:
        """选择最佳模板键"""
        templates = lib.get('templates', {})
        
        # 精确匹配
        key_exact = (int(center_su), tuple(hop1_ms), tuple(hop2_ms))
        if key_exact in templates:
            return key_exact, 'exact'
        
        # 近似匹配
        center_hop1_index = lib.get('center_hop1_index', {})
        center_degree_index = lib.get('center_degree_index', {})
        target_deg = int(len(tuple(hop1_ms)))

        # 优先匹配hop1相同的
        same_h1 = list(center_hop1_index.get((int(center_su), tuple(hop1_ms)), []) or [])
        mode = 'approx_h2' if same_h1 else 'approx_h1h2'
        if same_h1:
            use_pool = same_h1
        else:
            use_pool = list(center_degree_index.get((int(center_su), int(target_deg)), []) or [])
        if not use_pool:
            # Compatibility fallback for libraries built without the accelerated indexes.
            center_index = lib.get('center_index', {})
            cand_keys = center_index.get(int(center_su), [])
            fallback_pool = []
            for k in cand_keys:
                kt = tuple(k) if not isinstance(k, tuple) else k
                if len(kt) != 3 or kt not in templates:
                    continue
                try:
                    c, h1, h2 = kt
                    h1_t = tuple(int(x) for x in tuple(h1))
                    h2_t = tuple(int(x) for x in tuple(h2))
                except Exception:
                    continue
                if int(c) != int(center_su):
                    continue
                tpl = templates.get(kt, {})
                sc = int(tpl.get('sample_count', 0)) if isinstance(tpl, dict) else 0
                mu_tpl = None
                try:
                    if isinstance(tpl, dict) and tpl.get('center_mu') is not None:
                        mu_tpl = float(tpl.get('center_mu'))
                except Exception:
                    mu_tpl = None
                fallback_pool.append((kt, h1_t, h2_t, int(sc), mu_tpl))
            same_h1 = [it for it in fallback_pool if tuple(it[1]) == tuple(hop1_ms)]
            if same_h1:
                use_pool = same_h1
                mode = 'approx_h2'
            else:
                use_pool = [it for it in fallback_pool if int(len(tuple(it[1]))) == int(target_deg)]
                mode = 'approx_h1h2'
        if not use_pool:
            return None, 'missing'
        
        best = None
        for kt, h1_t, h2_t, sc, mu_tpl in use_pool:
            hop1_diff = multiset_diff_nodes(tuple(hop1_ms), tuple(h1_t))
            hop2_diff = multiset_diff_nodes(tuple(hop2_ms), tuple(h2_t))

            mu_diff = abs(float(mu_tpl) - float(mu_prior)) if mu_prior is not None and mu_tpl is not None else float('inf')
            mu_missing = int(not (mu_prior is not None and mu_tpl is not None))

            use_count = int((template_use_counter or Counter()).get(kt, 0))

            if same_h1:
                cand = (
                    int(hop2_diff),
                    int(mu_missing),
                    float(mu_diff),
                    -int(sc),
                    int(use_count),
                    kt,
                )
            else:
                cand = (
                    int(hop1_diff),
                    int(hop2_diff),
                    int(mu_missing),
                    float(mu_diff),
                    -int(sc),
                    int(use_count),
                    kt,
                )
            if best is None or cand < best:
                best = cand

        return best[-1] if best else None, mode
    
    # ========================================================================
    # z 解码
    # ========================================================================
    
    def _global_embed_from_elements(self, E_target: torch.Tensor) -> torch.Tensor:
        """从元素组成生成全局嵌入"""
        device = self.device
        e = E_target.to(device).view(1, -1)
        
        if float(e.max().item()) > 1.1:
            s = e.sum(dim=1, keepdim=True).clamp(min=1.0)
            e = e / s
        
        if self.vae is not None:
            try:
                g = self.vae.global_mlp(e)
                return 0.02 * g
            except:
                pass
        
        return torch.zeros((1, 2), dtype=torch.float, device=device)
    
    def _decode_mu_pi_from_z(self, center_su: int,
                              z_vec: torch.Tensor,
                              g_embed: torch.Tensor) -> Optional[Tuple[float, float]]:
        """从z向量解码mu/pi"""
        if self.vae is None:
            return None
        
        device = self.device
        try:
            su_feat = F.one_hot(
                torch.tensor([int(center_su)], dtype=torch.long, device=device),
                num_classes=NUM_SU_TYPES,
            ).float()
            
            z = z_vec.to(device).view(1, -1)
            pred = self.vae.decoder(su_feat, z, g_embed)
            
            mu_pred = float(pred[:, 0].detach().item())
            pi_pred = float(F.softplus(pred[:, 1]).detach().item())
            
            return mu_pred, pi_pred
        except:
            return None

    def _decode_mu_pi_batch(self,
                            center_su: int,
                            z_vecs: List[torch.Tensor],
                            g_embed: torch.Tensor) -> Optional[List[Tuple[float, float]]]:
        """批量从z向量解码mu/pi，避免同一模板组逐节点调用decoder。"""
        if self.vae is None or not z_vecs:
            return None

        device = self.device
        try:
            z = torch.stack([zv.to(device).view(-1) for zv in z_vecs], dim=0)
            su_feat = F.one_hot(
                torch.full((int(z.shape[0]),), int(center_su), dtype=torch.long, device=device),
                num_classes=NUM_SU_TYPES,
            ).float()
            g = g_embed.to(device)
            if int(g.shape[0]) == 1 and int(z.shape[0]) > 1:
                g = g.expand(int(z.shape[0]), -1)
            pred = self.vae.decoder(su_feat, z, g)
            mu_vals = pred[:, 0].detach().cpu().tolist()
            pi_vals = F.softplus(pred[:, 1]).detach().cpu().tolist()
            return [(float(mu), float(pi)) for mu, pi in zip(mu_vals, pi_vals)]
        except Exception:
            return None
    
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
                        max(1, len(nodes))
                    ),
                    'n_nodes': int(len(nodes)),
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
