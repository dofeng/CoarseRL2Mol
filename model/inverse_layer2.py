import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from .coarse_graph import SU_DEFS, E_SU, NUM_SU_TYPES, PPM_AXIS
from .inverse_common import (
    _NodeV3, lorentzian_spectrum, compute_segment_r2,
    visualize_spectrum_comparison, evaluate_spectrum_reconstruction
)

# ============================================================================
# 辅助函数
# ============================================================================

def multiset_from_counter(cnt: Counter) -> Tuple[int, ...]:
    """将Counter转换为排序的multiset元组"""
    ms = []
    for k, v in cnt.items():
        ms.extend([int(k)] * int(v))
    ms.sort()
    return tuple(ms)

def multiset_l1_distance(ms1: Tuple[int, ...], ms2: Tuple[int, ...]) -> int:
    """计算两个multiset的L1距离"""
    c1 = Counter(ms1)
    c2 = Counter(ms2)
    all_keys = set(c1.keys()) | set(c2.keys())
    return sum(abs(c1.get(k, 0) - c2.get(k, 0)) for k in all_keys)

def multiset_overlap_size(ms1: Tuple[int, ...], ms2: Tuple[int, ...]) -> int:
    c1 = Counter(ms1)
    c2 = Counter(ms2)
    all_keys = set(c1.keys()) | set(c2.keys())
    return sum(min(int(c1.get(k, 0)), int(c2.get(k, 0))) for k in all_keys)

def multiset_diff_nodes(ms1: Tuple[int, ...], ms2: Tuple[int, ...]) -> int:
    ov = multiset_overlap_size(ms1, ms2)
    return int(max(len(ms1), len(ms2)) - ov)

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
        
        # 路径配置
        base_dir = Path(__file__).resolve().parents[1] / 'z_library'
        self.lib_path = lib_path or str(base_dir / 'subgraph_library.pt')
        
        # VAE模型
        self.vae = vae_model

        self.intensity_scale = 1.0
        
        # 缓存
        self._template_cache = None
        
        # 统计
        self.stats = {
            'exact_match': 0,
            'approx_match': 0,
            'missing': 0,
        }

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
            if n_nodes <= 1:
                chosen_ids = [sorted_ids[n_samples // 2]]
            else:
                for pos in np.linspace(0, n_samples - 1, num=n_nodes):
                    idx = sorted_ids[int(round(float(pos)))]
                    chosen_ids.append(int(idx))

            for node, sample_idx in zip(ordered_nodes, chosen_ids):
                try:
                    z_cand = z_samples[int(sample_idx)].detach().clone().to(self.device)
                    mu_lib = float(mu_samples[int(sample_idx)].detach().cpu().item())
                    pi_lib = float(pi_samples[int(sample_idx)].detach().cpu().item())
                except Exception:
                    continue

                node.z_vec = z_cand
                decoded = self._decode_mu_pi_from_z(int(node.su_type), z_cand, g_embed)
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
            return nodes
        
        templates = lib.get('templates', {})
        g_embed = self._global_embed_from_elements(E_target)
        
        for n in nodes:
            center_su = int(n.su_type)
            hop1_ms = multiset_from_counter(n.hop1_su)
            hop2_ms = multiset_from_counter(n.hop2_su)

            mu_prior_val = None
            try:
                mu_prior_tmp = float(getattr(n, 'mu', 0.0))
                if mu_prior_tmp > 1e-6:
                    mu_prior_val = mu_prior_tmp
            except Exception as e:
                import logging
                logging.debug(f"Failed to get mu_prior for node {n.global_id}: {e}")
            
            # 查找最佳模板
            tpl_key, mode = self._select_template_key(center_su, hop1_ms, hop2_ms, mu_prior_val, lib)
            n.template_key = tpl_key

            try:
                if isinstance(getattr(n, 'score_components', None), dict):
                    n.score_components['layer2_mu_prior'] = mu_prior_val
                    n.score_components['layer2_match_mode'] = mode
            except Exception as e:
                import logging
                logging.debug(f"Failed to update score_components for node {n.global_id}: {e}")
            
            if mode == 'exact':
                self.stats['exact_match'] += 1
            elif mode.startswith('approx'):
                self.stats['approx_match'] += 1
            else:
                self.stats['missing'] += 1
            
            # 获取模板数据
            tpl = templates.get(tpl_key, None) if tpl_key else None

            try:
                if isinstance(getattr(n, 'score_components', None), dict) and isinstance(tpl, dict):
                    n.score_components['layer2_template_mu'] = float(tpl.get('center_mu')) if tpl.get('center_mu') is not None else None
                    n.score_components['layer2_template_sample_count'] = int(tpl.get('sample_count', 0))
            except Exception as e:
                import logging
                logging.debug(f"Failed to save template info for node {n.global_id}: {e}")
            
            # 初始化z向量
            if isinstance(tpl, dict) and tpl.get('center_z') is not None:
                try:
                    n.z_vec = tpl['center_z'].detach().clone().to(device)
                except:
                    pass
            
            # 解码mu/pi
            decoded = self._decode_mu_pi_from_z(center_su, n.z_vec, g_embed)
            if decoded is not None:
                n.mu, n.pi = decoded
            elif isinstance(tpl, dict):
                n.mu = float(tpl.get('center_mu', n.mu))
                n.pi = float(tpl.get('center_pi', n.pi))

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
            
            self._template_cache = {
                'templates': templates,
                'center_index': center_index,
            }
            return self._template_cache
        except Exception as e:
            print(f"  加载模板库失败: {e}")
            return None
    
    def _select_template_key(self, center_su: int,
                              hop1_ms: Tuple[int, ...],
                              hop2_ms: Tuple[int, ...],
                              mu_prior: Optional[float],
                              lib: Dict) -> Tuple[Optional[Tuple], str]:
        """选择最佳模板键"""
        templates = lib.get('templates', {})
        center_index = lib.get('center_index', {})
        
        # 精确匹配
        key_exact = (int(center_su), tuple(hop1_ms), tuple(hop2_ms))
        if key_exact in templates:
            return key_exact, 'exact'
        
        # 近似匹配
        cand_keys = center_index.get(int(center_su), [])
        pool = []
        
        for k in cand_keys:
            kt = tuple(k) if not isinstance(k, tuple) else k
            if len(kt) != 3:
                continue
            
            try:
                c, h1, h2 = kt
                c = int(c)
                h1_t = tuple(h1) if not isinstance(h1, tuple) else h1
                h2_t = tuple(h2) if not isinstance(h2, tuple) else h2
            except:
                continue
            
            if int(c) != int(center_su):
                continue
            
            pool.append((kt, h1_t, h2_t))
        
        if not pool:
            return None, 'missing'
        
        # 优先匹配hop1相同的
        same_h1 = [it for it in pool if tuple(it[1]) == tuple(hop1_ms)]
        mode = 'approx_h2' if same_h1 else 'approx_h1h2'
        use_pool = same_h1 if same_h1 else pool

        best = None
        for kt, h1_t, h2_t in use_pool:
            hop1_diff = multiset_diff_nodes(tuple(hop1_ms), tuple(h1_t))
            hop2_diff = multiset_diff_nodes(tuple(hop2_ms), tuple(h2_t))

            tpl = templates.get(kt, {})
            sc = int(tpl.get('sample_count', 0)) if isinstance(tpl, dict) else 0
            mu_tpl = None
            try:
                if isinstance(tpl, dict) and tpl.get('center_mu') is not None:
                    mu_tpl = float(tpl.get('center_mu'))
            except Exception as e:
                import logging
                logging.debug(f"Invalid template mu value: {e}")
                mu_tpl = None

            mu_diff = float('inf')
            if mu_prior is not None and mu_tpl is not None:
                mu_diff = abs(float(mu_tpl) - float(mu_prior))
            elif mu_prior is None:
                mu_diff = 0.0

            if same_h1:
                cand = (int(hop2_diff), float(mu_diff), -int(sc), kt)
            else:
                cand = (int(hop1_diff), int(hop2_diff), float(mu_diff), -int(sc), kt)
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
            is_carbon = float(self.E_SU[n.su_type, 0].item()) > 0
            if not is_carbon:
                continue
            
            mu = float(getattr(n, 'mu', 0.0))
            pi = float(getattr(n, 'pi', 0.0))
            
            if pi <= 0.0 or mu == 0.0:
                continue
            
            mus.append(mu)
            pis.append(pi)
        
        if not mus:
            return torch.zeros_like(ppm_axis)
        
        mu_t = torch.tensor(mus, dtype=torch.float, device=device)
        pi_t = torch.tensor(pis, dtype=torch.float, device=device)

        try:
            s = float(getattr(self, 'intensity_scale', 1.0))
        except Exception as e:
            import logging
            logging.debug(f"Failed to get intensity_scale: {e}")
            s = 1.0
        if float(s) != 1.0:
            pi_t = pi_t * float(s)
        
        return lorentzian_spectrum(mu_t, pi_t, ppm_axis, hwhm=hwhm)
    
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


if __name__ == '__main__':
    """
    测试 Layer2 Estimator 独立运行
    """
    print("\n" + "=" * 80)
    print("Layer2 Estimator 独立测试")
    print("=" * 80)
    
    # 创建测试节点
    nodes = []
    for i in range(10):
        n = _NodeV3(global_id=i, su_type=13)  # 芳香CH
        nodes.append(n)
    
    # 模拟1-hop连接
    for i in range(0, 10, 2):
        nodes[i].hop1_su[13] = 2
        nodes[i].hop1_ids = [i+1, (i+2) % 10]
        nodes[i+1].hop1_su[13] = 2
        nodes[i+1].hop1_ids = [i, (i+3) % 10]
    
    # 创建测试数据
    S_target = torch.randn(800).abs()
    E_target = torch.tensor([10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"\n输入: {len(nodes)} 个节点")
    
    # 创建估计器
    estimator = Layer2Estimator(device=torch.device('cpu'))
    
    # 运行Layer2
    nodes = estimator.layer2_assign(nodes, S_target, E_target)
    
    # 打印结果
    print(f"\n输出: {len(nodes)} 个节点")
    for n in nodes[:3]:
        print(f"  Node {n.global_id}: mu={n.mu:.2f}, hop2={dict(n.hop2_su)}")
    
    print("\n✓ Layer2 Estimator 测试完成")
