import torch
import pandas as pd
import copy
import numpy as np
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Sequence

# 导入项目内部模块
from ..shared.coarse_graph import (
    E_SU, SU_PPM_RANGES, NUM_SU_TYPES, PPM_AXIS
)
from ..forward.s2n.s2n_model import S2NConfig, S2NModel
from ..forward.g2s.g2s_model import G2SConfig, NMR_VAE
# 导入拆分模块
from ..shared.inverse_common import (
    _NodeV3, lorentzian_spectrum,
    multiset_from_counter,
    SU_DEFS,
    visualize_su_distribution, evaluate_spectrum_reconstruction, resample_spectrum_to_ppm_axis,
    normalize_spectrum_to_carbon_count,
    SPECIAL_DEGREE_PRIORS,
    get_effective_hist_element_vector,
    get_effective_nodes_element_vector,
    get_node_degree_hint,
)
from .layer0.inverse_layer0 import Layer0Estimator
from .layer1.inverse_layer1 import Layer1Assigner, DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS
from .layer2.inverse_layer2 import Layer2Estimator
from .layer3.inverse_layer3 import Layer3Estimator
from .layer4.inverse_layer4 import Layer4Adjuster
from .topology import assemble_topology_units, classify_topology_units
from .topology.round_context import normalize_stage3_product
# ============================================================================
# 主推理类
# ============================================================================

class InversePipelineV3:
    """
    改进的逆向推理主类
    
    核心改进：
    1. 软约束系统：约束按优先级分级，允许临时违反低优先级约束
    2. 合成模板生成：库缺失时动态生成化学合理的模板
    3. 增量优化：保留历史信息，避免全盘推翻
    4. 多目标评估：不仅看R²，还考虑分段匹配、尖峰抑制、元素平衡
    """
    
    def __init__(self, 
                 s2n_model: S2NModel,
                 vae_model: NMR_VAE,
                 templates: Optional[Any],
                 device: str = 'cuda',
                 nmr_intensity_scale: float = 1.0,
                 unit_peak_intensity: bool = True):
        """
        初始化推理管道
        
        Args:
            s2n_model: Spectrum-to-Node模型
            vae_model: NMR VAE模型（用于解码mu/pi）
            templates: 模板库路径或包含'_lib_path'的配置对象
            device: 计算设备
        """
        self.s2n = s2n_model.to(device).eval()
        self.vae = vae_model.to(device).eval()
        self.device = device
        self.nmr_intensity_scale = float(nmr_intensity_scale)
        self.unit_peak_intensity = bool(unit_peak_intensity)
        self.default_template_lib_path: Optional[str] = None
        if isinstance(templates, (str, Path)):
            self.default_template_lib_path = str(Path(templates))
        elif isinstance(templates, dict):
            try:
                lib_path = templates.get('_lib_path')
                if isinstance(lib_path, (str, Path)):
                    self.default_template_lib_path = str(Path(lib_path))
            except Exception:
                pass
        
        # 加载常量到设备
        self.E_SU = E_SU.to(device)
        self.SU_PPM_RANGES = SU_PPM_RANGES  # 字典，不需要.to(device)
        
        # 初始化分层组件（委托模式）
        self.layer0_estimator = Layer0Estimator(
            s2n_model=self.s2n,
            E_SU_tensor=self.E_SU,
            device=device
        )
        self.layer1_assigner = Layer1Assigner(
            device=device,
            E_SU_tensor=self.E_SU,
            layer0_estimator=self.layer0_estimator,  # 传递Layer0实例
            intensity_scale=float(self.nmr_intensity_scale),
            unit_peak_intensity=bool(self.unit_peak_intensity),
        )
        self.layer2_estimator = Layer2Estimator(
            device=torch.device(device),
            vae_model=self.vae
        )

        self.layer3_estimator = Layer3Estimator(
            device=torch.device(device),
            vae_model=self.vae,
            lib_path=self.default_template_lib_path,
        )
        if self.default_template_lib_path:
            self.layer2_estimator.lib_path = str(self.default_template_lib_path)

        try:
            self.layer2_estimator.intensity_scale = float(self.nmr_intensity_scale)
            self.layer2_estimator.unit_peak_intensity = bool(self.unit_peak_intensity)
        except Exception:
            pass
        try:
            self.layer3_estimator.intensity_scale = float(self.nmr_intensity_scale)
            self.layer3_estimator.unit_peak_intensity = bool(self.unit_peak_intensity)
        except Exception:
            pass
        
        self.layer4_adjuster = Layer4Adjuster(
             device=torch.device(device),
             layer0_estimator=self.layer0_estimator,
         )

    @staticmethod
    def _merge_stage_config(defaults: Dict[str, Any],
                            override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(defaults)
        if isinstance(override, dict):
            cfg.update(override)
        return cfg

    def _set_fixed_partition_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        """Keep Layer4 and Layer0 on the same fixed-anchor metadata snapshot."""
        fixed_meta = copy.deepcopy(meta or {})
        try:
            self.layer4_adjuster.fixed_partition_meta = copy.deepcopy(fixed_meta)
        except Exception:
            pass
        try:
            self.layer0_estimator.fixed_partition_meta = copy.deepcopy(fixed_meta)
            self.layer0_estimator.special_degree_meta = copy.deepcopy(fixed_meta.get('special_degree_meta', {}) or {})
            self.layer0_estimator.special_partition_meta = copy.deepcopy(fixed_meta.get('special_partition_meta', {}) or {})
        except Exception:
            pass

    def _sync_fixed_partition_meta_from_nodes(self, nodes: List[_NodeV3]) -> None:
        """Derive special-degree and fixed-anchor mode metadata from a real graph."""
        fixed_meta = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
        special_degree_meta: Dict[int, Dict[int, int]] = {
            int(su): {int(deg): 0 for deg in dict(priors).keys()}
            for su, priors in SPECIAL_DEGREE_PRIORS.items()
        }
        anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]] = {
            19: {
                'ether_single': {1: 0, 2: 0, 3: 0},
                'ether_double': {2: 0, 3: 0},
                'thio_single': {1: 0, 2: 0, 3: 0},
                'thio_double': {2: 0, 3: 0},
            },
            20: {
                'single': {1: 0, 2: 0, 3: 0},
                'double': {2: 0, 3: 0},
            },
            21: {
                'single': {2: 0, 3: 0},
                'double': {2: 0, 3: 0},
            },
        }

        for node in list(nodes or []):
            su_i = int(getattr(node, 'su_type', -1))
            if int(su_i) not in SPECIAL_DEGREE_PRIORS:
                continue
            degree = get_node_degree_hint(node)
            if degree is None or int(degree) not in special_degree_meta[int(su_i)]:
                continue
            deg_i = int(degree)
            special_degree_meta[int(su_i)][int(deg_i)] += 1
            fixed_cnt = 0
            try:
                fixed_cnt = max(0, int(getattr(node, 'target_fixed_anchor_count', 0) or 0))
            except Exception:
                fixed_cnt = 0
            if int(su_i) == 19:
                part = str(getattr(node, 'special_anchor_partition', None) or '')
                if str(part) == 'thio':
                    if int(fixed_cnt) >= 2 and int(deg_i) in {2, 3}:
                        anchor_mode_meta[19]['thio_double'][int(deg_i)] += 1
                    else:
                        anchor_mode_meta[19]['thio_single'][int(deg_i)] += 1
                elif str(part) == 'ether':
                    if int(fixed_cnt) >= 2 and int(deg_i) in {2, 3}:
                        anchor_mode_meta[19]['ether_double'][int(deg_i)] += 1
                    else:
                        anchor_mode_meta[19]['ether_single'][int(deg_i)] += 1
            elif int(su_i) == 20:
                if int(fixed_cnt) >= 2 and int(deg_i) in {2, 3}:
                    anchor_mode_meta[20]['double'][int(deg_i)] += 1
                else:
                    anchor_mode_meta[20]['single'][int(deg_i)] += 1
            elif int(su_i) == 21:
                if int(fixed_cnt) >= 2 and int(deg_i) in {2, 3}:
                    anchor_mode_meta[21]['double'][int(deg_i)] += 1
                else:
                    anchor_mode_meta[21]['single'][int(deg_i)] += 1

        ether_by_degree = {
            int(deg): (
                int(anchor_mode_meta[19]['ether_single'].get(int(deg), 0)) +
                int(anchor_mode_meta[19]['ether_double'].get(int(deg), 0))
            )
            for deg in [1, 2, 3]
        }
        thio_by_degree = {
            int(deg): (
                int(anchor_mode_meta[19]['thio_single'].get(int(deg), 0)) +
                int(anchor_mode_meta[19]['thio_double'].get(int(deg), 0))
            )
            for deg in [1, 2, 3]
        }
        fixed_meta['special_degree_meta'] = special_degree_meta
        fixed_meta['special_anchor_mode_meta'] = anchor_mode_meta
        fixed_meta['special_partition_meta'] = dict(fixed_meta.get('special_partition_meta', {}) or {})
        fixed_meta['special_partition_meta'][19] = {
            'ether': dict(ether_by_degree),
            'thio': dict(thio_by_degree),
        }
        fixed_meta['o_base_19'] = int(sum(int(v) for v in ether_by_degree.values()))
        fixed_meta['s_reserved_19'] = int(sum(int(v) for v in thio_by_degree.values()))
        fixed_meta['n19_total'] = int(fixed_meta['o_base_19'] + fixed_meta['s_reserved_19'])
        self._set_fixed_partition_meta(fixed_meta)

    def _build_layer_runtime_configs(self,
                                     eval_hwhm: float,
                                     eval_allow_approx: bool,
                                     enable_hop1_adjust: bool,
                                     hop1_adjust_iterations: int,
                                     hop1_neg_threshold: float,
                                     hop1_pos_threshold: float,
                                     layer1_config: Optional[Dict[str, Any]] = None,
                                     layer2_config: Optional[Dict[str, Any]] = None,
                                     layer3_config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        layer1_defaults = {
            'profile': 'initial',
            'eval_hwhm': float(eval_hwhm),
            'eval_allow_approx': bool(eval_allow_approx),
            'enable_carbonyl_joint_adjust': True,
            'carbonyl_joint_iterations': 10,
            'carbonyl_joint_max_adjustments': 3,
            'carbonyl_joint_pos_threshold': 0.1,
            'carbonyl_joint_neg_threshold': 0.1,
            'enable_hop1_adjust': bool(enable_hop1_adjust),
            'hop1_adjust_iterations': int(hop1_adjust_iterations),
            'hop1_neg_threshold': float(hop1_neg_threshold),
            'hop1_pos_threshold': float(hop1_pos_threshold),
            'initial_enable_spectral_refine': True,
            'initial_hop1_adjust_iterations': int(hop1_adjust_iterations),
            'initial_hop1_adjust_max_candidate_trials': 240,
            'hop1_adjust_max_candidate_nodes_per_su': 24,
            'hop1_adjust_max_alternatives_per_node': 4,
            'hop1_adjust_region_order': ['aromatic', 'carbonyl', 'unsaturated', 'aliphatic'],
            'hop1_adjust_max_moves_per_region': 8,
            'hop1_adjust_max_full_evals': 64,
            'build_attempts': 3,
            'local_swap_max_attempts': 64,
            'build_variant': 0,
        }
        layer2_defaults = {
            'eval_hwhm': float(eval_hwhm),
            'layer2_max_hop2_diff_nodes': 2,
            'layer2_use_g2s_decoder': False,
            'layer2_decoder_max_mu_delta': 8.0,
        }
        layer3_defaults = {
            'max_iters': 40,
            'hwhm': float(eval_hwhm),
            'pos_search_window': 15.0,
            'neg_assign_window': 2.0,
            'top_k_samples': 10,
            'neg_top_k_peaks': 15,
            'neg_peak_min_sep_ppm': 1.0,
            'enable_approx_hop2_template_adjust': False,
            'approx_hop2_max_iters': 20,
            'approx_hop2_max_diff_nodes': 8,
            'approx_hop2_top_k_templates': 80,
            'layer3_use_g2s_decoder': False,
            'layer3_verbose': True,
        }

        def _merge_strict(defaults: Dict[str, Any], user_cfg: Optional[Dict[str, Any]], label: str) -> Dict[str, Any]:
            cfg = dict(user_cfg or {})
            unknown = sorted(str(k) for k in cfg.keys() if str(k) not in defaults)
            if unknown:
                raise ValueError(f"{label} contains unsupported keys: {unknown}")
            out = dict(defaults)
            out.update(cfg)
            return out

        return (
            _merge_strict(layer1_defaults, layer1_config, 'layer1_config'),
            _merge_strict(layer2_defaults, layer2_config, 'layer2_config'),
            _merge_strict(layer3_defaults, layer3_config, 'layer3_config'),
        )

    def get_default_runtime_configs(self,
                                    eval_hwhm: float = 1.0,
                                    eval_allow_approx: bool = True,
                                    enable_hop1_adjust: bool = True,
                                    hop1_adjust_iterations: int = 10,
                                    hop1_neg_threshold: float = -1.0,
                                    hop1_pos_threshold: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """返回 Layer1/2/3 当前可调参数及默认值。"""
        layer1_cfg, layer2_cfg, layer3_cfg = self._build_layer_runtime_configs(
            eval_hwhm=eval_hwhm,
            eval_allow_approx=eval_allow_approx,
            enable_hop1_adjust=enable_hop1_adjust,
            hop1_adjust_iterations=hop1_adjust_iterations,
            hop1_neg_threshold=hop1_neg_threshold,
            hop1_pos_threshold=hop1_pos_threshold,
        )
        return {
            'layer1_config': layer1_cfg,
            'layer2_config': layer2_cfg,
            'layer3_config': layer3_cfg,
        }

    @staticmethod
    def _format_nonzero_histogram(H: torch.Tensor) -> str:
        try:
            vals = H.detach().cpu().long().tolist()
        except Exception:
            vals = [int(x) for x in list(H)]
        parts = [f"SU{idx}:{int(cnt)}" for idx, cnt in enumerate(vals) if int(cnt) > 0]
        return ", ".join(parts) if parts else "(empty)"

    @staticmethod
    def _format_special_degree_distribution(nodes: Optional[List[_NodeV3]]) -> Optional[str]:
        if not nodes:
            return None
        parts: List[str] = []
        for su_type in (19, 20, 21):
            deg_keys = sorted(int(k) for k in dict(SPECIAL_DEGREE_PRIORS.get(int(su_type), {})).keys())
            counts = {int(deg): 0 for deg in deg_keys}
            unknown = 0
            total = 0
            for node in nodes:
                if int(getattr(node, 'su_type', -1)) != int(su_type):
                    continue
                total += 1
                deg = get_node_degree_hint(node)
                if deg is None or int(deg) not in counts:
                    unknown += 1
                    continue
                counts[int(deg)] += 1
            if total <= 0:
                continue
            deg_txt = ", ".join(f"d{int(deg)}:{int(counts[deg])}" for deg in deg_keys)
            if int(unknown) > 0:
                deg_txt = f"{deg_txt}, d?:{int(unknown)}"
            parts.append(f"SU{int(su_type)}[{deg_txt}]")
        return "; ".join(parts) if parts else None

    def _print_histogram_and_elements(self,
                                      H: torch.Tensor,
                                      E_target: Optional[torch.Tensor],
                                      label: str,
                                      nodes: Optional[List[_NodeV3]] = None) -> None:
        try:
            H_cpu = H.detach().cpu().long()
            if nodes:
                E_pred = get_effective_nodes_element_vector(
                    nodes,
                    self.E_SU.detach().cpu(),
                    device=torch.device('cpu'),
                )
            else:
                try:
                    special_degree_meta = self.layer4_adjuster._get_special_degree_meta(H_cpu)
                except Exception:
                    special_degree_meta = None
                E_pred = get_effective_hist_element_vector(
                    H_cpu,
                    special_degree_meta=special_degree_meta,
                    E_SU_tensor=self.E_SU.detach().cpu(),
                    device=torch.device('cpu'),
                )
            if E_target is not None:
                E_t = E_target.detach().cpu().flatten().float()
                print(f"[{label}] H(nonzero) = {self._format_nonzero_histogram(H_cpu)}")
                degree_dist = self._format_special_degree_distribution(nodes)
                if degree_dist:
                    print(f"[{label}] SU19/20/21 degree_dist = {degree_dist}")
                print(
                    f"[{label}] Elements pred = {[float(x) for x in E_pred.tolist()]} "
                    f"target = {[float(x) for x in E_t.tolist()]}"
                )
            else:
                print(f"[{label}] H(nonzero) = {self._format_nonzero_histogram(H_cpu)}")
                degree_dist = self._format_special_degree_distribution(nodes)
                if degree_dist:
                    print(f"[{label}] SU19/20/21 degree_dist = {degree_dist}")
                print(f"[{label}] Elements pred = {[float(x) for x in E_pred.tolist()]}")
        except Exception as e:
            print(f"[{label}] Summary unavailable: {e}")

    def _print_effective_resource_summary(self,
                                          H: torch.Tensor,
                                          label: str) -> None:
        try:
            H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
            out = {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0}
            for idx in range(int(H_cpu.numel())):
                cnt = int(H_cpu[idx].item())
                if cnt <= 0:
                    continue
                if int(idx) in {5, 6, 7, 8, 9, 11}:
                    out['11'] += int(cnt)
                elif int(idx) in {1, 4, 16, 18, 22, 28, 32}:
                    out['22'] += int(cnt)
                elif int(idx) in {14, 24}:
                    out['24'] += int(cnt)
                elif int(idx) == 25:
                    out['25'] += int(cnt)
                elif int(idx) in {0, 2, 3, 15, 17, 19, 20, 21, 23, 27, 29, 31}:
                    out['23'] += int(cnt)
            print(
                f"[{label}] Effective resources = "
                f"11:{int(out['11'])} 22:{int(out['22'])} 23:{int(out['23'])} "
                f"24:{int(out['24'])} 25:{int(out['25'])}"
            )
        except Exception as e:
            print(f"[{label}] Effective resource summary unavailable: {e}")

    def _hist_element_counts_from_H(self, H: torch.Tensor) -> Dict[str, float]:
        labels = ('C', 'H', 'O', 'N', 'S', 'X')
        out = {str(label): 0.0 for label in labels}
        try:
            H_cpu = H.detach().cpu().flatten().long()
            vec = get_effective_hist_element_vector(
                H_cpu,
                E_SU_tensor=self.E_SU.detach().cpu(),
                device=torch.device('cpu'),
            ).detach().cpu().flatten().float()
            out = {
                str(label): float(vec[idx].item())
                for idx, label in enumerate(labels)
                if idx < int(vec.numel())
            }
        except Exception:
            pass
        return out

    def _element_error_summary(self,
                               H: Optional[torch.Tensor],
                               E_target: Optional[torch.Tensor],
                               nodes: Optional[List[_NodeV3]] = None) -> Dict[str, Any]:
        labels = ('C', 'H', 'O', 'N', 'S', 'X')
        pred = {str(label): 0.0 for label in labels}
        try:
            if nodes:
                vec = get_effective_nodes_element_vector(
                    list(nodes or []),
                    self.E_SU.detach().cpu(),
                    device=torch.device('cpu'),
                )
                pred = {
                    str(label): float(vec.detach().cpu().flatten()[idx].item())
                    for idx, label in enumerate(labels)
                    if idx < int(vec.numel())
                }
            elif H is not None:
                H_cpu = H.detach().cpu().long()
                try:
                    special_degree_meta = self.layer4_adjuster._get_special_degree_meta(H_cpu)
                except Exception:
                    special_degree_meta = None
                vec = get_effective_hist_element_vector(
                    H_cpu,
                    special_degree_meta=special_degree_meta,
                    E_SU_tensor=self.E_SU.detach().cpu(),
                    device=torch.device('cpu'),
                )
                pred = {
                    str(label): float(vec.detach().cpu().flatten()[idx].item())
                    for idx, label in enumerate(labels)
                    if idx < int(vec.numel())
                }
        except Exception:
            pred = self._hist_element_counts_from_H(H) if H is not None else pred
        target: Dict[str, float] = {}
        if E_target is not None:
            try:
                t = E_target.detach().cpu().flatten().float()
                target = {
                    str(label): float(t[idx].item())
                    for idx, label in enumerate(labels)
                    if idx < int(t.numel())
                }
            except Exception:
                target = {}
        errors = {
            str(label): float(pred.get(str(label), 0.0) - target.get(str(label), 0.0))
            for label in labels
            if str(label) in pred and str(label) in target
        }
        abs_total = float(sum(abs(float(v)) for v in errors.values()))
        return {
            'pred': pred,
            'target': target,
            'error': errors,
            'abs_total_error': abs_total,
            'h_error': float(errors.get('H', 0.0)),
        }

    def _run_layer12_cycle(self,
                           H_seed: torch.Tensor,
                           S_target: torch.Tensor,
                           E_target: torch.Tensor,
                           eval_lib_path: Optional[str],
                           layer1_runtime_cfg: Dict[str, Any],
                           layer2_runtime_cfg: Dict[str, Any],
                           identity_seed_nodes: Optional[List[_NodeV3]] = None,
                           eval_nmr: bool = False,
                           eval_output_dir: Optional[str] = None,
                           layer2_output_dir: Optional[str] = None) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, object], float]:
        try:
            current_fixed_meta = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
            if current_fixed_meta:
                self._set_fixed_partition_meta(current_fixed_meta)
        except Exception:
            current_fixed_meta = {}

        def _first_layer1_error(errors: List[str]) -> str:
            return str(errors[0]) if errors else 'unknown'

        def _run_layer1_once(cfg_local: Dict[str, Any],
                             attempt_name: str) -> Tuple[Optional[List[_NodeV3]], bool, List[str]]:
            try:
                if current_fixed_meta:
                    self._set_fixed_partition_meta(current_fixed_meta)
            except Exception:
                pass
            try:
                nodes_local = self.layer1_assign(
                    H_init=H_seed.detach().clone(),
                    S_target=S_target,
                    E_target=E_target,
                    eval_nmr=bool(eval_nmr),
                    eval_output_dir=eval_output_dir,
                    eval_lib_path=eval_lib_path,
                    **cfg_local,
                )
            except Exception as e:
                err = f"layer1_exception: {e}"
                print(f"  [Layer1 {attempt_name}失败] {err}")
                return None, False, [err]
            layer1_ok_local, layer1_errors_local = self.layer1_assigner.validate_graph_consistency(
                nodes=nodes_local,
                H=H_seed.detach().clone(),
                E_target=E_target,
                verbose=False,
            )
            layer1_accept_ok_local, layer1_accept_meta_local = self.layer1_assigner.assess_consistency_acceptance(
                list(layer1_errors_local or []),
                max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
            )
            if not bool(layer1_accept_ok_local):
                local_swap_diag = dict(getattr(self.layer1_assigner, '_last_local_swap_diagnostics', {}) or {})
                if local_swap_diag and not bool(local_swap_diag.get('ok', True)):
                    layer1_errors_local = list(layer1_errors_local or []) + [
                        f"local_swap_exhausted: {local_swap_diag}"
                    ]
            if not bool(layer1_accept_ok_local):
                print(
                    f"  [Layer1 {attempt_name}校验失败] "
                    f"{_first_layer1_error(list(layer1_errors_local or []))}"
                )
            elif (not bool(layer1_ok_local)) and int(layer1_accept_meta_local.get('n_soft', 0)) > 0:
                print(
                    f"  [Layer1 {attempt_name}软缺口容忍] "
                    f"soft={int(layer1_accept_meta_local.get('n_soft', 0))}/"
                    f"{int(layer1_accept_meta_local.get('max_soft_errors', 0))}"
                )
            return nodes_local, bool(layer1_accept_ok_local), list(layer1_errors_local or [])

        layer1_cfg_base = dict(layer1_runtime_cfg)
        profile = str(layer1_cfg_base.get('profile', 'rebuild') or 'rebuild').strip().lower()
        if profile not in {'initial', 'rebuild'}:
            raise ValueError(f"unsupported Layer1 profile: {profile}")
        build_attempts = max(1, int(layer1_cfg_base.pop('build_attempts', 3)))
        base_variant = int(layer1_cfg_base.get('build_variant', 0) or 0)
        nodes = None
        layer1_ok = False
        layer1_errors: List[str] = []
        attempt_errors: List[str] = []
        for attempt_idx in range(build_attempts):
            layer1_cfg_attempt = dict(layer1_cfg_base)
            layer1_cfg_attempt['build_variant'] = int(base_variant + 1009 * attempt_idx)
            nodes, layer1_ok, layer1_errors = _run_layer1_once(
                cfg_local=layer1_cfg_attempt,
                attempt_name=f'cold-start variant={int(layer1_cfg_attempt["build_variant"])}',
            )
            if bool(layer1_ok):
                break
            attempt_errors.append(_first_layer1_error(list(layer1_errors or [])))
        if not bool(layer1_ok):
            raise RuntimeError("layer1_graph_invalid: " + " | ".join(attempt_errors or [_first_layer1_error(list(layer1_errors or []))]))
        layer2_call_cfg = dict(layer2_runtime_cfg)
        layer1_baseline_diff: Dict[str, object] = {}
        if eval_lib_path and profile == 'initial':
            _, layer1_baseline_diff = self._materialize_layer1_library_baseline(
                nodes=copy.deepcopy(nodes),
                S_target=S_target,
                lib_path=eval_lib_path,
                hwhm=float(layer2_runtime_cfg.get('eval_hwhm', 1.0)),
                allow_approx=bool(layer1_runtime_cfg.get('eval_allow_approx', True)),
            )
        nodes = self.layer2_assign(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            lib_path=eval_lib_path,
            output_dir=layer2_output_dir,
            **layer2_call_cfg,
        )
        H_nodes = self._histogram_from_nodes(nodes).detach().clone()
        diff_info = self._compute_difference_spectrum_from_nodes_mu(
            nodes=nodes,
            S_target=S_target,
            hwhm=float(layer2_runtime_cfg.get('eval_hwhm', 1.0)),
        )
        r2 = float(diff_info.get('r2', 0.0))
        if layer1_baseline_diff:
            diff_info['layer1_library_baseline_r2'] = float(layer1_baseline_diff.get('r2', float('nan')))
            diff_info['layer1_library_baseline_alpha'] = float(layer1_baseline_diff.get('alpha', float('nan')))
        return nodes, H_nodes, diff_info, r2

    def _print_final_resource_summary(self,
                                      nodes: List[_NodeV3],
                                      S_target: torch.Tensor,
                                      E_target: torch.Tensor) -> None:
        try:
            alloc_diag = self.layer4_adjuster._evaluate_full_allocation_balance(
                nodes=nodes,
                S_target=S_target,
                E_target=E_target,
            )
            print(
                "[Final Allocation] "
                f"ok={bool(alloc_diag.get('ok', False))} "
                f"reason={alloc_diag.get('reason', 'unknown')}"
            )
        except Exception as e:
            print(f"[Final Allocation] Failed to summarize allocation: {e}")

    @staticmethod
    def _topology_pending_summary(assembly_dict: Dict[str, Any]) -> Dict[str, Any]:
        pending = list(dict(assembly_dict or {}).get('pending_units', []) or [])
        by_kind: Dict[str, int] = {}
        for item in pending:
            kind = str(dict(item or {}).get('pending_kind', ''))
            by_kind[kind] = int(by_kind.get(kind, 0)) + 1
        return {
            'count': int(len(pending)),
            'by_kind': dict(by_kind),
            'examples': pending[:10],
        }

    @staticmethod
    def _topology_improvement_metrics(assembly_dict: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics = dict(dict(assembly_dict or {}).get('diagnostics', {}) or {})
        flex_counts = dict(dict(assembly_dict or {}).get('flex_counts', {}) or {})
        flex_bounds = dict(dict(assembly_dict or {}).get('flex_bounds', {}) or {})
        pending = InversePipelineV3._topology_pending_summary(assembly_dict)
        deficit = {
            str(k): int(v)
            for k, v in dict(diagnostics.get('total_shortage', {}) or {}).items()
            if int(v) > 0
        }
        flex_min = int(flex_bounds.get('min', 0) or 0)
        flex_final = int(flex_counts.get('final', 0) or 0)
        return {
            'ok': bool(dict(assembly_dict or {}).get('ok', False)),
            'reason': str(dict(assembly_dict or {}).get('reason', '')),
            'pending_count': int(pending.get('count', 0)),
            'flex_final': int(flex_final),
            'flex_min': int(flex_min),
            'flex_deficit': max(0, int(flex_min - flex_final)),
            'resource_deficit_total': int(sum(int(v) for v in deficit.values())),
            'resource_deficit': deficit,
        }

    @staticmethod
    def _topology_improved(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[bool, str]:
        b = InversePipelineV3._topology_improvement_metrics(before)
        a = InversePipelineV3._topology_improvement_metrics(after)
        if bool(a.get('ok', False)):
            return True, 'assembled'
        non_worsening = [
            int(a.get('pending_count', 0)) <= int(b.get('pending_count', 0)),
            int(a.get('flex_deficit', 0)) <= int(b.get('flex_deficit', 0)),
            int(a.get('resource_deficit_total', 0)) <= int(b.get('resource_deficit_total', 0)),
        ]
        strict = [
            ('pending_count', int(a.get('pending_count', 0)) < int(b.get('pending_count', 0))),
            ('flex_deficit', int(a.get('flex_deficit', 0)) < int(b.get('flex_deficit', 0))),
            ('resource_deficit_total', int(a.get('resource_deficit_total', 0)) < int(b.get('resource_deficit_total', 0))),
        ]
        improved = [name for name, ok in strict if bool(ok)]
        if all(non_worsening) and improved:
            return True, 'improved:' + ','.join(improved)
        return False, f"no_topology_improvement before={b} after={a}"

    def _save_stage3_topology_debug(self,
                                    output_dir: Optional[str],
                                    outer_idx: int,
                                    cycle_idx: int,
                                    classification_dict: Dict[str, Any],
                                    assembly_dict: Dict[str, Any]) -> None:
        if not output_dir:
            return
        try:
            out = Path(output_dir) / f'outer_cycle_{int(outer_idx) + 1}' / 'stage3_topology_debug'
            out.mkdir(parents=True, exist_ok=True)
            prefix = out / f'stage3_topology_cycle_{int(cycle_idx) + 1}'
            pending = list(dict(assembly_dict or {}).get('pending_units', []) or [])
            products = list(dict(assembly_dict or {}).get('products', []) or [])
            summary = {
                'classification_summary': dict(dict(classification_dict or {}).get('summary', {}) or {}),
                'assembly_summary': self._topology_improvement_metrics(assembly_dict),
                'pending_summary': self._topology_pending_summary(assembly_dict),
                'resource_ledger': dict(dict(assembly_dict or {}).get('resource_ledger', {}) or {}),
                'diagnostics': dict(dict(assembly_dict or {}).get('diagnostics', {}) or {}),
            }
            with open(str(prefix) + '.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            if pending:
                pd.DataFrame(pending).to_csv(str(prefix) + '_pending.csv', index=False)
            if products:
                pd.DataFrame(products).head(200).to_csv(str(prefix) + '_products_head.csv', index=False)
        except Exception as e:
            print(f"[Stage3:topology] 保存诊断失败: {e}")

    @staticmethod
    def _symbolic_product_node_ids(product: Dict[str, Any]) -> List[int]:
        row = dict(product or {})
        return sorted({
            int(value)
            for value in list(row.get('fixed_source_node_ids', []) or [])
            + [
                int(value)
                for values in dict(row.get('claimed_resource_ids_by_su', {}) or {}).values()
                for value in list(values or [])
            ]
        })

    @staticmethod
    def _stage3_node_state_signature(nodes: Sequence[_NodeV3]) -> List[Dict[str, Any]]:
        return [
            {
                'node_id': int(getattr(node, 'global_id', -1)),
                'su_type': int(getattr(node, 'su_type', -1)),
                'hop1_ids': sorted(int(value) for value in list(getattr(node, 'hop1_ids', []) or [])),
                'fixed_hop1_ids': sorted(
                    int(value) for value in list(getattr(node, 'fixed_hop1_ids', []) or [])
                ),
                'target_hop1_degree': getattr(node, 'target_hop1_degree', None),
                'target_fixed_anchor_count': getattr(node, 'target_fixed_anchor_count', None),
                'special_anchor_mode': str(getattr(node, 'special_anchor_mode', '') or ''),
                'special_anchor_partition': str(
                    getattr(node, 'special_anchor_partition', '') or ''
                ),
            }
            for node in sorted(
                list(nodes or []),
                key=lambda item: int(getattr(item, 'global_id', -1)),
            )
        ]

    @staticmethod
    def _stage3_completion_ownership(ledger: Dict[str, Any]) -> Dict[str, Any]:
        ownership_states = [
            'fixed_topology_core',
            'pending_topology',
            'completed_topology',
            'claimed_closure_resource',
            'remaining_closure_resource',
            'remaining_aromatic_donor',
            'support_context',
            'unclassified_or_conflict',
        ]
        states = {state: [] for state in ownership_states}
        for raw in dict(ledger.get('ownership_by_node_id', {}) or {}).values():
            owner = dict(raw or {})
            state = str(owner.get('state', 'unclassified_or_conflict') or 'unclassified_or_conflict')
            states.setdefault(state, []).append(int(owner.get('node_id', -1)))
        states = {key: sorted(set(values)) for key, values in states.items()}
        available = {
            str(key): sorted({int(value) for value in list(values or [])})
            for key, values in dict(ledger.get('available_ids_by_su', {}) or {}).items()
        }
        return {
            'states': states,
            'closure_pool_ids_by_su': {
                str(su_type): list(available.get(str(su_type), []))
                for su_type in (11, 22, 23, 24)
            },
            'aromatic_donor_ids_by_su': {
                str(su_type): list(available.get(str(su_type), []))
                for su_type in (10, 12, 13)
            },
            'invariant_errors': [
                dict(item or {}) for item in list(ledger.get('conflicts', []) or [])
            ],
        }

    @staticmethod
    def _stage3_completion_resource_ledger(
        classification: Dict[str, Any],
        products: Sequence[Dict[str, Any]],
        ownership: Dict[str, Any],
    ) -> Dict[str, Any]:
        pool = dict(classification.get('resource_pool', {}) or {})
        initial: Dict[str, set[int]] = {}
        for raw_pool in (
            dict(pool.get('initial_ids_by_su', {}) or {}),
            dict(pool.get('su11_donor_ids_by_su', {}) or {}),
        ):
            for key, values in raw_pool.items():
                initial.setdefault(str(key), set()).update(
                    int(value) for value in list(values or [])
                )
        available: Dict[str, set[int]] = {}
        for raw_pool in (
            dict(ownership.get('closure_pool_ids_by_su', {}) or {}),
            dict(ownership.get('aromatic_donor_ids_by_su', {}) or {}),
        ):
            for key, values in raw_pool.items():
                available.setdefault(str(key), set()).update(
                    int(value) for value in list(values or [])
                )
        residual_claimed: Dict[str, set[int]] = {}
        for product in list(products or []):
            row = dict(product or {})
            product_type = str(row.get('product_type', row.get('target_role', '')) or '')
            for key, values in dict(row.get('claimed_resource_ids_by_su', {}) or {}).items():
                claimed_ids = {int(value) for value in list(values or [])}
                initial.setdefault(str(key), set()).update(claimed_ids)
                if product_type.startswith('residual_'):
                    residual_claimed.setdefault(str(key), set()).update(claimed_ids)
        for key, values in available.items():
            initial.setdefault(str(key), set()).update(values)
        keys = sorted(
            set(initial) | set(available),
            key=lambda value: (
                (0, int(value))
                if str(value).lstrip('-').isdigit()
                else (1, str(value))
            ),
        )
        consumed = {
            str(key): sorted(initial.get(str(key), set()) - available.get(str(key), set()))
            for key in keys
        }
        residual = {
            str(key): sorted(set(consumed.get(str(key), [])) & residual_claimed.get(str(key), set()))
            for key in keys
        }
        topology = {
            str(key): sorted(set(consumed.get(str(key), [])) - set(residual.get(str(key), [])))
            for key in keys
        }
        return {
            'initial_ids_by_su': {str(key): sorted(initial.get(str(key), set())) for key in keys},
            'consumed_ids_by_su': consumed,
            'remaining_ids_by_su': {
                str(key): sorted(available.get(str(key), set())) for key in keys
            },
            'initial_counts': {str(key): len(initial.get(str(key), set())) for key in keys},
            'consumed_counts': {str(key): len(consumed.get(str(key), [])) for key in keys},
            'topology_consumed_counts': {
                str(key): len(topology.get(str(key), [])) for key in keys
            },
            'residual_consumed_counts': {
                str(key): len(residual.get(str(key), [])) for key in keys
            },
            'remaining_counts': {str(key): len(available.get(str(key), set())) for key in keys},
            'shortage_counts': {},
            'snapshot_stage': 'stage3_complete_generation_ledger',
        }

    @staticmethod
    def _public_topology_rows_from_assembly(assembly_result: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if hasattr(assembly_result, "to_public_dict"):
            public = assembly_result.to_public_dict()
        else:
            public = dict(assembly_result or {})
        products = [dict(item or {}) for item in list(public.get('products', []) or [])]
        pending = [dict(item or {}) for item in list(public.get('pending_units', []) or [])]
        ledger = dict(public.get('resource_ledger', {}) or {})
        summary = {
            'ok': bool(public.get('ok', False)),
            'reason': str(public.get('reason', 'assembly_pending')),
            'topology_plan_feasible': bool(public.get('feasible', False)),
            'closed_topology_count': int(len(products)),
            'final_topology_count': int(len(products)),
            'pending_closure_total_count': int(len(pending)),
            'unclosed_candidates': pending,
            'resource_ledger': ledger,
            'diagnostics': {
                'reason': str(public.get('reason', 'assembly_pending')),
                'solver_reason': str(public.get('reason', 'assembly_pending')),
                'resource_ledger': ledger,
            },
        }
        return products, pending, summary

    def _final_topology_rows(
        self,
        nodes: List[_NodeV3],
        S_target: Optional[torch.Tensor] = None,
        E_target: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        classification = classify_topology_units(
            list(nodes or []),
            fixed_partition_meta=getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {},
        )
        assembly = assemble_topology_units(
            nodes=list(nodes or []),
            classification=classification,
            fixed_partition_meta=getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {},
            S_target=S_target,
            E_target=E_target,
        )
        classification_dict = classification.to_dict()
        assembly_dict = assembly.to_dict()
        public_rows, public_pending, public_summary = self._public_topology_rows_from_assembly(assembly)
        completion_diagnostics = dict(
            getattr(self, '_final_stage3_completion_diagnostics', {}) or {}
        )
        terminal_diagnostics = dict(
            getattr(self, '_final_stage3_terminal_diagnostics', {}) or {}
        )
        residual_infeasibility_proof = dict(
            completion_diagnostics.get('residual_infeasibility_proof', {}) or {}
        )
        current_node_state_signature = self._stage3_node_state_signature(nodes)
        completion_matches = bool(
            completion_diagnostics.get('stage3_complete', False)
            and list(completion_diagnostics.get('node_state_signature', []) or [])
            == current_node_state_signature
        )
        terminal_matches = bool(
            terminal_diagnostics.get('stop_reason') == 'phase_resource_infeasible'
            and list(terminal_diagnostics.get('node_state_signature', []) or [])
            == current_node_state_signature
        )
        infeasibility_proof = (
            dict(terminal_diagnostics.get('infeasibility_proof', {}) or {})
            if terminal_matches else {}
        )
        completion_ownership: Dict[str, Any] = {}
        completion_coverage: Dict[str, Any] = {}
        if completion_matches:
            public_rows = [
                dict(item or {})
                for item in list(completion_diagnostics.get('products', []) or [])
            ]
            public_pending = []
            completion_ownership = dict(
                completion_diagnostics.get('ownership_ledger', {}) or {}
            )
            completion_resource_ledger = self._stage3_completion_resource_ledger(
                classification_dict,
                public_rows,
                completion_ownership,
            )
            public_summary.update({
                'ok': True,
                'reason': 'stage3_complete',
                'topology_plan_feasible': True,
                'closed_topology_count': int(len(public_rows)),
                'final_topology_count': int(len(public_rows)),
                'pending_closure_total_count': 0,
                'unclosed_candidates': [],
                'resource_ledger': completion_resource_ledger,
            })
            states = dict(completion_ownership.get('states', {}) or {})
            all_ids = {int(getattr(node, 'global_id', -1)) for node in list(nodes or [])}
            completed_ids_for_coverage = {
                int(value)
                for state in ('completed_topology', 'claimed_closure_resource')
                for value in list(states.get(state, []) or [])
            }
            pending_ids_for_coverage = {
                int(value) for value in list(states.get('pending_topology', []) or [])
            }
            resource_ids_for_coverage = {
                int(value)
                for state in ('remaining_closure_resource', 'remaining_aromatic_donor')
                for value in list(states.get(state, []) or [])
            }
            conflict_ids_for_coverage = {
                int(value)
                for value in list(states.get('unclassified_or_conflict', []) or [])
            }
            completion_coverage = {
                'total_node_count': int(len(all_ids)),
                'completed_node_count': int(len(completed_ids_for_coverage)),
                'pending_node_count': int(len(pending_ids_for_coverage)),
                'remaining_pool_node_count': int(len(resource_ids_for_coverage)),
                'conflict_node_count': int(len(conflict_ids_for_coverage)),
                'conflict_node_ids': sorted(conflict_ids_for_coverage),
                'covered_node_count': int(sum(len(list(values or [])) for values in states.values())),
                'uncovered_node_ids': sorted(
                    all_ids
                    - completed_ids_for_coverage
                    - pending_ids_for_coverage
                    - resource_ids_for_coverage
                    - conflict_ids_for_coverage
                    - {int(value) for value in list(states.get('support_context', []) or [])}
                    - {int(value) for value in list(states.get('fixed_topology_core', []) or [])}
                ),
            }
        terminal_residual_complete = bool(
            completion_matches
            or (
                not public_pending
                and bool(public_summary.get('topology_plan_feasible', False))
                and residual_infeasibility_proof
                and residual_infeasibility_proof.get('executable') is False
            )
        )
        final_reason = (
            'stage3_complete'
            if terminal_residual_complete
            else (
                'phase_resource_infeasible'
                if infeasibility_proof
                else str(public_summary.get('reason', 'assembly_pending'))
            )
        )
        cls_summary = dict(classification_dict.get('summary', {}) or {})
        covered_product_node_ids = {
            int(node_id)
            for product in list(public_rows or [])
            for node_id in InversePipelineV3._symbolic_product_node_ids(dict(product or {}))
        }
        rejects = [
            dict(record or {})
            for record in list(classification_dict.get('classification_rejects', []) or [])
            if int(dict(record or {}).get('node_id', dict(record or {}).get('global_id', -1)))
            not in covered_product_node_ids
        ]
        unclassified_non_resource = [
            dict(rec or {})
            for rec in list(classification_dict.get('unclassified_nodes', []) or [])
            if not bool(dict(rec or {}).get('resource_pool_member', False))
            and int(dict(rec or {}).get('node_id', dict(rec or {}).get('global_id', -1)))
            not in covered_product_node_ids
        ]
        for rec in unclassified_non_resource:
            rec.setdefault('reason', 'unclassified_non_resource')
        summary = {
            'ok': bool(
                terminal_residual_complete
                or (not infeasibility_proof and public_summary.get('ok', False))
            ),
            'topology_plan_feasible': bool(
                not infeasibility_proof
                and public_summary.get('topology_plan_feasible', False)
            ),
            'closed_topology_count': int(public_summary.get('closed_topology_count', 0)),
            'final_topology_count': int(public_summary.get('final_topology_count', 0)),
            'pending_closure_total_count': int(public_summary.get('pending_closure_total_count', 0)),
            'unclosed_candidates': list(public_pending),
            'classification_rejects': rejects + unclassified_non_resource,
            'diagnostics': {
                'reason': str(final_reason),
                'solver_reason': str(final_reason),
                'infeasibility_proof': dict(infeasibility_proof),
                'residual_infeasibility_proof': dict(residual_infeasibility_proof),
                'class_counts': dict(cls_summary.get('class_counts', {}) or {}),
                'family_counts': dict(cls_summary.get('family_counts', {}) or {}),
                'resource_pool': dict(classification_dict.get('resource_pool', {}) or {}),
                'flex_bounds': dict(assembly_dict.get('flex_bounds', {}) or {}),
                'flex_counts': dict(assembly_dict.get('flex_counts', {}) or {}),
                'resource_ledger': dict(public_summary.get('resource_ledger', {}) or {}),
                'assembly': dict(assembly_dict.get('diagnostics', {}) or {}),
                'flex_target_plan': dict(assembly_dict.get('flex_target_plan', {}) or {}),
                'resource_demand': dict(assembly_dict.get('resource_demand', {}) or {}),
                'protected_resource_counts': dict(assembly_dict.get('protected_resource_counts', {}) or {}),
                'referenced_resource_counts': dict(assembly_dict.get('referenced_resource_counts', {}) or {}),
                'node_coverage': (
                    dict(completion_coverage)
                    if completion_matches
                    else dict(dict(assembly_dict.get('diagnostics', {}) or {}).get('node_coverage', {}) or {})
                ),
                'ownership_ledger': (
                    dict(completion_ownership)
                    if completion_matches
                    else dict(dict(assembly_dict.get('diagnostics', {}) or {}).get('ownership_ledger', {}) or {})
                ),
                'stage3_generation_id': int(
                    completion_diagnostics.get('generation_id', 0) or 0
                ) if completion_matches else int(
                    terminal_diagnostics.get('generation_id', 0) or 0
                ) if terminal_matches else 0,
                'stage3_terminal_stop_reason': str(
                    terminal_diagnostics.get('stop_reason', '') or ''
                ),
                'stage3_terminal_phase': str(
                    terminal_diagnostics.get('phase', '') or ''
                ),
                'stage3_terminal_snapshot_match': bool(terminal_matches),
            },
        }
        coverage = dict(summary['diagnostics'].get('node_coverage', {}) or {})
        if not coverage:
            completed_ids = {
                int(node_id)
                for row in public_rows
                for node_id in InversePipelineV3._symbolic_product_node_ids(dict(row or {}))
            }
            pending_ids = {
                int(node_id)
                for row in public_pending
                for node_id in list(dict(row or {}).get('source_node_ids', []) or [])
            }
            resource_ledger = dict(summary['diagnostics'].get('resource_ledger', {}) or {})
            ownership_states = dict(
                dict(summary['diagnostics'].get('ownership_ledger', {}) or {}).get('states', {}) or {}
            )
            resource_ids = {
                int(node_id)
                for state in ('remaining_closure_resource', 'remaining_aromatic_donor')
                for node_id in list(ownership_states.get(state, []) or [])
            }
            all_ids = {int(getattr(node, 'global_id', -1)) for node in list(nodes or [])}
            coverage = {
                'total_node_count': int(len(all_ids)),
                'completed_node_count': int(len(completed_ids)),
                'pending_node_count': int(len(pending_ids)),
                'remaining_pool_node_count': int(len(resource_ids)),
                'uncovered_node_ids': sorted(all_ids - completed_ids - pending_ids - resource_ids),
            }
            summary['diagnostics']['node_coverage'] = coverage
        summary['coverage_valid'] = not bool(list(coverage.get('uncovered_node_ids', []) or []))
        summary['classification_coverage_valid'] = not bool(
            list(coverage.get('conflict_node_ids', []) or [])
        )
        completed_ids = {
            int(node_id)
            for row in public_rows
            for node_id in InversePipelineV3._symbolic_product_node_ids(dict(row or {}))
        }
        pending_ids = {
            int(node_id)
            for row in public_pending
            for node_id in list(
                dict(row or {}).get('core_node_ids', [])
                or dict(row or {}).get('source_node_ids', [])
                or []
            )
        }
        resource_ledger = dict(summary['diagnostics'].get('resource_ledger', {}) or {})
        ownership_states = dict(
            dict(summary['diagnostics'].get('ownership_ledger', {}) or {}).get('states', {}) or {}
        )
        resource_ids = {
            int(node_id)
            for state in ('remaining_closure_resource', 'remaining_aromatic_donor')
            for node_id in list(ownership_states.get(state, []) or [])
        }
        invariant_conflicts: List[Dict[str, Any]] = []
        for left_name, left_ids, right_name, right_ids in (
            ('completed_topology', completed_ids, 'pending_topology', pending_ids),
            ('completed_topology', completed_ids, 'available_resource', resource_ids),
            ('pending_topology', pending_ids, 'available_resource', resource_ids),
        ):
            overlap = sorted(set(left_ids) & set(right_ids))
            if overlap:
                invariant_conflicts.append({
                    'reason': 'identity_overlap',
                    'states': [left_name, right_name],
                    'node_ids': overlap,
                })
        for row in public_rows:
            source_ids = InversePipelineV3._symbolic_product_node_ids(dict(row or {}))
            gid_ids = sorted({
                int(value)
                for value in re.findall(r'\d+', str(dict(row or {}).get('node_id_notation', '') or ''))
            })
            if source_ids != gid_ids:
                invariant_conflicts.append({
                    'reason': 'product_notation_source_mismatch',
                    'node_id_notation': str(dict(row or {}).get('node_id_notation', '')),
                    'source_node_ids': source_ids,
                    'gid_node_ids': gid_ids,
                })
            product_claimed_ids = {
                int(value)
                for values in dict(
                    dict(row or {}).get('claimed_resource_ids_by_su', {}) or {}
                ).values()
                for value in list(values or [])
            }
            for slot in (
                list(dict(row or {}).get('mainline_slots', []) or [])
                + list(dict(row or {}).get('branch_slots', []) or [])
            ):
                assigned_ids = [
                    int(value)
                    for value in list(dict(slot or {}).get('assigned_resource_ids', []) or [])
                ]
                if len(assigned_ids) != len(set(assigned_ids)):
                    invariant_conflicts.append({
                        'reason': 'duplicate_resource_within_slot',
                        'node_id_notation': str(dict(row or {}).get('node_id_notation', '')),
                        'slot_id': str(dict(slot or {}).get('slot_id', '')),
                        'node_ids': assigned_ids,
                    })
                unexpected_ids = sorted(set(assigned_ids) - product_claimed_ids)
                if unexpected_ids:
                    invariant_conflicts.append({
                        'reason': 'slot_resource_outside_product_ownership',
                        'node_id_notation': str(dict(row or {}).get('node_id_notation', '')),
                        'slot_id': str(dict(slot or {}).get('slot_id', '')),
                        'node_ids': unexpected_ids,
                    })
        for row in public_pending:
            source_ids = sorted({
                int(value)
                for value in list(
                    dict(row or {}).get('core_node_ids', [])
                    or dict(row or {}).get('source_node_ids', [])
                    or []
                )
            })
            gid_ids = sorted({int(value) for value in re.findall(r'\d+', str(dict(row or {}).get('real_gid', '') or ''))})
            infeasibility_proof = dict(dict(row or {}).get('infeasibility_proof', {}) or {})
            suppressed_support_ids = sorted({
                int(value)
                for value in list(infeasibility_proof.get('suppressed_support_node_ids', []) or [])
                if int(value) in set(source_ids)
            })
            expected_gid_ids = sorted(set(source_ids) - set(suppressed_support_ids))
            missing_gid_ids = sorted(set(expected_gid_ids) - set(gid_ids))
            extra_gid_ids = sorted(set(gid_ids) - set(source_ids))
            if missing_gid_ids or extra_gid_ids:
                invariant_conflicts.append({
                    'reason': 'pending_gid_source_mismatch',
                    'contract_id': int(dict(row or {}).get('contract_id', dict(row or {}).get('unit_id', -1)) or -1),
                    'source_node_ids': source_ids,
                    'gid_node_ids': gid_ids,
                    'suppressed_support_node_ids': suppressed_support_ids,
                    'missing_gid_node_ids': missing_gid_ids,
                    'extra_gid_node_ids': extra_gid_ids,
                })
        identity_by_node = {
            **{int(value): 'completed_topology' for value in completed_ids},
            **{int(value): 'pending_topology' for value in pending_ids},
            **{int(value): 'available_resource' for value in resource_ids},
        }
        for group in list(classification_dict.get('atomic_groups', []) or []):
            group_ids = [int(value) for value in list(dict(group or {}).get('node_ids', []) or [])]
            states = {identity_by_node.get(int(value), 'other') for value in group_ids}
            if len(states) > 1:
                invariant_conflicts.append({
                    'reason': 'atomic_pair_split',
                    'group_id': int(dict(group or {}).get('group_id', -1) or -1),
                    'node_ids': group_ids,
                    'states': sorted(states),
                })
        summary['diagnostics']['topology_invariant_conflicts'] = invariant_conflicts
        if invariant_conflicts:
            summary['ok'] = False
            summary['coverage_valid'] = False
            if not dict(summary['diagnostics'].get('infeasibility_proof', {}) or {}):
                summary['diagnostics']['solver_reason'] = 'topology_invariant_violation'
        return public_rows, summary

    @staticmethod
    def _compact_topology_export_rows(
        topology_rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                'actual_su': str(row.get('topology_notation', '')),
                'gid': str(row.get('node_id_notation', '')),
                'product_type': str(row.get('product_type', '')),
                'required_slots_satisfied': bool(row.get('closed', False)),
                'source_fragment_ids': json.dumps(
                    row.get('source_fragment_ids', []), ensure_ascii=False
                ),
                'fixed_source_node_ids': json.dumps(
                    row.get('fixed_source_node_ids', []), ensure_ascii=False
                ),
                'claimed_resource_ids': json.dumps(
                    row.get('claimed_resource_ids_by_su', {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for row in (dict(item or {}) for item in list(topology_rows or []))
        ]

    @staticmethod
    def _compact_pending_export_rows(
        pending_rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        compact_rows: List[Dict[str, Any]] = []
        for row in (dict(item or {}) for item in list(pending_rows or [])):
            fixed_ids = (
                row.get('claimed_source_node_ids', [])
                or row.get('core_node_ids', [])
                or row.get('source_node_ids', [])
                or []
            )
            compact_rows.append({
                'actual_su': str(row.get('real', '')),
                'gid': str(row.get('real_gid', '')),
                'pending_kind': str(row.get('pending_kind', '')),
                'phase': str(row.get('phase', '')),
                'contract_id': int(row.get('contract_id', -1) or -1),
                'source_fragment_ids': json.dumps(
                    row.get('source_fragment_ids', []), ensure_ascii=False
                ),
                'fixed_source_node_ids': json.dumps(fixed_ids, ensure_ascii=False),
                'required_counts': json.dumps(
                    row.get('required_counts', {}), ensure_ascii=False, sort_keys=True
                ),
                'available_counts': json.dumps(
                    row.get('available_counts', {}), ensure_ascii=False, sort_keys=True
                ),
                'reason': str(row.get('reason', '')),
            })
        return compact_rows

    @staticmethod
    def _topology_ownership_rows(
        nodes: List[_NodeV3],
        topology_rows: List[Dict[str, Any]],
        pending_rows: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        completed_by_node: Dict[int, Dict[str, Any]] = {}
        pending_by_node: Dict[int, Dict[str, Any]] = {}
        for row in list(topology_rows or []):
            product = dict(row or {})
            for node_id in InversePipelineV3._symbolic_product_node_ids(product):
                completed_by_node[int(node_id)] = {
                    'topology_id': str(product.get('node_id_notation', '')),
                    'contract_id': '',
                    'fragment_id': str(product.get('source_fragment_ids', '')),
                    'fragment_class': str(product.get('product_type', '')),
                    'role': 'completed_topology',
                }
        for row in list(pending_rows or []):
            for node_id in list(
                dict(row or {}).get('core_node_ids', [])
                or dict(row or {}).get('source_node_ids', [])
                or []
            ):
                pending_by_node[int(node_id)] = {
                    'topology_id': str(dict(row or {}).get('real_gid', '')),
                    'contract_id': str(dict(row or {}).get('contract_id', '')),
                    'fragment_id': str(dict(row or {}).get('source_fragment_ids', '')),
                    'fragment_class': str(
                        dict(row or {}).get('source_fragment_classes', [])
                        or dict(row or {}).get('pending_kind', '')
                    ),
                    'role': 'pending_topology',
                }
        diagnostics = dict(summary.get('diagnostics', {}) or {})
        ownership = dict(diagnostics.get('ownership_ledger', {}) or {})
        states = dict(ownership.get('states', {}) or {})
        state_by_node = {
            int(node_id): str(state)
            for state, node_ids in states.items()
            for node_id in list(node_ids or [])
        }
        rows: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = [
            dict(item or {}) for item in list(ownership.get('invariant_errors', []) or [])
        ]
        for node in sorted(list(nodes or []), key=lambda item: int(getattr(item, 'global_id', -1))):
            node_id = int(getattr(node, 'global_id', -1))
            state = str(state_by_node.get(node_id, 'unclassified_or_conflict'))
            if state in {'completed_topology', 'claimed_closure_resource'}:
                meta = completed_by_node.get(node_id, {'role': state})
            elif state == 'pending_topology':
                meta = pending_by_node.get(node_id, {'role': state})
            else:
                meta = {'role': state}
            meta = dict(meta or {})
            rows.append({
                'node_id': node_id,
                'su_type': int(getattr(node, 'su_type', -1)),
                'state': str(state),
                'topology_id': str(meta.get('topology_id', '')),
                'contract_id': str(meta.get('contract_id', '')),
                'fragment_id': str(meta.get('fragment_id', '')),
                'fragment_class': str(meta.get('fragment_class', '')),
                'role': str(meta.get('role', state)),
                'slot_id': str(meta.get('slot_id', '')),
                'resource_pool': bool(state in {'remaining_closure_resource', 'remaining_aromatic_donor'}),
            })
        return rows, conflicts

    def _print_final_topology_specs(
        self,
        nodes: List[_NodeV3],
        limit: int = 20,
        S_target: Optional[torch.Tensor] = None,
        E_target: Optional[torch.Tensor] = None,
    ) -> None:
        try:
            rows, summary = self._final_topology_rows(
                nodes,
                S_target=S_target,
                E_target=E_target,
            )
            print(
                "[Final Topology] "
                f"ok={bool(summary.get('ok', False))} "
                f"status={str(dict(summary.get('diagnostics', {}) or {}).get('solver_reason', ''))} "
                f"closed={int(summary.get('closed_topology_count', 0))} "
                f"final_specs={int(summary.get('final_topology_count', 0))} "
                f"pending={int(summary.get('pending_closure_total_count', 0))}"
            )
            ledger = dict(dict(summary.get('diagnostics', {}) or {}).get('resource_ledger', {}) or {})
            consumed_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('consumed_counts', {}) or {}).items()
            }
            topology_consumed_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('topology_consumed_counts', {}) or {}).items()
            }
            residual_consumed_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('residual_consumed_counts', {}) or {}).items()
            }
            remaining_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('remaining_counts', {}) or {}).items()
            }
            shortage_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('shortage_counts', {}) or {}).items()
                if int(v) > 0
            }
            initial_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('initial_counts', {}) or {}).items()
            }
            resource_keys = set(initial_counts.keys()) | set(consumed_counts.keys()) | set(remaining_counts.keys())

            def _resource_sort_key(value: str) -> Tuple[int, Any]:
                try:
                    return (0, int(value))
                except Exception:
                    return (1, str(value))

            ordered_resource_keys = sorted(resource_keys, key=_resource_sort_key)

            def _format_resource_counts(counts: Dict[str, int], keys: List[str]) -> str:
                if not keys:
                    return "{}"
                return "{" + ", ".join(
                    f"{key}: {int(counts.get(str(key), 0) or 0)}"
                    for key in list(keys or [])
                ) + "}"

            print(
                "  [Resource Ledger] "
                f"initial_total={int(sum(initial_counts.values()))} "
                f"consumed_total={int(sum(consumed_counts.values()))} "
                f"remaining_total={int(sum(remaining_counts.values()))} "
                f"shortage_total={int(sum(shortage_counts.values()))}"
            )
            print(f"    initial_counts={_format_resource_counts(initial_counts, ordered_resource_keys)}")
            print(f"    consumed_counts={_format_resource_counts(consumed_counts, ordered_resource_keys)}")
            print(f"    topology_consumed_counts={_format_resource_counts(topology_consumed_counts, ordered_resource_keys)}")
            print(f"    residual_consumed_counts={_format_resource_counts(residual_consumed_counts, ordered_resource_keys)}")
            print(f"    remaining_counts={_format_resource_counts(remaining_counts, ordered_resource_keys)}")
            print(f"    shortage_counts={_format_resource_counts(shortage_counts, sorted(shortage_counts.keys(), key=_resource_sort_key))}")
            print(f"    snapshot_stage={str(ledger.get('snapshot_stage', ''))}")
            shown = rows[:max(0, int(limit))]
            for idx, row in enumerate(shown, start=1):
                print(
                    "  [TopologySpec] "
                    f"#{int(idx)} "
                    f"topology={row.get('topology_notation', '')} "
                    f"node_ids={row.get('node_id_notation', '')} "
                    f"source_fragment_ids={row.get('source_fragment_ids', [])} "
                    f"fixed_source_node_ids={row.get('fixed_source_node_ids', [])} "
                    f"claimed_resource_ids_by_su={row.get('claimed_resource_ids_by_su', {})}"
                )
            if len(rows) > len(shown):
                print(f"  [TopologySpec] ... {len(rows) - len(shown)} more")
            unclosed = list(summary.get('unclosed_candidates', []) or [])
            if unclosed:
                print(f"  [Final Topology Pending] showing={min(8, len(unclosed))}/{len(unclosed)}")
                for idx, row in enumerate(unclosed[:8]):
                    rec = dict(row or {})
                    print(
                        f"    [{idx}] real={rec.get('real', '')} "
                        f"real_gid={rec.get('real_gid', '')}"
                    )
            rejects = list(summary.get('classification_rejects', []) or [])
            if rejects:
                print(f"  [Final Topology Unrecognized] showing={min(8, len(rejects))}/{len(rejects)}")
                for idx, row in enumerate(rejects[:8]):
                    rec = dict(row or {})
                    print(
                        f"    [{idx}] gid={rec.get('global_id', '')} "
                        f"su={rec.get('su_type', '')} "
                        f"reason={rec.get('reason', '')}"
                    )
        except Exception as e:
            print(f"[Final Topology] Failed to summarize topology specs: {e}")

    def _stage3_topology_signature(self,
                                   H: torch.Tensor,
                                   assembly: Dict[str, Any]) -> Tuple[Any, ...]:
        H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
        h_part = tuple(
            int(H_cpu[int(su)].item()) if int(su) < int(H_cpu.numel()) else 0
            for su in (10, 11, 12, 13, 22, 23, 24)
        )
        flex_counts = dict((assembly or {}).get('flex_counts', {}) or {})
        pending_count = len(list((assembly or {}).get('pending_units', []) or []))
        diag = dict((assembly or {}).get('diagnostics', {}) or {})
        total_shortage = dict(diag.get('total_shortage', {}) or {})
        residual = dict(diag.get('residual_resource_counts', {}) or {})
        shortage_part = tuple(
            int(total_shortage.get(str(su), total_shortage.get(int(su), 0)) or 0)
            for su in (11, 22, 23, 24)
        )
        residual_part = tuple(
            int(residual.get(str(su), residual.get(int(su), 0)) or 0)
            for su in (11, 22, 23, 24)
        )
        return (
            *h_part,
            int(flex_counts.get('final', 0) or 0),
            int(diag.get('explicit_flex_count', flex_counts.get('explicit_total', flex_counts.get('explicit', 0))) or 0),
            int(pending_count),
            shortage_part,
            residual_part,
            str((assembly or {}).get('reason', '')),
        )

    @staticmethod
    def _stage3_metrics_scalar(metrics: Dict[str, Any]) -> int:
        rec = dict(metrics or {})
        return (
            int(rec.get('pending_count', 0) or 0) +
            int(rec.get('flex_deficit', 0) or 0) +
            int(rec.get('resource_deficit_total', 0) or 0)
        )

    def _validate_stage3_candidate(self,
                                   nodes: List[_NodeV3],
                                   H: torch.Tensor,
                                   S_target: torch.Tensor,
                                   E_target: torch.Tensor,
                                   tag: str) -> Tuple[bool, str]:
        meta_before_sync = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
        try:
            graph_ok, graph_errors = self.layer1_assigner.validate_graph_consistency(
                nodes=nodes,
                H=H,
                E_target=E_target,
                verbose=False,
            )
        except Exception as e:
            print(f"[{tag}] 图校验失败: {e}")
            return False, 'graph_check_exception'
        graph_accept_ok, graph_accept_meta = self.layer1_assigner.assess_consistency_acceptance(
            list(graph_errors or []),
            max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
        )
        if not bool(graph_accept_ok):
            print(f"[{tag}] 回退: 图结构校验未通过，errors={len(graph_errors)}")
            try:
                self.layer1_assigner._print_consistency_errors(graph_errors, prefix=tag, limit=6)
            except Exception:
                pass
            return False, 'graph_invalid'
        if (not bool(graph_ok)) and int(graph_accept_meta.get('n_soft', 0)) > 0:
            print(
                f"[{tag}] 接受少量1-hop软缺口: "
                f"soft={int(graph_accept_meta.get('n_soft', 0))}/"
                f"{int(graph_accept_meta.get('max_soft_errors', 0))}"
            )
        try:
            self._sync_fixed_partition_meta_from_nodes(nodes)
        except Exception as e:
            self._set_fixed_partition_meta(meta_before_sync)
            print(f"[{tag}] 固定锚点元数据同步失败: {e}")
            return False, 'meta_sync_exception'
        try:
            hist_diag = self.layer4_adjuster._evaluate_required_hist_constraints(
                H,
                E_target,
                S_target=S_target,
                enforce_h_rel=True,
            )
        except Exception as e:
            self._set_fixed_partition_meta(meta_before_sync)
            print(f"[{tag}] 直方图校验失败: {e}")
            return False, 'hist_check_exception'
        if not bool(hist_diag.get('ok', False)):
            self._set_fixed_partition_meta(meta_before_sync)
            reason = str(hist_diag.get('reason', 'hist_invalid'))
            print(f"[{tag}] 回退: 直方图/固定连接校验未通过，reason={reason}")
            return False, reason
        return True, 'ok'

    @staticmethod
    def _parse_h_overflow_feedback(reason: str) -> Optional[Dict[str, Any]]:
        text = str(reason or "")
        if "元素H误差过大" not in text:
            return None
        predicted_h: Optional[float] = None
        target_h: Optional[float] = None
        try:
            import re
            match = re.search(
                r"预测(?:=)?([0-9]+(?:\.[0-9]+)?)(?:\s+vs\s+|,目标=?)"
                r"([0-9]+(?:\.[0-9]+)?)",
                text,
            )
            if match:
                predicted_h = float(match.group(1))
                target_h = float(match.group(2))
        except Exception:
            predicted_h = None
            target_h = None
        required_h_drop = 0
        if predicted_h is not None and target_h is not None:
            required_h_drop = max(0, int(round(predicted_h - target_h)))
        return {
            "reason": text,
            "predicted_h": predicted_h,
            "target_h": target_h,
            "required_h_drop": int(required_h_drop),
            "source": "layer1_cold_start",
        }

    def _run_topology_stage3_resource_cycles(self,
                                             nodes: List[_NodeV3],
                                             H: torch.Tensor,
                                             current_diff: Dict[str, object],
                                             S_target: torch.Tensor,
                                             E_target: torch.Tensor,
                                             eval_lib_path: Optional[str],
                                             layer1_runtime_cfg: Dict[str, Any],
                                             layer2_runtime_cfg: Dict[str, Any],
                                             layer2_eval_hwhm: float,
                                             outer_idx: int,
                                             max_stage3_iterations: int = 20,
                                             max_primary_adjustments_per_stage: int = 10,
                                             max_h_compensation_ops_per_adjustment: int = 8,
                                             residual_flex_preferred_23: int = 3,
                                             residual_flex_min_23: int = 2,
                                             residual_side_preferred_23: int = 2,
                                             residual_side_min_23: int = 1,
                                             h_tolerance: float = 0.06,
                                             enable_post_external_spectrum: bool = True,
                                             spectrum_candidate_limit: int = 10,
                                             spectrum_max_primary_adjustments: int = 8,
                                             spectrum_min_r2_gain: float = 0.002,
                                             spectrum_min_window_loss_gain: float = 0.1,
                                             spectrum_max_accepted_rounds: int = 2,
                                             iteration_offset: int = 0,
                                             output_dir: Optional[str] = None) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, object], Dict[str, Any]]:
        def rebuild_fn(H_seed: torch.Tensor,
                       phase: str,
                       cycle_idx: int,
                       identity_seed_nodes: Optional[List[_NodeV3]] = None) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, Any], float]:
            layer1_stage3_cfg = dict(layer1_runtime_cfg)
            try:
                base_variant = int(layer1_stage3_cfg.get('build_variant', 0))
            except Exception:
                base_variant = 0
            layer1_stage3_cfg['profile'] = 'rebuild'
            layer1_stage3_cfg['build_variant'] = (
                int(base_variant) +
                503 * int(outer_idx + 1) +
                29 * int(cycle_idx + 1)
            )
            cand_nodes, cand_H, cand_diff, cand_r2 = self._run_layer12_cycle(
                H_seed=H_seed.detach().clone(),
                S_target=S_target,
                E_target=E_target,
                eval_lib_path=eval_lib_path,
                layer1_runtime_cfg=layer1_stage3_cfg,
                layer2_runtime_cfg=layer2_runtime_cfg,
                identity_seed_nodes=None,
                eval_nmr=False,
            )
            cand_H = self._histogram_from_nodes(cand_nodes).detach().clone()
            cand_diff = dict(cand_diff or {})
            cand_diff['mode'] = 'stage3_resource_adjustment_layer1_layer2'
            cand_diff['preserves_node_ids'] = False
            cand_diff['cold_rebuild'] = True
            return cand_nodes, cand_H, cand_diff, float(cand_r2)

        def validate_fn(nodes: List[_NodeV3],
                        H: torch.Tensor,
                        tag: str) -> Tuple[bool, str]:
            return self._validate_stage3_candidate(
                nodes,
                H,
                S_target=S_target,
                E_target=E_target,
                tag=tag,
            )

        def context_snapshot_fn() -> Dict[str, Any]:
            return copy.deepcopy(
                getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {}
            )

        def context_restore_fn(snapshot: Dict[str, Any]) -> None:
            self._set_fixed_partition_meta(copy.deepcopy(snapshot or {}))

        workflow_result = self.layer4_adjuster.run_block_c_resource_adjustment_loop(
            H=H.detach().clone(),
            nodes=copy.deepcopy(nodes),
            E_target=E_target,
            S_target=S_target,
            rebuild_fn=rebuild_fn,
            validate_fn=validate_fn,
            context_snapshot_fn=context_snapshot_fn,
            context_restore_fn=context_restore_fn,
            output_dir=output_dir,
            max_stage3_iterations=int(max_stage3_iterations),
            max_primary_adjustments_per_stage=int(max_primary_adjustments_per_stage),
            max_h_compensation_ops_per_adjustment=int(max_h_compensation_ops_per_adjustment),
            residual_flex_preferred_23=int(residual_flex_preferred_23),
            residual_flex_min_23=int(residual_flex_min_23),
            residual_side_preferred_23=int(residual_side_preferred_23),
            residual_side_min_23=int(residual_side_min_23),
            h_tolerance=float(h_tolerance),
            outer_cycle=int(outer_idx) + 1,
            iteration_offset=max(0, int(iteration_offset)),
            difference_spectrum=dict(current_diff or {}),
            enable_post_external_spectrum=bool(enable_post_external_spectrum),
            spectrum_candidate_limit=int(spectrum_candidate_limit),
            spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
            spectrum_min_r2_gain=float(spectrum_min_r2_gain),
            spectrum_min_window_loss_gain=float(spectrum_min_window_loss_gain),
            spectrum_max_accepted_rounds=int(spectrum_max_accepted_rounds),
        )

        meta = workflow_result.to_dict()
        meta['stage'] = 'topology_resource_adjustment'
        meta['last_r2'] = float(dict(getattr(workflow_result, 'diff', {}) or {}).get('r2', float('nan')))
        meta['completion_diagnostics'] = {}
        meta['terminal_diagnostics'] = {}
        workflow_cycles = list(getattr(workflow_result, 'cycles', []) or [])
        if workflow_cycles:
            terminal_cycle = workflow_cycles[-1]
            terminal_plan = dict(getattr(terminal_cycle, 'plan', {}) or {})
            terminal_phase_result = dict(terminal_plan.get('phase_result', {}) or {})
            terminal_proof = dict(
                terminal_phase_result.get('infeasibility_proof', {})
                or dict(terminal_plan.get('diagnostics', {}) or {}).get(
                    'infeasibility_proof', {}
                )
                or {}
            )
            if (
                str(getattr(workflow_result, 'stop_reason', '') or '')
                == 'phase_resource_infeasible'
                and terminal_proof
            ):
                terminal_runtime = dict(terminal_plan.get('runtime', {}) or {})
                terminal_ledger = dict(
                    terminal_runtime.get('iteration_topology_ledger', {}) or {}
                )
                meta['terminal_diagnostics'] = {
                    'stop_reason': 'phase_resource_infeasible',
                    'phase': str(terminal_plan.get('phase', '') or 'unknown'),
                    'generation_id': int(
                        terminal_ledger.get(
                            'generation_id', getattr(terminal_cycle, 'cycle', 0)
                        )
                        or 0
                    ),
                    'infeasibility_proof': terminal_proof,
                    'node_state_signature': self._stage3_node_state_signature(
                        list(getattr(workflow_result, 'nodes', []) or [])
                    ),
                    'phase_result': terminal_phase_result,
                }
        for cycle in reversed(workflow_cycles):
            if str(getattr(cycle, 'stop_reason', '') or '') != 'stage3_complete':
                continue
            plan_dict = dict(getattr(cycle, 'plan', {}) or {})
            runtime = dict(plan_dict.get('runtime', {}) or {})
            ledger = dict(runtime.get('iteration_topology_ledger', {}) or {})
            generation_id = int(ledger.get('generation_id', getattr(cycle, 'cycle', 0)) or 0)
            completed_products = [
                normalize_stage3_product(
                    raw,
                    generation_id=generation_id,
                    index=index,
                ).to_dict()
                for index, raw in enumerate(list(ledger.get('completed_products', []) or []))
            ]
            meta['completion_diagnostics'] = {
                **dict(plan_dict.get('diagnostics', {}) or {}),
                'stage3_complete': True,
                'generation_id': int(generation_id),
                'products': completed_products,
                'ownership_ledger': self._stage3_completion_ownership(ledger),
                'node_state_signature': self._stage3_node_state_signature(
                    list(getattr(workflow_result, 'nodes', []) or [])
                ),
                'phase_result': dict(plan_dict.get('phase_result', {}) or {}),
            }
            break
        return (
            list(getattr(workflow_result, 'nodes', []) or []),
            torch.clamp(getattr(workflow_result, 'H', H).detach().clone(), min=0).long(),
            dict(getattr(workflow_result, 'diff', {}) or {}),
            meta,
        )

    def _flatten_stage3_resource_cycles(self,
                                        stage3_meta: Dict[str, Any],
                                        outer_idx: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        meta = dict(stage3_meta or {})
        cycles = list(meta.get('cycles', []) or [])
        if not cycles:
            return [{
                'outer_cycle': int(outer_idx) + 1,
                'cycle': 0,
                'phase': '',
                'applied': False,
                'stage3_stop_reason': str(meta.get('stop_reason', '')),
                'accepted_cycles_total': int(meta.get('accepted_cycles', 0) or 0),
                'accepted_state_retained': '',
                'rebuild_valid': '',
                'after_ok': '',
                'after_flex_range_reconciliation_ok': '',
                'after_external_closure_ok': '',
                'after_reason': '',
                'improve_reason': '',
                'h_error_before': '',
                'h_error_after': '',
                'element_abs_error_before': '',
                'element_abs_error_after': '',
                'pending_before': '',
                'pending_after': '',
                'flex_before': '',
                'flex_after': '',
                'flex_deficit_before': '',
                'flex_deficit_after': '',
                'resource_deficit_before': '',
                'resource_deficit_after': '',
                'layer2_r2': '',
                'spectrum_candidate_id': '',
                'spectrum_r2_gain': '',
                'spectrum_window_loss_gain': '',
                'spectrum_candidate_scores': '[]',
                'planned_conversions': '[]',
                'before_requirements': '{}',
                'conversion_pattern_key': '',
                'bundle_count': 0,
                'adjustment_bundles': '[]',
                'virtual_leftover_products': '[]',
                'committed_leftover_products': '[]',
                'primary_adjustment_count': 0,
                'physical_conversion_count': 0,
                'h_compensation_count': 0,
                'histogram_before': '{}',
                'histogram_after': '{}',
                'phase_inventory': '{}',
                'phase_demand': '{}',
                'gap_before': '{}',
                'gap_after': '{}',
                'state_signature': '',
                'cold_rebuild': False,
                'structural_blockers': '[]',
                'infeasibility_proof': '{}',
                'residual_infeasibility_proof': '{}',
            }]
        for item in cycles:
            rec = dict(item or {})
            before = dict(rec.get('before_metrics', {}) or {})
            after = dict(rec.get('after_metrics', {}) or {})
            plan_wrapper = dict(rec.get('plan', {}) or {})
            diagnostics = dict(rec.get('diagnostics', {}) or {})
            plan_diagnostics = dict(plan_wrapper.get('diagnostics', {}) or {})
            spectrum_diagnostics = (
                diagnostics
                if (
                    'selected_candidate' in diagnostics
                    or 'evaluated_candidates' in diagnostics
                )
                else plan_diagnostics
            )
            selected_spectrum_candidate = dict(
                spectrum_diagnostics.get('selected_candidate', {}) or {}
            )
            phase_result = dict(plan_wrapper.get('phase_result', {}) or {})
            rows.append({
                'outer_cycle': int(outer_idx) + 1,
                'cycle': int(rec.get('cycle', 0) or 0),
                'phase': str(rec.get('phase', '')),
                'applied': bool(rec.get('applied', False)),
                'stage3_stop_reason': str(rec.get('stop_reason', '') or meta.get('stop_reason', '')),
                'accepted_cycles_total': int(meta.get('accepted_cycles', 0) or 0),
                'accepted_state_retained': rec.get('accepted_state_retained', ''),
                'rebuild_valid': rec.get('rebuild_valid', ''),
                'after_ok': after.get('ok', ''),
                'after_flex_range_reconciliation_ok': diagnostics.get('after_flex_range_reconciliation_ok', ''),
                'after_external_closure_ok': diagnostics.get('after_external_closure_ok', ''),
                'after_reason': str(after.get('reason', '')),
                'improve_reason': str(diagnostics.get('improve_reason', '')),
                'h_error_before': diagnostics.get('h_error_before', ''),
                'h_error_after': diagnostics.get('h_error_after', ''),
                'element_abs_error_before': diagnostics.get('element_abs_error_before', ''),
                'element_abs_error_after': diagnostics.get('element_abs_error_after', ''),
                'pending_before': before.get('pending_count', ''),
                'pending_after': after.get('pending_count', ''),
                'flex_before': before.get('flex_final', ''),
                'flex_after': after.get('flex_final', ''),
                'flex_deficit_before': before.get('flex_deficit', ''),
                'flex_deficit_after': after.get('flex_deficit', ''),
                'resource_deficit_before': before.get('resource_deficit_total', ''),
                'resource_deficit_after': after.get('resource_deficit_total', ''),
                'layer2_r2': rec.get('layer2_r2', ''),
                'spectrum_candidate_id': str(
                    selected_spectrum_candidate.get('candidate_id', '')
                ),
                'spectrum_r2_gain': spectrum_diagnostics.get('r2_gain', ''),
                'spectrum_window_loss_gain': spectrum_diagnostics.get(
                    'window_loss_gain', ''
                ),
                'spectrum_candidate_scores': json.dumps(
                    spectrum_diagnostics.get(
                        'evaluated_candidates',
                        plan_diagnostics.get('rejected_spectrum_candidates', []),
                    ),
                    ensure_ascii=False,
                    default=str,
                ),
                'primary_adjustment_count': int(rec.get('primary_adjustment_count', 0) or 0),
                'physical_conversion_count': int(rec.get('physical_conversion_count', 0) or 0),
                'h_compensation_count': int(rec.get('h_compensation_count', 0) or 0),
                'histogram_before': json.dumps(rec.get('histogram_before', {}), ensure_ascii=False, sort_keys=True),
                'histogram_after': json.dumps(rec.get('histogram_after', {}), ensure_ascii=False, sort_keys=True),
                'phase_inventory': json.dumps(rec.get('phase_inventory', {}), ensure_ascii=False, sort_keys=True),
                'phase_demand': json.dumps(rec.get('phase_demand', {}), ensure_ascii=False, sort_keys=True),
                'gap_before': json.dumps(rec.get('gap_before', {}), ensure_ascii=False, sort_keys=True),
                'gap_after': json.dumps(rec.get('gap_after', {}), ensure_ascii=False, sort_keys=True),
                'state_signature': str(rec.get('state_signature', '')),
                'cold_rebuild': bool(rec.get('cold_rebuild', False)),
                'structural_blockers': json.dumps(rec.get('structural_blockers', []), ensure_ascii=False),
                'infeasibility_proof': json.dumps(
                    phase_result.get(
                        'infeasibility_proof',
                        dict(plan_wrapper.get('diagnostics', {}) or {}).get(
                            'infeasibility_proof', {}
                        ),
                    ),
                    ensure_ascii=False,
                    default=str,
                ),
                'residual_infeasibility_proof': json.dumps(
                    dict(plan_wrapper.get('diagnostics', {}) or {}).get('residual_infeasibility_proof', {}),
                    ensure_ascii=False,
                    default=str,
                ),
                'planned_conversions': json.dumps(
                    diagnostics.get('planned_conversions', []),
                    ensure_ascii=False,
                    default=str,
                ),
                'before_requirements': json.dumps(
                    diagnostics.get('before_requirements', {}),
                    ensure_ascii=False,
                    default=str,
                ),
                'conversion_pattern_key': str(diagnostics.get('conversion_pattern_key', '')),
                'bundle_count': len(list(diagnostics.get('adjustment_bundles', []) or [])),
                'adjustment_bundles': json.dumps(
                    diagnostics.get('adjustment_bundles', []),
                    ensure_ascii=False,
                    default=str,
                ),
                'virtual_leftover_products': json.dumps(
                    dict(plan_wrapper.get('assembly', {}) or {}).get('diagnostics', {}).get('virtual_leftover_products', []),
                    ensure_ascii=False,
                    default=str,
                ),
                'committed_leftover_products': json.dumps(
                    dict(plan_wrapper.get('assembly', {}) or {}).get('diagnostics', {}).get('committed_leftover_products', []),
                    ensure_ascii=False,
                    default=str,
                ),
                'transactions': json.dumps(
                    phase_result.get('transactions', []),
                    ensure_ascii=False,
                    default=str,
                ),
            })
        return rows

    def _save_final_stage3_resource_summary(self,
                                            rows: Sequence[Dict[str, Any]],
                                            output_dir: str) -> None:
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = final_dir / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        row_list = [dict(row or {}) for row in list(rows or [])]

        def _decoded_dict(value: Any) -> Dict[str, Any]:
            if isinstance(value, dict):
                return dict(value)
            try:
                decoded = json.loads(str(value or '{}'))
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
            return dict(decoded) if isinstance(decoded, dict) else {}

        cycle_columns = [
            'outer_cycle', 'cycle', 'phase', 'applied', 'stage3_stop_reason',
            'accepted_state_retained', 'rebuild_valid', 'after_ok', 'improve_reason',
            'h_error_before', 'h_error_after',
            'pending_before', 'pending_after',
            'flex_before', 'flex_after',
            'resource_deficit_before', 'resource_deficit_after',
            'layer2_r2', 'spectrum_candidate_id', 'spectrum_r2_gain',
            'spectrum_window_loss_gain',
            'primary_adjustment_count', 'physical_conversion_count', 'h_compensation_count',
            'infeasibility_reason', 'residual_infeasibility_reason',
        ]
        compact_cycle_rows = []
        for row in row_list:
            proof = _decoded_dict(row.get('infeasibility_proof', {}))
            residual_proof = _decoded_dict(row.get('residual_infeasibility_proof', {}))
            compact_cycle_rows.append({
                **{key: row.get(key, '') for key in cycle_columns},
                'infeasibility_reason': str(proof.get('reason', '')),
                'residual_infeasibility_reason': str(residual_proof.get('reason', '')),
            })
        pd.DataFrame(compact_cycle_rows, columns=cycle_columns).to_csv(
            final_dir / 'final_stage3_resource_cycles.csv',
            index=False,
        )
        with open(debug_dir / 'final_stage3_resource_cycle_details.jsonl', 'w', encoding='utf-8') as handle:
            for row in row_list:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')

        transaction_details: List[Dict[str, Any]] = []
        transaction_rows_by_signature: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for row in row_list:
            raw_transactions = row.get('transactions', '[]')
            try:
                transactions = json.loads(raw_transactions) if isinstance(raw_transactions, str) else list(raw_transactions or [])
            except (TypeError, ValueError, json.JSONDecodeError):
                transactions = []
            for transaction in list(transactions or []):
                item = dict(transaction or {})
                outer_cycle = int(item.get('outer_cycle', row.get('outer_cycle', 0)) or 0)
                iteration = int(item.get('iteration', row.get('cycle', 0)) or 0)
                detail = {
                    **item,
                    'outer_cycle': outer_cycle,
                    'iteration': iteration,
                    'phase': str(item.get('phase', row.get('phase', ''))),
                }
                transaction_details.append(detail)
                compact = {
                    'first_outer_cycle': outer_cycle,
                    'last_outer_cycle': outer_cycle,
                    'first_iteration': iteration,
                    'last_iteration': iteration,
                    'phase': str(item.get('phase', row.get('phase', ''))),
                    'contract_id': int(item.get('contract_id', -1) or -1),
                    'target_role': str(item.get('target_role', '')),
                    'source_node_ids': json.dumps(item.get('source_node_ids', []), ensure_ascii=False),
                    'converted_node_ids': json.dumps(item.get('converted_node_ids', []), ensure_ascii=False),
                    'compensation_node_ids': json.dumps(item.get('compensation_node_ids', []), ensure_ascii=False),
                    'claimed_resource_ids': json.dumps(item.get('claimed_resource_ids', {}), ensure_ascii=False, sort_keys=True),
                    'product_node_ids': json.dumps(item.get('product_node_ids', []), ensure_ascii=False),
                    'commit_scope': str(item.get('commit_scope', '')),
                    'committed': bool(item.get('committed', False)),
                    'rollback_reason': str(item.get('rollback_reason', '')),
                    'occurrences': 1,
                }
                signature_fields = [
                    'phase', 'contract_id', 'target_role', 'source_node_ids',
                    'converted_node_ids', 'compensation_node_ids', 'claimed_resource_ids',
                    'product_node_ids', 'commit_scope', 'committed', 'rollback_reason',
                ]
                signature = tuple(compact[key] for key in signature_fields)
                existing = transaction_rows_by_signature.get(signature)
                if existing is None:
                    transaction_rows_by_signature[signature] = compact
                else:
                    existing['last_outer_cycle'] = outer_cycle
                    existing['last_iteration'] = iteration
                    existing['occurrences'] = int(existing.get('occurrences', 1)) + 1
        transaction_rows = list(transaction_rows_by_signature.values())
        transaction_columns = [
            'first_outer_cycle', 'last_outer_cycle', 'first_iteration', 'last_iteration',
            'phase', 'contract_id', 'target_role', 'source_node_ids',
            'converted_node_ids', 'compensation_node_ids', 'claimed_resource_ids',
            'product_node_ids', 'commit_scope', 'committed', 'rollback_reason', 'occurrences',
        ]
        pd.DataFrame(transaction_rows, columns=transaction_columns).to_csv(
            final_dir / 'final_stage3_topology_transactions.csv',
            index=False,
        )
        with open(debug_dir / 'final_stage3_topology_transaction_details.jsonl', 'w', encoding='utf-8') as handle:
            for row in transaction_details:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        summary = {
            'cycle_rows': int(len(row_list)),
            'applied_cycle_count': int(sum(1 for row in row_list if bool(row.get('applied', False)))),
            'accepted_cycle_count': int(sum(
                1 for row in row_list
                if str(row.get('stage3_stop_reason', '')) == 'accepted'
                and bool(row.get('rebuild_valid', False))
            )),
            'committed_transaction_count': int(sum(
                1 for row in transaction_details if bool(row.get('committed', False))
            )),
            'rolled_back_transaction_count': int(sum(
                1 for row in transaction_details
                if not bool(row.get('committed', False)) and bool(str(row.get('rollback_reason', '')))
            )),
            'transaction_detail_rows': int(len(transaction_details)),
            'transaction_summary_rows': int(len(transaction_rows)),
            'outer_cycles_with_stage3': sorted({
                int(row.get('outer_cycle', 0) or 0)
                for row in row_list
                if int(row.get('outer_cycle', 0) or 0) > 0
            }),
            'last_stop_reason': str(row_list[-1].get('stage3_stop_reason', '')) if row_list else '',
            'accepted_spectrum_cycle_count': int(sum(
                1 for row in row_list
                if str(row.get('phase', '')) == 'remaining_pool_spectrum_rebalance'
                and str(row.get('stage3_stop_reason', '')) == 'accepted'
            )),
            'infeasibility_proof': (
                _decoded_dict(row_list[-1].get('infeasibility_proof', {}))
                if row_list else {}
            ),
            'residual_infeasibility_proof': (
                _decoded_dict(row_list[-1].get('residual_infeasibility_proof', {}))
                if row_list else {}
            ),
        }
        with open(final_dir / 'final_stage3_resource_summary.json', 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'summary': summary,
                    'cycles': compact_cycle_rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
    
    def infer(self, S_target: torch.Tensor, E_target: torch.Tensor,
              save_intermediates: bool = True,
              output_dir: str = 'inverse_result',
              eval_lib_path: Optional[str] = None,
              eval_hwhm: float = 1.0,
              eval_allow_approx: bool = True,
              enable_hop1_adjust: bool = True,
              hop1_adjust_iterations: int = 6,
              hop1_neg_threshold: float = -0.8,
              hop1_pos_threshold: float = 0.8,
              layer1_config: Optional[Dict[str, Any]] = None,
              layer2_config: Optional[Dict[str, Any]] = None,
              layer3_config: Optional[Dict[str, Any]] = None,
              stage_configs: Optional[Dict[str, Dict]] = None,
              outer_max_cycles: int = 8,
              outer_improve_eps: float = 1e-4) -> Tuple[List[_NodeV3], torch.Tensor]:
        """
        逆向推理主入口
        
        流程：
        1. Layer0 初始化 SU 直方图
        2. Layer1 -> Layer2 -> Layer3 得到初始完整状态
        3. 进入外循环：
           - block_a / block_b:
             作为候选 H 调整，只在重跑 Layer1/2 后通过结构校验且
             Layer2 R2 超过当前接受参考时才采纳
           - block_c_resource:
             每轮执行“分类 -> 组装 -> 精确需求 -> 合法转化 -> 冷启动 Layer1/2”
             的 Stage3 组装循环；保持合法即向前探索，不做短期回退
           - 每轮结束后执行一次 Layer3 细化
        4. 返回当前外循环结束时的状态
        
        Args:
            stage_configs: Layer4 正式阶段配置，推荐格式如:
                {
                    'block_a': {...},
                    'block_b': {...},
                    'block_c_resource': {
                        'enabled': True,
                        'max_stage3_iterations': 20,
                        'max_primary_adjustments_per_stage': 10,
                    },
                }
            layer1_config 可调参数:
                - eval_hwhm, eval_allow_approx, build_variant
                - enable_carbonyl_joint_adjust, carbonyl_joint_iterations
                - carbonyl_joint_max_adjustments
                - carbonyl_joint_pos_threshold, carbonyl_joint_neg_threshold
                - enable_hop1_adjust, hop1_adjust_iterations
                - hop1_neg_threshold, hop1_pos_threshold
                - hop1_adjust_region_order, hop1_adjust_max_candidate_trials
                - hop1_adjust_max_moves_per_region
                - hop1_adjust_max_full_evals
                - profile, build_attempts, local_swap_max_attempts
            layer2_config 可调参数:
                - eval_hwhm
                - layer2_max_hop2_diff_nodes
                - layer2_use_g2s_decoder, layer2_decoder_max_mu_delta
            layer3_config 可调参数:
                - max_iters, hwhm, pos_search_window, neg_assign_window
                - top_k_samples, neg_top_k_peaks, neg_peak_min_sep_ppm
                - enable_approx_hop2_template_adjust
                - approx_hop2_max_iters, approx_hop2_max_diff_nodes
                - approx_hop2_top_k_templates
                - layer3_use_g2s_decoder, layer3_verbose
            outer_max_cycles:
                - Layer4 -> Layer1/2 -> Layer3 宏循环次数上限
            outer_improve_eps:
                - block_a / block_b 的最小 Layer2 R2 提升阈值
        """
        print("=" * 80)
        print("InversePipelineV3: 开始逆向推理")
        print("=" * 80)

        layer1_runtime_cfg, layer2_runtime_cfg, layer3_runtime_cfg = self._build_layer_runtime_configs(
            eval_hwhm=eval_hwhm,
            eval_allow_approx=eval_allow_approx,
            enable_hop1_adjust=enable_hop1_adjust,
            hop1_adjust_iterations=hop1_adjust_iterations,
            hop1_neg_threshold=hop1_neg_threshold,
            hop1_pos_threshold=hop1_pos_threshold,
            layer1_config=layer1_config,
            layer2_config=layer2_config,
            layer3_config=layer3_config,
        )
        layer2_eval_hwhm = float(layer2_runtime_cfg.get('eval_hwhm', eval_hwhm))
        save_round_outputs = bool(save_intermediates)

        default_stage_configs = {
            'block_a': {
                'max_cycles': 1,
                'max_moves': 6,
                'carbonyl_max_moves': 4,
                'score_rel_threshold': 0.02,
                'peak_rel_threshold': 0.01,
            },
            'block_b': {
                'max_cycles': 1,
                'max_moves_each': 4,
                'peak_rel_threshold': 0.01,
            },
            'block_c_resource': {
                'enabled': True,
                'max_stage3_iterations': 20,
                'max_primary_adjustments_per_stage': 10,
                'max_h_compensation_ops_per_adjustment': 8,
                'residual_flex_preferred_23': 3,
                'residual_flex_min_23': 2,
                'residual_side_preferred_23': 2,
                'residual_side_min_23': 1,
                'h_tolerance': 0.06,
                'enable_post_external_spectrum': True,
                'spectrum_candidate_limit': 10,
                'spectrum_max_primary_adjustments': 8,
                'spectrum_min_r2_gain': 0.002,
                'spectrum_min_window_loss_gain': 0.1,
                'spectrum_max_accepted_rounds': 2,
            },
        }
        supported_block_b_keys = set(default_stage_configs['block_b'])
        if stage_configs is not None:
            for stage, cfg in stage_configs.items():
                if stage in default_stage_configs:
                    default_stage_configs[stage].update(cfg)
                else:
                    raise ValueError(f"未知 stage_configs 键: {stage}")
        block_b_user_cfg = dict((stage_configs or {}).get('block_b', {}) or {})
        unknown_block_b_keys = set(block_b_user_cfg) - supported_block_b_keys
        if unknown_block_b_keys:
            raise ValueError(
                "block_b contains unknown keys: "
                + ", ".join(sorted(str(key) for key in unknown_block_b_keys))
            )
        resource_cfg = dict(default_stage_configs.get('block_c_resource', {}) or {})
        resource_enabled = bool(resource_cfg.pop('enabled', True))
        supported_resource_keys = {
            'max_stage3_iterations',
            'max_primary_adjustments_per_stage',
            'max_h_compensation_ops_per_adjustment',
            'residual_flex_preferred_23',
            'residual_flex_min_23',
            'residual_side_preferred_23',
            'residual_side_min_23',
            'h_tolerance',
            'enable_post_external_spectrum',
            'spectrum_candidate_limit',
            'spectrum_max_primary_adjustments',
            'spectrum_min_r2_gain',
            'spectrum_min_window_loss_gain',
            'spectrum_max_accepted_rounds',
        }
        unknown_resource_keys = set(resource_cfg) - supported_resource_keys
        if unknown_resource_keys:
            raise ValueError(
                "block_c_resource contains unknown keys: "
                + ", ".join(sorted(str(key) for key in unknown_resource_keys))
            )

        print("\n>>> Layer0: 估计SU直方图")
        H_init = self.estimate_su_histogram(S_target, E_target)
        try:
            self._set_fixed_partition_meta(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
        except Exception:
            pass
        self._print_histogram_and_elements(H_init, E_target, label='Layer0 Summary')

        print("\n>>> 初始化 Layer1 -> Layer2 -> Layer3")
        initial_layer12_nodes, initial_layer12_H, initial_layer12_diff, initial_layer2_r2 = self._run_layer12_cycle(
            H_seed=H_init.detach().clone(),
            S_target=S_target,
            E_target=E_target,
            eval_lib_path=eval_lib_path,
            layer1_runtime_cfg={**layer1_runtime_cfg, 'profile': 'initial'},
            layer2_runtime_cfg=layer2_runtime_cfg,
            eval_nmr=True,
            eval_output_dir=(str(Path(output_dir) / 'initial_layer12') if save_intermediates and output_dir else None),
            layer2_output_dir=(str(Path(output_dir) / 'initial_layer12') if save_intermediates and output_dir else None),
        )
        print(f"[Initial Layer2] r2={initial_layer2_r2:.4f}")

        current_nodes = copy.deepcopy(initial_layer12_nodes)
        current_H = initial_layer12_H.detach().clone()
        current_diff = copy.deepcopy(initial_layer12_diff)
        accepted_layer2_r2 = float(initial_layer2_r2)
        historical_best_layer2_r2 = float(initial_layer2_r2)

        try:
            current_nodes, _ = self.layer3_adjust_templates(
                nodes=copy.deepcopy(current_nodes),
                S_target=S_target,
                E_target=E_target,
                lib_path=eval_lib_path,
                output_dir=None,
                **layer3_runtime_cfg,
            )
        except Exception as e:
            print(f"[Initial Layer3] 失败: {e}")

        current_H = self._histogram_from_nodes(current_nodes).detach().clone()
        current_diff = self._compute_difference_spectrum_from_nodes_mu(
            nodes=current_nodes,
            S_target=S_target,
            hwhm=float(layer2_eval_hwhm),
        )
        initial_post_layer3_r2 = float(current_diff.get('r2', float('nan')))
        print(
            f"[Initial Layer3] r2={initial_post_layer3_r2:.4f}"
            if np.isfinite(initial_post_layer3_r2) else
            "[Initial Layer3] r2=nan"
        )

        print("\n>>> Outer Loop(Layer4 -> Layer1-2 -> Layer3)")
        stage3_resource_cycle_rows: List[Dict[str, Any]] = []
        self._final_stage3_completion_diagnostics = {}
        self._final_stage3_terminal_diagnostics = {}
        outer_cycle_count = max(1, int(outer_max_cycles))
        for outer_idx in range(outer_cycle_count):
            print("\n" + "=" * 80)
            print(f"Outer Cycle {outer_idx + 1}/{outer_cycle_count}")
            print("=" * 80)

            cycle_best_layer2_r2 = float(accepted_layer2_r2)
            cycle_pre_layer3_r2 = float('nan')
            def _candidate_structurally_valid(nodes_local: List[_NodeV3],
                                              H_local: torch.Tensor,
                                              tag: str) -> Tuple[bool, str]:
                meta_before_sync = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                try:
                    graph_ok, graph_errors = self.layer1_assigner.validate_graph_consistency(
                        nodes=nodes_local,
                        H=H_local,
                        E_target=E_target,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"[{tag}] 图校验失败: {e}")
                    return False, 'graph_check_exception'
                graph_accept_ok, graph_accept_meta = self.layer1_assigner.assess_consistency_acceptance(
                    list(graph_errors or []),
                    max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
                )
                if not bool(graph_accept_ok):
                    print(f"[{tag}] 回退: 图结构校验未通过，errors={len(graph_errors)}")
                    self.layer1_assigner._print_consistency_errors(graph_errors, prefix=tag, limit=6)
                    return False, 'graph_invalid'
                if (not bool(graph_ok)) and int(graph_accept_meta.get('n_soft', 0)) > 0:
                    print(
                        f"[{tag}] 接受少量1-hop软缺口: "
                        f"soft={int(graph_accept_meta.get('n_soft', 0))}/"
                        f"{int(graph_accept_meta.get('max_soft_errors', 0))}"
                    )
                try:
                    self._sync_fixed_partition_meta_from_nodes(nodes_local)
                except Exception as e:
                    self._set_fixed_partition_meta(meta_before_sync)
                    print(f"[{tag}] 固定锚点元数据同步失败: {e}")
                    return False, 'meta_sync_exception'
                try:
                    hist_diag = self.layer4_adjuster._evaluate_required_hist_constraints(
                        H_local,
                        E_target,
                        S_target=S_target,
                        enforce_h_rel=True,
                    )
                except Exception as e:
                    self._set_fixed_partition_meta(meta_before_sync)
                    print(f"[{tag}] 直方图校验失败: {e}")
                    return False, 'hist_check_exception'
                if not bool(hist_diag.get('ok', False)):
                    self._set_fixed_partition_meta(meta_before_sync)
                    print(f"[{tag}] 回退: 直方图/固定连接校验未通过，reason={hist_diag.get('reason', 'unknown')}")
                    return False, str(hist_diag.get('reason', 'hist_invalid'))
                return True, 'ok'

            def _candidate_hist_feasible(H_local: torch.Tensor, tag: str) -> Tuple[bool, str]:
                try:
                    hist_diag = self.layer4_adjuster._evaluate_required_hist_constraints(
                        H_local,
                        E_target,
                        S_target=S_target,
                        enforce_h_rel=True,
                    )
                except Exception as e:
                    print(f"[{tag}] 直方图预校验失败: {e}")
                    return False, 'hist_precheck_exception'
                if not bool(hist_diag.get('ok', False)):
                    reason = str(hist_diag.get('reason', 'hist_invalid'))
                    print(f"[{tag}] 回退: 直方图/固定连接预校验未通过，reason={reason}")
                    return False, reason
                return True, 'ok'

            # Stage A: compare with current best layer2 R2, reject if not improved.
            block_a_cfg = dict(default_stage_configs['block_a'])
            block_a_cycles = max(1, int(block_a_cfg.pop('max_cycles', 1)))
            for block_a_iter in range(block_a_cycles):
                H_block_a_base = current_H.detach().clone()
                nodes_block_a_base = copy.deepcopy(current_nodes)
                diff_block_a_base = copy.deepcopy(current_diff)
                meta_block_a_base = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                try:
                    H_block_a, moves_a, _meta_a = self.layer4_adjuster.adjust_by_stage(
                        H=H_block_a_base,
                        ppm=diff_block_a_base.get('ppm'),
                        diff=diff_block_a_base.get('diff'),
                        E_target=E_target,
                        S_target=S_target,
                        stage='block_a',
                        nodes=nodes_block_a_base,
                        **block_a_cfg,
                    )
                except Exception as e:
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    H_block_a, moves_a = H_block_a_base, []
                    print(f"[Block A #{block_a_iter + 1}] 失败: {e}")

                if not moves_a:
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    if int(block_a_iter) == 0:
                        print("[Block A] 无有效调整")
                    break

                hist_ok, _hist_reason = _candidate_hist_feasible(H_block_a, tag=f'BlockA #{block_a_iter + 1}')
                if not bool(hist_ok):
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    break

                try:
                    cand_nodes, cand_H, cand_diff, cand_r2 = self._run_layer12_cycle(
                        H_seed=H_block_a.detach().clone(),
                        S_target=S_target,
                        E_target=E_target,
                        eval_lib_path=eval_lib_path,
                        layer1_runtime_cfg={**layer1_runtime_cfg, 'profile': 'rebuild'},
                        layer2_runtime_cfg=layer2_runtime_cfg,
                        # After Layer4 changes the histogram, rebuild 1-hop
                        # from the current constraints.
                        eval_nmr=False,
                    )
                except Exception as e:
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    print(f"[Block A #{block_a_iter + 1}] 回退: layer1/2 failed ({e})")
                    break
                cand_valid, cand_reason = _candidate_structurally_valid(cand_nodes, cand_H, tag='BlockA')
                if cand_valid and float(cand_r2) > float(cycle_best_layer2_r2) + float(outer_improve_eps):
                    current_nodes = cand_nodes
                    current_H = cand_H.detach().clone()
                    current_diff = copy.deepcopy(cand_diff)
                    cycle_pre_layer3_r2 = float(cand_r2)
                    accepted_layer2_r2 = float(cand_r2)
                    historical_best_layer2_r2 = max(float(historical_best_layer2_r2), float(cand_r2))
                    cycle_best_layer2_r2 = float(cand_r2)
                    print(f"[Block A #{block_a_iter + 1}] 接受: layer2_r2={cand_r2:.4f}")
                    self._print_histogram_and_elements(current_H, E_target, label=f'Block A #{block_a_iter + 1} Accepted', nodes=current_nodes)
                    self._print_effective_resource_summary(current_H, label=f'Block A #{block_a_iter + 1} Accepted')
                    continue

                if not bool(cand_valid):
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    print(f"[Block A #{block_a_iter + 1}] 回退: candidate invalid ({cand_reason})")
                    break
                try:
                    self._set_fixed_partition_meta(meta_block_a_base)
                except Exception:
                    pass
                print(
                    f"[Block A #{block_a_iter + 1}] 回退: candidate_r2={cand_r2:.4f} "
                    f"<= accept_ref_layer2_r2={cycle_best_layer2_r2:.4f}"
                )
                break

            # Stage B: evaluate each hetero-anchor substage immediately.  The
            # following substage must see the accepted graph/diff from the
            # previous one; otherwise fixed-anchor slots and target demands drift.
            block_b_cfg = dict(default_stage_configs['block_b'])
            block_b_cycles = max(1, int(block_b_cfg.pop('max_cycles', 1)))
            block_b_peak_rel_threshold = float(block_b_cfg.pop('peak_rel_threshold', 0.01))
            block_b_families = [
                ('ether', ['ether_2829', 'ether', 'ether_count19', 'ether_mode19']),
                ('amine', ['amine', 'amine_count20', 'amine_mode20']),
                ('thioether', ['thioether', 'thio_count19', 'thio_mode19']),
                # SU21 can only attach to one terminal SU32, so there is no
                # single/double count stage for halogenated aliphatic carbon.
                ('halogen', ['halogen', 'halogen_mode21']),
            ]
            for block_b_iter in range(block_b_cycles):
                block_b_progress = False
                block_b_attempted = False
                for family_idx, (family_name, family_substages) in enumerate(block_b_families):
                    for substage_idx, substage in enumerate(family_substages):
                        nodes_stage_base = copy.deepcopy(current_nodes)
                        H_stage_base = current_H.detach().clone()
                        diff_stage_base = copy.deepcopy(current_diff)
                        meta_stage_base = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                        self.layer4_adjuster.fixed_assignment_precheck = lambda H_candidate: (
                            self.layer1_assigner.precheck_fixed_assignment_feasibility(
                                H_candidate.detach().clone(),
                                E_target,
                                build_variant=int(layer1_runtime_cfg.get('build_variant', 0) or 0),
                            )
                        )
                        try:
                            H_stage, moves_b, _meta_b = self.layer4_adjuster.adjust_by_stage(
                                H=H_stage_base,
                                ppm=diff_stage_base.get('ppm'),
                                diff=diff_stage_base.get('diff'),
                                E_target=E_target,
                                S_target=S_target,
                                stage='block_b',
                                nodes=nodes_stage_base,
                                block_b_substage=str(substage),
                                peak_rel_threshold=block_b_peak_rel_threshold,
                                **block_b_cfg,
                            )
                        except Exception as e:
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            H_stage, moves_b = H_stage_base, []
                            print(f"[Block B:{substage} #{block_b_iter + 1}] 失败: {e}")
                            continue

                        if not moves_b and bool(torch.equal(H_stage.detach().cpu(), H_stage_base.detach().cpu())):
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            continue

                        block_b_attempted = True
                        hist_ok, _hist_reason = _candidate_hist_feasible(
                            H_stage,
                            tag=f'BlockB:{family_name}/{substage} #{block_b_iter + 1}',
                        )
                        if not bool(hist_ok):
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            continue

                        fixed_precheck_ok, fixed_precheck_reason = self.layer1_assigner.precheck_fixed_assignment_feasibility(
                            H_stage.detach().clone(),
                            E_target,
                            build_variant=int(layer1_runtime_cfg.get('build_variant', 0) or 0),
                        )
                        if not bool(fixed_precheck_ok):
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            print(
                                f"[Block B:{family_name}/{substage} #{block_b_iter + 1}] "
                                f"前置固定连接预校验拒绝: {fixed_precheck_reason}"
                            )
                            continue

                        try:
                            layer1_stage_cfg = dict(layer1_runtime_cfg)
                            layer1_stage_cfg['profile'] = 'rebuild'
                            try:
                                base_variant = int(layer1_stage_cfg.get('build_variant', 0))
                            except Exception:
                                base_variant = 0
                            layer1_stage_cfg['build_variant'] = (
                                int(base_variant) +
                                101 * int(block_b_iter) +
                                17 * int(family_idx) +
                                int(substage_idx)
                            )
                            cand_nodes, cand_H, cand_diff, cand_r2 = self._run_layer12_cycle(
                                H_seed=H_stage.detach().clone(),
                                S_target=S_target,
                                E_target=E_target,
                                eval_lib_path=eval_lib_path,
                                layer1_runtime_cfg=layer1_stage_cfg,
                                layer2_runtime_cfg=layer2_runtime_cfg,
                                eval_nmr=False,
                            )
                        except Exception as e:
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            print(f"[Block B:{family_name}/{substage} #{block_b_iter + 1}] 回退: layer1/2 failed ({e})")
                            continue

                        cand_valid, cand_reason = _candidate_structurally_valid(
                            cand_nodes,
                            cand_H,
                            tag=f'BlockB:{family_name}/{substage}',
                        )
                        if cand_valid and float(cand_r2) > float(cycle_best_layer2_r2) + float(outer_improve_eps):
                            current_nodes = cand_nodes
                            current_H = cand_H.detach().clone()
                            current_diff = copy.deepcopy(cand_diff)
                            cycle_pre_layer3_r2 = float(cand_r2)
                            accepted_layer2_r2 = float(cand_r2)
                            historical_best_layer2_r2 = max(float(historical_best_layer2_r2), float(cand_r2))
                            cycle_best_layer2_r2 = float(cand_r2)
                            block_b_progress = True
                            print(f"[Block B:{family_name}/{substage} #{block_b_iter + 1}] 接受: layer2_r2={cand_r2:.4f}")
                            self._print_histogram_and_elements(
                                current_H,
                                E_target,
                                label=f'Block B:{family_name}/{substage} #{block_b_iter + 1} Accepted',
                                nodes=current_nodes,
                            )
                            self._print_effective_resource_summary(
                                current_H,
                                label=f'Block B:{family_name}/{substage} #{block_b_iter + 1} Accepted',
                            )
                            continue

                        try:
                            self._set_fixed_partition_meta(meta_stage_base)
                        except Exception:
                            pass
                        if not bool(cand_valid):
                            print(f"[Block B:{family_name}/{substage} #{block_b_iter + 1}] 回退: candidate invalid ({cand_reason})")
                            continue
                        print(
                            f"[Block B:{family_name}/{substage} #{block_b_iter + 1}] 回退: candidate_r2={cand_r2:.4f} "
                            f"<= accept_ref_layer2_r2={cycle_best_layer2_r2:.4f}"
                        )
                if not bool(block_b_progress) and not bool(block_b_attempted):
                    break

            if bool(resource_enabled):
                try:
                    stage3_nodes, stage3_H, stage3_diff, stage3_meta = self._run_topology_stage3_resource_cycles(
                        nodes=copy.deepcopy(current_nodes),
                        H=current_H.detach().clone(),
                        current_diff=copy.deepcopy(current_diff),
                        S_target=S_target,
                        E_target=E_target,
                        eval_lib_path=eval_lib_path,
                        layer1_runtime_cfg={**layer1_runtime_cfg, 'profile': 'rebuild'},
                        layer2_runtime_cfg=layer2_runtime_cfg,
                        layer2_eval_hwhm=float(layer2_eval_hwhm),
                        outer_idx=int(outer_idx),
                        max_stage3_iterations=int(resource_cfg.get('max_stage3_iterations', 20)),
                        max_primary_adjustments_per_stage=int(resource_cfg.get('max_primary_adjustments_per_stage', 10)),
                        max_h_compensation_ops_per_adjustment=int(resource_cfg.get('max_h_compensation_ops_per_adjustment', 8)),
                        residual_flex_preferred_23=int(resource_cfg.get('residual_flex_preferred_23', 3)),
                        residual_flex_min_23=int(resource_cfg.get('residual_flex_min_23', 2)),
                        residual_side_preferred_23=int(resource_cfg.get('residual_side_preferred_23', 2)),
                        residual_side_min_23=int(resource_cfg.get('residual_side_min_23', 1)),
                        h_tolerance=float(resource_cfg.get('h_tolerance', 0.06)),
                        enable_post_external_spectrum=bool(resource_cfg.get('enable_post_external_spectrum', True)),
                        spectrum_candidate_limit=int(resource_cfg.get('spectrum_candidate_limit', 10)),
                        spectrum_max_primary_adjustments=int(resource_cfg.get('spectrum_max_primary_adjustments', 8)),
                        spectrum_min_r2_gain=float(resource_cfg.get('spectrum_min_r2_gain', 0.002)),
                        spectrum_min_window_loss_gain=float(resource_cfg.get('spectrum_min_window_loss_gain', 0.1)),
                        spectrum_max_accepted_rounds=int(resource_cfg.get('spectrum_max_accepted_rounds', 2)),
                        output_dir=(
                            str(Path(output_dir) / f'outer_cycle_{outer_idx + 1}' / 'block_c')
                            if save_round_outputs else None
                        ),
                    )
                    stage3_resource_cycle_rows.extend(
                        self._flatten_stage3_resource_cycles(stage3_meta, outer_idx=int(outer_idx))
                    )
                    current_nodes = copy.deepcopy(stage3_nodes)
                    current_H = stage3_H.detach().clone()
                    current_diff = copy.deepcopy(stage3_diff)
                    self._final_stage3_terminal_diagnostics = dict(
                        stage3_meta.get('terminal_diagnostics', {}) or {}
                    )
                    if str(stage3_meta.get('stop_reason', '') or '') == 'stage3_complete':
                        self._final_stage3_completion_diagnostics = dict(
                            stage3_meta.get('completion_diagnostics', {}) or {}
                        )
                    else:
                        self._final_stage3_completion_diagnostics = {}
                    print(
                        f"[Block C:resource] 完成: accepted_cycles={int(stage3_meta.get('accepted_cycles', 0) or 0)} "
                        f"stop={stage3_meta.get('stop_reason', '')}"
                    )
                except Exception as e:
                    self._final_stage3_completion_diagnostics = {}
                    self._final_stage3_terminal_diagnostics = {}
                    print(f"[Block C:resource] 失败，保持当前状态: {e}")

            try:
                current_nodes, _ = self.layer3_adjust_templates(
                    nodes=copy.deepcopy(current_nodes),
                    S_target=S_target,
                    E_target=E_target,
                    lib_path=eval_lib_path,
                    output_dir=None,
                    **layer3_runtime_cfg,
                )
            except Exception as e:
                print(f"[Outer Cycle {outer_idx + 1}] Layer3 失败: {e}")

            current_H = self._histogram_from_nodes(current_nodes).detach().clone()
            current_diff = self._compute_difference_spectrum_from_nodes_mu(
                nodes=current_nodes,
                S_target=S_target,
                hwhm=float(layer2_eval_hwhm),
            )
            cycle_post_layer3_r2 = float(current_diff.get('r2', float('nan')))
            cycle_pre_layer3_txt = (
                f"{cycle_pre_layer3_r2:.4f}"
                if np.isfinite(cycle_pre_layer3_r2) else
                "n/a"
            )
            cycle_post_layer3_txt = (
                f"{cycle_post_layer3_r2:.4f}"
                if np.isfinite(cycle_post_layer3_r2) else
                "nan"
            )
            print(
                f"[Outer Cycle {outer_idx + 1}] Summary: "
                f"accept_ref_layer2_r2={accepted_layer2_r2:.4f}, "
                f"historical_best_layer2_r2={historical_best_layer2_r2:.4f}, "
                f"pre_layer3_r2={cycle_pre_layer3_txt}, "
                f"post_layer3_r2={cycle_post_layer3_txt}"
            )
            self._print_histogram_and_elements(
                current_H,
                E_target,
                label=f'Outer Cycle {outer_idx + 1} Post-Layer3',
                nodes=current_nodes,
            )
            self._print_effective_resource_summary(
                current_H,
                label=f'Outer Cycle {outer_idx + 1} Post-Layer3',
            )

            if save_round_outputs:
                try:
                    outer_dir = str(Path(output_dir) / f"outer_cycle_{outer_idx + 1}")
                    Path(outer_dir).mkdir(parents=True, exist_ok=True)
                    visualize_su_distribution(
                        current_H.long().detach().cpu(),
                        f'Outer_{outer_idx + 1}',
                        save_dir=outer_dir,
                    )
                    self._save_final_nmr_comparison(
                        nodes=current_nodes,
                        S_target=S_target,
                        E_target=E_target,
                        output_dir=str(Path(outer_dir) / 'nmr_comparison'),
                        hwhm=float(layer2_eval_hwhm),
                        save_full_outputs=False,
                    )
                except Exception:
                    pass

        print(
            "[Final Topology] using final Post-Layer3 node state; no extra Layer1/2 rebuild",
            flush=True,
        )

        nodes = copy.deepcopy(current_nodes)
        H_final = self._histogram_from_nodes(nodes).detach().clone()

        try:
            visualize_su_distribution(H_final.long().detach().cpu(), 'Final', save_dir=output_dir)
        except Exception:
            pass
        try:
            self._save_final_nmr_comparison(
                nodes=nodes,
                S_target=S_target,
                E_target=E_target,
                output_dir=str(Path(output_dir) / 'final_outputs'),
                hwhm=float(layer2_eval_hwhm),
            )
        except Exception as e:
            print(f"[Final NMR Compare] 保存失败: {e}")
        try:
            final_output_dir = Path(output_dir) / 'final_outputs'
            self._save_final_stage3_resource_summary(
                rows=stage3_resource_cycle_rows,
                output_dir=str(final_output_dir),
            )
        except Exception as e:
            print(f"[Final Stage3 Resource Summary] 保存失败: {e}")

        print("\n" + "=" * 80)
        print("最终结果汇总")
        print("=" * 80)
        final_post_layer3_r2 = float(current_diff.get('r2', float('nan')))
        final_post_layer3_txt = (
            f"{final_post_layer3_r2:.4f}"
            if np.isfinite(final_post_layer3_r2) else
            "nan"
        )
        print(
            f"[Final R2] layer2_accept_ref={accepted_layer2_r2:.4f}, "
            f"historical_best_layer2_r2={historical_best_layer2_r2:.4f}, "
            f"final_post_layer3_r2={final_post_layer3_txt}"
        )
        stage3_applied = int(sum(1 for row in stage3_resource_cycle_rows if bool(row.get('applied', False))))
        stage3_last_stop = str(stage3_resource_cycle_rows[-1].get('stage3_stop_reason', '')) if stage3_resource_cycle_rows else ''
        print(
            f"[Final Stage3 Resource] cycles={len(stage3_resource_cycle_rows)} "
            f"accepted={stage3_applied} "
            f"last_stop={stage3_last_stop}"
        )
        self._print_histogram_and_elements(H_final, E_target, label='Final Summary', nodes=nodes)
        self._print_final_resource_summary(nodes, S_target, E_target)
        self._print_final_topology_specs(
            nodes,
            S_target=S_target,
            E_target=E_target,
        )
        print("\n" + "=" * 80)
        print("逆向推理完成！")
        print("=" * 80)

        return nodes, H_final

    def _compute_difference_spectrum_from_nodes_mu(self,
                                                    nodes: List[_NodeV3],
                                                    S_target: torch.Tensor,
                                                    hwhm: float) -> Dict[str, object]:
        device = self.device
        S_target = S_target.to(device).flatten()
        ppm_axis = PPM_AXIS.to(device).flatten()

        S_recon = self.reconstruct_spectrum(nodes, hwhm=float(hwhm)).to(device).flatten()
        eval_info = evaluate_spectrum_reconstruction(
            S_target,
            S_recon,
            ppm_axis=ppm_axis,
            fit_scale=True,
            nonnegative_alpha=True,
        )
        ppm_eval = eval_info.get('ppm_axis', ppm_axis)
        S_target_eval = eval_info['S_target']
        S_fit = eval_info['S_fit']
        diff = (S_target_eval - S_fit).detach().cpu().numpy()

        return {
            'ppm': ppm_eval.detach().cpu().numpy(),
            'diff': diff,
            'r2': float(eval_info.get('r2', 0.0)),
            'alpha': float(eval_info.get('alpha', 1.0)),
        }

    def _materialize_layer1_library_baseline(self,
                                             nodes: List[_NodeV3],
                                             S_target: torch.Tensor,
                                             lib_path: Optional[str],
                                             hwhm: float,
                                             allow_approx: bool) -> Tuple[List[_NodeV3], Dict[str, object]]:
        baseline_nodes = copy.deepcopy(nodes)
        try:
            snapshot = self.layer1_assigner._nmr_evaluator.build_eval_snapshot(
                nodes=baseline_nodes,
                S_target=S_target,
                lib_path=lib_path,
                hwhm=float(hwhm),
                allow_approx=bool(allow_approx),
            )
            assignments = dict(snapshot.get('assignments', {}) or {})
            for node in baseline_nodes:
                rec = assignments.get(int(getattr(node, 'global_id', -1)))
                if not isinstance(rec, dict):
                    continue
                mu = rec.get('mu')
                pi = rec.get('pi')
                if mu is None or pi is None:
                    continue
                try:
                    node.mu = float(mu)
                    node.pi = float(max(1e-6, float(pi)))
                    if isinstance(getattr(node, 'score_components', None), dict):
                        node.score_components['layer1_library_baseline'] = True
                except Exception:
                    continue
            diff = {
                'ppm': np.asarray(snapshot.get('ppm', []), dtype=np.float64),
                'diff': np.asarray(snapshot.get('diff', []), dtype=np.float64),
                'r2': float(snapshot.get('r2', 0.0)),
                'alpha': float(snapshot.get('alpha', 1.0)),
                'matched_ratio': float(snapshot.get('matched_ratio', 0.0)),
                'matched_cnt': int(snapshot.get('matched_cnt', 0)),
                'carbon_cnt': int(snapshot.get('carbon_cnt', 0)),
                'layer1_library_baseline': True,
            }
            print(
                "  [Layer1 library baseline] "
                f"r2={float(diff.get('r2', 0.0)):.4f} "
                f"matched={int(diff.get('matched_cnt', 0))}/{int(diff.get('carbon_cnt', 0))}"
            )
            return baseline_nodes, diff
        except Exception as e:
            print(f"  [Layer1 library baseline失败] {e}")
            diff = self._compute_difference_spectrum_from_nodes_mu(
                nodes=baseline_nodes,
                S_target=S_target,
                hwhm=float(hwhm),
            )
            diff['layer1_library_baseline'] = False
            return baseline_nodes, diff

    def _histogram_from_nodes(self, nodes: List[_NodeV3]) -> torch.Tensor:
        device = self.device
        H = torch.zeros(NUM_SU_TYPES, dtype=torch.long, device=device)
        for n in nodes:
            su = int(n.su_type)
            if 0 <= su < NUM_SU_TYPES:
                H[su] += 1
        return H
    
    def estimate_su_histogram(self, S_target: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """委托给Layer0Estimator执行SU直方图估计"""
        return self.layer0_estimator.estimate_su_histogram(S_target, E_target)
    
    def layer1_assign(self, H_init: torch.Tensor, S_target: torch.Tensor, E_target: torch.Tensor,
                      eval_nmr: bool = True, eval_output_dir: str = 'inverse_result',
                      eval_lib_path: Optional[str] = None, eval_hwhm: float = 1.0,
                      eval_allow_approx: bool = True,
                      build_variant: int = 0,
                      enable_carbonyl_joint_adjust: bool = True,
                      carbonyl_joint_iterations: int = 5,
                      carbonyl_joint_max_adjustments: int = 4,
                      carbonyl_joint_pos_threshold: float = 0.1,
                      carbonyl_joint_neg_threshold: float = 0.1,
                      enable_hop1_adjust: bool = True,
                      hop1_adjust_iterations: int = 4,
                      hop1_neg_threshold: float = -0.7,
                      hop1_pos_threshold: float = 0.7,
                      hop1_adjust_max_candidate_trials: int = 240,
                      hop1_adjust_max_candidate_nodes_per_su: int = 24,
                      hop1_adjust_max_alternatives_per_node: int = 4,
                      hop1_adjust_region_order: Optional[List[str]] = None,
                      hop1_adjust_max_moves_per_region: int = 8,
                      hop1_adjust_max_full_evals: int = 32,
                      profile: str = 'initial',
                      local_swap_max_attempts: int = 64,
                      initial_enable_spectral_refine: bool = True,
                      initial_hop1_adjust_iterations: int = 6,
                      initial_hop1_adjust_max_candidate_trials: int = 240) -> List[_NodeV3]:
        """委托给Layer1Assigner执行1-hop分配"""
        eval_lib_path = eval_lib_path or self.default_template_lib_path
        return self.layer1_assigner.layer1_assign(
            H_init=H_init, S_target=S_target, E_target=E_target,
            eval_nmr=eval_nmr, eval_output_dir=eval_output_dir,
            eval_lib_path=eval_lib_path, eval_hwhm=eval_hwhm,
            eval_allow_approx=eval_allow_approx,
            build_variant=int(build_variant),
            enable_carbonyl_joint_adjust=enable_carbonyl_joint_adjust,
            carbonyl_joint_iterations=carbonyl_joint_iterations,
            carbonyl_joint_max_adjustments=carbonyl_joint_max_adjustments,
            carbonyl_joint_pos_threshold=carbonyl_joint_pos_threshold,
            carbonyl_joint_neg_threshold=carbonyl_joint_neg_threshold,
            enable_hop1_adjust=enable_hop1_adjust,
            hop1_adjust_iterations=hop1_adjust_iterations,
            hop1_neg_threshold=hop1_neg_threshold,
            hop1_pos_threshold=hop1_pos_threshold,
            hop1_adjust_max_candidate_trials=hop1_adjust_max_candidate_trials,
            hop1_adjust_max_candidate_nodes_per_su=hop1_adjust_max_candidate_nodes_per_su,
            hop1_adjust_max_alternatives_per_node=hop1_adjust_max_alternatives_per_node,
            hop1_adjust_region_order=hop1_adjust_region_order,
            hop1_adjust_max_moves_per_region=hop1_adjust_max_moves_per_region,
            hop1_adjust_max_full_evals=hop1_adjust_max_full_evals,
            profile=profile,
            local_swap_max_attempts=local_swap_max_attempts,
            initial_enable_spectral_refine=initial_enable_spectral_refine,
            initial_hop1_adjust_iterations=initial_hop1_adjust_iterations,
            initial_hop1_adjust_max_candidate_trials=initial_hop1_adjust_max_candidate_trials,
        )
    
    def _compute_layer1_difference_spectrum(self, nodes: List[_NodeV3], S_target: torch.Tensor,
                                            lib_path: Optional[str], hwhm: float,
                                            allow_approx: bool) -> Dict[str, object]:
        """委托给Layer1Assigner计算差谱"""
        lib_path = lib_path or self.default_template_lib_path
        return self.layer1_assigner._compute_layer1_difference_spectrum(
            nodes=nodes, S_target=S_target, lib_path=lib_path,
            hwhm=hwhm, allow_approx=allow_approx
        )
    
    def evaluate_layer1_nmr_with_library(self, nodes: List[_NodeV3], S_target: torch.Tensor,
                                         lib_path: Optional[str] = None, output_dir: str = 'inverse_result',
                                         hwhm: float = 1.0, allow_approx: bool = True) -> Dict[str, float]:
        """委托给Layer1Assigner评估Layer1 NMR"""
        lib_path = lib_path or self.default_template_lib_path
        return self.layer1_assigner.evaluate_layer1_nmr_with_library(
            nodes=nodes, S_target=S_target, lib_path=lib_path,
            output_dir=output_dir, hwhm=hwhm, allow_approx=allow_approx
        )

    # ========================================================================
    # Layer2: 2-hop分配（改进版）
    # ========================================================================
    
    def layer2_assign(self, nodes: List[_NodeV3],
                      S_target: torch.Tensor, E_target: torch.Tensor,
                      lib_path: Optional[str] = None,
                      output_dir: Optional[str] = None,
                      eval_hwhm: float = 1.0,
                      layer2_max_hop2_diff_nodes: int = 2,
                      layer2_use_g2s_decoder: bool = False,
                      layer2_decoder_max_mu_delta: float = 8.0) -> List[_NodeV3]:
        """
        Layer2: 委托给Layer2Estimator进行2-hop推导和模板检索
        
        功能：
        1. 从1-hop邻居推导2-hop邻居
        2. 模板精确匹配和近似匹配
        3. z向量初始化和mu/pi解码
        4. NMR谱图重建和评估
        """
        lib_path = lib_path or self.default_template_lib_path
        if lib_path:
            try:
                if getattr(self.layer2_estimator, 'lib_path', None) != lib_path:
                    self.layer2_estimator.lib_path = lib_path
                    if hasattr(self.layer2_estimator, '_template_cache'):
                        self.layer2_estimator._template_cache = None
            except Exception:
                pass
        # 委托给Layer2Estimator
        return self.layer2_estimator.layer2_assign(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            output_dir=output_dir,
            hwhm=eval_hwhm,
            layer2_max_hop2_diff_nodes=int(layer2_max_hop2_diff_nodes),
            layer2_use_g2s_decoder=bool(layer2_use_g2s_decoder),
            layer2_decoder_max_mu_delta=float(layer2_decoder_max_mu_delta),
        )

    def _share_layer2_template_cache_with_layer3(self, lib_path: Optional[str]) -> bool:
        resolved_path = str(lib_path or self.default_template_lib_path or "")
        layer2_path = str(getattr(self.layer2_estimator, "lib_path", "") or "")
        layer2_cache = getattr(self.layer2_estimator, "_template_cache", None)
        if not resolved_path or layer2_path != resolved_path or not isinstance(layer2_cache, dict):
            return False
        self.layer3_estimator.lib_path = resolved_path
        self.layer3_estimator._template_cache = layer2_cache
        return True
    
    # ========================================================================
    # Layer3: z调优（改进版）
    # ========================================================================
    
    def layer3_adjust_templates(self,
                                nodes: List[_NodeV3],
                                S_target: torch.Tensor,
                                E_target: torch.Tensor,
                                max_iters: int = 150,
                                lib_path: Optional[str] = None,
                                output_dir: Optional[str] = None,
                                hwhm: float = 1.0,
                                pos_search_window: float = 15.0,
                                neg_assign_window: float = 2.0,
                                top_k_samples: int = 8,
                                neg_top_k_peaks: int = 10,
                                neg_peak_min_sep_ppm: float = 0.8,
                                enable_approx_hop2_template_adjust: bool = True,
                                approx_hop2_max_iters: Optional[int] = None,
                                approx_hop2_max_diff_nodes: int = 5,
                                approx_hop2_top_k_templates: int = 80,
                                layer3_use_g2s_decoder: bool = False,
                                layer3_verbose: bool = True) \
                                -> Tuple[List[_NodeV3], float]:
        """
        Layer3: 通过调整z向量优化谱图匹配
        
        改进点：
        1. 分段优先优化：集中火力攻克尖峰区域
        2. 候选池扩展：增加多样性采样
        3. 早停机制：R²达标提前终止
        """
        return self.layer3_adjust_templates_impl(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            max_iters=int(max_iters),
            lib_path=lib_path,
            output_dir=output_dir,
            hwhm=float(hwhm),
            pos_search_window=float(pos_search_window),
            neg_assign_window=float(neg_assign_window),
            top_k_samples=int(top_k_samples),
            neg_top_k_peaks=int(neg_top_k_peaks),
            neg_peak_min_sep_ppm=float(neg_peak_min_sep_ppm),
            enable_approx_hop2_template_adjust=bool(enable_approx_hop2_template_adjust),
            approx_hop2_max_iters=(int(approx_hop2_max_iters) if approx_hop2_max_iters is not None else None),
            approx_hop2_max_diff_nodes=int(approx_hop2_max_diff_nodes),
            approx_hop2_top_k_templates=int(approx_hop2_top_k_templates),
            use_g2s_decoder=bool(layer3_use_g2s_decoder),
            verbose=bool(layer3_verbose),
        )

    def layer3_adjust_templates_impl(
        self,
        nodes: List[_NodeV3],
        S_target: torch.Tensor,
        E_target: torch.Tensor,
        max_iters: int = 150,
        lib_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        hwhm: float = 1.0,
        pos_search_window: float = 15.0,
        neg_assign_window: float = 2.0,
        top_k_samples: int = 8,
        neg_top_k_peaks: int = 10,
        neg_peak_min_sep_ppm: float = 0.8,
        enable_approx_hop2_template_adjust: bool = True,
        approx_hop2_max_iters: Optional[int] = None,
        approx_hop2_max_diff_nodes: int = 5,
        approx_hop2_top_k_templates: int = 80,
        use_g2s_decoder: bool = False,
        verbose: bool = True,
    ) -> Tuple[List[_NodeV3], float]:
        lib_path = lib_path or self.default_template_lib_path
        if lib_path:
            try:
                if getattr(self.layer3_estimator, 'lib_path', None) != lib_path:
                    self.layer3_estimator.lib_path = lib_path
                    if hasattr(self.layer3_estimator, '_template_cache'):
                        self.layer3_estimator._template_cache = None
            except Exception:
                pass
        shared_template_cache = self._share_layer2_template_cache_with_layer3(lib_path)
        if verbose:
            print(
                "[Layer3 template library] "
                + ("reusing Layer2 in-memory cache" if shared_template_cache else "loading Layer3 cache")
            )
        return self.layer3_estimator.layer3_adjust_templates(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            max_iters=int(max_iters),
            hwhm=float(hwhm),
            output_dir=output_dir,
            pos_search_window=float(pos_search_window),
            neg_assign_window=float(neg_assign_window),
            top_k_samples=int(top_k_samples),
            neg_top_k_peaks=int(neg_top_k_peaks),
            neg_peak_min_sep_ppm=float(neg_peak_min_sep_ppm),
            enable_approx_hop2_template_adjust=bool(enable_approx_hop2_template_adjust),
            approx_hop2_max_iters=(int(approx_hop2_max_iters) if approx_hop2_max_iters is not None else None),
            approx_hop2_max_diff_nodes=int(approx_hop2_max_diff_nodes),
            approx_hop2_top_k_templates=int(approx_hop2_top_k_templates),
            use_g2s_decoder=bool(use_g2s_decoder),
            verbose=bool(verbose),
        )

    def _save_final_nmr_comparison(self,
                                   nodes: List[_NodeV3],
                                   S_target: torch.Tensor,
                                   E_target: torch.Tensor,
                                   output_dir: str,
                                   hwhm: float = 1.0,
                                   save_full_outputs: bool = True) -> None:
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        diff_info = self._compute_difference_spectrum_from_nodes_mu(
            nodes=nodes,
            S_target=S_target,
            hwhm=float(hwhm),
        )
        ppm_np = np.asarray(diff_info.get('ppm', []), dtype=np.float64)
        alpha = float(diff_info.get('alpha', 1.0))
        S_target_np = np.asarray(S_target.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        S_recon_np = self.reconstruct_spectrum(
            nodes,
            hwhm=float(hwhm),
        ).detach().cpu().numpy().astype(np.float64).reshape(-1)
        n = min(len(ppm_np), len(S_target_np), len(S_recon_np))
        ppm_np = ppm_np[:n]
        S_target_np = S_target_np[:n]
        S_fit_np = (alpha * S_recon_np[:n]).astype(np.float64)

        pd.DataFrame({
            'ppm': ppm_np,
            'target': S_target_np,
            'reconstructed': S_fit_np,
        }).to_csv(final_dir / 'final_nmr_comparison.csv', index=False)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.fill_between(ppm_np, 0, S_target_np, alpha=0.25, color='#1f77b4', label='Target')
            ax.fill_between(ppm_np, 0, S_fit_np, alpha=0.25, color='#ff7f0e', label='Reconstructed')
            ax.plot(ppm_np, S_target_np, lw=1.8, color='#1f77b4', label='Target')
            ax.plot(ppm_np, S_fit_np, lw=1.8, color='#ff7f0e', linestyle='--', label='Reconstructed')
            ax.invert_xaxis()
            ax.set_xlabel('Chemical Shift (ppm)')
            ax.set_ylabel('Intensity')
            ax.set_title('Final Target vs Reconstructed NMR')
            ax.legend(frameon=False)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(final_dir / 'final_nmr_comparison.png', dpi=300)
            plt.close(fig)
        except Exception:
            pass

        if bool(save_full_outputs):
            try:
                self._save_final_node_exports(nodes=nodes, output_dir=str(final_dir))
            except Exception as e:
                print(f"[Final Node Export] 保存失败: {e}")
            try:
                self._save_final_result_summary(
                    nodes=nodes,
                    E_target=E_target,
                    S_target=S_target,
                    output_dir=str(final_dir),
                )
            except Exception as e:
                print(f"[Final Result Summary] 保存失败: {e}")

    def _build_final_node_rows(self, nodes: List[_NodeV3]) -> List[Dict[str, Any]]:
        su_names = [name for name, _ in SU_DEFS]
        rows: List[Dict[str, Any]] = []
        for n in nodes:
            su_idx = int(getattr(n, 'su_type', -1))
            hop1_ms = tuple(multiset_from_counter(getattr(n, 'hop1_su', {}) or {}))
            hop2_ms = tuple(multiset_from_counter(getattr(n, 'hop2_su', {}) or {}))
            hop1_ids = [int(x) for x in list(getattr(n, 'hop1_ids', []) or [])]
            fixed_ids = sorted(int(x) for x in set(getattr(n, 'fixed_hop1_ids', set()) or set()))
            score_components = dict(getattr(n, 'score_components', {}) or {})
            z_vec = getattr(n, 'z_vec', None)
            z_norm = None
            z_head4 = None
            if isinstance(z_vec, torch.Tensor) and int(z_vec.numel()) > 0:
                try:
                    z_norm = float(z_vec.detach().float().norm().item())
                    z_head4 = ",".join(
                        f"{float(v):.6f}" for v in z_vec.detach().flatten()[:4].cpu().tolist()
                    )
                except Exception:
                    z_norm = None
                    z_head4 = None
            rows.append({
                'global_id': int(getattr(n, 'global_id', -1)),
                'su_type': int(su_idx),
                'su_name': su_names[su_idx] if 0 <= su_idx < len(su_names) else str(su_idx),
                'special_anchor_partition': str(getattr(n, 'special_anchor_partition', '')) if getattr(n, 'special_anchor_partition', None) is not None else '',
                'special_anchor_mode': str(getattr(n, 'special_anchor_mode', '')) if getattr(n, 'special_anchor_mode', None) is not None else '',
                'target_hop1_degree': int(getattr(n, 'target_hop1_degree', -1)) if getattr(n, 'target_hop1_degree', None) is not None else None,
                'target_fixed_anchor_count': int(getattr(n, 'target_fixed_anchor_count', -1)) if getattr(n, 'target_fixed_anchor_count', None) is not None else None,
                'actual_hop1_degree': int(sum((getattr(n, 'hop1_su', {}) or {}).values())) if getattr(n, 'hop1_su', None) is not None else 0,
                'hop1_ids': str(hop1_ids),
                'hop1_ms': str(list(hop1_ms)),
                'hop2_ms': str(list(hop2_ms)),
                'mu': float(getattr(n, 'mu', 0.0) or 0.0),
                'pi': float(getattr(n, 'pi', 0.0) or 0.0),
                'template_key': str(getattr(n, 'template_key', None)),
            })
        return rows

    def _save_final_node_exports(self, nodes: List[_NodeV3], output_dir: str) -> None:
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

        rows = self._build_final_node_rows(nodes)
        node_columns = [
            'global_id', 'su_type', 'su_name',
            'special_anchor_partition', 'special_anchor_mode',
            'target_hop1_degree', 'target_fixed_anchor_count', 'actual_hop1_degree',
            'hop1_ids', 'hop1_ms', 'hop2_ms',
            'mu', 'pi', 'template_key',
        ]
        df_nodes = pd.DataFrame(rows)
        if not df_nodes.empty:
            df_nodes.sort_values(['global_id', 'su_type'], kind='stable').to_csv(
                final_dir / 'final_nodes_by_global_id.csv', index=False
            )
            df_nodes.sort_values(['su_type', 'global_id'], kind='stable').to_csv(
                final_dir / 'final_nodes_by_su_then_global.csv', index=False
            )
        else:
            pd.DataFrame(columns=node_columns).to_csv(final_dir / 'final_nodes_by_global_id.csv', index=False)
            pd.DataFrame(columns=node_columns).to_csv(final_dir / 'final_nodes_by_su_then_global.csv', index=False)

        su_names = [name for name, _ in SU_DEFS]
        counts = [0] * len(su_names)
        for n in nodes:
            try:
                su_idx = int(getattr(n, 'su_type', -1))
            except Exception:
                continue
            if 0 <= su_idx < len(counts):
                counts[su_idx] += 1
        df_dist = pd.DataFrame([
            {
                'su_type': int(idx),
                'su_name': su_names[idx] if 0 <= idx < len(su_names) else str(idx),
                'count': int(cnt),
            }
            for idx, cnt in enumerate(counts)
        ])
        df_dist.sort_values(['su_type'], kind='stable').to_csv(
            final_dir / 'final_su_distribution.csv', index=False
        )

    def _save_final_result_summary(self,
                                   nodes: List[_NodeV3],
                                   E_target: torch.Tensor,
                                   S_target: Optional[torch.Tensor],
                                   output_dir: str) -> None:
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

        H_final = self._histogram_from_nodes(nodes).detach().cpu().long()
        E_pred_effective = get_effective_nodes_element_vector(
            nodes,
            self.E_SU.detach().cpu(),
            device=torch.device('cpu'),
        )
        E_pred_raw = get_effective_hist_element_vector(
            H_final,
            E_SU_tensor=self.E_SU.detach().cpu(),
            device=torch.device('cpu'),
        )
        e_target = E_target.detach().cpu().float().flatten()
        elem_names = ['C', 'H', 'O', 'N', 'S', 'X']
        rows = []
        for idx, name in enumerate(elem_names):
            tgt = float(e_target[idx].item()) if idx < int(e_target.numel()) else 0.0
            pred = float(E_pred_effective[idx].item()) if idx < int(E_pred_effective.numel()) else 0.0
            raw_pred = float(E_pred_raw[idx].item()) if idx < int(E_pred_raw.numel()) else 0.0
            diff = float(pred - tgt)
            rel = float(abs(diff) / tgt) if tgt > 1e-8 else 0.0
            rows.append({
                'element': str(name),
                'predicted': float(pred),
                'raw_hist_predicted': float(raw_pred),
                'target': float(tgt),
                'diff': float(diff),
                'rel_error': float(rel),
            })
        pd.DataFrame(rows).to_csv(final_dir / 'final_element_composition.csv', index=False)

        try:
            topology_rows, topology_summary = self._final_topology_rows(
                nodes,
                S_target=S_target,
                E_target=E_target,
            )
            debug_dir = final_dir / 'debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            topology_columns = [
                'actual_su', 'gid', 'product_type', 'required_slots_satisfied',
                'source_fragment_ids', 'fixed_source_node_ids', 'claimed_resource_ids',
            ]
            pd.DataFrame(
                self._compact_topology_export_rows(topology_rows),
                columns=topology_columns,
            ).to_csv(final_dir / 'final_topology_specs.csv', index=False)
            with open(debug_dir / 'final_topology_slot_details.jsonl', 'w', encoding='utf-8') as handle:
                for row in topology_rows:
                    handle.write(json.dumps(dict(row or {}), ensure_ascii=False, default=str) + '\n')
            topo_diag = dict(topology_summary.get('diagnostics', {}) or {})
            topo_scalar = {
                'ok': bool(topology_summary.get('ok', False)),
                'topology_plan_feasible': bool(topology_summary.get('topology_plan_feasible', False)),
                'required_slots_satisfied_count': int(topology_summary.get('closed_topology_count', 0)),
                'final_topology_count': int(topology_summary.get('final_topology_count', 0)),
                'pending_closure_total_count': int(topology_summary.get('pending_closure_total_count', 0)),
                'solver_reason': str(topo_diag.get('solver_reason', '')),
                'stage3_terminal_stop_reason': str(
                    topo_diag.get('stage3_terminal_stop_reason', '') or ''
                ),
                'stage3_terminal_phase': str(
                    topo_diag.get('stage3_terminal_phase', '') or ''
                ),
                'stage3_terminal_snapshot_match': bool(
                    topo_diag.get('stage3_terminal_snapshot_match', False)
                ),
            }
            resource_ledger = dict(topo_diag.get('resource_ledger', {}) or {})
            initial_counts = {
                str(key): int(value)
                for key, value in dict(resource_ledger.get('initial_counts', {}) or {}).items()
            }
            topology_consumed_counts = {
                str(key): int(value)
                for key, value in dict(resource_ledger.get('topology_consumed_counts', {}) or {}).items()
            }
            residual_consumed_counts = {
                str(key): int(value)
                for key, value in dict(resource_ledger.get('residual_consumed_counts', {}) or {}).items()
            }
            remaining_counts = {
                str(key): int(value)
                for key, value in dict(resource_ledger.get('remaining_counts', {}) or {}).items()
            }
            closure_keys = ('11', '22', '23', '24')
            aromatic_donor_keys = ('10', '12', '13')
            topo_scalar.update({
                'resource_snapshot_stage': str(resource_ledger.get('snapshot_stage', '')),
                'initial_closure_resource_count': int(sum(
                    initial_counts.get(key, 0) for key in closure_keys
                )),
                'topology_consumed_closure_resource_count': int(sum(
                    topology_consumed_counts.get(key, 0) for key in closure_keys
                )),
                'residual_consumed_closure_resource_count': int(sum(
                    residual_consumed_counts.get(key, 0) for key in closure_keys
                )),
                'remaining_closure_resource_count': int(sum(
                    remaining_counts.get(key, 0) for key in closure_keys
                )),
                'remaining_aromatic_donor_count': int(sum(
                    remaining_counts.get(key, 0) for key in aromatic_donor_keys
                )),
                'remaining_resource_pool_count': int(sum(remaining_counts.values())),
            })
            unclosed_rows = list(topology_summary.get('unclosed_candidates', []) or [])
            pending_columns = [
                'actual_su', 'gid', 'pending_kind', 'phase', 'contract_id',
                'source_fragment_ids', 'fixed_source_node_ids', 'required_counts',
                'available_counts', 'reason',
            ]
            pd.DataFrame(
                self._compact_pending_export_rows(unclosed_rows),
                columns=pending_columns,
            ).to_csv(
                final_dir / 'final_topology_pending.csv',
                index=False,
            )
            (final_dir / 'final_topology_unclosed_candidates.csv').unlink(missing_ok=True)
            with open(debug_dir / 'final_topology_pending_details.jsonl', 'w', encoding='utf-8') as handle:
                for row in unclosed_rows:
                    handle.write(json.dumps(dict(row or {}), ensure_ascii=False, default=str) + '\n')
            ownership_rows, ownership_conflicts = self._topology_ownership_rows(
                nodes=nodes,
                topology_rows=topology_rows,
                pending_rows=unclosed_rows,
                summary=topology_summary,
            )
            pd.DataFrame(ownership_rows).to_csv(
                final_dir / 'final_topology_node_ownership.csv',
                index=False,
            )
            resource_rows = [
                {
                    'node_id': int(row.get('node_id', -1)),
                    'su_type': int(row.get('su_type', -1)),
                    'state': str(row.get('state', '')),
                }
                for row in ownership_rows
                if bool(row.get('resource_pool', False))
            ]
            pd.DataFrame(resource_rows, columns=['node_id', 'su_type', 'state']).to_csv(
                final_dir / 'final_topology_resource_pool.csv',
                index=False,
            )
            topo_scalar.update({
                'total_node_count': int(len(ownership_rows)),
                'ownership_conflict_count': int(len(ownership_conflicts)),
                'unclassified_or_conflict_count': int(sum(
                    1 for row in ownership_rows if row.get('state') == 'unclassified_or_conflict'
                )),
                'remaining_pool_node_count': int(len(resource_rows)),
                'coverage_valid': bool(topology_summary.get('coverage_valid', False)),
            })
            pd.DataFrame([topo_scalar]).to_csv(final_dir / 'final_topology_summary.csv', index=False)
            with open(debug_dir / 'final_topology_diagnostics.json', 'w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'summary': topology_summary,
                        'ownership_conflicts': ownership_conflicts,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            reject_rows = list(topology_summary.get('classification_rejects', []) or [])
            reject_columns = sorted({str(key) for row in reject_rows for key in dict(row or {}).keys()}) or [
                'node_id', 'su_type', 'family', 'reason'
            ]
            pd.DataFrame(reject_rows, columns=reject_columns).to_csv(
                final_dir / 'final_topology_classification_rejects.csv',
                index=False,
            )
        except Exception as e:
            print(f"[Final Topology Export] 保存失败: {e}")
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def reconstruct_spectrum(self, nodes: List[_NodeV3], 
                             hwhm: float = 1.0) -> torch.Tensor:
        """从节点列表重构NMR谱图"""
        device = self.device
        ppm_axis = PPM_AXIS.to(device)

        mus = []
        pis = []
        for n in nodes:
            try:
                center_su = int(n.su_type)
            except Exception:
                continue
            try:
                is_carbon = float(self.E_SU[center_su, 0].detach().cpu().item()) > 0
            except Exception:
                is_carbon = False
            if not bool(is_carbon):
                continue

            try:
                mu = float(getattr(n, 'mu', 0.0))
                pi = float(getattr(n, 'pi', 0.0))
            except Exception:
                continue
            if float(pi) <= 0.0:
                continue
            if float(mu) == 0.0:
                continue
            mus.append(float(mu))
            if bool(getattr(self, 'unit_peak_intensity', False)):
                pis.append(1.0)
            else:
                pis.append(float(pi))

        if not mus:
            return torch.zeros_like(ppm_axis)

        mu_t = torch.tensor(mus, dtype=torch.float, device=device)
        pi_t = torch.tensor(pis, dtype=torch.float, device=device)
        try:
            s = float(getattr(self, 'nmr_intensity_scale', 1.0))
        except Exception:
            s = 1.0
        if float(s) != 1.0:
            pi_t = pi_t * float(s)
        if bool(getattr(self, 'unit_peak_intensity', False)):
            pi_t = torch.ones_like(pi_t)
        return lorentzian_spectrum(mu_t, pi_t, ppm_axis, hwhm=float(hwhm))

# ============================================================================
# 工具函数（用于外部调用）
# ============================================================================

def _extract_checkpoint_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return dict(ckpt["model_state_dict"]), dict(ckpt)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return dict(ckpt["state_dict"]), dict(ckpt)
    return dict(ckpt), {"legacy_state_dict_only": True}


def _normalize_g2s_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in dict(state_dict or {}).items():
        key_s = str(key)
        if key_s.startswith("decoder.decoder."):
            key_s = "decoder." + key_s[len("decoder.decoder."):]
        normalized[key_s] = value
    return normalized


def _g2s_predict_intensity_from_state_dict(state_dict: Dict[str, torch.Tensor],
                                           fallback: bool = False) -> bool:
    weight = state_dict.get("decoder.net.5.weight")
    if weight is None:
        return bool(fallback)
    try:
        return bool(int(weight.shape[0]) > 1)
    except Exception:
        return bool(fallback)


def load_pipeline(s2n_ckpt: str, vae_ckpt: str, templates_pkl: str,
                  device: str = 'cuda') -> InversePipelineV3:
    """加载完整推理管道，兼容新旧 S2N/G2S checkpoint 格式。"""
    s2n_ckpt_obj = torch.load(s2n_ckpt, map_location=device, weights_only=False)
    s2n_state, s2n_meta = _extract_checkpoint_state_dict(s2n_ckpt_obj)
    s2n_model = S2NModel.from_config(S2NConfig.from_dict(s2n_meta.get("config")))
    s2n_model.load_state_dict(s2n_state)

    vae_ckpt_obj = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae_state_raw, vae_meta = _extract_checkpoint_state_dict(vae_ckpt_obj)
    vae_state = _normalize_g2s_state_dict(vae_state_raw)
    g2s_cfg = G2SConfig.from_dict(vae_meta.get("config"))
    g2s_cfg.predict_intensity = _g2s_predict_intensity_from_state_dict(
        vae_state,
        fallback=bool(getattr(g2s_cfg, "predict_intensity", False)),
    )
    vae_model = NMR_VAE.from_config(g2s_cfg)
    vae_model.load_state_dict(vae_state)

    template_path = str(Path(templates_pkl))
    pipeline = InversePipelineV3(s2n_model, vae_model, template_path, device)
    return pipeline


def read_spectrum_csv(path: str) -> torch.Tensor:
    """从CSV读取谱图数据"""
    df = pd.read_csv(path, sep=r'[;, \t]', engine='python', header=None)
    if df.shape[1] < 2:
        raise ValueError("CSV需要两列: ppm, intensity")
    ppm = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    intensity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    return resample_spectrum_to_ppm_axis(ppm, intensity, ppm_axis=PPM_AXIS)


def parse_elements(expr: str) -> torch.Tensor:
    """解析元素表达式，如 'C=100 H=150 O=10 N=2'"""
    import re
    matches = dict(re.findall(r"([CHONSX])\s*=\s*(\d+)", expr.upper()))
    return torch.tensor([int(matches.get(sym, 0)) for sym in ['C', 'H', 'O', 'N', 'S', 'X']], 
                       dtype=torch.float)


# ============================================================================
# 命令行接口（用于测试）
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='InversePipelineV3: 改进的逆向推理')
    parser.add_argument('--s2n_ckpt', type=str, required=True, help='S2N模型检查点路径')
    parser.add_argument('--vae_ckpt', type=str, required=True, help='VAE模型检查点路径')
    parser.add_argument('--templates', type=str, required=True, help='模板库pkl路径')
    parser.add_argument('--spectrum', type=str, required=True, help='目标谱图CSV路径')
    parser.add_argument('--elements', type=str, required=True, help='元素组成，如 C=100 H=150 O=10')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--output', type=str, default='inverse_result', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载管道
    print(f"加载模型: S2N={args.s2n_ckpt}, VAE={args.vae_ckpt}")
    pipeline = load_pipeline(args.s2n_ckpt, args.vae_ckpt, args.templates, args.device)
    
    # 读取输入
    print(f"读取谱图: {args.spectrum}")
    S_target_raw = read_spectrum_csv(args.spectrum)
    
    print(f"解析元素: {args.elements}")
    E_target = parse_elements(args.elements)
    S_target = normalize_spectrum_to_carbon_count(S_target_raw, float(E_target[0].item()))
    
    print(f"目标谱图维度: {S_target.shape}, 元素组成: {E_target.tolist()}")
    
    # 执行推理
    nodes, H_final = pipeline.infer(
        S_target,
        E_target,
        save_intermediates=True,
        output_dir=args.output,
    )
    
    print(f"\n推理完成！结果已保存至 {args.output}/")
    print(f"最终SU数量: {len(nodes)}")
    print(f"SU直方图: {H_final.long().tolist()}")
