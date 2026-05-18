import torch
import pandas as pd
import copy
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# 导入项目内部模块
from ..shared.coarse_graph import (
    E_SU, SU_PPM_RANGES, NUM_SU_TYPES, PPM_AXIS
)
from ..forward.s2n.s2n_model import S2NModel
from ..forward.g2s.g2s_model import NMR_VAE
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
            'build_variant': 0,
        }
        layer2_defaults = {
            'eval_hwhm': float(eval_hwhm),
        }
        layer3_defaults = {
            'max_iters': 150,
            'hwhm': float(eval_hwhm),
            'pos_search_window': 15.0,
            'neg_assign_window': 2.0,
            'top_k_samples': 10,
            'neg_top_k_peaks': 15,
            'neg_peak_min_sep_ppm': 1.0,
            'enable_approx_hop2_template_adjust': True,
            'approx_hop2_max_iters': None,
            'approx_hop2_max_diff_nodes': 8,
            'approx_hop2_top_k_templates': 80,
        }
        return (
            self._merge_stage_config(layer1_defaults, layer1_config),
            self._merge_stage_config(layer2_defaults, layer2_config),
            self._merge_stage_config(layer3_defaults, layer3_config),
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

    def _run_layer12_cycle(self,
                           H_seed: torch.Tensor,
                           S_target: torch.Tensor,
                           E_target: torch.Tensor,
                           eval_lib_path: Optional[str],
                           layer1_runtime_cfg: Dict[str, Any],
                           layer2_runtime_cfg: Dict[str, Any],
                           seed_nodes: Optional[List[_NodeV3]] = None,
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

        def _run_layer1_once(seed_local: Optional[List[_NodeV3]],
                             cfg_local: Dict[str, Any],
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
                    seed_nodes=seed_local,
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

        layer1_cfg_warm = dict(layer1_runtime_cfg)
        nodes, layer1_ok, layer1_errors = _run_layer1_once(
            seed_local=seed_nodes,
            cfg_local=layer1_cfg_warm,
            attempt_name='warm-start' if seed_nodes is not None else 'cold-start',
        )
        if (not bool(layer1_ok)) and seed_nodes is not None:
            print("  [Layer1 fallback] warm-start 拓扑不可行，改用 cold-start 重建候选图")
            layer1_cfg_cold = dict(layer1_runtime_cfg)
            try:
                layer1_cfg_cold['build_variant'] = int(layer1_cfg_cold.get('build_variant', 0)) + 1009
            except Exception:
                layer1_cfg_cold['build_variant'] = 1009
            nodes, layer1_ok, layer1_errors = _run_layer1_once(
                seed_local=None,
                cfg_local=layer1_cfg_cold,
                attempt_name='cold-start',
            )
        if not bool(layer1_ok):
            raise RuntimeError(f"layer1_graph_invalid: {_first_layer1_error(list(layer1_errors or []))}")
        nodes = self.layer2_assign(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            lib_path=eval_lib_path,
            output_dir=layer2_output_dir,
            **layer2_runtime_cfg,
        )
        H_nodes = self._histogram_from_nodes(nodes).detach().clone()
        diff_info = self._compute_difference_spectrum_from_nodes_mu(
            nodes=nodes,
            S_target=S_target,
            hwhm=float(layer2_runtime_cfg.get('eval_hwhm', 1.0)),
        )
        r2 = float(diff_info.get('r2', 0.0))
        return nodes, H_nodes, diff_info, r2

    def _print_final_resource_summary(self,
                                      nodes: List[_NodeV3],
                                      S_target: torch.Tensor,
                                      E_target: torch.Tensor) -> None:
        try:
            alloc_diag = self.layer4_adjuster._evaluate_full_allocation_balance(
                nodes=nodes,
                flex_ratio=0.80,
                flex_lower_extra=1,
                S_target=S_target,
                E_target=E_target,
            )
            print(
                "[Final Allocation] "
                f"ok={bool(alloc_diag.get('ok', False))} "
                f"reason={alloc_diag.get('reason', 'unknown')}"
            )
            alloc_res = alloc_diag.get('allocation_result', None)
            if alloc_res is not None:
                self.layer4_adjuster._print_allocation_details(
                    alloc_res,
                    header="最终资源分配详情",
                )
        except Exception as e:
            print(f"[Final Allocation] Failed to summarize allocation: {e}")
    
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
           - block_a / block_b / block_c_tail:
             作为候选 H 调整，只在重跑 Layer1/2 后通过结构校验且
             Layer2 R2 超过当前接受参考时才采纳
           - block_c_branch / block_c_extra:
             以资源分配与拓扑修复为目标，不走上述 improvement gate；
             只要本阶段成功产出并完成 Layer1/2 重跑，就直接采纳阶段结果
           - 每轮结束后执行一次 Layer3 细化
        4. 返回当前外循环结束时的状态
        
        Args:
            stage_configs: Layer4 正式阶段配置，推荐格式如:
                {
                    'block_a': {...},
                    'block_b': {...},
                    'block_c_tail': {
                        'max_moves': 3,
                        'peak_rel_threshold': 0.01,
                        'carbonyl_couple': True,
                        'h_tolerance': 0.08,
                    },
                    'block_c_branch': {
                        'max_steps': 150,
                    },
                    'block_c_extra': {
                        'guided_max_steps': 180,
                        'relaxed_flexible_ratio': 0.82,
                    },
                }
            约定:
                - 传入 'block_c' 时会映射到 'block_c_tail'
            layer1_config 可调参数:
                - eval_hwhm, eval_allow_approx, build_variant
                - enable_carbonyl_joint_adjust, carbonyl_joint_iterations
                - carbonyl_joint_max_adjustments
                - carbonyl_joint_pos_threshold, carbonyl_joint_neg_threshold
                - enable_hop1_adjust, hop1_adjust_iterations
                - hop1_neg_threshold, hop1_pos_threshold
            layer2_config 可调参数:
                - eval_hwhm
            layer3_config 可调参数:
                - max_iters, hwhm, pos_search_window, neg_assign_window
                - top_k_samples, neg_top_k_peaks, neg_peak_min_sep_ppm
                - enable_approx_hop2_template_adjust
                - approx_hop2_max_iters, approx_hop2_max_diff_nodes
                - approx_hop2_top_k_templates
            outer_max_cycles:
                - Layer4 -> Layer1/2 -> Layer3 宏循环次数上限
            outer_improve_eps:
                - block_a / block_b / block_c_tail 的最小 Layer2 R2 提升阈值
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
                'max_moves_each': 6,
                'peak_rel_threshold': 0.01,
            },
            'block_c_tail': {
                'max_cycles': 1,
                'max_moves': 3,
                'peak_rel_threshold': 0.01,
                'carbonyl_couple': True,
                'h_tolerance': 0.08,
            },
            'block_c_branch': {
                'max_steps': 150,
            },
            'block_c_extra': {
                'guided_max_steps': 180,
                'relaxed_flexible_ratio': 0.82,
            },
        }
        block_c_group_cfg = {
            'max_cycles': 1,
        }
        if stage_configs is not None:
            for stage, cfg in stage_configs.items():
                if stage == 'block_c':
                    if isinstance(cfg, dict):
                        if 'max_cycles' in cfg:
                            block_c_group_cfg['max_cycles'] = int(cfg.get('max_cycles', 1))
                        tail_compat_cfg = {k: v for k, v in cfg.items() if str(k) != 'max_cycles'}
                        if tail_compat_cfg:
                            default_stage_configs['block_c_tail'].update(tail_compat_cfg)
                elif stage in default_stage_configs:
                    default_stage_configs[stage].update(cfg)

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
            layer1_runtime_cfg=layer1_runtime_cfg,
            layer2_runtime_cfg=layer2_runtime_cfg,
            seed_nodes=None,
            eval_nmr=True,
            eval_output_dir=None,
            layer2_output_dir=None,
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
        for outer_idx in range(max(1, int(outer_max_cycles))):
            print("\n" + "=" * 80)
            print(f"Outer Cycle {outer_idx + 1}/{max(1, int(outer_max_cycles))}")
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
                        layer1_runtime_cfg=layer1_runtime_cfg,
                        layer2_runtime_cfg=layer2_runtime_cfg,
                        # After Layer4 changes the histogram, rebuild 1-hop
                        # topology from scratch instead of restoring a stale seed.
                        seed_nodes=None,
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

                        try:
                            layer1_stage_cfg = dict(layer1_runtime_cfg)
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
                                seed_nodes=None,
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

            # Stage C group: tail -> branch -> extra, repeated by block_c.max_cycles.
            block_c_group_cycles = max(1, int(block_c_group_cfg.get('max_cycles', 1)))
            for block_c_iter in range(block_c_group_cycles):
                block_c_group_progress = False
                block_c_group_attempted = False

                block_c_tail_cfg = dict(default_stage_configs['block_c_tail'])
                block_c_tail_cycles = max(1, int(block_c_tail_cfg.pop('max_cycles', 1)))
                for tail_iter in range(block_c_tail_cycles):
                    nodes_tail_base = copy.deepcopy(current_nodes)
                    H_tail_base = current_H.detach().clone()
                    diff_tail_base = copy.deepcopy(current_diff)
                    meta_tail_base = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                    try:
                        H_tail, moves_tail, _meta_tail = self.layer4_adjuster.adjust_by_stage(
                            H=H_tail_base,
                            ppm=diff_tail_base.get('ppm'),
                            diff=diff_tail_base.get('diff'),
                            E_target=E_target,
                            S_target=S_target,
                            stage='block_c',
                            nodes=nodes_tail_base,
                            **block_c_tail_cfg,
                        )
                    except Exception as e:
                        try:
                            self._set_fixed_partition_meta(meta_tail_base)
                        except Exception:
                            pass
                        H_tail, moves_tail = H_tail_base, []
                        print(f"[Block C:tail #{block_c_iter + 1}.{tail_iter + 1}] 失败: {e}")

                    if not moves_tail:
                        try:
                            self._set_fixed_partition_meta(meta_tail_base)
                        except Exception:
                            pass
                        if int(tail_iter) == 0 and int(block_c_iter) == 0:
                            print("[Block C:tail] 无有效调整")
                        break

                    block_c_group_attempted = True
                    hist_ok, _hist_reason = _candidate_hist_feasible(
                        H_tail,
                        tag=f'Block C:tail #{block_c_iter + 1}.{tail_iter + 1}',
                    )
                    if not bool(hist_ok):
                        try:
                            self._set_fixed_partition_meta(meta_tail_base)
                        except Exception:
                            pass
                        break

                    try:
                        layer1_tail_cfg = dict(layer1_runtime_cfg)
                        try:
                            base_variant = int(layer1_tail_cfg.get('build_variant', 0))
                        except Exception:
                            base_variant = 0
                        layer1_tail_cfg['build_variant'] = int(base_variant) + 211 * int(block_c_iter) + int(tail_iter)
                        cand_nodes, cand_H, cand_diff, cand_r2 = self._run_layer12_cycle(
                            H_seed=H_tail.detach().clone(),
                            S_target=S_target,
                            E_target=E_target,
                            eval_lib_path=eval_lib_path,
                            layer1_runtime_cfg=layer1_tail_cfg,
                            layer2_runtime_cfg=layer2_runtime_cfg,
                            seed_nodes=None,
                            eval_nmr=False,
                        )
                    except Exception as e:
                        try:
                            self._set_fixed_partition_meta(meta_tail_base)
                        except Exception:
                            pass
                        print(f"[Block C:tail #{block_c_iter + 1}.{tail_iter + 1}] 回退: layer1/2 failed ({e})")
                        break
                    cand_valid, cand_reason = _candidate_structurally_valid(cand_nodes, cand_H, tag='BlockC:tail')
                    if cand_valid and float(cand_r2) > float(cycle_best_layer2_r2) + float(outer_improve_eps):
                        current_nodes = cand_nodes
                        current_H = cand_H.detach().clone()
                        current_diff = copy.deepcopy(cand_diff)
                        cycle_pre_layer3_r2 = float(cand_r2)
                        accepted_layer2_r2 = float(cand_r2)
                        historical_best_layer2_r2 = max(float(historical_best_layer2_r2), float(cand_r2))
                        cycle_best_layer2_r2 = float(cand_r2)
                        block_c_group_progress = True
                        print(f"[Block C:tail #{block_c_iter + 1}.{tail_iter + 1}] 接受: layer2_r2={cand_r2:.4f}")
                        self._print_histogram_and_elements(
                            current_H,
                            E_target,
                            label=f'Block C:tail #{block_c_iter + 1}.{tail_iter + 1} Accepted',
                            nodes=current_nodes,
                        )
                        self._print_effective_resource_summary(
                            current_H,
                            label=f'Block C:tail #{block_c_iter + 1}.{tail_iter + 1} Accepted',
                        )
                        continue

                    if not bool(cand_valid):
                        try:
                            self._set_fixed_partition_meta(meta_tail_base)
                        except Exception:
                            pass
                        print(f"[Block C:tail #{block_c_iter + 1}.{tail_iter + 1}] 回退: candidate invalid ({cand_reason})")
                        break
                    try:
                        self._set_fixed_partition_meta(meta_tail_base)
                    except Exception:
                        pass
                    print(
                        f"[Block C:tail #{block_c_iter + 1}.{tail_iter + 1}] 回退: candidate_r2={cand_r2:.4f} "
                        f"<= accept_ref_layer2_r2={cycle_best_layer2_r2:.4f}"
                    )
                    break

                branch_cfg = dict(default_stage_configs['block_c_branch'])
                nodes_branch_base = copy.deepcopy(current_nodes)
                H_branch_base = current_H.detach().clone()
                diff_branch_base = copy.deepcopy(current_diff)
                meta_branch_base = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                try:
                    H_branch, _moves_branch, _meta_branch = self.layer4_adjuster.adjust_by_stage(
                        H=H_branch_base,
                        ppm=diff_branch_base.get('ppm'),
                        diff=diff_branch_base.get('diff'),
                        E_target=E_target,
                        S_target=S_target,
                        stage='block_c_branch',
                        nodes=nodes_branch_base,
                        **branch_cfg,
                    )
                    branch_changed = bool(_moves_branch) or not bool(
                        torch.equal(H_branch.detach().cpu(), H_branch_base.detach().cpu())
                    )
                    if branch_changed:
                        block_c_group_attempted = True
                    if not bool((_meta_branch or {}).get('ok', False)):
                        reason = str((_meta_branch or {}).get('final_scenario', 'branch_not_ok'))
                        print(
                            f"[Block C:branch #{block_c_iter + 1}] 强制继续: "
                            f"branch meta pending ({reason})，转入 Layer1-2 重建"
                        )
                    if branch_changed:
                        layer1_branch_cfg = dict(layer1_runtime_cfg)
                        try:
                            base_variant = int(layer1_branch_cfg.get('build_variant', 0))
                        except Exception:
                            base_variant = 0
                        layer1_branch_cfg['build_variant'] = int(base_variant) + 307 * int(block_c_iter)
                        cand_nodes, cand_H, cand_diff, branch_r2 = self._run_layer12_cycle(
                            H_seed=H_branch.detach().clone(),
                            S_target=S_target,
                            E_target=E_target,
                            eval_lib_path=eval_lib_path,
                            layer1_runtime_cfg=layer1_branch_cfg,
                            layer2_runtime_cfg=layer2_runtime_cfg,
                            seed_nodes=None,
                            eval_nmr=False,
                        )
                        cand_valid, _cand_reason = _candidate_structurally_valid(cand_nodes, cand_H, tag='BlockC:branch')
                        if cand_valid:
                            current_nodes = cand_nodes
                            current_H = cand_H.detach().clone()
                            current_diff = copy.deepcopy(cand_diff)
                            cycle_pre_layer3_r2 = float(branch_r2)
                            accepted_layer2_r2 = float(branch_r2)
                            historical_best_layer2_r2 = max(float(historical_best_layer2_r2), float(branch_r2))
                            cycle_best_layer2_r2 = float(branch_r2)
                            block_c_group_progress = True
                            print(f"[Block C:branch #{block_c_iter + 1}] 直接采纳: layer2_r2={branch_r2:.4f}")
                            self._print_histogram_and_elements(
                                current_H,
                                E_target,
                                label=f'Block C:branch #{block_c_iter + 1} Accepted',
                                nodes=current_nodes,
                            )
                            self._print_effective_resource_summary(
                                current_H,
                                label=f'Block C:branch #{block_c_iter + 1} Accepted',
                            )
                        else:
                            try:
                                self._set_fixed_partition_meta(meta_branch_base)
                            except Exception:
                                pass
                            print(f"[Block C:branch #{block_c_iter + 1}] 回退: 结构/约束校验未通过")
                    else:
                        try:
                            self._set_fixed_partition_meta(meta_branch_base)
                        except Exception:
                            pass
                except Exception as e:
                    try:
                        self._set_fixed_partition_meta(meta_branch_base)
                    except Exception:
                        pass
                    print(f"[Block C:branch #{block_c_iter + 1}] 失败，保持当前状态: {e}")

                extra_cfg = dict(default_stage_configs['block_c_extra'])
                nodes_extra_base = copy.deepcopy(current_nodes)
                H_extra_base = current_H.detach().clone()
                diff_extra_base = copy.deepcopy(current_diff)
                meta_extra_base = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                try:
                    H_extra, _moves_extra, _meta_extra = self.layer4_adjuster.adjust_by_stage(
                        H=H_extra_base,
                        ppm=diff_extra_base.get('ppm'),
                        diff=diff_extra_base.get('diff'),
                        E_target=E_target,
                        S_target=S_target,
                        stage='block_c_extra',
                        nodes=nodes_extra_base,
                        **extra_cfg,
                    )
                    extra_changed = bool(_moves_extra) or not bool(
                        torch.equal(H_extra.detach().cpu(), H_extra_base.detach().cpu())
                    )
                    if extra_changed:
                        block_c_group_attempted = True
                    if not bool((_meta_extra or {}).get('ok', False)):
                        reason = str((_meta_extra or {}).get('final_scenario', 'extra_not_ok'))
                        print(
                            f"[Block C:extra #{block_c_iter + 1}] 强制继续: "
                            f"extra meta pending ({reason})，转入 Layer1-2 重建"
                        )
                    if extra_changed:
                        layer1_extra_cfg = dict(layer1_runtime_cfg)
                        try:
                            base_variant = int(layer1_extra_cfg.get('build_variant', 0))
                        except Exception:
                            base_variant = 0
                        layer1_extra_cfg['build_variant'] = int(base_variant) + 401 * int(block_c_iter)
                        cand_nodes, cand_H, cand_diff, extra_r2 = self._run_layer12_cycle(
                            H_seed=H_extra.detach().clone(),
                            S_target=S_target,
                            E_target=E_target,
                            eval_lib_path=eval_lib_path,
                            layer1_runtime_cfg=layer1_extra_cfg,
                            layer2_runtime_cfg=layer2_runtime_cfg,
                            seed_nodes=None,
                            eval_nmr=False,
                        )
                        cand_valid, _cand_reason = _candidate_structurally_valid(cand_nodes, cand_H, tag='BlockC:extra')
                        if cand_valid:
                            current_nodes = cand_nodes
                            current_H = cand_H.detach().clone()
                            current_diff = copy.deepcopy(cand_diff)
                            cycle_pre_layer3_r2 = float(extra_r2)
                            accepted_layer2_r2 = float(extra_r2)
                            historical_best_layer2_r2 = max(float(historical_best_layer2_r2), float(extra_r2))
                            cycle_best_layer2_r2 = float(extra_r2)
                            block_c_group_progress = True
                            print(f"[Block C:extra #{block_c_iter + 1}] 直接采纳: layer2_r2={extra_r2:.4f}")
                            self._print_histogram_and_elements(
                                current_H,
                                E_target,
                                label=f'Block C:extra #{block_c_iter + 1} Accepted',
                                nodes=current_nodes,
                            )
                            self._print_effective_resource_summary(
                                current_H,
                                label=f'Block C:extra #{block_c_iter + 1} Accepted',
                            )
                        else:
                            try:
                                self._set_fixed_partition_meta(meta_extra_base)
                            except Exception:
                                pass
                            print(f"[Block C:extra #{block_c_iter + 1}] 回退: 结构/约束校验未通过")
                    else:
                        try:
                            self._set_fixed_partition_meta(meta_extra_base)
                        except Exception:
                            pass
                except Exception as e:
                    try:
                        self._set_fixed_partition_meta(meta_extra_base)
                    except Exception:
                        pass
                    print(f"[Block C:extra #{block_c_iter + 1}] 失败，保持当前状态: {e}")

                if (not bool(block_c_group_progress)) and (not bool(block_c_group_attempted)):
                    break

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
                        output_dir=str(Path(outer_dir) / 'final_outputs'),
                        hwhm=float(layer2_eval_hwhm),
                    )
                except Exception:
                    pass

        nodes = copy.deepcopy(current_nodes)
        H_final = current_H.detach().clone()

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
        self._print_histogram_and_elements(H_final, E_target, label='Final Summary', nodes=nodes)
        self._print_final_resource_summary(nodes, S_target, E_target)
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
                      seed_nodes: Optional[List[_NodeV3]] = None,
                      enable_carbonyl_joint_adjust: bool = True,
                      carbonyl_joint_iterations: int = 5,
                      carbonyl_joint_max_adjustments: int = 4,
                      carbonyl_joint_pos_threshold: float = 0.1,
                      carbonyl_joint_neg_threshold: float = 0.1,
                      enable_hop1_adjust: bool = True,
                      hop1_adjust_iterations: int = 4,
                      hop1_neg_threshold: float = -0.7,
                      hop1_pos_threshold: float = 0.7) -> List[_NodeV3]:
        """委托给Layer1Assigner执行1-hop分配"""
        eval_lib_path = eval_lib_path or self.default_template_lib_path
        return self.layer1_assigner.layer1_assign(
            H_init=H_init, S_target=S_target, E_target=E_target,
            eval_nmr=eval_nmr, eval_output_dir=eval_output_dir,
            eval_lib_path=eval_lib_path, eval_hwhm=eval_hwhm,
            eval_allow_approx=eval_allow_approx,
            build_variant=int(build_variant),
            seed_nodes=seed_nodes,
            enable_carbonyl_joint_adjust=enable_carbonyl_joint_adjust,
            carbonyl_joint_iterations=carbonyl_joint_iterations,
            carbonyl_joint_max_adjustments=carbonyl_joint_max_adjustments,
            carbonyl_joint_pos_threshold=carbonyl_joint_pos_threshold,
            carbonyl_joint_neg_threshold=carbonyl_joint_neg_threshold,
            enable_hop1_adjust=enable_hop1_adjust,
            hop1_adjust_iterations=hop1_adjust_iterations,
            hop1_neg_threshold=hop1_neg_threshold,
            hop1_pos_threshold=hop1_pos_threshold,
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
                      eval_hwhm: float = 1.0) -> List[_NodeV3]:
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
            hwhm=eval_hwhm
        )
    
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
                                approx_hop2_top_k_templates: int = 80) \
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
        )

    def _save_final_nmr_comparison(self,
                                   nodes: List[_NodeV3],
                                   S_target: torch.Tensor,
                                   E_target: torch.Tensor,
                                   output_dir: str,
                                   hwhm: float = 1.0) -> None:
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
        E_pred_raw = torch.matmul(H_final.float(), self.E_SU.detach().cpu())
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
            alloc_diag = self.layer4_adjuster._evaluate_full_allocation_balance(
                nodes,
                flex_ratio=0.80,
                flex_lower_extra=1,
                S_target=S_target,
                E_target=E_target,
            )
        except Exception as e:
            alloc_diag = {'ok': False, 'reason': f'alloc_error:{e}'}

        scalar_summary = {}
        for key, value in dict(alloc_diag or {}).items():
            if isinstance(value, (bool, int, float, str)):
                scalar_summary[str(key)] = value
        pd.DataFrame([scalar_summary]).to_csv(final_dir / 'final_allocation_summary.csv', index=False)

        ledger_rows = []
        for ledger_name, ledger in dict((alloc_diag or {}).get('resource_ledger', {}) or {}).items():
            if not isinstance(ledger, dict):
                continue
            for sub_key, sub_val in ledger.items():
                ledger_rows.append({
                    'ledger': str(ledger_name),
                    'item': str(sub_key),
                    'value': int(sub_val),
                })
        pd.DataFrame(ledger_rows).to_csv(final_dir / 'final_resource_ledger.csv', index=False)

        detail_rows = []
        try:
            alloc_details = dict((alloc_diag or {}).get('allocation_details', {}) or {})
            for section_name, row_key in (
                ('bridge', 'bridge_rows'),
                ('side', 'side_rows'),
                ('branch', 'branch_rows'),
            ):
                for idx, text in enumerate(list(alloc_details.get(row_key, []) or [])):
                    detail_rows.append({
                        'section': str(section_name),
                        'index': int(idx),
                        'detail': str(text),
                    })
        except Exception:
            detail_rows = []
        pd.DataFrame(detail_rows, columns=['section', 'index', 'detail']).to_csv(
            final_dir / 'final_allocation_details.csv',
            index=False,
        )
    
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

def load_pipeline(s2n_ckpt: str, vae_ckpt: str, templates_pkl: str,
                  device: str = 'cuda') -> InversePipelineV3:
    """加载完整推理管道"""
    # 加载模型
    s2n_model = S2NModel()
    s2n_model.load_state_dict(torch.load(s2n_ckpt, map_location=device))
    
    vae_model = NMR_VAE()
    vae_model.load_state_dict(torch.load(vae_ckpt, map_location=device))
    
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
