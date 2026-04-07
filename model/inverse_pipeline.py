import torch
import pandas as pd
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# 导入项目内部模块
from .coarse_graph import (
    E_SU, SU_PPM_RANGES, NUM_SU_TYPES, PPM_AXIS
)
from .s2n_model import S2NModel
from .g2s_model import NMR_VAE
# 导入拆分模块
from .inverse_common import (
    _NodeV3, L4_CONFIG, lorentzian_spectrum,
    visualize_su_distribution, evaluate_spectrum_reconstruction, resample_spectrum_to_ppm_axis,
    normalize_spectrum_to_carbon_count,
)
from .inverse_layer0 import Layer0Estimator
from .inverse_layer1 import Layer1Assigner
from .inverse_layer2 import Layer2Estimator
from .inverse_layer3 import Layer3Estimator
from .inverse_layer4 import Layer4Adjuster

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
                 nmr_intensity_scale: float = 0.95):
        """
        初始化推理管道
        
        Args:
            s2n_model: Spectrum-to-Node模型
            vae_model: NMR VAE模型（用于解码mu/pi）
            templates: 预建模板库
            device: 计算设备
        """
        self.s2n = s2n_model.to(device).eval()
        self.vae = vae_model.to(device).eval()
        self.device = device
        self.nmr_intensity_scale = float(nmr_intensity_scale)
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
        
        # Layer4配置
        self.r2_target = L4_CONFIG['r2_target']
        self.max_outer_iters = L4_CONFIG['max_outer_iters']
        self.spike_drop_min = L4_CONFIG['spike_drop_min']
        
        # 初始化分层组件（委托模式）
        self.layer0_estimator = Layer0Estimator(
            s2n_model=self.s2n,
            E_SU_tensor=self.E_SU,
            device=device
        )
        self.layer1_assigner = Layer1Assigner(
            device=device,
            vae_model=self.vae,
            E_SU_tensor=self.E_SU,
            layer0_estimator=self.layer0_estimator,  # 传递Layer0实例
            intensity_scale=float(self.nmr_intensity_scale),
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
        except Exception:
            pass
        try:
            self.layer3_estimator.intensity_scale = float(self.nmr_intensity_scale)
        except Exception:
            pass
        
        self.layer4_adjuster = Layer4Adjuster(
             device=torch.device(device),
             layer0_estimator=self.layer0_estimator,
         )
    
    def infer(self, S_target: torch.Tensor, E_target: torch.Tensor,
              save_intermediates: bool = True,
              output_dir: str = 'inverse_result',
              eval_lib_path: Optional[str] = None,
              eval_hwhm: float = 1.0,
              eval_allow_approx: bool = True,
              enable_hop1_adjust: bool = True,
              hop1_adjust_iterations: int = 3,
              hop1_neg_threshold: float = -0.5,
              hop1_pos_threshold: float = 0.5,
              stage_configs: Optional[Dict[str, Dict]] = None,
              enable_su22_adjust: bool = True,
              su22_ratio: float = 0.1,
              su22_h_tol: float = 0.03,
              outer_max_cycles: int = 8,
              outer_patience: int = 3,
              outer_improve_eps: float = 1e-4) -> Tuple[List[_NodeV3], torch.Tensor]:
        """
        逆向推理主入口
        
        流程：
        Layer0: 预测+初始化SU直方图
        Loop:
            Layer1: 分配1-hop + hop1差谱微调
            Layer2: 分配2-hop + 模板匹配
            Layer3: 分配z向量 + 预测mu/pi + 重构精确谱图
            Layer4: 基于差谱(target-reconstructed)调整SU直方图 → Layer0修正
            → 循环回 Layer1
        
        Args:
            stage_configs: Layer4 各阶段调整配置，格式如:
                {
                    'carbonyl': {'max_cycles': 3, 'max_moves': 5, ...},
                    'su9': {'max_cycles': 3, 'max_moves': 5, ...},
                    'ether': {'max_cycles': 2, 'max_moves': 10, ...},
                    'amine': {'max_cycles': 2, 'max_moves': 5, ...},
                    'thioether': {'max_cycles': 2, 'max_moves': 5, ...},
                    'halogen': {'max_cycles': 2, 'max_moves': 5, ...},
                }
        """
        print("=" * 80)
        print("InversePipelineV3: 开始逆向推理")
        print("=" * 80)
        
        # Layer0: 估计SU直方图
        print("\n>>> Layer0: 估计SU直方图")
        H_init = self.estimate_su_histogram(S_target, E_target)
        if save_intermediates:
            visualize_su_distribution(H_init, 'Layer0', save_dir=output_dir)

        # Loop(L1→L2→L3→L4): 多阶段调整
        print("\n>>> Loop(L1→L2→L3→L4): 1-hop分配 + 2-hop模板 + z微调 + SU调整")
        nodes, H_final, loop_summary = self.optimize_layer1_2_3_4_multistage(
            H_init=H_init,
            S_target=S_target,
            E_target=E_target,
            output_dir=output_dir,
            eval_lib_path=eval_lib_path,
            eval_hwhm=eval_hwhm,
            eval_allow_approx=eval_allow_approx,
            enable_hop1_adjust=enable_hop1_adjust,
            hop1_adjust_iterations=hop1_adjust_iterations,
            hop1_neg_threshold=hop1_neg_threshold,
            hop1_pos_threshold=hop1_pos_threshold,
            stage_configs=stage_configs,
            enable_su22_adjust=enable_su22_adjust,
            su22_ratio=su22_ratio,
            su22_h_tol=su22_h_tol,
            outer_max_cycles=outer_max_cycles,
            outer_patience=outer_patience,
            outer_improve_eps=outer_improve_eps,
        )

        try:
            print(f"[Loop] best_r2={float(loop_summary.get('best_r2', 0.0)):.4f}")
        except Exception:
            pass

        if save_intermediates:
            try:
                visualize_su_distribution(H_final.long().detach().cpu(), 'Final', save_dir=output_dir)
            except Exception:
                pass
        
        print("\n" + "=" * 80)
        print("逆向推理完成！")
        print("=" * 80)
        
        return nodes, H_final

    def _compute_difference_spectrum_from_nodes_mu(self,
                                                    nodes: List[_NodeV3],
                                                    S_target: torch.Tensor,
                                                    E_target: torch.Tensor,
                                                    hwhm: float) -> Dict[str, object]:
        device = self.device
        S_target = S_target.to(device).flatten()
        ppm_axis = PPM_AXIS.to(device).flatten()

        S_recon = self.reconstruct_spectrum(nodes, E_target=E_target, hwhm=float(hwhm)).to(device).flatten()
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

    def optimize_layer1_2_3_4_multistage(self,
                                         H_init: torch.Tensor,
                                         S_target: torch.Tensor,
                                         E_target: torch.Tensor,
                                         output_dir: str,
                                         eval_lib_path: Optional[str],
                                         eval_hwhm: float,
                                         eval_allow_approx: bool,
                                         enable_hop1_adjust: bool,
                                         hop1_adjust_iterations: int,
                                         hop1_neg_threshold: float,
                                         hop1_pos_threshold: float,
                                         stage_configs: Optional[Dict[str, Dict]] = None,
                                         enable_su22_adjust: bool = True,
                                         su22_ratio: float = 0.1,
                                         su22_h_tol: float = 0.03,
                                         outer_max_cycles: int = 8,
                                         outer_patience: int = 3,
                                         outer_improve_eps: float = 1e-4) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, object]]:
        """
        多阶段调整流程：
        - 小循环仅执行 L1→L2→差谱→Layer4(stage)
        - Layer3 只在所有阶段结束后做一次最终精修
        
        Args:
            stage_configs: 每个阶段的配置，格式如:
                {
                    'carbonyl': {'max_cycles': 3, 'max_moves': 5, 'window_12': 5.0, ...},
                    'su9': {'max_cycles': 3, 'max_moves': 5, ...},
                    'ether': {'max_cycles': 2, 'max_moves': 10, ...},
                    ...
                }
        """
        device = self.device
        S_target = S_target.to(device)
        E_target = E_target.to(device)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 默认阶段配置
        default_stage_configs = {
            'block_a': {'max_cycles': 3, 'max_moves': 6, 'carbonyl_max_moves': 2, 'score_rel_threshold': 0.02, 'peak_rel_threshold': 0.01},
            'block_b': {'max_cycles': 2, 'max_moves_each': 3, 'peak_rel_threshold': 0.01},
            'block_c': {'max_cycles': 3, 'max_moves': 6, 'peak_rel_threshold': 0.01, 'enable_su22_adjust': False},
            'skeleton': {'max_cycles': 10, 'max_steps': 4},  # Resource-allocation-based skeleton SU adjustment
        }
        
        if stage_configs is not None:
            for stage, cfg in stage_configs.items():
                if stage in default_stage_configs:
                    default_stage_configs[stage].update(cfg)
        
        stages = ['block_a', 'block_b', 'block_c', 'skeleton']
        
        H_curr = H_init.detach().clone()
        best_nodes: Optional[List[_NodeV3]] = None
        best_H: Optional[torch.Tensor] = None
        best_r2 = -1e9
        global_history: List[Dict[str, object]] = []
        
        print("\n" + "=" * 80)
        print("开始多阶段 Layer4 调整流程")
        print("阶段顺序: " + " → ".join(stages))
        print("=" * 80)
        
        for stage_idx, stage in enumerate(stages):
            stage_cfg = default_stage_configs.get(stage, {})
            stage_max_cycles = int(stage_cfg.get('max_cycles', 2))
            
            if stage_max_cycles <= 0:
                print(f"\n[跳过] {stage.upper()} 阶段 (max_cycles=0)")
                continue
            
            print("\n" + "=" * 80)
            print(f"阶段 {stage_idx+1}/{len(stages)}: {stage.upper()}")
            print(f"最大循环次数: {stage_max_cycles}")
            print("=" * 80)
            
            stage_no_improve = 0
            
            for cycle in range(stage_max_cycles):
                cycle_dir = str(Path(output_dir) / f"{stage}_cycle_{cycle}")
                Path(cycle_dir).mkdir(parents=True, exist_ok=True)
                
                print(f"\n[{stage.upper()}] cycle {cycle+1}/{stage_max_cycles}")
                
                # L1: 1-hop分配
                nodes = self.layer1_assign(
                    H_init=H_curr,
                    S_target=S_target,
                    E_target=E_target,
                    eval_nmr=True,
                    eval_output_dir=str(Path(cycle_dir) / 'layer1_eval'),
                    eval_lib_path=eval_lib_path,
                    eval_hwhm=eval_hwhm,
                    eval_allow_approx=eval_allow_approx,
                    enable_hop1_adjust=enable_hop1_adjust,
                    hop1_adjust_iterations=hop1_adjust_iterations,
                    hop1_neg_threshold=hop1_neg_threshold,
                    hop1_pos_threshold=hop1_pos_threshold,
                )
                
                # L2: 2-hop推导
                try:
                    nodes = self.layer2_assign(
                        nodes=nodes,
                        H_center=self._histogram_from_nodes(nodes),
                        S_target=S_target,
                        E_target=E_target,
                        lib_path=eval_lib_path,
                        output_dir=str(Path(cycle_dir) / 'layer2_eval'),
                        eval_hwhm=float(eval_hwhm),
                    )
                except Exception as e:
                    print(f"  [Layer2] 失败: {e}")
                
                # 计算 Layer1-2 差谱
                diff_info = self._compute_difference_spectrum_from_nodes_mu(
                    nodes=nodes,
                    S_target=S_target,
                    E_target=E_target,
                    hwhm=float(eval_hwhm),
                )
                r2_before = float(diff_info.get('r2', 0.0))
                print(f"  [L1-2] r2={r2_before:.4f}")
                
                # Layer4: 当前阶段的SU调整
                H_base = self._histogram_from_nodes(nodes)
                prev_H = H_curr.detach().clone()
                prev_nodes = copy.deepcopy(nodes)
                moves = []
                meta: Dict[str, object] = {}
                try:
                    # 如果是 Skeleton 阶段，首次循环仅进行纯资源分配修复（不传差谱），
                    # 修复好资源拓扑并重构图之后，后续循环才有有意义的差谱。
                    if stage == 'skeleton' and cycle == 0:
                        eval_ppm = None
                        eval_diff = None
                    else:
                        eval_ppm = diff_info.get('ppm')
                        eval_diff = diff_info.get('diff')
                        
                    H_adjusted, moves, meta = self.layer4_adjuster.adjust_by_stage(
                        H=H_base,
                        ppm=eval_ppm,
                        diff=eval_diff,
                        E_target=E_target,
                        S_target=S_target,
                        stage=stage,
                        nodes=nodes,  # Pass nodes to Layer4 for skeleton allocation evaluation
                        **stage_cfg
                    )
                    n_moves = len(moves)
                    print(f"  [Layer4-{stage.upper()}] 调整: {n_moves} 次变更")
                except Exception as e:
                    print(f"  [Layer4-{stage.upper()}] 失败: {e}")
                    H_adjusted = H_base
                    n_moves = 0
                
                counts_changed = not bool(torch.equal(H_adjusted.detach().cpu(), H_base.detach().cpu()))
                topology_changed = bool(int(n_moves) > 0)
                changed = bool(counts_changed or topology_changed)
                candidate_nodes = nodes
                candidate_H = H_base.detach().clone()
                candidate_r2 = r2_before

                if changed:
                    try:
                        candidate_H = H_adjusted.detach().clone()

                        if stage == 'skeleton' and topology_changed and not counts_changed:
                            candidate_nodes = copy.deepcopy(nodes)
                        else:
                            candidate_nodes = self.layer1_assign(
                                H_init=candidate_H,
                                S_target=S_target,
                                E_target=E_target,
                                eval_nmr=False,
                                eval_output_dir=str(Path(cycle_dir) / 'layer1_post_stage'),
                                eval_lib_path=eval_lib_path,
                                eval_hwhm=eval_hwhm,
                                eval_allow_approx=eval_allow_approx,
                                enable_hop1_adjust=enable_hop1_adjust,
                                hop1_adjust_iterations=hop1_adjust_iterations,
                                hop1_neg_threshold=hop1_neg_threshold,
                                hop1_pos_threshold=hop1_pos_threshold,
                            )
                        candidate_nodes = self.layer2_assign(
                            nodes=candidate_nodes,
                            H_center=self._histogram_from_nodes(candidate_nodes),
                            S_target=S_target,
                            E_target=E_target,
                            lib_path=eval_lib_path,
                            output_dir=str(Path(cycle_dir) / 'layer2_post_stage'),
                            eval_hwhm=float(eval_hwhm),
                        )
                        candidate_diff = self._compute_difference_spectrum_from_nodes_mu(
                            nodes=candidate_nodes,
                            S_target=S_target,
                            E_target=E_target,
                            hwhm=float(eval_hwhm),
                        )
                        candidate_r2 = float(candidate_diff.get('r2', 0.0))
                        print(f"  [Layer4-{stage.upper()} 后 L1-2] r2={candidate_r2:.4f}")
                    except Exception as e:
                        print(f"  [Layer4-{stage.upper()} 后重建失败] {e}")
                        candidate_nodes = prev_nodes
                        candidate_H = H_base.detach().clone()
                        candidate_r2 = r2_before
                        changed = False
                
                # 记录历史
                global_history.append({
                    'stage': stage,
                    'cycle': cycle + 1,
                    'r2': candidate_r2,
                    'n_moves': n_moves,
                    'changed': changed,
                })
                
                # 更新最佳结果
                is_accepted = False
                if stage == 'skeleton':
                    candidate_alloc = {}
                    prev_alloc_ok = True
                    prev_req_ok = True
                    candidate_req_ok = True
                    try:
                        candidate_alloc = dict((meta or {}).get('final_allocation', {}) or {})
                    except Exception:
                        candidate_alloc = {}
                    try:
                        prev_alloc_diag = self.layer4_adjuster._evaluate_full_allocation_balance(
                            prev_nodes,
                            S_target=S_target,
                            E_target=E_target,
                        )
                        prev_alloc_ok = bool(prev_alloc_diag.get('ok', True))
                    except Exception:
                        prev_alloc_ok = True
                    try:
                        prev_req_diag = self.layer4_adjuster._evaluate_required_hist_constraints(
                            H_base,
                            E_target,
                            su22_ratio=float(stage_cfg.get('su22_ratio', 0.1)),
                            su22_h_tol=float(stage_cfg.get('su22_h_tol', 0.03)),
                        )
                        prev_req_ok = bool(prev_req_diag.get('ok', True))
                    except Exception:
                        prev_req_ok = True
                    try:
                        candidate_req_diag = self.layer4_adjuster._evaluate_required_hist_constraints(
                            candidate_H,
                            E_target,
                            su22_ratio=float(stage_cfg.get('su22_ratio', 0.1)),
                            su22_h_tol=float(stage_cfg.get('su22_h_tol', 0.03)),
                        )
                        candidate_req_ok = bool(candidate_req_diag.get('ok', True))
                    except Exception:
                        candidate_req_ok = True

                    alloc_ok = bool(candidate_alloc.get('ok', False))
                    post_ok = bool(not bool((meta or {}).get('recheck_required', False)))
                    mandatory_fix = bool((not prev_alloc_ok and alloc_ok) or (not prev_req_ok and candidate_req_ok))
                    if changed and alloc_ok and candidate_req_ok and post_ok:
                        if mandatory_fix:
                            is_accepted = True
                        else:
                            is_accepted = bool(candidate_r2 >= (r2_before - float(outer_improve_eps)))
                else:
                    if candidate_r2 > best_r2 + outer_improve_eps:
                        is_accepted = True

                if is_accepted:
                    H_curr = candidate_H.detach().clone()
                    nodes = candidate_nodes
                    if candidate_r2 > best_r2:
                        best_r2 = candidate_r2
                    best_nodes = copy.deepcopy(candidate_nodes)
                    best_H = candidate_H.detach().clone()
                    stage_no_improve = 0
                    print(f"  ✓ 接受 ({stage} 策略): r2={candidate_r2:.4f}, 历史 best_r2={best_r2:.4f}")
                else:
                    stage_no_improve += 1
                    print(f"  ✗ 拒绝 ({stage} 策略): r2={candidate_r2:.4f}, 历史 best_r2={best_r2:.4f}, no_improve={stage_no_improve}/{outer_patience}")
                    H_curr = prev_H
                    nodes = prev_nodes
                
                # 本阶段提前终止条件
                if not changed and stage_no_improve >= outer_patience:
                    # 对于 skeleton，如果资源分配仍不合理或仍有操作，不能因为 R2 没改善就停止
                    if stage == 'skeleton' and (meta.get('final_scenario', 'ok') != 'ok' or len(moves) > 0):
                        print(f"  [{stage.upper()}] 正在强制调整拓扑分配，继续循环。")
                    else:
                        print(f"  [{stage.upper()}] 阶段收敛，提前结束")
                        break
            
            print(f"\n[{stage.upper()}] 阶段完成，best_r2={best_r2:.4f}")
        
        print("\n" + "=" * 80)
        print("多阶段 Layer4 调整流程完成！")
        print(f"最终 best_r2={best_r2:.4f}")
        print("=" * 80)
        
        out_nodes = best_nodes if best_nodes is not None else nodes
        out_H = best_H if best_H is not None else H_curr
        try:
            final_nodes = self.layer1_assign(
                H_init=out_H,
                S_target=S_target,
                E_target=E_target,
                eval_nmr=False,
                eval_output_dir=str(Path(output_dir) / 'final_layer1'),
                eval_lib_path=eval_lib_path,
                eval_hwhm=eval_hwhm,
                eval_allow_approx=eval_allow_approx,
                enable_hop1_adjust=enable_hop1_adjust,
                hop1_adjust_iterations=hop1_adjust_iterations,
                hop1_neg_threshold=hop1_neg_threshold,
                hop1_pos_threshold=hop1_pos_threshold,
            )
            final_nodes = self.layer2_assign(
                nodes=final_nodes,
                H_center=self._histogram_from_nodes(final_nodes),
                S_target=S_target,
                E_target=E_target,
                lib_path=eval_lib_path,
                output_dir=str(Path(output_dir) / 'final_layer2'),
                eval_hwhm=float(eval_hwhm),
            )
            final_nodes, final_r2 = self.layer3_adjust_templates(
                nodes=final_nodes,
                S_target=S_target,
                E_target=E_target,
                max_iters=30,
                lib_path=eval_lib_path,
                output_dir=str(Path(output_dir) / 'final_layer3'),
                hwhm=float(eval_hwhm),
            )
            out_nodes = final_nodes
            out_H = self._histogram_from_nodes(final_nodes).detach().clone()
            best_r2 = max(float(best_r2), float(final_r2))
        except Exception as e:
            print(f"[Final Layer3] 失败: {e}")
        summary = {
            'best_r2': float(best_r2),
            'total_stages': len(stages),
            'history': global_history,
        }
        return out_nodes, out_H, summary

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
                      enable_carbonyl_joint_adjust: bool = True,
                      carbonyl_joint_iterations: int = 3,
                      carbonyl_joint_max_adjustments: int = 3,
                      carbonyl_joint_pos_threshold: float = 0.08,
                      carbonyl_joint_neg_threshold: float = 0.08,
                      enable_hop1_adjust: bool = False,
                      hop1_adjust_iterations: int = 3,
                      hop1_neg_threshold: float = -0.5,
                      hop1_pos_threshold: float = 0.5) -> List[_NodeV3]:
        """委托给Layer1Assigner执行1-hop分配"""
        eval_lib_path = eval_lib_path or self.default_template_lib_path
        return self.layer1_assigner.layer1_assign(
            H_init=H_init, S_target=S_target, E_target=E_target,
            eval_nmr=eval_nmr, eval_output_dir=eval_output_dir,
            eval_lib_path=eval_lib_path, eval_hwhm=eval_hwhm,
            eval_allow_approx=eval_allow_approx,
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
    
    def layer2_assign(self, nodes: List[_NodeV3], H_center: torch.Tensor,
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
                                max_iters: int = 30,
                                lib_path: Optional[str] = None,
                                output_dir: Optional[str] = None,
                                hwhm: float = 1.0,
                                pos_search_window: float = 10.0,
                                neg_assign_window: float = 1.5,
                                top_k_samples: int = 5,
                                enable_approx_hop2_template_adjust: bool = False,
                                approx_hop2_max_iters: Optional[int] = None,
                                approx_hop2_max_diff_nodes: int = 3,
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
        max_iters: int = 30,
        lib_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        hwhm: float = 1.0,
        pos_search_window: float = 10.0,
        neg_assign_window: float = 1.5,
        top_k_samples: int = 5,
        enable_approx_hop2_template_adjust: bool = False,
        approx_hop2_max_iters: Optional[int] = None,
        approx_hop2_max_diff_nodes: int = 3,
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
            enable_approx_hop2_template_adjust=bool(enable_approx_hop2_template_adjust),
            approx_hop2_max_iters=(int(approx_hop2_max_iters) if approx_hop2_max_iters is not None else None),
            approx_hop2_max_diff_nodes=int(approx_hop2_max_diff_nodes),
            approx_hop2_top_k_templates=int(approx_hop2_top_k_templates),
        )
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def reconstruct_spectrum(self, nodes: List[_NodeV3], 
                             E_target: torch.Tensor,
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
