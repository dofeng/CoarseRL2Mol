import sys
import copy
import json
import io
from pathlib import Path
from typing import Optional, Dict, Any
from collections import Counter
from contextlib import redirect_stdout

try:
    import numpy as np
    import pandas as pd
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "test_v1.py requires numpy, pandas, and torch. Please use the project "
        "environment that has these dependencies installed."
    ) from exc

sys.path.append(str(Path(__file__).parent))

from model.s2n_model import S2NModel
from model.g2s_model import NMR_VAE
from model.inverse_pipeline import InversePipelineV3, read_spectrum_csv, parse_elements
from model.coarse_graph import NUM_SU_TYPES, E_SU, SU_DEFS
from model.inverse_common import (
    PPM_AXIS, visualize_spectrum_comparison, visualize_su_distribution,
    SU_CONNECTION_DEGREE, normalize_spectrum_to_carbon_count
)

SU_NAMES = [name for name, _ in SU_DEFS]
AB_STAGE_ORDER = ['block_a', 'block_b']

# ========================================================================
# Default Configuration — all parameters in one place
# ========================================================================
DEFAULT_CONFIG = {
    # Layer3
    'layer3_max_iters': 240,
    'layer3_pos_window': 15.0,
    'layer3_neg_window': 2.0,
    'layer3_top_k': 12,
    'layer3_approx_hop2': False,
    'layer3_approx_hop2_max_iters': None,
    'layer3_approx_hop2_max_diff_nodes': 8,
    'layer3_approx_hop2_top_k': 15,
    # Hop1
    'hop1_adjust': False,
    'hop1_iterations': 5,
    'hop1_neg_threshold': -0.5,
    'hop1_pos_threshold': 0.5,
    'enable_carbonyl_joint_adjust': True,
    'carbonyl_joint_iterations': 3,
    'carbonyl_joint_max_adjustments': 3,
    'carbonyl_joint_pos_threshold': 0.08,
    'carbonyl_joint_neg_threshold': 0.08,
    # NMR eval
    'eval_lib': 'z_library/subgraph_library.pt',
    'eval_hwhm': 1.0,
    'eval_allow_approx': True,
    # Layer4 block cycles (0 = disabled)
    'block_a_cycles': 4,
    'block_b_cycles': 2,
    'cd_inner_max_loops': 3,
    'skeleton_max_steps': 40,
    'extra_max_steps': 100,
    'extra_flexible_ratio': 0.80,
    'extra_flexible_lower_extra': 1,
    'extra_relaxed_flexible_ratio': 0.82,
    'extra_relaxed_lower_extra': 0,
    # Block A/B/C
    'block_a_max_moves': 6,
    'block_a_carbonyl_max_moves': 2,
    'block_a_score_rel_threshold': 0.02,
    'block_a_peak_rel_threshold': 0.01,
    'block_a_min_keep': 0,
    'block_b_max_moves_each': 3,
    'block_b_peak_rel_threshold': 0.01,
    'block_c_max_moves': 6,
    'block_c_peak_rel_threshold': 0.01,
    'block_c_min_keep_22': 1,
    'block_c_min_keep_23': 0,
    'block_c_min_keep_24': 0,
    'block_c_min_keep_25': 0,
    'block_c_carbonyl_couple': True,
    'block_c_h_tolerance': 0.04,
    # SU22
    'su22_ratio': 0.1,
    'su22_h_tol': 0.03,
    # Outer loop
    'outer_max_cycles': 3,
    'outer_patience': 2,
    'outer_improve_eps': 1e-4,
    # Final smoothing
    'final_smooth_sigma_ppm': None,
    'final_smooth_passes': 1,
    'final_smooth_radius_factor': 4.0,
}


def _get_stage_params(cfg):
    """Build per-stage kwargs dict for adjust_by_stage."""
    return {
        'block_a': {
            'max_moves': cfg['block_a_max_moves'],
            'carbonyl_max_moves': cfg['block_a_carbonyl_max_moves'],
            'score_rel_threshold': cfg['block_a_score_rel_threshold'],
            'peak_rel_threshold': cfg['block_a_peak_rel_threshold'],
            'min_keep': cfg['block_a_min_keep'],
        },
        'block_b': {
            'max_moves_each': cfg['block_b_max_moves_each'],
            'peak_rel_threshold': cfg['block_b_peak_rel_threshold'],
        },
        'block_c': {
            'max_moves': cfg['block_c_max_moves'],
            'peak_rel_threshold': cfg['block_c_peak_rel_threshold'],
            'min_keep_22': cfg['block_c_min_keep_22'],
            'min_keep_23': cfg['block_c_min_keep_23'],
            'min_keep_24': cfg['block_c_min_keep_24'],
            'min_keep_25': cfg['block_c_min_keep_25'],
            'carbonyl_couple': bool(cfg['block_c_carbonyl_couple']),
            'h_tolerance': cfg['block_c_h_tolerance'],
            'enable_su22_adjust': False,
        },
        'skeleton': {
            'max_steps': cfg['skeleton_max_steps'],
            'extra_max_steps': cfg['extra_max_steps'],
            'extra_flexible_ratio': cfg['extra_flexible_ratio'],
            'extra_flexible_lower_extra': cfg['extra_flexible_lower_extra'],
            'extra_relaxed_flexible_ratio': cfg['extra_relaxed_flexible_ratio'],
            'extra_relaxed_lower_extra': cfg['extra_relaxed_lower_extra'],
        },
    }


def _window_abs_from_diff(diff_info: Dict[str, Any], lo: float, hi: float) -> float:
    ppm = np.asarray(diff_info.get('ppm', []), dtype=np.float64)
    diff = np.asarray(diff_info.get('diff', []), dtype=np.float64)
    if ppm.size == 0 or diff.size == 0:
        return 1e12
    mask = (ppm >= float(lo)) & (ppm <= float(hi))
    if not np.any(mask):
        return 1e12
    return float(np.sum(np.abs(diff[mask])))


def _compute_stage_score(diff_info: Dict[str, Any], stage: str) -> Dict[str, float]:
    r2 = float(diff_info.get('r2', 0.0))
    loss_160_170 = _window_abs_from_diff(diff_info, 160.0, 170.0)
    loss_172_180 = _window_abs_from_diff(diff_info, 172.0, 180.0)
    loss_186_205 = _window_abs_from_diff(diff_info, 186.0, 205.0)
    loss_8_18 = _window_abs_from_diff(diff_info, 8.0, 18.0)
    loss_18_35 = _window_abs_from_diff(diff_info, 18.0, 35.0)
    loss_32_50 = _window_abs_from_diff(diff_info, 32.0, 50.0)
    loss_45_65 = _window_abs_from_diff(diff_info, 45.0, 65.0)
    loss_100_128 = _window_abs_from_diff(diff_info, 100.0, 128.0)
    loss_126_142 = _window_abs_from_diff(diff_info, 126.0, 142.0)
    loss_134_165 = _window_abs_from_diff(diff_info, 134.0, 165.0)
    loss_global = _window_abs_from_diff(diff_info, 0.1, 240.0)

    carbonyl_loss = 1.6 * loss_160_170 + 2.2 * loss_172_180 + 1.0 * loss_186_205
    tail_loss = 1.4 * loss_8_18 + 1.2 * loss_18_35 + 1.6 * loss_32_50 + 1.1 * loss_45_65
    aromatic_support_loss = 1.0 * loss_100_128 + 1.3 * loss_126_142 + 1.1 * loss_134_165
    global_loss = 0.35 * loss_global

    if stage == 'block_a':
        score = - (1.0 * carbonyl_loss + 0.35 * tail_loss + 0.35 * aromatic_support_loss + global_loss) + 8.0 * r2
    elif stage == 'block_c':
        score = - (0.65 * carbonyl_loss + 1.0 * tail_loss + 0.9 * aromatic_support_loss + global_loss) + 7.0 * r2
    elif stage == 'skeleton':
        score = - (0.55 * carbonyl_loss + 0.8 * tail_loss + 1.0 * aromatic_support_loss + global_loss) + 6.5 * r2
    elif stage == 'block_b':
        score = - (0.8 * carbonyl_loss + 0.8 * tail_loss + 0.45 * aromatic_support_loss + global_loss) + 6.0 * r2
    else:
        score = - (0.9 * carbonyl_loss + 0.9 * tail_loss + 0.8 * aromatic_support_loss + global_loss) + 6.5 * r2

    return {
        'score': float(score),
        'r2': float(r2),
        'carbonyl_loss': float(carbonyl_loss),
        'tail_loss': float(tail_loss),
        'aromatic_support_loss': float(aromatic_support_loss),
        'global_loss': float(global_loss),
    }


def _compute_overall_score(diff_info: Dict[str, Any]) -> Dict[str, float]:
    r2 = float(diff_info.get('r2', 0.0))
    loss_160_170 = _window_abs_from_diff(diff_info, 160.0, 170.0)
    loss_172_180 = _window_abs_from_diff(diff_info, 172.0, 180.0)
    loss_186_205 = _window_abs_from_diff(diff_info, 186.0, 205.0)
    loss_8_18 = _window_abs_from_diff(diff_info, 8.0, 18.0)
    loss_18_35 = _window_abs_from_diff(diff_info, 18.0, 35.0)
    loss_32_50 = _window_abs_from_diff(diff_info, 32.0, 50.0)
    loss_45_65 = _window_abs_from_diff(diff_info, 45.0, 65.0)
    loss_100_128 = _window_abs_from_diff(diff_info, 100.0, 128.0)
    loss_126_142 = _window_abs_from_diff(diff_info, 126.0, 142.0)
    loss_134_165 = _window_abs_from_diff(diff_info, 134.0, 165.0)
    loss_global = _window_abs_from_diff(diff_info, 0.1, 240.0)

    carbonyl_loss = 1.6 * loss_160_170 + 2.2 * loss_172_180 + 1.0 * loss_186_205
    tail_loss = 1.4 * loss_8_18 + 1.2 * loss_18_35 + 1.6 * loss_32_50 + 1.1 * loss_45_65
    aromatic_support_loss = 1.0 * loss_100_128 + 1.3 * loss_126_142 + 1.1 * loss_134_165
    global_loss = 0.35 * loss_global
    score = - (1.0 * carbonyl_loss + 0.9 * tail_loss + 0.85 * aromatic_support_loss + global_loss) + 7.5 * r2
    return {
        'score': float(score),
        'r2': float(r2),
        'carbonyl_loss': float(carbonyl_loss),
        'tail_loss': float(tail_loss),
        'aromatic_support_loss': float(aromatic_support_loss),
        'global_loss': float(global_loss),
    }


def _evaluate_required_hist_constraints(H, E_target, cfg) -> Dict[str, Any]:
    H_cpu = H.detach().cpu().long()
    E_target_cpu = E_target.detach().cpu().float()
    E_pred = torch.matmul(H_cpu.float(), E_SU.cpu())

    target_H = float(E_target_cpu[1].item())
    current_H = float(E_pred[1].item())
    h_tol = max(0.04, float(cfg.get('su22_h_tol', 0.03)))
    if target_H > 1e-8:
        h_rel = abs(current_H - target_H) / target_H
        h_ok = bool(h_rel <= h_tol + 1e-9)
    else:
        h_rel = 0.0
        h_ok = True

    n22 = int(H_cpu[22].item()) if int(H_cpu.numel()) > 22 else 0
    n23 = int(H_cpu[23].item()) if int(H_cpu.numel()) > 23 else 0
    su22_ratio = max(0.0, float(cfg.get('su22_ratio', 0.1)))
    req22 = max(1, int(np.ceil(float(su22_ratio) * float(n23)))) if n23 > 0 else 0
    su22_ok = True if n23 <= 0 else bool(n22 >= req22)

    even10_ok = True
    if int(H_cpu.numel()) > 10:
        even10_ok = bool(int(H_cpu[10].item()) % 2 == 0)

    unsat_even_ok = True
    if int(H_cpu.numel()) > 16:
        unsat_total = int(H_cpu[14].item()) + int(H_cpu[15].item()) + int(H_cpu[16].item())
        unsat_even_ok = bool(int(unsat_total) % 2 == 0)
    else:
        unsat_total = 0

    return {
        'ok': bool(h_ok and su22_ok and even10_ok and unsat_even_ok),
        'h_ok': bool(h_ok),
        'h_rel': float(h_rel),
        'h_tol': float(h_tol),
        'su22_ok': bool(su22_ok),
        'req22': int(req22),
        'n22': int(n22),
        'n23': int(n23),
        'even10_ok': bool(even10_ok),
        'unsat_even_ok': bool(unsat_even_ok),
        'unsat_total': int(unsat_total),
    }


# ========================================================================
# Helper Functions
# ========================================================================

class _PlaceholderDecoder(torch.nn.Module):
    """Compatibility stub for pipeline initialization when G2S is unavailable."""

    def forward(self, su_feat, z, g_embed):
        batch = int(su_feat.size(0))
        return torch.zeros((batch, 2), dtype=su_feat.dtype, device=su_feat.device)


class _PlaceholderVAE(torch.nn.Module):
    """Minimal VAE-compatible stub; used only when Layer2/3 are disabled."""

    def __init__(self):
        super().__init__()
        self.decoder = _PlaceholderDecoder()
        self.global_mlp = torch.nn.Sequential(torch.nn.Linear(6, 2), torch.nn.ReLU())
        for p in self.parameters():
            torch.nn.init.zeros_(p)

    def forward(self, data):
        raise RuntimeError("Placeholder VAE does not support forward(); disable Layer2/3.")


def _load_models(s2n_path, g2s_path, device):
    """Load S2N and G2S models, return (s2n_model, vae_model, g2s_available)."""
    s2n_ckpt = Path(s2n_path)
    if not s2n_ckpt.exists():
        print(f"Error: S2N checkpoint not found: {s2n_ckpt}")
        return None, None, False

    s2n_model = S2NModel(hid=256, n_su_types=NUM_SU_TYPES)
    try:
        ckpt = torch.load(s2n_ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(s2n_ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        s2n_model.load_state_dict(ckpt['model_state_dict'])
    else:
        s2n_model.load_state_dict(ckpt)
    s2n_model = s2n_model.to(device).eval()
    print(f"  S2N loaded: {s2n_ckpt}")

    g2s_ckpt = Path(g2s_path)
    if not g2s_ckpt.exists():
        print(f"  Warning: G2S not found: {g2s_ckpt}, using compatibility placeholder")
        vae_model = _PlaceholderVAE().to(device).eval()
        g2s_available = False
    else:
        vae_model = NMR_VAE(n_node_feat=39, n_edge_feat=2, n_su_feat=NUM_SU_TYPES, latent_dim=16, hid=384)
        try:
            ckpt = torch.load(g2s_ckpt, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(g2s_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            vae_model.load_state_dict(ckpt['model_state_dict'])
        else:
            vae_model.load_state_dict(ckpt)
        vae_model = vae_model.to(device).eval()
        print(f"  G2S loaded: {g2s_ckpt}")
        g2s_available = True

    return s2n_model, vae_model, g2s_available


def _print_element_comparison(H, E_target, device):
    """Print SU distribution and element comparison, return E_final."""
    E_final = torch.matmul(H.float(), E_SU.to(device))
    elem_names = ['C', 'H', 'O', 'N', 'S', 'X']

    print("\n[SU分布] (非零项):")
    for i in range(NUM_SU_TYPES):
        c = int(H[i].item())
        if c > 0:
            print(f"  {i:2d} {SU_NAMES[i]:30s}: {c:3d}")
    print(f"\n总SU数量: {int(H.sum().item())}")

    print(f"\n{'元素':<6} {'目标':>10} {'实际':>10} {'误差':>10} {'相对误差':>10}")
    print("-" * 50)
    for i, name in enumerate(elem_names):
        t, a = float(E_target[i].item()), float(E_final[i].item())
        d = a - t
        r = abs(d) / (t + 1e-6) * 100
        print(f"{name:<6} {t:>10.1f} {a:>10.1f} {d:>+10.1f} {r:>9.2f}%")

    # Connection verification
    co = int(H[0].item()) + int(H[1].item()) + int(H[2].item()) + int(H[3].item()) * 2
    print(f"\n  C=O连接: 需求={co}, SU9={int(H[9].item())}")
    o_conn = int(H[2].item()) + int(H[28].item()) + int(H[29].item()) * 2
    print(f"  -O-连接: 需求={o_conn}, SU5+19={int(H[5].item()) + int(H[19].item())}")
    nh = int(H[0].item()) + int(H[27].item()) * 2
    print(f"  -NH-连接: 需求={nh}, SU6+20={int(H[6].item()) + int(H[20].item())}")
    s_conn = int(H[31].item()) * 2
    print(f"  -S-连接: 需求={s_conn}, SU7={int(H[7].item())}")
    x_conn = int(H[32].item())
    print(f"  -X连接: 需求={x_conn}, SU8+21={int(H[8].item()) + int(H[21].item())}")

    return E_final


def _run_layer12(pipeline, H, S_target, E_target, cfg, out_dir, enable_layer2):
    """Run Layer1 → Layer2 (optional) → compute diff."""
    eval_lib = cfg.get('_resolved_eval_lib', cfg['eval_lib'])
    hwhm = float(cfg['eval_hwhm'])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    layer1_buf = io.StringIO()
    with redirect_stdout(layer1_buf):
        nodes = pipeline.layer1_assign(
            H_init=H, S_target=S_target, E_target=E_target,
            eval_nmr=True,
            eval_output_dir=str(Path(out_dir) / 'layer1_eval'),
            eval_lib_path=eval_lib, eval_hwhm=hwhm,
            eval_allow_approx=bool(cfg['eval_allow_approx']),
            enable_carbonyl_joint_adjust=bool(cfg['enable_carbonyl_joint_adjust']),
            carbonyl_joint_iterations=int(cfg['carbonyl_joint_iterations']),
            carbonyl_joint_max_adjustments=int(cfg['carbonyl_joint_max_adjustments']),
            carbonyl_joint_pos_threshold=float(cfg['carbonyl_joint_pos_threshold']),
            carbonyl_joint_neg_threshold=float(cfg['carbonyl_joint_neg_threshold']),
            enable_hop1_adjust=bool(cfg['hop1_adjust']),
            hop1_adjust_iterations=int(cfg['hop1_iterations']),
            hop1_neg_threshold=float(cfg['hop1_neg_threshold']),
            hop1_pos_threshold=float(cfg['hop1_pos_threshold']),
        )
    for line in layer1_buf.getvalue().splitlines():
        if line.startswith("[Layer1-NMR-Eval] carbon_nodes=") or line.startswith("[Layer1-NMR-Eval] R2="):
            print(line)

    if enable_layer2:
        try:
            nodes = pipeline.layer2_assign(
                nodes=nodes, H_center=pipeline._histogram_from_nodes(nodes),
                S_target=S_target, E_target=E_target,
                lib_path=eval_lib, output_dir=str(Path(out_dir) / 'layer2_eval'),
                eval_hwhm=hwhm)
        except Exception as e:
            print(f"  [Layer2] 失败: {e}")

    if enable_layer2:
        diff_info = pipeline._compute_difference_spectrum_from_nodes_mu(
            nodes=nodes, S_target=S_target, E_target=E_target, hwhm=hwhm)
    else:
        diff_info = pipeline._compute_layer1_difference_spectrum(
            nodes=nodes,
            S_target=S_target,
            lib_path=eval_lib,
            hwhm=hwhm,
            allow_approx=bool(cfg['eval_allow_approx']),
        )
    return nodes, diff_info


def _run_layer3_refine(pipeline, nodes, S_target, E_target, cfg, out_dir, enable_layer2, enable_layer3):
    """Run Layer3 once on top of the current Layer1-2 result."""
    eval_lib = cfg.get('_resolved_eval_lib', cfg['eval_lib'])
    hwhm = float(cfg['eval_hwhm'])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if enable_layer3 and enable_layer2:
        try:
            nodes, _ = pipeline.layer3_adjust_templates(
                nodes=nodes, S_target=S_target, E_target=E_target,
                max_iters=int(cfg['layer3_max_iters']),
                lib_path=eval_lib,
                output_dir=str(Path(out_dir) / 'layer3_eval'),
                hwhm=hwhm,
                pos_search_window=float(cfg['layer3_pos_window']),
                neg_assign_window=float(cfg['layer3_neg_window']),
                top_k_samples=int(cfg['layer3_top_k']),
                enable_approx_hop2_template_adjust=bool(cfg['layer3_approx_hop2']),
                approx_hop2_max_iters=(int(cfg['layer3_approx_hop2_max_iters'])
                                       if cfg['layer3_approx_hop2_max_iters'] is not None else None),
                approx_hop2_max_diff_nodes=int(cfg['layer3_approx_hop2_max_diff_nodes']),
                approx_hop2_top_k_templates=int(cfg['layer3_approx_hop2_top_k']),
            )
        except Exception as e:
            print(f"  [Layer3] 失败: {e}")
    elif enable_layer3 and not enable_layer2:
        print("  [Layer3] 已跳过: Layer3 依赖 Layer2 的模板/z 初始化")

    if enable_layer2:
        diff_info = pipeline._compute_difference_spectrum_from_nodes_mu(
            nodes=nodes, S_target=S_target, E_target=E_target, hwhm=hwhm)
    else:
        diff_info = pipeline._compute_layer1_difference_spectrum(
            nodes=nodes,
            S_target=S_target,
            lib_path=eval_lib,
            hwhm=hwhm,
            allow_approx=bool(cfg['eval_allow_approx']),
        )
    return nodes, diff_info

def _run_single_stage_cycle(pipeline, stage, H_curr, nodes, diff_info, S_target, E_target,
                            cfg, stage_params, work_dir, enable_layer2, enable_layer3,
                            outer_cycle, stage_cycle_idx):
    cycle_dir = Path(work_dir) / stage / f"outer_{outer_cycle + 1}" / f"cycle_{stage_cycle_idx + 1}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    H_base = pipeline._histogram_from_nodes(nodes)
    prev_H = H_curr.detach().clone()
    prev_nodes = copy.deepcopy(nodes)
    prev_diff_info = copy.deepcopy(diff_info)
    prev_r2 = float(diff_info.get('r2', 0.0))
    prev_stage_score_info = _compute_stage_score(diff_info, stage)
    prev_stage_score = float(prev_stage_score_info['score'])
    prev_overall_score_info = _compute_overall_score(diff_info)
    prev_overall_score = float(prev_overall_score_info['score'])
    prev_h_rotation_state = int(getattr(pipeline.layer4_adjuster, '_h_rotation_state', 0))
    prev_block_c_rotation = int(getattr(pipeline.layer4_adjuster, '_block_c_aro23_rotation', 0))

    kwargs = dict(stage_params.get(stage, {}))
    kwargs['enable_su22_adjust'] = False if stage == 'block_c' else True
    kwargs['su22_ratio'] = cfg['su22_ratio']
    kwargs['su22_h_tol'] = cfg['su22_h_tol']

    try:
        with redirect_stdout(io.StringIO()):
            H_next, moves, meta = pipeline.layer4_adjuster.adjust_by_stage(
                H=H_base, ppm=diff_info.get('ppm'), diff=diff_info.get('diff'),
                E_target=E_target, S_target=S_target, stage=stage, nodes=nodes, **kwargs)
        n_moves = len(moves)
        _print_h_rotation_summary(meta.get('h_rotation_meta'), header=f'{stage.upper()} H轮动')
        if stage == 'skeleton':
            post_meta = meta.get('post_meta', {}) or {}
            _print_h_rotation_summary(post_meta.get('post_constraint_h_rotation'), header='SKELETON 后约束H轮动')
        _print_stage_h_changes(stage.upper(), H_base, H_next)
        _print_stage_move_summary(stage.upper(), moves)
    except Exception as e:
        print(f"  [Layer4-{stage.upper()}] 失败: {e}")
        import traceback; traceback.print_exc()
        return {
            'accepted': False,
            'changed': False,
            'H_curr': prev_H,
            'nodes': prev_nodes,
            'diff_info': prev_diff_info,
            'r2': prev_r2,
            'overall_score_info': prev_overall_score_info,
            'meta': {'ok': False, 'error': str(e)},
            'n_moves': 0,
        }

    if n_moves == 0:
        print("  无调整，提前结束")
        return {
            'accepted': False,
            'changed': False,
            'H_curr': prev_H,
            'nodes': prev_nodes,
            'diff_info': prev_diff_info,
            'r2': prev_r2,
            'overall_score_info': prev_overall_score_info,
            'meta': meta,
            'n_moves': 0,
        }

    H_work = H_next.detach().clone()
    new_nodes, new_diff_info = _run_layer12(
        pipeline, H_work, S_target, E_target, cfg, str(cycle_dir), enable_layer2)
    new_r2 = float(new_diff_info.get('r2', 0.0))
    new_stage_score_info = _compute_stage_score(new_diff_info, stage)
    new_stage_score = float(new_stage_score_info['score'])
    new_overall_score_info = _compute_overall_score(new_diff_info)
    new_overall_score = float(new_overall_score_info['score'])

    if stage == 'skeleton':
        accept = True
    else:
        accept = bool(
            new_stage_score > prev_stage_score + max(float(cfg['outer_improve_eps']), 1e-4)
            and new_overall_score >= prev_overall_score - 0.5
        )

    if accept:
        if stage == 'skeleton':
            print("\n[Skeleton Candidate Summary] ACCEPTED")
            _print_skeleton_meta_summary(meta)
        print(
            f"  [{stage} 第{stage_cycle_idx+1}轮] 调整={n_moves}次, 接受, "
            f"R²={new_r2:.4f}, stage_score={new_stage_score:.4f}, overall={new_overall_score:.4f}"
        )
        return {
            'accepted': True,
            'changed': True,
            'H_curr': H_work,
            'nodes': new_nodes,
            'diff_info': new_diff_info,
            'r2': new_r2,
            'overall_score_info': new_overall_score_info,
            'meta': meta,
            'n_moves': n_moves,
        }

    if stage == 'skeleton':
        print("\n[Skeleton Candidate Summary] REJECTED")
        _print_skeleton_meta_summary(meta)
    print(
        f"  [{stage} 第{stage_cycle_idx+1}轮] 调整={n_moves}次, 拒绝, "
        f"R²={new_r2:.4f}, stage_score={new_stage_score:.4f} (prev={prev_stage_score:.4f}), "
        f"overall={new_overall_score:.4f} (prev={prev_overall_score:.4f})"
    )
    pipeline.layer4_adjuster._h_rotation_state = int(prev_h_rotation_state)
    pipeline.layer4_adjuster._block_c_aro23_rotation = int(prev_block_c_rotation)
    return {
        'accepted': False,
        'changed': True,
        'H_curr': prev_H,
        'nodes': prev_nodes,
        'diff_info': prev_diff_info,
        'r2': prev_r2,
        'overall_score_info': prev_overall_score_info,
        'meta': meta,
        'n_moves': n_moves,
    }


def _evaluate_allocation_balance(pipeline, nodes, S_target, E_target, cfg):
    strict_diag = pipeline.layer4_adjuster._evaluate_full_allocation_balance(
        nodes,
        flex_ratio=float(cfg['extra_flexible_ratio']),
        flex_lower_extra=int(cfg['extra_flexible_lower_extra']),
        S_target=S_target,
        E_target=E_target,
    )
    relaxed_diag = pipeline.layer4_adjuster._evaluate_full_allocation_balance(
        nodes,
        flex_ratio=float(cfg['extra_relaxed_flexible_ratio']),
        flex_lower_extra=int(cfg['extra_relaxed_lower_extra']),
        S_target=S_target,
        E_target=E_target,
    )
    if bool(strict_diag.get('ok', False)):
        selected = dict(strict_diag)
        selected['selected_mode'] = 'strict'
    elif bool(relaxed_diag.get('ok', False)):
        selected = dict(relaxed_diag)
        selected['selected_mode'] = 'relaxed'
    else:
        selected = dict(strict_diag)
        selected['selected_mode'] = 'strict'
    selected['strict_ok'] = bool(strict_diag.get('ok', False))
    selected['relaxed_ok'] = bool(relaxed_diag.get('ok', False))
    selected['strict_reason'] = str(strict_diag.get('reason', 'unknown'))
    selected['relaxed_reason'] = str(relaxed_diag.get('reason', 'unknown'))
    return selected


def _run_cd_coupled_cycle(pipeline, H_curr, nodes, diff_info, S_target, E_target, cfg,
                          stage_params, work_dir, enable_layer2, outer_cycle, inner_idx):
    cycle_dir = Path(work_dir) / "cd_coupled" / f"outer_{outer_cycle + 1}" / f"round_{inner_idx + 1}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    prev_H = H_curr.detach().clone()
    prev_nodes = copy.deepcopy(nodes)
    prev_diff_info = copy.deepcopy(diff_info)
    prev_r2 = float(diff_info.get('r2', 0.0))
    prev_stage_score_info = _compute_stage_score(diff_info, 'block_c')
    prev_stage_score = float(prev_stage_score_info['score'])
    prev_overall_score_info = _compute_overall_score(diff_info)
    prev_overall_score = float(prev_overall_score_info['score'])
    prev_h_rotation_state = int(getattr(pipeline.layer4_adjuster, '_h_rotation_state', 0))
    prev_block_c_rotation = int(getattr(pipeline.layer4_adjuster, '_block_c_aro23_rotation', 0))
    prev_alloc_diag = _evaluate_allocation_balance(pipeline, nodes, S_target, E_target, cfg)
    prev_req_diag = _evaluate_required_hist_constraints(prev_H, E_target, cfg)

    H_stage = pipeline._histogram_from_nodes(nodes)
    tmp_nodes = copy.deepcopy(nodes)
    tmp_diff = copy.deepcopy(diff_info)
    total_moves = 0
    skeleton_moves_total = 0
    meta = {'block_c': [], 'skeleton': []}

    kwargs = dict(stage_params.get('block_c', {}))
    kwargs['su22_ratio'] = cfg['su22_ratio']
    kwargs['su22_h_tol'] = cfg['su22_h_tol']
    H_before = H_stage.detach().clone()
    block_c_buf = io.StringIO()
    with redirect_stdout(block_c_buf):
        H_next, moves, submeta = pipeline.layer4_adjuster.adjust_by_stage(
            H=H_stage,
            ppm=tmp_diff.get('ppm'),
            diff=tmp_diff.get('diff'),
            E_target=E_target,
            S_target=S_target,
            stage='block_c',
            nodes=tmp_nodes,
            **kwargs,
        )
    meta['block_c'].append(submeta)
    if moves:
        total_moves += len(moves)
        H_stage = H_next.detach().clone()
        tmp_nodes, tmp_diff = _run_layer12(
            pipeline, H_stage, S_target, E_target, cfg,
            str(cycle_dir / 'block_c'), enable_layer2)

    kwargs = dict(stage_params.get('skeleton', {}))
    kwargs['su22_ratio'] = cfg['su22_ratio']
    kwargs['su22_h_tol'] = cfg['su22_h_tol']
    H_base = pipeline._histogram_from_nodes(tmp_nodes)
    skeleton_buf = io.StringIO()
    with redirect_stdout(skeleton_buf):
        H_next, moves, submeta = pipeline.layer4_adjuster.adjust_by_stage(
            H=H_base,
            ppm=tmp_diff.get('ppm'),
            diff=tmp_diff.get('diff'),
            E_target=E_target,
            S_target=S_target,
            stage='skeleton',
            nodes=tmp_nodes,
            **kwargs,
        )
    meta['skeleton'].append(submeta)
    _print_stage_h_changes('SKELETON', H_base, H_next)
    if moves:
        total_moves += len(moves)
        skeleton_moves_total += len(moves)
        H_stage = H_next.detach().clone()
        tmp_nodes, tmp_diff = _run_layer12(
            pipeline, H_stage, S_target, E_target, cfg,
            str(cycle_dir / 'skeleton'), enable_layer2)

    if total_moves == 0:
        print("  [CD Coupled] 无调整，提前结束")
        return {
            'accepted': False,
            'changed': False,
            'H_curr': prev_H,
            'nodes': prev_nodes,
            'diff_info': prev_diff_info,
            'r2': prev_r2,
            'overall_score_info': prev_overall_score_info,
            'meta': meta,
            'n_moves': 0,
        }

    alloc_diag = _evaluate_allocation_balance(pipeline, tmp_nodes, S_target, E_target, cfg)
    new_req_diag = _evaluate_required_hist_constraints(H_stage, E_target, cfg)
    new_r2 = float(tmp_diff.get('r2', 0.0))
    new_stage_score_info = _compute_stage_score(tmp_diff, 'block_c')
    new_stage_score = float(new_stage_score_info['score'])
    new_overall_score_info = _compute_overall_score(tmp_diff)
    new_overall_score = float(new_overall_score_info['score'])
    mandatory_alloc_fix = bool((not prev_alloc_diag.get('ok', False)) and alloc_diag.get('ok', False))
    mandatory_hist_fix = bool((not prev_req_diag.get('ok', False)) and new_req_diag.get('ok', False))
    mandatory_accept = bool(mandatory_alloc_fix or mandatory_hist_fix)
    mandatory_failed = bool((not alloc_diag.get('ok', False)) or (not new_req_diag.get('ok', False)))
    score_improved = bool(
        new_stage_score > prev_stage_score + max(float(cfg['outer_improve_eps']), 1e-4)
        and new_overall_score >= prev_overall_score - 0.5
    )
    if mandatory_failed:
        accept = False
        accept_reason = 'constraint_fail'
    elif mandatory_accept:
        accept = True
        reasons = []
        if mandatory_alloc_fix:
            reasons.append('alloc_fix')
        if mandatory_hist_fix:
            reasons.append('hist_fix')
        accept_reason = '+'.join(reasons) if reasons else 'mandatory'
    else:
        accept = bool(score_improved)
        accept_reason = 'score_improve' if accept else 'score_regress'
    meta['allocation_check'] = alloc_diag
    meta['required_hist_before'] = prev_req_diag
    meta['required_hist_after'] = new_req_diag
    meta['accept_reason'] = str(accept_reason)

    if accept:
        print(
            f"  [CD Coupled 第{inner_idx + 1}轮] 调整={total_moves}次, 接受, "
            f"R²={new_r2:.4f}, stage_score={new_stage_score:.4f}, overall={new_overall_score:.4f}, "
            f"alloc={alloc_diag.get('selected_mode', 'strict')}/{alloc_diag.get('reason', 'ok')}, "
            f"reason={accept_reason}"
        )
        return {
            'accepted': True,
            'changed': True,
            'H_curr': H_stage,
            'nodes': tmp_nodes,
            'diff_info': tmp_diff,
            'r2': new_r2,
            'overall_score_info': new_overall_score_info,
            'meta': meta,
            'n_moves': total_moves,
        }

    print(
        f"  [CD Coupled 第{inner_idx + 1}轮] 调整={total_moves}次, 拒绝, "
        f"R²={new_r2:.4f}, stage_score={new_stage_score:.4f} (prev={prev_stage_score:.4f}), "
        f"overall={new_overall_score:.4f} (prev={prev_overall_score:.4f}), "
        f"alloc={alloc_diag.get('selected_mode', 'strict')}/{alloc_diag.get('reason', 'unknown')}, "
        f"reason={accept_reason}"
    )
    print("  [CD Coupled] 候选已回滚，以上 Skeleton 资源分配结果不代表最终采用结果")
    pipeline.layer4_adjuster._h_rotation_state = int(prev_h_rotation_state)
    pipeline.layer4_adjuster._block_c_aro23_rotation = int(prev_block_c_rotation)
    return {
        'accepted': False,
        'changed': True,
        'H_curr': prev_H,
        'nodes': prev_nodes,
        'diff_info': prev_diff_info,
        'r2': prev_r2,
        'overall_score_info': prev_overall_score_info,
        'meta': meta,
        'n_moves': total_moves,
    }


def _run_cd_inner_loop(pipeline, H_curr, nodes, diff_info, S_target, E_target, cfg,
                       stage_params, work_dir, enable_layer2, enable_layer3, outer_cycle):
    cd_loops = max(1, int(cfg['cd_inner_max_loops']))
    improved = False
    current_H = H_curr
    current_nodes = nodes
    current_diff = diff_info
    current_r2 = float(diff_info.get('r2', 0.0))
    current_overall = _compute_overall_score(diff_info)

    for inner_idx in range(cd_loops):
        print(f"\n{'-'*60}")
        print(f"CD 耦合内循环 {inner_idx + 1}/{cd_loops}")
        print(f"{'-'*60}")

        res = _run_cd_coupled_cycle(
            pipeline, current_H, current_nodes, current_diff, S_target, E_target,
            cfg, stage_params, work_dir, enable_layer2, outer_cycle, inner_idx)
        current_H = res['H_curr']
        current_nodes = res['nodes']
        current_diff = res['diff_info']
        current_r2 = res['r2']
        current_overall = res['overall_score_info']

        if res['accepted']:
            improved = True
        elif not res['changed']:
            print("[CD内循环] 无进一步改善，停止")
            break
        else:
            print("[CD内循环] 候选被拒绝，停止")
            break

    return current_H, current_nodes, current_diff, current_r2, current_overall, improved


def _print_layer1_analysis(nodes):
    """Print Layer1 node statistics."""
    print("\n" + "=" * 80)
    print("Layer1 分析结果")
    print("=" * 80)

    su_counts = Counter(n.su_type for n in nodes)
    complete = sum(1 for n in nodes if sum(n.hop1_su.values()) >= int(SU_CONNECTION_DEGREE.get(n.su_type, 4) if not isinstance(SU_CONNECTION_DEGREE.get(n.su_type, 4), tuple) else SU_CONNECTION_DEGREE.get(n.su_type, 4)[1]))
    total = max(1, len(nodes))

    print(f"\n  总节点数: {len(nodes)}")
    print(f"  已完成hop1: {complete}/{len(nodes)} ({complete / total * 100:.1f}%)")

    print(f"\n  各SU类型hop1状态:")
    for st in sorted(su_counts.keys()):
        su_nodes = [n for n in nodes if n.su_type == st]
        md = SU_CONNECTION_DEGREE.get(st, 4)
        if isinstance(md, tuple):
            md = md[1]
        full = sum(1 for n in su_nodes if sum(n.hop1_su.values()) == md)
        incomplete = sum(1 for n in su_nodes if 0 < sum(n.hop1_su.values()) < md)
        empty = sum(1 for n in su_nodes if sum(n.hop1_su.values()) == 0)
        status = "✓" if incomplete == 0 and empty == 0 else "⚠"
        print(f"  {status} SU{st:2d} ({SU_NAMES[st]:25s}): n={len(su_nodes):3d} full={full:3d} incomplete={incomplete:3d} empty={empty:3d} (deg≤{md})")


def _print_captured_stage_output(stage: str, captured: str):
    lines = [line.rstrip("\n") for line in str(captured).splitlines()]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return
    print(f"\n  [{stage}] 详细过程:")
    for line in lines:
        print(line)


def _print_h_rotation_summary(h_meta: Optional[Dict[str, Any]], header: str = "H轮动"):
    if not h_meta:
        return
    ops = list(h_meta.get('ops', []) or [])
    applied = bool(h_meta.get('applied', False)) or bool(ops)
    if not applied:
        return

    def _ratio_pct(v):
        try:
            return float(v) * 100.0
        except Exception:
            return 0.0

    before = _ratio_pct(h_meta.get('before_ratio', 0.0))
    after = _ratio_pct(h_meta.get('after_ratio', 0.0))
    state = int(h_meta.get('rotation_state', -1))
    print(f"    [{header}] before={before:+.2f}% -> after={after:+.2f}% | rotation_state={state}")
    if ops:
        print(f"      ops: {' | '.join(str(op) for op in ops)}")


def _summarize_h_changes(H_before, H_after):
    rows = []
    for idx in range(int(min(H_before.numel(), H_after.numel()))):
        before = int(H_before[idx].item())
        after = int(H_after[idx].item())
        if before != after:
            rows.append((idx, before, after, after - before))
    return rows


def _print_stage_h_changes(stage: str, H_before, H_after, limit: int = 12):
    rows = _summarize_h_changes(H_before, H_after)
    if not rows:
        print(f"  [{stage}] H无变化")
        return
    print(f"  [{stage}] 结构单元调整:")
    for idx, before, after, delta in rows[:limit]:
        print(f"    SU{idx:02d} {SU_NAMES[idx]:25s}: {before} -> {after} ({delta:+d})")
    if len(rows) > limit:
        print(f"    ... 其余 {len(rows) - limit} 项略")


def _print_stage_move_summary(stage: str, moves, limit: int = 10):
    if not moves:
        return
    print(f"  [{stage}] 调整动作:")
    for mv in moves[:limit]:
        if isinstance(mv, dict):
            parts = []
            if 'stage' in mv:
                parts.append(f"stage={mv['stage']}")
            if 'op' in mv:
                parts.append(str(mv['op']))
            if 'from' in mv and 'to' in mv:
                parts.append(f"{mv['from']}->{mv['to']}")
            if 'delta' in mv:
                delta = mv.get('delta', {}) or {}
                delta_txt = ", ".join(f"{int(k)}:{int(v):+d}" for k, v in delta.items())
                if delta_txt:
                    parts.append(f"delta[{delta_txt}]")
            if 'block' in mv:
                parts.append(f"block={mv['block']}")
            if 'substage' in mv:
                parts.append(f"sub={mv['substage']}")
            print(f"    - {' | '.join(parts) if parts else str(mv)}")
        else:
            print(f"    - {mv}")
    if len(moves) > limit:
        print(f"    ... 其余 {len(moves) - limit} 条略")


def _print_skeleton_meta_summary(meta: Optional[Dict[str, Any]]):
    """Print a compact final summary for Layer4 skeleton allocation diagnostics."""
    if not meta:
        print("\n[Skeleton Summary] meta为空")
        return

    branch_meta = meta.get('branch_meta', {}) or {}
    extra_meta = meta.get('extra_meta', {}) or {}
    align_meta = meta.get('align_meta', {}) or {}
    final_alloc = meta.get('final_allocation', {}) or {}
    phase_hists = meta.get('phase_hists', {}) or {}
    phase_moves = meta.get('phase_moves', {}) or {}

    def _fmt_bool(v):
        return 'OK' if bool(v) else 'FAIL'

    def _fmt_float(v, default=0.0, digits=4):
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return f"{float(default):.{digits}f}"

    def _print_phase_delta(tag: str, before_key: str, after_key: str):
        before = phase_hists.get(before_key)
        after = phase_hists.get(after_key)
        if before is None or after is None:
            return
        _print_stage_h_changes(f'SKELETON-{tag}', before, after)
        _print_stage_move_summary(f'SKELETON-{tag}', phase_moves.get(tag.lower(), []))

    print("\n" + "=" * 80)
    print("Skeleton 资源分配最终汇总")
    print("=" * 80)
    print(
        f"  总体: scenario={meta.get('final_scenario', 'unknown')} | "
        f"branch={_fmt_bool(meta.get('branch_ok', meta.get('ok', False)))} | "
        f"extra={_fmt_bool(meta.get('extra_ok', meta.get('ok', False)))} | "
        f"moves={int(meta.get('n_moves', 0))} | "
        f"H偏差={_fmt_float(meta.get('final_h_ratio', 0.0), digits=4)}"
    )
    post_meta = meta.get('post_meta', {}) or {}
    _print_h_rotation_summary(post_meta.get('post_constraint_h_rotation'), header='Skeleton 最终H轮动')

    cluster_meta = (
        final_alloc.get('cluster_meta', {}) or
        align_meta.get('after', {}) or
        extra_meta.get('final_diag', {}).get('cluster_meta', {}) or
        {}
    )
    if cluster_meta:
        kind_counts = cluster_meta.get('cluster_kind_counts', {}) or {}
        kind_desc = ", ".join(f"{k}={int(v)}" for k, v in kind_counts.items()) if kind_counts else "none"
        print("\n  [Aromatic Conversion]")
        print(
            f"    Converted 13={_fmt_float(cluster_meta.get('converted_13', 0.0), digits=1)} | "
            f"Converted 12={int(cluster_meta.get('converted_12', 0))} | "
            f"used 12->13={int(cluster_meta.get('used_12_to_13', 0))} | "
            f"used 13->12={int(cluster_meta.get('used_13_to_12', 0))}"
        )
        print(
            f"    Clusters={int(cluster_meta.get('cluster_count', 0)) if 'cluster_count' in cluster_meta else int(final_alloc.get('cluster_count', 0))} | "
            f"kinds: {kind_desc}"
        )

    print("\n  [Skeleton 子阶段结构单元变化]")
    _print_phase_delta('Branch', 'input', 'after_branch')
    _print_phase_delta('Extra', 'after_branch', 'after_extra')
    _print_phase_delta('Align', 'after_extra', 'after_align')
    _print_phase_delta('Post', 'after_align', 'after_post')

    print("\n  [Branch 阶段]")
    print(
        f"    status={_fmt_bool(branch_meta.get('ok', False))} | "
        f"scenario={branch_meta.get('final_scenario', branch_meta.get('phase', 'branch'))} | "
        f"moves={int(branch_meta.get('n_moves', 0))} | "
        f"H偏差={_fmt_float(branch_meta.get('final_h_ratio', 0.0), digits=4)}"
    )
    branch_diag = branch_meta.get('final_diag', {}) or {}
    if branch_diag:
        rem = branch_diag.get('remaining', {}) or {}
        print(
            f"    branch诊断: shortage={branch_diag.get('shortage_type', 'none')} | "
            f"unallocated_branch={int(branch_diag.get('unallocated_branch', 0))} | "
            f"req11={int(branch_diag.get('req_11', 0))} "
            f"req22={int(branch_diag.get('req_22', 0))} "
            f"req23={int(branch_diag.get('req_23', 0))}"
        )
        if rem:
            print(
                f"    branch剩余: 11×{int(rem.get('11', 0))} "
                f"22×{int(rem.get('22', 0))} "
                f"23×{int(rem.get('23', 0))} "
                f"24×{int(rem.get('24', 0))} "
                f"25×{int(rem.get('25', 0))}"
            )
        branch_rows = list(branch_diag.get('branch_chains', []) or [])
        if branch_rows:
            print("    branch结构:")
            for idx, row in enumerate(branch_rows):
                print(f"      [{idx}] {row}")

    print("\n  [Extra 阶段]")
    print(
        f"    status={_fmt_bool(extra_meta.get('ok', False))} | "
        f"scenario={extra_meta.get('final_scenario', extra_meta.get('reason', 'extra'))} | "
        f"moves={int(extra_meta.get('n_moves', 0))} | "
        f"H偏差={_fmt_float(extra_meta.get('final_h_ratio', 0.0), digits=4)}"
    )
    extra_diag = extra_meta.get('final_diag', {}) or {}
    if extra_diag:
        rem = extra_diag.get('remaining', {}) or {}
        print(
            f"    extra诊断: reason={extra_diag.get('reason', 'ok')} | "
            f"clusters={int(extra_diag.get('cluster_count', 0))} | "
            f"rigid10={int(extra_diag.get('rigid_pairs', 0))} | "
            f"flex={int(extra_diag.get('flexible_bridge_count', 0))}/"
            f"[{int(extra_diag.get('flexible_bridge_min', 0))},{int(extra_diag.get('flexible_bridge_limit', 0))}] | "
            f"side22={int(extra_diag.get('side_to_22_count', 0))} | "
            f"ali={int(extra_diag.get('aliphatic_total', 0))}/"
            f"[{int(extra_diag.get('aliphatic_min_total', 0))},{int(extra_diag.get('aliphatic_max_total', 0))}]"
        )
        if rem:
            print(
                f"    extra剩余: 11×{int(rem.get('11', 0))} "
                f"22×{int(rem.get('22', 0))} "
                f"23×{int(rem.get('23', 0))} "
                f"24×{int(rem.get('24', 0))} "
                f"25×{int(rem.get('25', 0))}"
            )

    print("\n  [Align 阶段]")
    print(
        f"    applied={_fmt_bool(align_meta.get('applied', False))} | "
        f"req12->13={int(align_meta.get('requested_12_to_13', 0))} | "
        f"req13->12={int(align_meta.get('requested_13_to_12', 0))} | "
        f"done12->13={int(align_meta.get('applied_12_to_13', 0))} | "
        f"done13->12={int(align_meta.get('applied_13_to_12', 0))}"
    )

    print("\n  [Final Allocation]")
    if final_alloc:
        print(
            f"    未分配: bridge={int(final_alloc.get('unallocated_bridge', 0))} | "
            f"branch={int(final_alloc.get('unallocated_branch', 0))}"
        )
        print(
            f"    评估模式: selected={final_alloc.get('selected_mode', 'unknown')} | "
            f"strict={_fmt_bool(final_alloc.get('strict_ok', False))} "
            f"({final_alloc.get('strict_reason', 'n/a')}) | "
            f"relaxed={_fmt_bool(final_alloc.get('relaxed_ok', False))} "
            f"({final_alloc.get('relaxed_reason', 'n/a')})"
        )
        print(
            f"    额外需求: 11×{int(final_alloc.get('required_extra_11', 0))} "
            f"22×{int(final_alloc.get('required_extra_22', 0))} "
            f"23×{int(final_alloc.get('required_extra_23', 0))}"
        )
        print(
            f"    最终剩余: 11×{int(final_alloc.get('remaining_11', 0))} "
            f"22×{int(final_alloc.get('remaining_22', 0))} "
            f"23×{int(final_alloc.get('remaining_23', 0))} "
            f"24×{int(final_alloc.get('remaining_24', 0))} "
            f"25×{int(final_alloc.get('remaining_25', 0))}"
        )
        print(
            f"    平衡评估: reason={final_alloc.get('reason', 'ok')} "
            f"clusters={int(final_alloc.get('cluster_count', 0))} "
            f"effective_clusters={int(final_alloc.get('effective_cluster_count', 0))} "
            f"rigid_clusters={int(final_alloc.get('rigid_cluster_count', 0))} "
            f"rigid10={int(final_alloc.get('rigid_pairs', 0))} "
            f"flex={int(final_alloc.get('flexible_bridge_count', 0))}/"
            f"[{int(final_alloc.get('flexible_bridge_min', 0))},{int(final_alloc.get('flexible_bridge_limit', 0))}] "
            f"side22={int(final_alloc.get('side_to_22_count', 0))} "
            f"ali={int(final_alloc.get('aliphatic_total', 0))}/"
            f"[{int(final_alloc.get('aliphatic_min_total', 0))},{int(final_alloc.get('aliphatic_max_total', 0))}]"
        )
        alloc_details = final_alloc.get('allocation_details', {}) or {}
        bridge_rows = list(alloc_details.get('bridge_rows', []) or [])
        side_rows = list(alloc_details.get('side_rows', []) or [])
        branch_rows = list(alloc_details.get('branch_rows', []) or [])
        print(
            f"    结构明细: bridge={len(bridge_rows)} | "
            f"side={len(side_rows)} | branch={len(branch_rows)}"
        )
        if bridge_rows:
            print("    [Bridge 明细]")
            for idx, row in enumerate(bridge_rows):
                print(f"      [{idx}] {row}")
        if side_rows:
            print("    [Side 明细]")
            for idx, row in enumerate(side_rows):
                print(f"      [{idx}] {row}")
        if branch_rows:
            print("    [Branch 明细]")
            for idx, row in enumerate(branch_rows):
                print(f"      [{idx}] {row}")
    else:
        print("    无 final_allocation 数据")
    print("=" * 80)


def _print_final_allocation_summary(final_alloc: Optional[Dict[str, Any]]):
    if not final_alloc:
        print("\n[Final Allocation] 无数据")
        return

    def _fmt_bool(v):
        return 'OK' if bool(v) else 'FAIL'

    print("\n" + "=" * 80)
    print("最终节点对应的资源分配校验")
    print("=" * 80)
    print(
        f"  mode={final_alloc.get('selected_mode', 'unknown')} | "
        f"ok={_fmt_bool(final_alloc.get('ok', False))} | "
        f"reason={final_alloc.get('reason', 'unknown')}"
    )
    print(
        f"  strict={_fmt_bool(final_alloc.get('strict_ok', False))} "
        f"({final_alloc.get('strict_reason', 'n/a')}) | "
        f"relaxed={_fmt_bool(final_alloc.get('relaxed_ok', False))} "
        f"({final_alloc.get('relaxed_reason', 'n/a')})"
    )
    print(
        f"  未分配: bridge={int(final_alloc.get('unallocated_bridge', 0))} | "
        f"branch={int(final_alloc.get('unallocated_branch', 0))}"
    )
    print(
        f"  额外需求: 11×{int(final_alloc.get('required_extra_11', 0))} "
        f"22×{int(final_alloc.get('required_extra_22', 0))} "
        f"23×{int(final_alloc.get('required_extra_23', 0))}"
    )
    print(
        f"  最终剩余: 11×{int(final_alloc.get('remaining_11', 0))} "
        f"22×{int(final_alloc.get('remaining_22', 0))} "
        f"23×{int(final_alloc.get('remaining_23', 0))} "
        f"24×{int(final_alloc.get('remaining_24', 0))} "
        f"25×{int(final_alloc.get('remaining_25', 0))}"
    )
    print(
        f"  平衡评估: clusters={int(final_alloc.get('cluster_count', 0))} "
        f"effective_clusters={int(final_alloc.get('effective_cluster_count', 0))} "
        f"rigid10={int(final_alloc.get('rigid_pairs', 0))} "
        f"flex={int(final_alloc.get('flexible_bridge_count', 0))}/"
        f"[{int(final_alloc.get('flexible_bridge_min', 0))},{int(final_alloc.get('flexible_bridge_limit', 0))}] "
        f"side22={int(final_alloc.get('side_to_22_count', 0))} "
        f"ali={int(final_alloc.get('aliphatic_total', 0))}/"
        f"[{int(final_alloc.get('aliphatic_min_total', 0))},{int(final_alloc.get('aliphatic_max_total', 0))}]"
    )
    cluster_meta = final_alloc.get('cluster_meta', {}) or {}
    if cluster_meta:
        kind_counts = cluster_meta.get('cluster_kind_counts', {}) or {}
        kind_desc = ", ".join(f"{k}={int(v)}" for k, v in kind_counts.items()) if kind_counts else "none"
        try:
            converted_13 = float(cluster_meta.get('converted_13', 0.0))
        except Exception:
            converted_13 = 0.0
        converted_13_txt = f"{converted_13:.1f}" if abs(converted_13 - round(converted_13)) > 1e-6 else str(int(round(converted_13)))
        print(
            f"  Aromatic conversion: 13={converted_13_txt} "
            f"12={int(cluster_meta.get('converted_12', 0))} "
            f"| used 12->13={int(cluster_meta.get('used_12_to_13', 0))} "
            f"used 13->12={int(cluster_meta.get('used_13_to_12', 0))}"
        )
        print(f"  Cluster kinds: {kind_desc}")
    alloc_details = final_alloc.get('allocation_details', {}) or {}
    bridge_rows = list(alloc_details.get('bridge_rows', []) or [])
    side_rows = list(alloc_details.get('side_rows', []) or [])
    branch_rows = list(alloc_details.get('branch_rows', []) or [])
    print(
        f"  Allocation details: bridge={len(bridge_rows)} | "
        f"side={len(side_rows)} | branch={len(branch_rows)}"
    )
    if bridge_rows:
        print("  [Bridge 明细]")
        for idx, row in enumerate(bridge_rows):
            print(f"    [{idx}] {row}")
    if side_rows:
        print("  [Side 明细]")
        for idx, row in enumerate(side_rows):
            print(f"    [{idx}] {row}")
    if branch_rows:
        print("  [Branch 明细]")
        for idx, row in enumerate(branch_rows):
            print(f"    [{idx}] {row}")
    print("=" * 80)


def _to_jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k) == 'allocation_result':
                continue
            out[str(k)] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, '__dict__'):
        try:
            return _to_jsonable(vars(obj))
        except Exception:
            return str(obj)
    return str(obj)


def _save_final_outputs(pipeline, nodes, H_final, S_target, E_target, output_dir, cfg, enable_layer2,
                        final_alloc: Optional[Dict[str, Any]] = None):
    """Save final output files."""
    final_dir = Path(output_dir) / "final_outputs"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Run config
    pd.DataFrame([cfg]).to_csv(final_dir / "final_run_config.csv", index=False)

    # SU histogram
    try:
        visualize_su_distribution(H_final.long().cpu(), 'Final', save_dir=str(final_dir))
    except Exception:
        pass

    su_data = [{'su_idx': i, 'su_name': SU_NAMES[i], 'count': int(H_final[i].item())}
               for i in range(NUM_SU_TYPES) if int(H_final[i].item()) > 0]
    pd.DataFrame(su_data).to_csv(final_dir / "final_su_histogram.csv", index=False)

    # Node details
    node_rows = []
    for n in nodes:
        node_rows.append({
            'global_id': int(n.global_id),
            'su_type': int(n.su_type),
            'su_name': SU_NAMES[int(n.su_type)] if int(n.su_type) < len(SU_NAMES) else str(n.su_type),
            'mu': float(getattr(n, 'mu', 0.0)),
            'pi': float(getattr(n, 'pi', 0.0)),
            'hop1_degree': int(sum((getattr(n, 'hop1_su', {}) or {}).values())),
            'hop2_degree': int(sum((getattr(n, 'hop2_su', {}) or {}).values())),
            'template_key': str(getattr(n, 'template_key', None)),
        })
    pd.DataFrame(node_rows).to_csv(final_dir / "final_nodes.csv", index=False)

    if final_alloc:
        summary_row = {
            'ok': bool(final_alloc.get('ok', False)),
            'reason': str(final_alloc.get('reason', 'unknown')),
            'selected_mode': str(final_alloc.get('selected_mode', 'unknown')),
            'strict_ok': bool(final_alloc.get('strict_ok', False)),
            'strict_reason': str(final_alloc.get('strict_reason', 'unknown')),
            'relaxed_ok': bool(final_alloc.get('relaxed_ok', False)),
            'relaxed_reason': str(final_alloc.get('relaxed_reason', 'unknown')),
            'unallocated_bridge': int(final_alloc.get('unallocated_bridge', 0)),
            'unallocated_branch': int(final_alloc.get('unallocated_branch', 0)),
            'required_extra_11': int(final_alloc.get('required_extra_11', 0)),
            'required_extra_22': int(final_alloc.get('required_extra_22', 0)),
            'required_extra_23': int(final_alloc.get('required_extra_23', 0)),
            'remaining_11': int(final_alloc.get('remaining_11', 0)),
            'remaining_22': int(final_alloc.get('remaining_22', 0)),
            'remaining_23': int(final_alloc.get('remaining_23', 0)),
            'remaining_24': int(final_alloc.get('remaining_24', 0)),
            'remaining_25': int(final_alloc.get('remaining_25', 0)),
            'cluster_count': int(final_alloc.get('cluster_count', 0)),
            'rigid_pairs': int(final_alloc.get('rigid_pairs', 0)),
            'flexible_bridge_count': int(final_alloc.get('flexible_bridge_count', 0)),
            'flexible_bridge_min': int(final_alloc.get('flexible_bridge_min', 0)),
            'flexible_bridge_limit': int(final_alloc.get('flexible_bridge_limit', 0)),
            'side_to_22_count': int(final_alloc.get('side_to_22_count', 0)),
            'aliphatic_total': int(final_alloc.get('aliphatic_total', 0)),
            'aliphatic_min_total': int(final_alloc.get('aliphatic_min_total', 0)),
        }
        pd.DataFrame([summary_row]).to_csv(final_dir / "final_allocation_summary.csv", index=False)
        with open(final_dir / "final_allocation_details.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(final_alloc), f, ensure_ascii=False, indent=2)

    # Spectrum comparison + smoothing
    hwhm = float(cfg.get('eval_hwhm', 1.0))
    S_target_np = S_target.detach().cpu().numpy().astype(np.float64).flatten()
    eval_lib = cfg.get('_resolved_eval_lib', cfg['eval_lib'])

    has_node_mu = any(abs(float(getattr(n, 'mu', 0.0))) > 1e-8 and float(getattr(n, 'pi', 0.0)) > 1e-8 for n in nodes)
    if enable_layer2 and has_node_mu:
        diff_info = pipeline._compute_difference_spectrum_from_nodes_mu(
            nodes=nodes, S_target=S_target, E_target=E_target, hwhm=hwhm)
        ppm_np = np.asarray(diff_info.get('ppm'), dtype=np.float64)
        alpha = float(diff_info.get('alpha', 1.0))
        S_recon_np = pipeline.reconstruct_spectrum(
            nodes, E_target=E_target, hwhm=hwhm).detach().cpu().numpy().astype(np.float64).flatten()
        n = min(ppm_np.size, S_target_np.size, S_recon_np.size)
        ppm_np, S_target_np = ppm_np[:n], S_target_np[:n]
        S_fit_np = (alpha * S_recon_np[:n]).astype(np.float64)
        diff_np = S_target_np - S_fit_np
    else:
        diff_info = pipeline._compute_layer1_difference_spectrum(
            nodes=nodes,
            S_target=S_target,
            lib_path=eval_lib,
            hwhm=hwhm,
            allow_approx=bool(cfg['eval_allow_approx']),
        )
        ppm_np = np.asarray(diff_info.get('ppm'), dtype=np.float64)
        alpha = float(diff_info.get('alpha', 1.0))
        diff_np = np.asarray(diff_info.get('diff'), dtype=np.float64)
        n = min(ppm_np.size, S_target_np.size, diff_np.size)
        ppm_np, S_target_np, diff_np = ppm_np[:n], S_target_np[:n], diff_np[:n]
        S_fit_np = (S_target_np - diff_np).astype(np.float64)

    step = float(np.median(np.abs(np.diff(ppm_np)))) if ppm_np.size > 3 else 1.0
    sigma_ppm_val = cfg['final_smooth_sigma_ppm']
    sigma_ppm = float(sigma_ppm_val) if sigma_ppm_val is not None else max(step, 0.25)
    sigma_pts = max(1.0, sigma_ppm / max(1e-12, step))
    radius = int(max(3, round(float(cfg['final_smooth_radius_factor']) * sigma_pts)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-xs ** 2 / (2.0 * sigma_pts ** 2))
    w /= max(1e-12, float(w.sum()))

    S_fit_smooth = S_fit_np.copy()
    for _ in range(max(1, int(cfg['final_smooth_passes']))):
        S_fit_smooth = np.convolve(S_fit_smooth, w, mode='same')
    diff_smooth = S_target_np - S_fit_smooth

    pd.DataFrame({
        'ppm': ppm_np, 'target': S_target_np,
        'reconstructed': S_fit_np, 'reconstructed_smooth': S_fit_smooth,
        'difference': diff_np, 'difference_smooth': diff_smooth,
    }).to_csv(final_dir / "final_spectrum_comparison.csv", index=False)

    for label, recon in [('Final', S_fit_np), ('FinalSmooth', S_fit_smooth)]:
        try:
            visualize_spectrum_comparison(
                S_target=torch.tensor(S_target_np, dtype=torch.float32),
                S_recon=torch.tensor(recon, dtype=torch.float32),
                ppm_axis=torch.tensor(ppm_np, dtype=torch.float32),
                layer_name=label, save_dir=str(final_dir))
        except Exception:
            pass

    r2 = float(diff_info.get('r2', 0.0))
    print(f"\n[Final Outputs] 已保存: {final_dir}")
    print(f"  R2={r2:.4f}, alpha={alpha:.4f}, smooth_sigma={sigma_ppm:.2f}ppm")
    return r2


# ========================================================================
# Main Pipeline Function
# ========================================================================

def run_inverse_pipeline(
    spectrum_csv_path: str,
    elements_str: str,
    output_dir: str = "test_results",
    s2n_model_path: str = "checkpoints_s2n/best_s2n_model.pt",
    g2s_model_path: str = "checkpoints_g2s/g2s_best_model.pt",
    enable_layer2: bool = True,
    enable_layer3: bool = True,
    eval_nmr: bool = True,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Run the inverse NMR pipeline with multistage Layer4 adjustment.

    All tunable parameters are in `config` dict (see DEFAULT_CONFIG).
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    device = torch.device('cpu')
    print(f"\nDevice: {device}\n")

    # ---- 1. Load spectrum ----
    print("\n" + "=" * 80)
    print("数据加载")
    print("=" * 80)

    spectrum_csv = Path(spectrum_csv_path)
    if not spectrum_csv.exists():
        print(f"  Warning: spectrum not found: {spectrum_csv}, generating dummy")
        S_target_raw = torch.clamp(torch.randn(2400) * 0.1 + 0.5, min=0)
    else:
        S_target_raw = read_spectrum_csv(str(spectrum_csv))
    E_target = parse_elements(elements_str)

    S_target = normalize_spectrum_to_carbon_count(S_target_raw, float(E_target[0].item()))

    print(f"  谱图: {S_target.shape}, 元素: C={int(E_target[0])},H={int(E_target[1])},O={int(E_target[2])},N={int(E_target[3])},S={int(E_target[4])},X={int(E_target[5])}")

    # ---- 2. Load models ----
    print("\n" + "=" * 80)
    print("模型加载")
    print("=" * 80)
    s2n_model, vae_model, g2s_available = _load_models(s2n_model_path, g2s_model_path, device)
    if s2n_model is None:
        return None, None, None

    if not g2s_available and (enable_layer2 or enable_layer3):
        print("  Warning: G2S unavailable, automatically disabling Layer2/Layer3 for this test run.")
        enable_layer2 = False
        enable_layer3 = False

    # ---- 3. Create pipeline ----
    pipeline = InversePipelineV3(s2n_model, vae_model, {}, device)

    # ---- 4. Layer0: SU histogram estimation ----
    H_layer0 = pipeline.estimate_su_histogram(S_target, E_target).to(device)
    E_final = _print_element_comparison(H_layer0, E_target, device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve eval_lib path
    eval_lib = cfg['eval_lib']
    if eval_lib and not Path(eval_lib).is_absolute():
        eval_lib = str((Path(__file__).resolve().parent / eval_lib).resolve())
    cfg['_resolved_eval_lib'] = eval_lib

    hwhm = float(cfg['eval_hwhm'])

    # ---- 4.5 初始评估 (Layer1-2-3) ----
    print("\n" + "=" * 80)
    print("初始图结构评估 (基于Layer0预测分布)")
    print("=" * 80)
    
    work_dir = output_dir / "multistage"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    init_dir = work_dir / "initial_eval"
    init_dir.mkdir(parents=True, exist_ok=True)
    
    H_curr = H_layer0.detach().clone()
    nodes, diff_info = _run_layer12(
        pipeline, H_curr, S_target, E_target, cfg,
        str(init_dir), enable_layer2)
    last_r2 = float(diff_info.get('r2', 0.0))
    current_score = _compute_overall_score(diff_info)
    print(f"\n初始结构评估完成 (Layer1-2): R²={last_r2:.4f}")
    
    # ---- 5. Multistage optimization ----
    stage_params = _get_stage_params(cfg)
    stage_cycles = {
        'block_a': int(cfg.get('block_a_cycles', 0)),
        'block_b': int(cfg.get('block_b_cycles', 0)),
    }

    print("\n" + "=" * 80)
    print(
        "新版本宏循环: "
        f"A({stage_cycles['block_a']}) -> B({stage_cycles['block_b']}) -> "
        "[C(1) <-> D(1)] "
        f"inner×{int(cfg['cd_inner_max_loops'])}"
    )
    print("=" * 80)

    outer_patience_left = int(cfg['outer_patience'])
    total_outer = max(1, int(cfg['outer_max_cycles']))
    for outer_cycle in range(total_outer):
        print(f"\n{'='*80}")
        print(f"宏循环轮次 {outer_cycle + 1}/{total_outer}")
        print(f"{'='*80}")

        outer_improved = False

        for stage in AB_STAGE_ORDER:
            n_cycles = int(stage_cycles.get(stage, 0))
            if n_cycles <= 0:
                continue

            print(f"\n{'='*60}")
            print(f"阶段: {stage.upper()} (最多{n_cycles}轮)")
            print(f"{'='*60}")

            for cycle in range(n_cycles):
                res = _run_single_stage_cycle(
                    pipeline, stage, H_curr, nodes, diff_info, S_target, E_target,
                    cfg, stage_params, work_dir, enable_layer2, enable_layer3, outer_cycle, cycle)
                H_curr = res['H_curr']
                nodes = res['nodes']
                diff_info = res['diff_info']
                last_r2 = res['r2']
                current_score = res['overall_score_info']
                if res['accepted']:
                    outer_improved = True
                elif not res['changed']:
                    break
                else:
                    break

        H_curr, nodes, diff_info, last_r2, current_score, cd_improved = _run_cd_inner_loop(
            pipeline, H_curr, nodes, diff_info, S_target, E_target, cfg,
            stage_params, work_dir, enable_layer2, enable_layer3, outer_cycle)
        outer_improved = outer_improved or bool(cd_improved)

        if enable_layer3 and enable_layer2:
            nodes, diff_info = _run_layer3_refine(
                pipeline, nodes, S_target, E_target, cfg,
                str(work_dir / f"outer_{outer_cycle + 1}" / "layer3_refine"),
                enable_layer2, enable_layer3)
            last_r2 = float(diff_info.get('r2', 0.0))
            current_score = _compute_overall_score(diff_info)
            print(f"[宏循环] Layer3末尾精修完成: R²={last_r2:.4f}")

        print(f"\n[宏循环 {outer_cycle + 1}] 资源分配结果")
        outer_alloc_diag = _evaluate_allocation_balance(pipeline, nodes, S_target, E_target, cfg)
        _print_final_allocation_summary(outer_alloc_diag)

        if outer_improved:
            outer_patience_left = int(cfg['outer_patience'])
        else:
            outer_patience_left -= 1
            print(f"\n[宏循环] 本轮无接受更新, patience -> {outer_patience_left}")
            if outer_patience_left <= 0:
                print("[宏循环] 提前停止")
                break

    H_final = H_curr.to(device)
    H_final = pipeline._histogram_from_nodes(nodes).to(device)
    E_final = torch.matmul(H_final.float(), E_SU.to(device))
    final_alloc_diag = _evaluate_allocation_balance(pipeline, nodes, S_target, E_target, cfg)

    print("\n" + "=" * 80)
    print(f"多阶段优化完成 - 最终R²={last_r2:.4f}")
    print("=" * 80)
    _print_element_comparison(H_final, E_target, device)

    # ---- 6. NMR eval ----
    if eval_nmr:
        eval_dir = output_dir / "layer1_library_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        lib_str = eval_lib if eval_lib and Path(eval_lib).exists() else None
        if lib_str:
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    metrics = pipeline.evaluate_layer1_nmr_with_library(
                        nodes=nodes, S_target=S_target, lib_path=lib_str,
                        output_dir=str(eval_dir), hwhm=hwhm,
                        allow_approx=bool(cfg['eval_allow_approx']))
                captured = buf.getvalue().splitlines()
                for line in captured:
                    if line.startswith("[Layer1-NMR-Eval] carbon_nodes=") or line.startswith("[Layer1-NMR-Eval] R2="):
                        print(line)
                pd.DataFrame([metrics]).to_csv(eval_dir / "layer1_library_eval_metrics.csv", index=False)
            except Exception as e:
                print(f"  [NMR Eval] 失败: {e}")

    # ---- 7. Save final outputs ----
    _save_final_outputs(
        pipeline, nodes, H_final, S_target, E_target,
        str(output_dir), cfg, enable_layer2,
        final_alloc=final_alloc_diag,
    )

    return H_final, E_final, nodes


# ========================================================================
# CLI
# ========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Inverse NMR Pipeline — Multistage Layer4 Adjustment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Essential ---
    g = parser.add_argument_group('Essential')
    g.add_argument('--spectrum_csv', type=str, default="/mnt/e/NN/GA/HSQ/standardized_nmr.csv",
                   help='Path to spectrum CSV')
    g.add_argument('--elements', type=str, default="C=460,H=400,O=60,N=2,S=2,X=0",
                   help='Target element composition')
    g.add_argument('--output_dir', type=str, default="test_results", help='Output directory')
    g.add_argument('--s2n_model', type=str, default="checkpoints_s2n/best_s2n_model.pt")
    g.add_argument('--g2s_model', type=str, default="checkpoints_g2s/g2s_best_model.pt")

    # --- Layer control ---
    g = parser.add_argument_group('Layer Control')
    g.add_argument('--disable_layer2', action='store_true')
    g.add_argument('--disable_layer3', action='store_true')
    g.add_argument('--eval_nmr', action='store_true', help='Enable NMR evaluation')
    g.add_argument('--no_eval_nmr', action='store_true', help='Disable NMR evaluation')

    # --- Hop1 ---
    g = parser.add_argument_group('Hop1 Adjustment')
    g.add_argument('--enable_hop1_adjust', action='store_true')
    g.add_argument('--hop1_iterations', type=int, default=DEFAULT_CONFIG['hop1_iterations'])
    g.add_argument('--hop1_neg_threshold', type=float, default=DEFAULT_CONFIG['hop1_neg_threshold'])
    g.add_argument('--hop1_pos_threshold', type=float, default=DEFAULT_CONFIG['hop1_pos_threshold'])
    g.add_argument('--disable_carbonyl_joint_adjust', action='store_true')
    g.add_argument('--carbonyl_joint_iterations', type=int, default=DEFAULT_CONFIG['carbonyl_joint_iterations'])
    g.add_argument('--carbonyl_joint_max_adjustments', type=int, default=DEFAULT_CONFIG['carbonyl_joint_max_adjustments'])
    g.add_argument('--carbonyl_joint_pos_threshold', type=float, default=DEFAULT_CONFIG['carbonyl_joint_pos_threshold'])
    g.add_argument('--carbonyl_joint_neg_threshold', type=float, default=DEFAULT_CONFIG['carbonyl_joint_neg_threshold'])

    # --- NMR eval ---
    g = parser.add_argument_group('NMR Evaluation')
    g.add_argument('--eval_lib', type=str, default=DEFAULT_CONFIG['eval_lib'])
    g.add_argument('--eval_hwhm', type=float, default=DEFAULT_CONFIG['eval_hwhm'])
    g.add_argument('--eval_no_approx', action='store_true')

    # --- Layer3 ---
    g = parser.add_argument_group('Layer3')
    g.add_argument('--layer3_max_iters', type=int, default=DEFAULT_CONFIG['layer3_max_iters'])
    g.add_argument('--layer3_pos_window', type=float, default=DEFAULT_CONFIG['layer3_pos_window'])
    g.add_argument('--layer3_neg_window', type=float, default=DEFAULT_CONFIG['layer3_neg_window'])
    g.add_argument('--layer3_top_k', type=int, default=DEFAULT_CONFIG['layer3_top_k'])
    g.add_argument('--enable_layer3_approx_hop2', action='store_true')
    g.add_argument('--layer3_approx_hop2_max_iters', type=int, default=DEFAULT_CONFIG['layer3_approx_hop2_max_iters'])
    g.add_argument('--layer3_approx_hop2_max_diff_nodes', type=int, default=DEFAULT_CONFIG['layer3_approx_hop2_max_diff_nodes'])
    g.add_argument('--layer3_approx_hop2_top_k', type=int, default=DEFAULT_CONFIG['layer3_approx_hop2_top_k'])

    # --- New Macro Loop Blocks ---
    g = parser.add_argument_group('Layer4 Macro Blocks')
    g.add_argument('--block_a_cycles', type=int, default=DEFAULT_CONFIG['block_a_cycles'])
    g.add_argument('--block_b_cycles', type=int, default=DEFAULT_CONFIG['block_b_cycles'])

    # --- Layer4 block parameters ---
    g = parser.add_argument_group('Layer4 Block Parameters')
    g.add_argument('--block_a_max_moves', type=int, default=DEFAULT_CONFIG['block_a_max_moves'])
    g.add_argument('--block_a_carbonyl_max_moves', type=int, default=DEFAULT_CONFIG['block_a_carbonyl_max_moves'])
    g.add_argument('--block_a_score_rel_threshold', type=float, default=DEFAULT_CONFIG['block_a_score_rel_threshold'])
    g.add_argument('--block_a_peak_rel_threshold', type=float, default=DEFAULT_CONFIG['block_a_peak_rel_threshold'])
    g.add_argument('--block_a_min_keep', type=int, default=DEFAULT_CONFIG['block_a_min_keep'])
    g.add_argument('--block_b_max_moves_each', type=int, default=DEFAULT_CONFIG['block_b_max_moves_each'])
    g.add_argument('--block_b_peak_rel_threshold', type=float, default=DEFAULT_CONFIG['block_b_peak_rel_threshold'])
    g.add_argument('--block_c_max_moves', type=int, default=DEFAULT_CONFIG['block_c_max_moves'])
    g.add_argument('--block_c_peak_rel_threshold', type=float, default=DEFAULT_CONFIG['block_c_peak_rel_threshold'])
    g.add_argument('--block_c_min_keep_22', type=int, default=DEFAULT_CONFIG['block_c_min_keep_22'])
    g.add_argument('--block_c_min_keep_23', type=int, default=DEFAULT_CONFIG['block_c_min_keep_23'])
    g.add_argument('--block_c_min_keep_24', type=int, default=DEFAULT_CONFIG['block_c_min_keep_24'])
    g.add_argument('--block_c_min_keep_25', type=int, default=DEFAULT_CONFIG['block_c_min_keep_25'])
    g.add_argument('--block_c_h_tolerance', type=float, default=DEFAULT_CONFIG['block_c_h_tolerance'])
    g.add_argument('--block_c_no_carbonyl_couple', action='store_true')
    g.add_argument('--su22_ratio', type=float, default=DEFAULT_CONFIG['su22_ratio'])
    g.add_argument('--su22_h_tol', type=float, default=DEFAULT_CONFIG['su22_h_tol'])
    g.add_argument('--extra_max_steps', type=int, default=DEFAULT_CONFIG['extra_max_steps'])
    g.add_argument('--extra_flexible_ratio', type=float, default=DEFAULT_CONFIG['extra_flexible_ratio'])
    g.add_argument('--extra_flexible_lower_extra', type=int, default=DEFAULT_CONFIG['extra_flexible_lower_extra'])
    g.add_argument('--extra_relaxed_flexible_ratio', type=float, default=DEFAULT_CONFIG['extra_relaxed_flexible_ratio'])
    g.add_argument('--extra_relaxed_lower_extra', type=int, default=DEFAULT_CONFIG['extra_relaxed_lower_extra'])
    g.add_argument('--skeleton_max_steps', type=int, default=DEFAULT_CONFIG['skeleton_max_steps'])

    # --- Outer loop ---
    g = parser.add_argument_group('Outer Loop')
    g.add_argument('--outer_max_cycles', type=int, default=DEFAULT_CONFIG['outer_max_cycles'])
    g.add_argument('--outer_patience', type=int, default=DEFAULT_CONFIG['outer_patience'])
    g.add_argument('--outer_improve_eps', type=float, default=DEFAULT_CONFIG['outer_improve_eps'])
    g.add_argument('--cd_inner_max_loops', type=int, default=DEFAULT_CONFIG['cd_inner_max_loops'])

    # --- Final smoothing ---
    g = parser.add_argument_group('Final Smoothing')
    g.add_argument('--final_smooth_sigma_ppm', type=float, default=DEFAULT_CONFIG['final_smooth_sigma_ppm'])
    g.add_argument('--final_smooth_passes', type=int, default=DEFAULT_CONFIG['final_smooth_passes'])
    g.add_argument('--final_smooth_radius_factor', type=float, default=DEFAULT_CONFIG['final_smooth_radius_factor'])

    args = parser.parse_args()

    # Build config from CLI args — auto-map matching keys
    config = {}
    for key in DEFAULT_CONFIG:
        if hasattr(args, key):
            config[key] = getattr(args, key)

    # Special flags
    config['hop1_adjust'] = bool(args.enable_hop1_adjust)
    config['enable_carbonyl_joint_adjust'] = not bool(args.disable_carbonyl_joint_adjust)
    config['layer3_approx_hop2'] = bool(args.enable_layer3_approx_hop2)
    config['eval_allow_approx'] = not bool(args.eval_no_approx)
    config['block_c_carbonyl_couple'] = not bool(args.block_c_no_carbonyl_couple)

    eval_nmr_flag = args.eval_nmr or (not args.no_eval_nmr)

    try:
        H_final, E_final, nodes = run_inverse_pipeline(
            spectrum_csv_path=args.spectrum_csv,
            elements_str=args.elements,
            output_dir=args.output_dir,
            s2n_model_path=args.s2n_model,
            g2s_model_path=args.g2s_model,
            enable_layer2=not args.disable_layer2,
            enable_layer3=not args.disable_layer3,
            eval_nmr=eval_nmr_flag,
            config=config,
        )
        if nodes:
            print(f"\n✓✓✓ 完成: {len(nodes)}个节点, {len(set(n.su_type for n in nodes))}种SU类型 ✓✓✓\n")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
