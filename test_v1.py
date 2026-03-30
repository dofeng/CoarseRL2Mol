import sys
from pathlib import Path
from typing import Optional, Dict, Any
from collections import Counter

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
from model.inverse_common import PPM_AXIS, visualize_spectrum_comparison, visualize_su_distribution, SU_CONNECTION_DEGREE

SU_NAMES = [name for name, _ in SU_DEFS]
STAGE_ORDER = ['carbonyl', 'su9', 'ether', 'amine', 'thioether', 'halogen', 'skeleton']

# ========================================================================
# Default Configuration — all parameters in one place
# ========================================================================
DEFAULT_CONFIG = {
    # Layer3
    'layer3_max_iters': 200,
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
    # NMR eval
    'eval_lib': 'z_library/subgraph_library.pt',
    'eval_hwhm': 1.0,
    'eval_allow_approx': True,
    # Layer4 stage cycles (0 = disabled)
    'carbonyl_cycles': 4,
    'su9_cycles': 3,
    'ether_cycles': 3,
    'amine_cycles': 1,
    'thioether_cycles': 1,
    'halogen_cycles': 1,
    'skeleton_cycles': 1,
    'skeleton_max_steps': 40,
    'extra_max_steps': 100,
    'extra_flexible_ratio': 0.80,
    'extra_flexible_lower_extra': 1,
    'extra_relaxed_flexible_ratio': 0.82,
    'extra_relaxed_lower_extra': 0,
    # Carbonyl (1/2/3)
    'carbonyl_max_moves': 3,
    'carbonyl_window_12': 5.0,
    'carbonyl_window_3': 13.0,
    'carbonyl_score_rel_threshold': 0.02,
    'carbonyl_min_keep': 1,
    # SU9
    'su9_max_moves': 5,
    'su9_window': 2.5,
    'su9_score_rel_threshold': 0.01,
    'su9_min_keep': 0,
    # Ether O (5/19)
    'o_519_max_moves': 5,
    'o_519_window_5': 3.0,
    'o_519_window_19': 3.0,
    'o_519_peak_rel_threshold': 0.01,
    'o_519_min_keep': 1,
    # Amine N (6/20)
    'n_620_max_moves': 5,
    'n_620_window_6': 3.0,
    'n_620_window_20': 3.0,
    'n_620_peak_rel_threshold': 0.01,
    'n_620_min_keep': 0,
    # Thioether S (7/19)
    's_719_max_moves': 5,
    's_719_window_7': 3.0,
    's_719_window_19': 3.0,
    's_719_peak_rel_threshold': 0.01,
    's_719_min_keep': 0,
    # Halogen X (8/21)
    'x_821_max_moves': 5,
    'x_821_window_8': 3.0,
    'x_821_window_21': 3.0,
    'x_821_peak_rel_threshold': 0.01,
    'x_821_min_keep': 0,
    # SU22
    'su22_ratio': 0.1,
    'su22_h_tol': 0.03,
    # Outer loop
    'outer_max_cycles': 1,
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
        'carbonyl': {
            'max_moves': cfg['carbonyl_max_moves'],
            'score_rel_threshold': cfg['carbonyl_score_rel_threshold'],
            'window_12': cfg['carbonyl_window_12'],
            'window_3': cfg['carbonyl_window_3'],
            'min_keep': cfg['carbonyl_min_keep'],
        },
        'su9': {
            'max_moves': cfg['su9_max_moves'],
            'score_rel_threshold': cfg['su9_score_rel_threshold'],
            'window': cfg['su9_window'],
            'min_keep': cfg['su9_min_keep'],
        },
        'ether': {
            'max_moves': cfg['o_519_max_moves'],
            'peak_rel_threshold': cfg['o_519_peak_rel_threshold'],
            'window_5': cfg['o_519_window_5'],
            'window_19': cfg['o_519_window_19'],
            'min_keep': cfg['o_519_min_keep'],
        },
        'amine': {
            'max_moves': cfg['n_620_max_moves'],
            'peak_rel_threshold': cfg['n_620_peak_rel_threshold'],
            'window_6': cfg['n_620_window_6'],
            'window_20': cfg['n_620_window_20'],
            'min_keep': cfg['n_620_min_keep'],
        },
        'thioether': {
            'max_moves': cfg['s_719_max_moves'],
            'peak_rel_threshold': cfg['s_719_peak_rel_threshold'],
            'window_7': cfg['s_719_window_7'],
            'window_19': cfg['s_719_window_19'],
            'min_keep': cfg['s_719_min_keep'],
        },
        'halogen': {
            'max_moves': cfg['x_821_max_moves'],
            'peak_rel_threshold': cfg['x_821_peak_rel_threshold'],
            'window_8': cfg['x_821_window_8'],
            'window_21': cfg['x_821_window_21'],
            'min_keep': cfg['x_821_min_keep'],
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


def _run_layer123(pipeline, H, S_target, E_target, cfg, out_dir, enable_layer2, enable_layer3):
    """Run Layer1 → Layer2 → Layer3 (optional) → compute diff."""
    eval_lib = cfg.get('_resolved_eval_lib', cfg['eval_lib'])
    hwhm = float(cfg['eval_hwhm'])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    nodes = pipeline.layer1_assign(
        H_init=H, S_target=S_target, E_target=E_target,
        eval_nmr=True,
        eval_output_dir=str(Path(out_dir) / 'layer1_eval'),
        eval_lib_path=eval_lib, eval_hwhm=hwhm,
        eval_allow_approx=bool(cfg['eval_allow_approx']),
        enable_hop1_adjust=bool(cfg['hop1_adjust']),
        hop1_adjust_iterations=int(cfg['hop1_iterations']),
        hop1_neg_threshold=float(cfg['hop1_neg_threshold']),
        hop1_pos_threshold=float(cfg['hop1_pos_threshold']),
    )

    if enable_layer2:
        try:
            nodes = pipeline.layer2_assign(
                nodes=nodes, H_center=pipeline._histogram_from_nodes(nodes),
                S_target=S_target, E_target=E_target,
                lib_path=eval_lib, output_dir=str(Path(out_dir) / 'layer2_eval'),
                eval_hwhm=hwhm)
        except Exception as e:
            print(f"  [Layer2] 失败: {e}")

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


def _print_skeleton_meta_summary(meta: Optional[Dict[str, Any]]):
    """Print a compact final summary for Layer4 skeleton allocation diagnostics."""
    if not meta:
        print("\n[Skeleton Summary] meta为空")
        return

    branch_meta = meta.get('branch_meta', {}) or {}
    extra_meta = meta.get('extra_meta', {}) or {}
    final_alloc = meta.get('final_allocation', {}) or {}

    def _fmt_bool(v):
        return 'OK' if bool(v) else 'FAIL'

    def _fmt_float(v, default=0.0, digits=4):
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return f"{float(default):.{digits}f}"

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
            f"{int(extra_diag.get('aliphatic_min_total', 0))}"
        )
        if rem:
            print(
                f"    extra剩余: 11×{int(rem.get('11', 0))} "
                f"22×{int(rem.get('22', 0))} "
                f"23×{int(rem.get('23', 0))} "
                f"24×{int(rem.get('24', 0))} "
                f"25×{int(rem.get('25', 0))}"
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
            f"rigid_clusters={int(final_alloc.get('rigid_cluster_count', 0))} "
            f"rigid10={int(final_alloc.get('rigid_pairs', 0))} "
            f"flex={int(final_alloc.get('flexible_bridge_count', 0))}/"
            f"[{int(final_alloc.get('flexible_bridge_min', 0))},{int(final_alloc.get('flexible_bridge_limit', 0))}] "
            f"side22={int(final_alloc.get('side_to_22_count', 0))} "
            f"ali={int(final_alloc.get('aliphatic_total', 0))}/"
            f"{int(final_alloc.get('aliphatic_min_total', 0))}"
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


def _save_final_outputs(pipeline, nodes, H_final, S_target, E_target, output_dir, cfg, enable_layer2):
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

    if E_target[0] > 0:
        target_area = float(torch.pi) * float(E_target[0].item())
        current_area = float(S_target_raw.sum().item()) * 0.1
        S_target = S_target_raw * (target_area / current_area) if current_area > 1e-6 else S_target_raw
    else:
        S_target = S_target_raw

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
    nodes, diff_info = _run_layer123(
        pipeline, H_curr, S_target, E_target, cfg,
        str(init_dir), enable_layer2, enable_layer3)
    last_r2 = float(diff_info.get('r2', 0.0))
    print(f"\n初始结构评估完成: R²={last_r2:.4f}")
    
    # ---- 5. Multistage optimization ----
    stage_params = _get_stage_params(cfg)
    stage_cycles = {s: int(cfg.get(f'{s}_cycles', 0)) for s in STAGE_ORDER}

    print("\n" + "=" * 80)
    active = [f"{s}({stage_cycles[s]})" for s in STAGE_ORDER if stage_cycles[s] > 0]
    print(f"多阶段优化循环: {' → '.join(active)}")
    print("=" * 80)

    for stage in STAGE_ORDER:
        n_cycles = stage_cycles.get(stage, 0)
        if n_cycles <= 0:
            continue

        stage_dir = work_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"阶段: {stage.upper()} (最多{n_cycles}轮)")
        print(f"{'='*60}")

        for cycle in range(n_cycles):
            cycle_dir = stage_dir / f"cycle_{cycle + 1}"

            # Layer4 stage adjustment (基于当前的 nodes 和 diff_info)
            H_base = pipeline._histogram_from_nodes(nodes)
            try:
                kwargs = dict(stage_params.get(stage, {}))
                if stage == 'ether':
                    kwargs['reserved_19'] = max(0, int(2 * H_base[31].item()) - int(H_base[7].item()))
                kwargs['enable_su22_adjust'] = True
                kwargs['su22_ratio'] = cfg['su22_ratio']
                kwargs['su22_h_tol'] = cfg['su22_h_tol']
                
                H_next, moves, meta = pipeline.layer4_adjuster.adjust_by_stage(
                    H=H_base, ppm=diff_info.get('ppm'), diff=diff_info.get('diff'),
                    E_target=E_target, S_target=S_target, stage=stage,
                    nodes=nodes,
                    **kwargs)
                n_moves = len(moves)
                if stage == 'skeleton':
                    _print_skeleton_meta_summary(meta)
            except Exception as e:
                print(f"  [Layer4-{stage.upper()}] 失败: {e}")
                import traceback; traceback.print_exc()
                H_next = H_base
                n_moves = 0

            if n_moves == 0:
                print(f"  无调整，提前结束")
                break
                
            H_curr = H_next.detach().clone()
            
            # Layer1 → Layer2 → Layer3 → diff (评估调整后的结果)
            nodes, diff_info = _run_layer123(
                pipeline, H_curr, S_target, E_target, cfg,
                str(cycle_dir), enable_layer2, enable_layer3)
            last_r2 = float(diff_info.get('r2', 0.0))
            
            print(f"  [{stage} 第{cycle+1}轮] 调整={n_moves}次, 更新后R²={last_r2:.4f}")

    H_final = H_curr.to(device)
    E_final = torch.matmul(H_final.float(), E_SU.to(device))

    if nodes is None:
        nodes, diff_info = _run_layer123(
            pipeline, H_final, S_target, E_target, cfg,
            str(output_dir / 'final_eval'), enable_layer2, enable_layer3)

    print("\n" + "=" * 80)
    print(f"多阶段优化完成 - 最终R²={last_r2:.4f}")
    print("=" * 80)
    _print_element_comparison(H_final, E_target, device)

    # ---- 6. Layer1 analysis ----
    _print_layer1_analysis(nodes)

    # ---- 7. NMR eval ----
    if eval_nmr:
        eval_dir = output_dir / "layer1_library_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        lib_str = eval_lib if eval_lib and Path(eval_lib).exists() else None
        if lib_str:
            try:
                metrics = pipeline.evaluate_layer1_nmr_with_library(
                    nodes=nodes, S_target=S_target, lib_path=lib_str,
                    output_dir=str(eval_dir), hwhm=hwhm,
                    allow_approx=bool(cfg['eval_allow_approx']))
                print("\n[Layer1-NMR-Eval]")
                for k in ['r2', 'r2_carbonyl', 'r2_aromatic', 'r2_aliphatic', 'matched_ratio']:
                    if k in metrics:
                        print(f"  {k}: {metrics[k]}")
                pd.DataFrame([metrics]).to_csv(eval_dir / "layer1_library_eval_metrics.csv", index=False)
            except Exception as e:
                print(f"  [NMR Eval] 失败: {e}")

    # ---- 8. Save final outputs ----
    _save_final_outputs(pipeline, nodes, H_final, S_target, E_target, str(output_dir), cfg, enable_layer2)

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

    # --- Layer4 stage cycles (set to 0 to disable a stage) ---
    g = parser.add_argument_group('Layer4 Stage Cycles (set 0 to disable)')
    g.add_argument('--carbonyl_cycles', type=int, default=DEFAULT_CONFIG['carbonyl_cycles'])
    g.add_argument('--su9_cycles', type=int, default=DEFAULT_CONFIG['su9_cycles'])
    g.add_argument('--ether_cycles', type=int, default=DEFAULT_CONFIG['ether_cycles'])
    g.add_argument('--amine_cycles', type=int, default=DEFAULT_CONFIG['amine_cycles'])
    g.add_argument('--thioether_cycles', type=int, default=DEFAULT_CONFIG['thioether_cycles'])
    g.add_argument('--halogen_cycles', type=int, default=DEFAULT_CONFIG['halogen_cycles'])
    g.add_argument('--skeleton_cycles', type=int, default=DEFAULT_CONFIG['skeleton_cycles'])

    # --- Layer4 stage parameters ---
    g = parser.add_argument_group('Layer4 Stage Parameters')
    g.add_argument('--carbonyl_max_moves', type=int, default=DEFAULT_CONFIG['carbonyl_max_moves'])
    g.add_argument('--carbonyl_window_12', type=float, default=DEFAULT_CONFIG['carbonyl_window_12'])
    g.add_argument('--carbonyl_window_3', type=float, default=DEFAULT_CONFIG['carbonyl_window_3'])
    g.add_argument('--carbonyl_score_rel_threshold', type=float, default=DEFAULT_CONFIG['carbonyl_score_rel_threshold'])
    g.add_argument('--carbonyl_min_keep', type=int, default=DEFAULT_CONFIG['carbonyl_min_keep'])
    g.add_argument('--su9_max_moves', type=int, default=DEFAULT_CONFIG['su9_max_moves'])
    g.add_argument('--su9_window', type=float, default=DEFAULT_CONFIG['su9_window'])
    g.add_argument('--su9_score_rel_threshold', type=float, default=DEFAULT_CONFIG['su9_score_rel_threshold'])
    g.add_argument('--su9_min_keep', type=int, default=DEFAULT_CONFIG['su9_min_keep'])
    g.add_argument('--o_519_max_moves', type=int, default=DEFAULT_CONFIG['o_519_max_moves'])
    g.add_argument('--o_519_window_5', type=float, default=DEFAULT_CONFIG['o_519_window_5'])
    g.add_argument('--o_519_window_19', type=float, default=DEFAULT_CONFIG['o_519_window_19'])
    g.add_argument('--o_519_peak_rel_threshold', type=float, default=DEFAULT_CONFIG['o_519_peak_rel_threshold'])
    g.add_argument('--o_519_min_keep', type=int, default=DEFAULT_CONFIG['o_519_min_keep'])
    g.add_argument('--n_620_max_moves', type=int, default=DEFAULT_CONFIG['n_620_max_moves'])
    g.add_argument('--n_620_window_6', type=float, default=DEFAULT_CONFIG['n_620_window_6'])
    g.add_argument('--n_620_window_20', type=float, default=DEFAULT_CONFIG['n_620_window_20'])
    g.add_argument('--n_620_peak_rel_threshold', type=float, default=DEFAULT_CONFIG['n_620_peak_rel_threshold'])
    g.add_argument('--n_620_min_keep', type=int, default=DEFAULT_CONFIG['n_620_min_keep'])
    g.add_argument('--s_719_max_moves', type=int, default=DEFAULT_CONFIG['s_719_max_moves'])
    g.add_argument('--s_719_window_7', type=float, default=DEFAULT_CONFIG['s_719_window_7'])
    g.add_argument('--s_719_window_19', type=float, default=DEFAULT_CONFIG['s_719_window_19'])
    g.add_argument('--s_719_peak_rel_threshold', type=float, default=DEFAULT_CONFIG['s_719_peak_rel_threshold'])
    g.add_argument('--s_719_min_keep', type=int, default=DEFAULT_CONFIG['s_719_min_keep'])
    g.add_argument('--x_821_max_moves', type=int, default=DEFAULT_CONFIG['x_821_max_moves'])
    g.add_argument('--x_821_window_8', type=float, default=DEFAULT_CONFIG['x_821_window_8'])
    g.add_argument('--x_821_window_21', type=float, default=DEFAULT_CONFIG['x_821_window_21'])
    g.add_argument('--x_821_peak_rel_threshold', type=float, default=DEFAULT_CONFIG['x_821_peak_rel_threshold'])
    g.add_argument('--x_821_min_keep', type=int, default=DEFAULT_CONFIG['x_821_min_keep'])
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
    config['layer3_approx_hop2'] = bool(args.enable_layer3_approx_hop2)
    config['eval_allow_approx'] = not bool(args.eval_no_approx)

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
