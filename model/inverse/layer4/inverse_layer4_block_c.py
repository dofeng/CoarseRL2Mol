import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...shared.inverse_common import SU_AROMATIC, _NodeV3, get_effective_hist_element_vector
from RL_MTCS.RL_allocator import FlexAllocator


def _node_target_degree(node: _NodeV3) -> Optional[int]:
    try:
        val = getattr(node, 'target_hop1_degree', None)
        return int(val) if val is not None else None
    except Exception:
        return None


def _endpoint_class_for_block_c_node(node: _NodeV3) -> str:
    su = int(getattr(node, 'su_type', -1))
    deg = _node_target_degree(node)
    if su in set(int(x) for x in SU_AROMATIC) or su in {26, 30}:
        return 'aromatic'
    if su in {1, 4, 16, 18, 22, 28, 32}:
        return 'terminal'
    if su in {19, 20}:
        if int(deg or 0) == 1:
            return 'terminal'
        return 'aliphatic'
    return 'aliphatic'


def _is_effective_24_like_node(node: _NodeV3) -> bool:
    su = int(getattr(node, 'su_type', -1))
    deg = _node_target_degree(node)
    if su in {14, 24}:
        return True
    if su in {19, 20, 21} and int(deg or 0) == 3:
        return True
    return False


def _classify_24_like_family(adjuster: Any,
                             node: _NodeV3,
                             nodes_local: List[_NodeV3]) -> str:
    node_lookup = {int(getattr(n, 'global_id', -1)): n for n in list(nodes_local or [])}
    hop1_nodes: List[_NodeV3] = []
    for nid in list(getattr(node, 'hop1_ids', []) or []):
        nb = node_lookup.get(int(nid))
        if nb is not None:
            hop1_nodes.append(nb)
    if not hop1_nodes:
        raw_hop1 = set(adjuster._current_neighbor_types(node, nodes_local))
        has_aro = any(int(h) in set(int(x) for x in SU_AROMATIC) or int(h) in {26, 30} for h in raw_hop1)
        has_22 = any(int(h) in {1, 4, 16, 18, 22, 28, 32} for h in raw_hop1)
    else:
        classes = [str(_endpoint_class_for_block_c_node(nb)) for nb in hop1_nodes]
        has_aro = any(str(cls) == 'aromatic' for cls in classes)
        has_22 = any(str(cls) == 'terminal' for cls in classes)
    if has_aro and not has_22:
        return '24_A'
    if has_aro and has_22:
        return '24_B'
    if (not has_aro) and (not has_22):
        return '24_C'
    return '24_D'


def _collect_tail_window_stats(adjuster: Any,
                               ppm_arr: np.ndarray,
                               diff_arr: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'core': {},
        'wide': {},
        'core_windows': {},
        'wide_windows': {},
        'need': {},
    }
    defaults = {
        22: 19.81,
        23: 29.48,
        24: 39.97,
    }
    for su_idx, fallback_mu in defaults.items():
        lo_w, hi_w, mu_w = adjuster._get_su_common_window(
            int(su_idx),
            fallback_mu=float(fallback_mu),
            min_half_width=6.0,
        )
        lo_c, hi_c, mu_c = adjuster._get_su_tail_core_window(
            int(su_idx),
            fallback_mu=float(fallback_mu),
            max_half_width=8.0,
        )
        stats_w = adjuster._window_stats(ppm_arr, diff_arr, lo_w, hi_w)
        stats_c = adjuster._window_stats(ppm_arr, diff_arr, lo_c, hi_c)
        out['wide'][int(su_idx)] = stats_w
        out['core'][int(su_idx)] = stats_c
        out['wide_windows'][int(su_idx)] = (float(lo_w), float(hi_w), float(mu_w))
        out['core_windows'][int(su_idx)] = (float(lo_c), float(hi_c), float(mu_c))
        out['need'][int(su_idx)] = float(stats_c['pos']) - float(stats_c['neg'])
    return out


def _prefer_preserve_24_from_tail_stats(tail_stats: Dict[str, Any]) -> bool:
    core_24 = dict((tail_stats or {}).get('core', {}).get(24, {}) or {})
    pos_24 = float(core_24.get('pos', 0.0))
    neg_24 = float(core_24.get('neg', 0.0))
    need_24 = float(dict((tail_stats or {}).get('need', {}) or {}).get(24, 0.0))
    if float(need_24) >= 0.0:
        return True
    return bool(float(pos_24) >= 0.70 * float(neg_24))


def _allow_reduce_24_from_tail_stats(tail_stats: Dict[str, Any], thr: float) -> bool:
    core_24 = dict((tail_stats or {}).get('core', {}).get(24, {}) or {})
    pos_24 = float(core_24.get('pos', 0.0))
    neg_24 = float(core_24.get('neg', 0.0))
    need_24 = float(dict((tail_stats or {}).get('need', {}) or {}).get(24, 0.0))
    return bool(float(need_24) < -float(thr) and float(neg_24) > float(pos_24) + 0.35 * float(thr))


def adjust_block_c_aliphatic_tail_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    E_target: Optional[torch.Tensor] = None,
    max_moves: int = 6,
    peak_rel_threshold: float = 0.01,
    min_keep_22: int = 1,
    min_keep_23: int = 0,
    min_keep_24: int = 0,
    carbonyl_couple: bool = True,
    h_tolerance: float = 0.08,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[Block C] 脂肪尾部 22/23/24 与 13/23 联合调整")
    # Keep tail moves single-step so each accepted change is re-evaluated by the
    # outer Layer1/2 loop instead of compounding several stale-diff decisions.
    max_moves = min(int(max_moves), 1)

    if ppm is None or diff is None:
        return H, [], {'reason': 'missing_diff'}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        return H, [], {'reason': 'empty_diff'}

    tail_stats = _collect_tail_window_stats(adjuster, ppm_arr, diff_arr)
    s22 = dict(tail_stats['wide'][22])
    s23 = dict(tail_stats['wide'][23])
    s24 = dict(tail_stats['wide'][24])
    s22_core = dict(tail_stats['core'][22])
    s23_core = dict(tail_stats['core'][23])
    s24_core = dict(tail_stats['core'][24])
    need22 = float(tail_stats['need'][22])
    need23 = float(tail_stats['need'][23])
    need24 = float(tail_stats['need'][24])
    s23_wide = adjuster._window_stats(ppm_arr, diff_arr, 15.0, 45.0)
    s12_13 = adjuster._window_stats(ppm_arr, diff_arr, 115.0, 135.0)
    carb_windows = adjuster._get_carb_joint_windows()
    low_lo, low_hi = carb_windows['low']
    mid_lo, mid_hi = carb_windows['mid']
    low = adjuster._window_stats(ppm_arr, diff_arr, low_lo, low_hi)
    mid = adjuster._window_stats(ppm_arr, diff_arr, mid_lo, mid_hi)

    tail_mask = (ppm_arr >= 8.0) & (ppm_arr <= 65.0)
    tail_abs = float(np.sum(np.abs(diff_arr[tail_mask]))) if np.any(tail_mask) else float(np.sum(np.abs(diff_arr)))
    thr = float(peak_rel_threshold) * max(1e-8, tail_abs)
    preserve_24 = _prefer_preserve_24_from_tail_stats(tail_stats)
    allow_reduce_24 = (not bool(preserve_24)) and _allow_reduce_24_from_tail_stats(tail_stats, thr)

    move_order: List[Tuple[str, Dict[int, int]]] = []
    if float(need24) > float(thr) and float(need23) < -float(thr):
        move_order.append(('C_23to24', {23: -1, 24: +1}))
    if float(need24) > float(thr) and float(need22) < -float(thr):
        move_order.append(('C_22to24', {22: -1, 24: +1}))
    if float(need23) > float(thr) and float(need22) < -float(thr):
        move_order.append(('C_22to23', {22: -1, 23: +1}))
    if float(s23_core['neg']) > float(thr) and float(s12_13['pos']) > float(thr):
        move_order.append(('C_23to13', {23: -1, 13: +1}))
    if float(need22) > float(thr) and float(need23) < -float(thr):
        move_order.append(('C_23to22', {23: -1, 22: +1}))
    if float(s12_13['neg']) > float(thr) and float(s23_wide['pos']) > float(thr):
        move_order.append(('C_13to23', {13: -1, 23: +1}))
    if bool(allow_reduce_24) and float(need23) > float(thr):
        move_order.append(('C_24to23', {24: -1, 23: +1}))
    if bool(allow_reduce_24) and float(need22) > float(thr):
        move_order.append(('C_24to22', {24: -1, 22: +1}))

    if bool(carbonyl_couple) and float(mid['pos']) > float(thr) and float(low['neg']) > float(thr):
        coupled = []
        if float(need24) > float(thr) and float(need23) < -float(thr):
            coupled.append(('C_couple_23to24', {23: -1, 24: +1}))
        if float(need24) > float(thr) and float(need22) < -float(thr):
            coupled.append(('C_couple_22to24', {22: -1, 24: +1}))
        if float(need23) > float(thr) and float(need22) < -float(thr):
            coupled.append(('C_couple_22to23', {22: -1, 23: +1}))
        if float(s12_13['neg']) > float(thr) and float(s23_wide['pos']) > float(thr):
            coupled.append(('C_couple_13to23', {13: -1, 23: +1}))
        move_order = coupled + move_order

    H_work = torch.clamp(H, min=0).long().clone()
    all_moves: List[Dict[str, Any]] = []
    keep_24 = int(min_keep_24)
    if bool(preserve_24):
        keep_24 = max(int(keep_24), 1)
    keep = {22: int(min_keep_22), 23: int(min_keep_23), 24: int(keep_24), 13: 0}

    def _current_h(hh: torch.Tensor) -> float:
        try:
            meta = adjuster._get_special_degree_meta(hh)
            eff = get_effective_hist_element_vector(hh, special_degree_meta=meta, E_SU_tensor=adjuster.E_SU.cpu())
            return float(eff[1].item())
        except Exception:
            return float(torch.matmul(hh.float(), adjuster.E_SU)[1].item())

    def _h_within_tolerance(hh_before: torch.Tensor, hh_after: torch.Tensor) -> bool:
        if E_target is None:
            return True
        try:
            target_h = float(E_target.to(hh_after.device)[1].item())
        except Exception:
            return True
        if target_h <= 1e-8:
            return True
        before = float(_current_h(hh_before))
        after = float(_current_h(hh_after))
        tol = max(0.0, float(h_tolerance))
        after_rel = abs(after - target_h) / target_h
        before_rel = abs(before - target_h) / target_h
        return bool(after_rel <= tol or after_rel <= before_rel + 1e-9)

    def _try_balance_h(base_h: torch.Tensor, cand_h: torch.Tensor) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
        if E_target is None:
            return cand_h, []
        if _h_within_tolerance(base_h, cand_h):
            return cand_h, []
        H_bal, h_moves, _h_meta = adjuster._apply_h_rotation_to_counts(cand_h, E_target, max_ops=None)
        if _h_within_tolerance(base_h, H_bal):
            return H_bal.detach().clone().cpu(), list(h_moves)
        return None, []

    for _ in range(max(0, int(max_moves))):
        applied = False
        for name, delta in move_order:
            H_try = adjuster._apply_count_delta(H_work, delta, min_keep=keep)
            if H_try is None:
                continue
            H_bal, h_moves = _try_balance_h(H_work, H_try)
            if H_bal is None:
                continue
            H_work = H_bal
            move_rec = {'block': 'C', 'op': name, 'delta': dict(delta)}
            if h_moves:
                move_rec['h_moves'] = [str(mv.get('op', '')) for mv in h_moves]
            all_moves.append(move_rec)
            applied = True
            break
        if not applied:
            break

    meta = {
        'threshold': float(thr),
        'windows': {
            '22': s22, '23': s23, '24': s24,
            '22_core': s22_core, '23_core': s23_core, '24_core': s24_core,
            '23_20_45': s23_wide, '12_13_115_130': s12_13,
            f'{low_lo:.1f}_{low_hi:.1f}': low,
            f'{mid_lo:.1f}_{mid_hi:.1f}': mid,
        },
        'tail_needs': {
            '22': float(need22),
            '23': float(need23),
            '24': float(need24),
        },
        'tail_core_windows': {
            str(k): list(v) for k, v in dict(tail_stats.get('core_windows', {}) or {}).items()
        },
        'tail_wide_windows': {
            str(k): list(v) for k, v in dict(tail_stats.get('wide_windows', {}) or {}).items()
        },
        'move_order': [name for name, _ in move_order],
        'h_tolerance': float(h_tolerance),
        'preserve_24': bool(preserve_24),
        'allow_reduce_24': bool(allow_reduce_24),
        'block_c_phase': 'tail_diff',
    }
    return H_work, all_moves, meta


def adjust_block_c_branch_phase_impl(
    adjuster: Any,
    H: torch.Tensor,
    E_target: Optional[torch.Tensor],
    S_target: Optional[torch.Tensor] = None,
    ppm: Optional[np.ndarray] = None,
    diff: Optional[np.ndarray] = None,
    max_steps: int = 50,
    nodes: Optional[List[_NodeV3]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print(f"\n[Skeleton-Alloc Adjust] 开始")
    if nodes is None:
        print("  [ERROR] nodes 必须提供以进行真实拓扑分配评估！")
        return H, [], {'n_moves': 0, 'ok': False}

    import copy
    H_work = H.cpu().clone()
    adjuster.E_target = E_target
    tmp_nodes = copy.deepcopy(nodes)
    adjuster._refresh_node_counters(tmp_nodes)

    tail_bias = {'need_22': 0.0, 'need_23': 0.0, 'need_24': 0.0, 'prefer_preserve_24': False}
    if ppm is not None and diff is not None:
        try:
            ppm_arr = np.asarray(ppm, dtype=np.float64)
            diff_arr = np.asarray(diff, dtype=np.float64)
            if int(ppm_arr.size) > 0 and int(diff_arr.size) > 0:
                tail_stats = _collect_tail_window_stats(adjuster, ppm_arr, diff_arr)
                tail_bias = {
                    'need_22': float(tail_stats['need'][22]),
                    'need_23': float(tail_stats['need'][23]),
                    'need_24': float(tail_stats['need'][24]),
                    'prefer_preserve_24': bool(_prefer_preserve_24_from_tail_stats(tail_stats)),
                }
        except Exception:
            tail_bias = {'need_22': 0.0, 'need_23': 0.0, 'need_24': 0.0, 'prefer_preserve_24': False}

    _, _h_ratio, _check_h, _ali_total = adjuster._make_h_helpers()
    rot_idx = int(getattr(adjuster, '_h_rotation_state', 0))
    moves: List[Dict[str, Any]] = []

    def _can_grow_aliphatic(by_count: int = 1) -> bool:
        cap = getattr(adjuster, '_h_rotation_aliphatic_cap', None)
        if cap is None:
            return True
        return int(_ali_total(H_work)) + int(by_count) <= int(cap)

    def _log_move(op_desc, stage):
        moves.append({
            'op': op_desc,
            'stage': stage,
            'h_ratio_before_h_adjust': float(_h_ratio(H_work)),
        })
        print(f"    -> {op_desc} (H偏差: {_h_ratio(H_work)*100:.1f}%)")

    def _chain_text(ch) -> str:
        comp = '-'.join(str(int(x)) for x in getattr(ch, 'composition', []))
        ctype = getattr(ch, 'chain_type', '?')
        origin = getattr(ch, 'origin_type', '?')
        src = getattr(ch, 'source_ids', [])
        return f"{ctype}/{origin}: {comp} src={src}"

    def _print_alloc_snapshot(tag: str, res: Dict[str, Any], chain_limit: int = 8):
        closed = res.get('closed_consumed', {})
        opened = res.get('open_consumed', {})
        pre = res.get('pre_branch_available', {})
        rem = res.get('remaining', {})
        chains = res.get('branch_chains', []) or []
        unsupported_count = int(res.get('unsupported_special_count', 0) or 0)
        unsupported_blocked = int(res.get('unsupported_special_blocked_count', 0) or 0)
        unsupported_reasons = dict(res.get('unsupported_special_reasons', {}) or {})
        if closed or opened or pre:
            print(
                f"    [{tag}资源] closed消耗 11×{closed.get('11', 0)} 23×{closed.get('23', 0)} 22×{closed.get('22', 0)} | "
                f"open消耗 11×{opened.get('11', 0)} 23×{opened.get('23', 0)} 22×{opened.get('22', 0)} | "
                f"branch前剩余 11×{pre.get('11', 0)} 23×{pre.get('23', 0)} 22×{pre.get('22', 0)}"
            )
        if unsupported_count > 0:
            reason_txt = ", ".join(f"{str(k)}={int(v)}" for k, v in sorted(unsupported_reasons.items()))
            print(
                f"    [{tag}特殊拓扑] unsupported={unsupported_count} blocked={unsupported_blocked}"
                + (f" | {reason_txt}" if reason_txt else "")
            )
        if chains:
            print(f"    [{tag}已分配结构] {len(chains)}个")
            for idx, ch in enumerate(chains[:chain_limit]):
                print(f"      [{idx}] {_chain_text(ch)}")
            if len(chains) > chain_limit:
                print(f"      ... 其余 {len(chains) - chain_limit} 个结构未展开")
        if rem:
            print(
                f"    [{tag}分支后剩余] 11×{rem.get('11', 0)} 23×{rem.get('23', 0)} 22×{rem.get('22', 0)} "
                f"24×{rem.get('24', 0)} 25×{rem.get('25', 0)}"
            )

    def _pick_24_node_for_conversion() -> Optional[Tuple[_NodeV3, str]]:
        buckets: Dict[str, List[_NodeV3]] = {'24_A': [], '24_B': [], '24_C': [], '24_D': []}
        for n in tmp_nodes:
            if not _is_effective_24_like_node(n):
                continue
            label = _classify_24_like_family(adjuster, n, tmp_nodes)
            buckets[label].append(n)
        total_24_like = sum(len(v) for v in buckets.values())
        if int(total_24_like) <= 2:
            return None
        for label in ['24_B', '24_D', '24_A', '24_C']:
            native = [n for n in buckets[label] if int(n.su_type) == 24]
            if native:
                return native[0], label
            derived = [n for n in buckets[label] if int(n.su_type) in {14, 19, 20, 21}]
            if derived:
                return derived[0], label
        return None

    def _can_convert_25_to_24(min_ratio: float = 0.01) -> bool:
        try:
            keep_25 = max(1, int(math.ceil(float(min_ratio) * float(_ali_total(H_work)))))
            return int(H_work[25].item()) > int(keep_25)
        except Exception:
            return int(H_work[25].item()) > 1

    def _convert_node(node: _NodeV3, dst_su: int) -> None:
        src_su = int(node.su_type)
        H_work[src_su] -= 1
        node.su_type = int(dst_su)
        H_work[int(dst_su)] += 1
        if int(dst_su) not in {19, 20, 21}:
            # The actual topology will be rebuilt by a later Layer1 pass.
            # Clear stale special-node semantics on ordinary converted nodes.
            for attr_name in (
                'target_hop1_degree',
                'init_target_hop1_degree',
                'special_degree_source',
                'special_anchor_partition',
                'target_fixed_anchor_count',
                'init_target_fixed_anchor_count',
                'special_anchor_mode',
            ):
                try:
                    setattr(node, attr_name, None)
                except Exception:
                    pass

    def _apply_2x13_to_23_12() -> bool:
        if not _can_grow_aliphatic(1):
            return False
        picked = [n for n in tmp_nodes if int(n.su_type) == 13]
        picked.sort(key=lambda n: int(n.global_id))
        if len(picked) < 2:
            return False
        _convert_node(picked[0], 23)
        _convert_node(picked[1], 12)
        return True

    def _apply_n_simple(src: int, dst: int, max_count: int) -> int:
        applied = 0
        picked = [n for n in tmp_nodes if int(n.su_type) == int(src)]
        picked.sort(key=lambda n: int(n.global_id))
        for node in picked[:max(0, int(max_count))]:
            _convert_node(node, int(dst))
            applied += 1
        return int(applied)

    allocator = FlexAllocator(nodes=tmp_nodes)

    print(f"\n  [Step 0] 完整分支调度评估")
    s1_iter = 0
    toggle_11 = 0

    while s1_iter < max_steps:
        s1_iter += 1
        print(f"    [Step 0] 评估轮次 {s1_iter}/{max_steps}")
        res_24 = allocator.evaluate_su24_branches(tmp_nodes, quiet=True)
        proxy_diag_24 = _block_c_eval(adjuster, tmp_nodes, S_target, E_target)
        print(
            f"    [Step 0诊断] shortage={res_24.get('shortage_type', 'none')} "
            f"req22={res_24.get('req_22', 0)} req11={res_24.get('req_11', 0)} req23={res_24.get('req_23', 0)} "
            f"su11miss={proxy_diag_24.get('su11_missing_external', 0)} "
            f"specialGap={proxy_diag_24.get('special_degree_gap', 0)}"
        )
        _print_alloc_snapshot('Step 0', res_24)

        if res_24['ok']:
            print("    [Step 0] 分支调度通过")
            break

        shortage = res_24['shortage_type']
        req_22 = res_24.get('req_22', 0)
        req_11 = res_24.get('req_11', 0)
        req_23 = res_24.get('req_23', 0)
        op = ''

        if shortage == '22_shortage':
            if _can_convert_25_to_24(min_ratio=0.01):
                for n in tmp_nodes:
                    if int(n.su_type) == 25:
                        _convert_node(n, 24)
                        op = 'S0_25->24'
                        break
            elif int(req_22) < 4:
                applied = _apply_n_simple(23, 22, min(max(1, int(req_22)), 3))
                if applied > 0:
                    op = f'S0_23->22x{applied}'
            elif int(req_22) <= 8:
                picked = None if bool(tail_bias.get('prefer_preserve_24', False)) else _pick_24_node_for_conversion()
                if picked:
                    chosen_node, p = picked
                    _convert_node(chosen_node, 22)
                    op = f'S0_24({p})->22'
                else:
                    for n in tmp_nodes:
                        if int(n.su_type) == 23:
                            _convert_node(n, 22)
                            op = 'S0_fallback_23->22'
                            break
            else:
                picked = None if bool(tail_bias.get('prefer_preserve_24', False)) else _pick_24_node_for_conversion()
                if picked:
                    chosen_node, p = picked
                    _convert_node(chosen_node, 23)
                    op = f'S0_24({p})->23'
                else:
                    for n in tmp_nodes:
                        if int(n.su_type) == 23:
                            _convert_node(n, 22)
                            op = 'S0_fallback_23->22'
                            break

        elif shortage == '11_shortage' or req_11 > 0:
            strategies = [12, 13] if toggle_11 % 2 == 0 else [13, 12]
            for src in strategies:
                if int(H_work[src].item()) > 0:
                    desired = min(max(1, int(req_11)), 4)
                    applied = _apply_n_simple(int(src), 11, desired)
                    if applied > 0:
                        op = f'S0_{src}->11x{applied}'
                        break
            toggle_11 += 1

        elif shortage == '23_shortage' or req_23 > 0:
            applied_pairs = 0
            for _ in range(min(max(1, int(req_23)), 3)):
                if not _apply_2x13_to_23_12():
                    break
                applied_pairs += 1
            if applied_pairs > 0:
                op = f'S0_2x13->23+12 x{applied_pairs}'

        if not op:
            print(f"    [Step 0] 无法处理短缺: {shortage}")
            break

        _log_move(op, 'S0')
        moves[-1]['diagnostic_before'] = {
            'shortage_type': str(shortage),
            'req_22': int(res_24.get('req_22', 0)),
            'req_11': int(res_24.get('req_11', 0)),
            'req_23': int(res_24.get('req_23', 0)),
        }
        print("      [Step 0] 开始H修正与重评估准备")
        h_ops, rot_idx = adjuster._h_rotation_adjust(
            tmp_nodes,
            H_work,
            _h_ratio,
            rot_idx,
            max_ops=8,
            max_aliphatic_total=getattr(adjuster, '_h_rotation_aliphatic_cap', None),
            max_ordinary_aliphatic_total=getattr(adjuster, '_h_rotation_ordinary_aliphatic_cap', None),
            h_tolerance=float(getattr(adjuster, '_h_tolerance', 0.08)),
        )
        if h_ops:
            print(f"      H调整: {' + '.join(h_ops)}")
            moves[-1]['h_ops'] = list(h_ops)
        moves[-1]['h_ratio_after_h_adjust'] = float(_h_ratio(H_work))
        adjuster._refresh_node_counters(tmp_nodes)

        print("      [Step 0] H修正完成，重建 allocator")
        allocator = FlexAllocator(nodes=tmp_nodes)

    if not res_24['ok']:
        print(
            f"  [Skeleton-Alloc] 分支仍未全部分配: "
            f"unallocated_branch={res_24.get('unallocated_branch', 0)} "
            f"req22={res_24.get('req_22', 0)} req11={res_24.get('req_11', 0)} req23={res_24.get('req_23', 0)}"
        )
    final_proxy_diag = _block_c_eval(adjuster, tmp_nodes, S_target, E_target)
    branch_ok = (
        bool(res_24['ok']) and
        bool(final_proxy_diag.get('hist_proxy_ok', True))
    )
    branch_fail_parts: List[str] = []
    if not bool(res_24['ok']):
        branch_fail_parts.append(
            "allocation"
            f"(unallocated={int(res_24.get('unallocated_branch', 0))},"
            f"req22={int(res_24.get('req_22', 0))},"
            f"req11={int(res_24.get('req_11', 0))},"
            f"req23={int(res_24.get('req_23', 0))})"
        )
    if not bool(final_proxy_diag.get('hist_proxy_ok', True)):
        branch_fail_parts.append(
            "hist"
            f"(reason={str(final_proxy_diag.get('hist_proxy_reason', final_proxy_diag.get('reason', 'unknown')))})"
        )
    runtime_proxy_deferred = not bool(final_proxy_diag.get('runtime_layer1_ok', True))
    if bool(runtime_proxy_deferred):
        print(
            "  [Skeleton-Alloc] 分支候选的 runtime proxy 将在 Layer1 重建后再校验: "
            f"su11miss={int(final_proxy_diag.get('su11_missing_external', 0))} "
            f"specialMiss={int(final_proxy_diag.get('special_missing_external', 0))} "
            f"fixedGap={int(final_proxy_diag.get('special_fixed_anchor_gap', 0))} "
            f"degreeGap={int(final_proxy_diag.get('special_degree_gap', 0))}"
        )
    if bool(branch_ok):
        adjuster._h_rotation_state = int(rot_idx)
        adjuster._refresh_node_counters(tmp_nodes)
        for i, tn in enumerate(tmp_nodes):
            nodes[i].su_type = tn.su_type
            nodes[i].hop1_su = Counter(tn.hop1_su)
            nodes[i].hop2_su = Counter(tn.hop2_su)
            nodes[i].target_hop1_degree = getattr(tn, 'target_hop1_degree', None)
            nodes[i].special_anchor_partition = getattr(tn, 'special_anchor_partition', None)
            nodes[i].target_fixed_anchor_count = getattr(tn, 'target_fixed_anchor_count', None)
            nodes[i].special_anchor_mode = getattr(tn, 'special_anchor_mode', None)
    else:
        fail_txt = "; ".join(branch_fail_parts) if branch_fail_parts else "unknown"
        print(f"  [Skeleton-Alloc] 分支候选未通过分支资源/直方图约束，跳过写回 seed topology: {fail_txt}")
    print(f"  [Skeleton-Alloc] 最终H偏差: {_h_ratio(H_work)*100:.2f}%")
    final_scenario = 'ok' if bool(branch_ok) else 'branch_not_ok'
    if not bool(branch_ok) and branch_fail_parts:
        final_scenario = f"{final_scenario}: {'; '.join(branch_fail_parts)}"
    return H_work, moves, {
        'n_moves': len(moves),
        'ok': bool(branch_ok),
        'final_h_ratio': float(_h_ratio(H_work)),
        'records': moves,
        'final_diag': final_proxy_diag,
        'fail_reasons': list(branch_fail_parts),
        'phase': 'branch',
        'block_c_phase': 'branch_topology',
        'final_scenario': str(final_scenario),
    }


def _count_su(nodes: List[_NodeV3], su_type: int) -> int:
    return sum(1 for n in nodes if int(getattr(n, 'su_type', -1)) == int(su_type))


def _find_nodes_by_type(nodes: List[_NodeV3], su_type: int, count: int) -> List[_NodeV3]:
    picked = [n for n in nodes if int(getattr(n, 'su_type', -1)) == int(su_type)]
    picked.sort(key=lambda n: int(getattr(n, 'global_id', 0)))
    return picked[:max(0, int(count))]


def _block_c_bounds(adjuster: Any, nodes: List[_NodeV3]) -> Dict[str, int]:
    cluster_meta = adjuster._compute_aromatic_cluster_metrics(nodes)
    x = int(cluster_meta.get('cluster_count', 0))
    y = max(0, _count_su(nodes, 10) // 2)
    z = int(max(0, int(x) - int(y)))
    lower = int(z + 1)
    upper_raw = int(math.floor(0.8 * float(max(x, 0))))
    return {
        'X': int(x),
        'Y': int(y),
        'Z': int(z),
        'flex_lower_raw': int(lower),
        'flex_upper_raw': int(upper_raw),
    }


def _block_c_eval(adjuster: Any,
                  nodes: List[_NodeV3],
                  S_target: Optional[torch.Tensor],
                  E_target: Optional[torch.Tensor]) -> Dict[str, Any]:
    try:
        diag = adjuster._evaluate_full_allocation_balance(
            nodes,
            flex_ratio=0.80,
            flex_lower_extra=1,
            S_target=S_target,
            E_target=E_target,
        )
    except Exception as e:
        diag = {
            'ok': False,
            'reason': f'alloc_eval_error:{e}',
            'warnings': [],
            'cluster_count': 0,
            'effective_cluster_count': 0,
            'rigid_pairs': 0,
            'rigid_cluster_count': 0,
            'flexible_bridge_count': 0,
            'flexible_bridge_min': 0,
            'flexible_bridge_limit': 0,
            'side_to_22_count': 0,
            'aliphatic_total': 0,
            'aliphatic_min_total': 0,
            'aliphatic_max_total': 10**9,
            'unallocated_bridge': 0,
            'unallocated_branch': 0,
            'required_extra_11': 0,
            'required_extra_22': 0,
            'required_extra_23': 0,
            'remaining': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'native_remaining': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'proxy_remaining': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'native_total': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'proxy_total': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'native_consumed': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'proxy_consumed': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'alloc_eval_error': str(e),
        }
    bounds = _block_c_bounds(adjuster, nodes)
    diag = dict(diag)
    diag.update(bounds)
    diag['flex_count'] = int(diag.get('flexible_bridge_count', 0))
    diag['flex_lower'] = int(bounds['flex_lower_raw'])
    diag['flex_upper'] = int(bounds['flex_upper_raw'])
    diag['remaining_11'] = int((diag.get('remaining', {}) or {}).get('11', 0))
    diag['remaining_22'] = int((diag.get('remaining', {}) or {}).get('22', 0))
    diag['remaining_23'] = int((diag.get('remaining', {}) or {}).get('23', 0))
    diag['remaining_24'] = int((diag.get('remaining', {}) or {}).get('24', 0))
    diag['remaining_25'] = int((diag.get('remaining', {}) or {}).get('25', 0))
    diag['native_remaining_11'] = int((diag.get('native_remaining', {}) or {}).get('11', 0))
    diag['native_remaining_22'] = int((diag.get('native_remaining', {}) or {}).get('22', 0))
    diag['native_remaining_23'] = int((diag.get('native_remaining', {}) or {}).get('23', 0))
    diag['native_remaining_24'] = int((diag.get('native_remaining', {}) or {}).get('24', 0))
    diag['native_remaining_25'] = int((diag.get('native_remaining', {}) or {}).get('25', 0))
    diag['proxy_remaining_11'] = int((diag.get('proxy_remaining', {}) or {}).get('11', 0))
    diag['proxy_remaining_22'] = int((diag.get('proxy_remaining', {}) or {}).get('22', 0))
    diag['proxy_remaining_23'] = int((diag.get('proxy_remaining', {}) or {}).get('23', 0))
    diag['proxy_remaining_24'] = int((diag.get('proxy_remaining', {}) or {}).get('24', 0))
    diag['proxy_remaining_25'] = int((diag.get('proxy_remaining', {}) or {}).get('25', 0))
    diag['adjustable_remaining_11'] = int(diag.get('native_remaining_11', 0))
    diag['adjustable_remaining_22'] = int(diag.get('native_remaining_22', 0))
    diag['adjustable_remaining_23'] = int(diag.get('native_remaining_23', 0))
    diag['adjustable_remaining_24'] = int(diag.get('native_remaining_24', 0))
    diag['adjustable_remaining_25'] = int(diag.get('native_remaining_25', 0))
    diag['bounds_ok'] = bool(int(diag['flex_lower']) <= int(diag['flex_upper']))
    diag['flex_short'] = bool(int(diag['flex_count']) < int(diag['flex_lower']))
    diag['flex_excess'] = bool(int(diag['flex_count']) > int(diag['flex_upper']))
    runtime_proxy = adjuster._evaluate_runtime_layer1_proxy(nodes)
    diag.update(runtime_proxy)

    H_nodes = torch.zeros(33, dtype=torch.long)
    for node in list(nodes or []):
        try:
            su_i = int(getattr(node, 'su_type', -1))
        except Exception:
            su_i = -1
        if 0 <= int(su_i) < int(H_nodes.numel()):
            H_nodes[int(su_i)] += 1
    hist_proxy = adjuster._evaluate_required_hist_constraints(H_nodes, E_target, S_target=S_target)
    diag['hist_proxy_ok'] = bool(hist_proxy.get('ok', False))
    diag['hist_proxy_reason'] = str(hist_proxy.get('reason', 'unknown'))
    diag['hist_proxy_reasons'] = list(hist_proxy.get('reasons', []) or [])
    diag['hist_su11_external_ok'] = bool(hist_proxy.get('su11_external_ok', True))
    diag['hist_su11_required'] = int(hist_proxy.get('su11_required', 0))
    diag['hist_su11_external_slots'] = int(hist_proxy.get('su11_external_slots', 0))
    diag['hist_special_generic_ok'] = bool(hist_proxy.get('special_generic_ok', True))
    diag['hist_special_generic_demand'] = int(hist_proxy.get('special_generic_demand', 0))
    diag['hist_special_generic_partner_slots'] = int(hist_proxy.get('special_generic_partner_slots', 0))
    arom_meta = dict(hist_proxy.get('aromatic_balance', {}) or {})
    diag['aromatic_balance_ok'] = bool(hist_proxy.get('aromatic_balance_ok', True))
    diag['su12'] = int(arom_meta.get('su12', 0))
    diag['su13'] = int(arom_meta.get('su13', 0))
    diag['su13_min'] = int(arom_meta.get('su13_min', 0))
    diag['su12_max'] = int(arom_meta.get('su12_max', 10**9))
    diag['aromatic_ch_target'] = int(arom_meta.get('aromatic_ch_target', 0))
    return diag


def _block_c_penalty(diag: Dict[str, Any]) -> Tuple[int, ...]:
    flex_count = int(diag.get('flex_count', 0))
    flex_lo = int(diag.get('flex_lower', 0))
    flex_hi = int(diag.get('flex_upper', 0))
    req11 = max(0, int(diag.get('required_extra_11', 0)))
    req22 = max(0, int(diag.get('required_extra_22', 0)))
    req23 = max(0, int(diag.get('required_extra_23', 0)))
    remaining_11 = max(0, int(diag.get('remaining_11', 0)))
    remaining_22 = max(0, int(diag.get('remaining_22', 0)))
    remaining_23 = max(0, int(diag.get('remaining_23', 0)))
    remaining_24 = max(0, int(diag.get('remaining_24', 0)))
    remaining_25 = max(0, int(diag.get('remaining_25', 0)))
    native_tail_residual = (
        max(0, int(diag.get('native_remaining_11', 0))) +
        max(0, int(diag.get('native_remaining_22', 0))) +
        max(0, int(diag.get('native_remaining_23', 0)))
    )
    unexpected_tail_residual = int(remaining_22 + remaining_24 + remaining_25)
    return (
        max(0, int(diag.get('unallocated_branch', 0))),
        max(0, int(diag.get('unallocated_bridge', 0))),
        0 if bool(diag.get('hist_proxy_ok', True)) else 1,
        int(req11 + req22 + req23),
        int(req11),
        int(req22),
        int(req23),
        int(remaining_24 + remaining_25),
        max(0, int(flex_count - flex_hi)),
        max(0, int(flex_lo - flex_count)),
        int(native_tail_residual),
        int(unexpected_tail_residual),
        max(0, int(remaining_11 + remaining_23)),
        max(0, int(remaining_11 - 10)),
        max(0, int(diag.get('aliphatic_min_total', 0) - diag.get('aliphatic_total', 0))),
        max(0, int(diag.get('aliphatic_total', 0) - diag.get('aliphatic_max_total', 0))),
        max(0, int(diag.get('ordinary_aliphatic_min_total', 0) - diag.get('ordinary_aliphatic_total', 0))),
        max(0, int(diag.get('ordinary_aliphatic_total', 0) - diag.get('ordinary_aliphatic_max_total', 0))),
        max(0, int(diag.get('oxygenated_aliphatic_min_total', 0) - diag.get('oxygenated_aliphatic_total', 0))),
        max(0, int(diag.get('oxygenated_aliphatic_total', 0) - diag.get('oxygenated_aliphatic_max_total', 0))),
        0 if bool(diag.get('aromatic_balance_ok', True)) else 1,
        max(0, int(diag.get('su13_min', 0)) - int(diag.get('su13', 0))),
        max(0, int(diag.get('su12', 0)) - int(diag.get('su12_max', 10**9))),
        max(0, int(diag.get('flex_lower', 0) - int(diag.get('flex_upper', 0)))),
    )


def _block_c_unexpected_residual(diag: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        max(0, int(diag.get('remaining_22', 0))),
        max(0, int(diag.get('remaining_24', 0))),
        max(0, int(diag.get('remaining_25', 0))),
    )


def _block_c_residual_11_23_done(diag: Dict[str, Any]) -> bool:
    rem11 = int(diag.get('remaining_11', 0))
    rem23 = int(diag.get('remaining_23', 0))
    unexpected = _block_c_unexpected_residual(diag)
    return (
        int(rem11) == 0 and
        int(rem23) == 0 and
        int(sum(int(x) for x in unexpected)) == 0
    )


def _get_h_ratio(adjuster: Any, H: torch.Tensor, E_target: Optional[torch.Tensor]) -> float:
    if E_target is None:
        return 0.0
    try:
        target_h = float(E_target.detach().cpu().flatten()[1].item())
    except Exception:
        return 0.0
    if target_h <= 0.0:
        return 0.0
    H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
    try:
        adjuster.E_target = E_target.detach().cpu()
        _, h_ratio_fn, _, _ = adjuster._make_h_helpers()
        return float(h_ratio_fn(H_cpu))
    except Exception:
        pass
    pred = torch.matmul(H_cpu.float(), adjuster.E_SU.cpu())
    current_h = float(pred[1].item())
    return float((current_h - target_h) / target_h)


def adjust_block_c_extra_phase_impl(
    adjuster: Any,
    H: torch.Tensor,
    E_target: Optional[torch.Tensor],
    S_target: Optional[torch.Tensor] = None,
    ppm: Optional[np.ndarray] = None,
    diff: Optional[np.ndarray] = None,
    guided_max_steps: int = 150,
    relaxed_flexible_ratio: float = 0.82,
    nodes: Optional[List[_NodeV3]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n  [Step 2] BLOCK_C 第三阶段: 柔性链/团簇/后校正")
    if nodes is None:
        print("    [Step 2] 缺少 nodes，无法进行第三阶段评估")
        return H, [], {'n_moves': 0, 'ok': False, 'phase': 'extra', 'reason': 'missing_nodes'}

    import copy
    H_work = torch.clamp(H, min=0).long().clone().cpu()
    tmp_nodes = copy.deepcopy(nodes)
    adjuster._refresh_node_counters(tmp_nodes)
    adjuster.E_target = E_target
    tail_bias = {'need_22': 0.0, 'need_23': 0.0, 'need_24': 0.0, 'prefer_preserve_24': False}
    if ppm is not None and diff is not None:
        try:
            ppm_arr = np.asarray(ppm, dtype=np.float64)
            diff_arr = np.asarray(diff, dtype=np.float64)
            if int(ppm_arr.size) > 0 and int(diff_arr.size) > 0:
                tail_stats = _collect_tail_window_stats(adjuster, ppm_arr, diff_arr)
                tail_bias = {
                    'need_22': float(tail_stats['need'][22]),
                    'need_23': float(tail_stats['need'][23]),
                    'need_24': float(tail_stats['need'][24]),
                    'prefer_preserve_24': bool(_prefer_preserve_24_from_tail_stats(tail_stats)),
                }
        except Exception:
            tail_bias = {'need_22': 0.0, 'need_23': 0.0, 'need_24': 0.0, 'prefer_preserve_24': False}

    moves: List[Dict[str, Any]] = []
    phase_moves: Dict[str, List[Dict[str, Any]]] = {
        'extra': [],
        'align': [],
        'post': [],
    }

    def _log(op: str,
             stage: str,
             before_diag: Dict[str, Any],
             extra_payload: Optional[Dict[str, Any]] = None):
        rec = {
            'op': str(op),
            'stage': str(stage),
            'diagnostic_before': {
                'flex_count': int(before_diag.get('flex_count', 0)),
                'flex_lower': int(before_diag.get('flex_lower', 0)),
                'flex_upper': int(before_diag.get('flex_upper', 0)),
                'remaining_11': int(before_diag.get('remaining_11', 0)),
                'remaining_22': int(before_diag.get('remaining_22', 0)),
                'remaining_23': int(before_diag.get('remaining_23', 0)),
                'remaining_24': int(before_diag.get('remaining_24', 0)),
                'remaining_25': int(before_diag.get('remaining_25', 0)),
                'native_remaining_11': int(before_diag.get('native_remaining_11', 0)),
                'native_remaining_22': int(before_diag.get('native_remaining_22', 0)),
                'native_remaining_23': int(before_diag.get('native_remaining_23', 0)),
                'native_remaining_24': int(before_diag.get('native_remaining_24', 0)),
                'native_remaining_25': int(before_diag.get('native_remaining_25', 0)),
                'proxy_remaining_11': int(before_diag.get('proxy_remaining_11', 0)),
                'proxy_remaining_22': int(before_diag.get('proxy_remaining_22', 0)),
                'proxy_remaining_23': int(before_diag.get('proxy_remaining_23', 0)),
                'proxy_remaining_24': int(before_diag.get('proxy_remaining_24', 0)),
                'proxy_remaining_25': int(before_diag.get('proxy_remaining_25', 0)),
                'required_extra_11': int(before_diag.get('required_extra_11', 0)),
                'required_extra_22': int(before_diag.get('required_extra_22', 0)),
                'required_extra_23': int(before_diag.get('required_extra_23', 0)),
                'ordinary_aliphatic_total': int(before_diag.get('ordinary_aliphatic_total', 0)),
                'ordinary_aliphatic_max_total': int(before_diag.get('ordinary_aliphatic_max_total', 0)),
                'su12': int(before_diag.get('su12', 0)),
                'su13': int(before_diag.get('su13', 0)),
                'su13_min': int(before_diag.get('su13_min', 0)),
                'su12_max': int(before_diag.get('su12_max', 0)),
                'X': int(before_diag.get('X', 0)),
                'Y': int(before_diag.get('Y', 0)),
                'Z': int(before_diag.get('Z', 0)),
            },
            'h_ratio_before': float(_get_h_ratio(adjuster, H_work, E_target)),
        }
        if isinstance(extra_payload, dict):
            rec.update(dict(extra_payload))
        moves.append(rec)
        phase_moves['extra'].append(dict(rec))
        print(
            f"    -> {op} | flex={before_diag.get('flex_count', 0)}/"
            f"[{before_diag.get('flex_lower', 0)},{before_diag.get('flex_upper', 0)}] "
            f"native11={before_diag.get('native_remaining_11', 0)} "
            f"proxy11={before_diag.get('proxy_remaining_11', 0)} "
            f"native22={before_diag.get('native_remaining_22', 0)} "
            f"native23={before_diag.get('native_remaining_23', 0)} "
            f"req11={before_diag.get('required_extra_11', 0)} "
            f"req22={before_diag.get('required_extra_22', 0)} "
            f"req23={before_diag.get('required_extra_23', 0)} "
            f"su12/13={before_diag.get('su12', 0)}/{before_diag.get('su13', 0)}"
        )

    def _refresh() -> None:
        adjuster._refresh_node_counters(tmp_nodes)
        H_work.zero_()
        for n in tmp_nodes:
            su = int(getattr(n, 'su_type', -1))
            if 0 <= su < int(H_work.numel()):
                H_work[su] += 1

    def _apply_node_h_rotation(nodes_local: List[_NodeV3],
                               H_local: torch.Tensor) -> List[str]:
        h_tol = float(getattr(adjuster, '_h_tolerance', 0.08))
        if E_target is None or abs(float(_get_h_ratio(adjuster, H_local, E_target))) <= float(h_tol):
            return []
        _, h_ratio_fn, _, _ = adjuster._make_h_helpers()
        rot_ops, rot_idx_new = adjuster._h_rotation_adjust(
            nodes_local,
            H_local,
            h_ratio_fn,
            int(getattr(adjuster, '_h_rotation_state', 0)),
            max_ops=None,
            max_aliphatic_total=getattr(adjuster, '_h_rotation_aliphatic_cap', None),
            max_ordinary_aliphatic_total=getattr(adjuster, '_h_rotation_ordinary_aliphatic_cap', None),
            h_tolerance=float(h_tol),
        )
        adjuster._h_rotation_state = int(rot_idx_new)
        return [str(op) for op in list(rot_ops or [])]

    def _convert_n(nodes_local: List[_NodeV3],
                   H_local: torch.Tensor,
                   src: int,
                   dst: int,
                   count: int) -> bool:
        picked = _find_nodes_by_type(nodes_local, int(src), int(count))
        if len(picked) < int(count):
            return False
        for node in picked:
            H_local[int(src)] -= 1
            node.su_type = int(dst)
            H_local[int(dst)] += 1
            if int(dst) not in {19, 20, 21}:
                for attr_name in (
                    'target_hop1_degree',
                    'init_target_hop1_degree',
                    'special_degree_source',
                    'special_anchor_partition',
                    'target_fixed_anchor_count',
                    'init_target_fixed_anchor_count',
                    'special_anchor_mode',
                ):
                    try:
                        setattr(node, attr_name, None)
                    except Exception:
                        pass
        return True

    def _unique_k_values(desired: int) -> List[int]:
        desired_i = max(0, int(desired))
        vals: List[int] = []
        for v in (desired_i, int(math.ceil(float(desired_i) / 2.0)), 1):
            v_i = int(v)
            if v_i > 0 and v_i not in vals:
                vals.append(v_i)
        return vals

    def _have_for_conversions(H_local: torch.Tensor,
                              conversions: List[Tuple[int, int, int]],
                              k: int) -> bool:
        need: Dict[int, int] = {}
        for src, dst, mult in list(conversions or []):
            src_i = int(src)
            dst_i = int(dst)
            if src_i == 25 or dst_i == 25:
                return False
            need[src_i] = int(need.get(src_i, 0) + int(mult) * int(k))
        for src_i, cnt in need.items():
            if int(H_local[int(src_i)].item()) < int(cnt):
                return False
        return True

    def _ordinary_delta(conversions: List[Tuple[int, int, int]], k: int) -> int:
        ordinary = {22, 23, 24, 25}
        delta = 0
        for src, dst, mult in list(conversions or []):
            amount = int(mult) * int(k)
            if int(src) in ordinary:
                delta -= int(amount)
            if int(dst) in ordinary:
                delta += int(amount)
        return int(delta)

    def _apply_conversion_bundle(nodes_local: List[_NodeV3],
                                 H_local: torch.Tensor,
                                 conversions: List[Tuple[int, int, int]],
                                 k: int) -> bool:
        if not _have_for_conversions(H_local, conversions, k):
            return False
        for src, dst, mult in list(conversions or []):
            count = int(mult) * int(k)
            if count <= 0:
                continue
            if not _convert_n(nodes_local, H_local, int(src), int(dst), int(count)):
                return False
        return True

    def _try_guided_bundle(op: str,
                           stage: str,
                           desired_k: int,
                           conversions: List[Tuple[int, int, int]],
                           before_diag: Dict[str, Any]) -> bool:
        nonlocal H_work, tmp_nodes
        if int(desired_k) <= 0:
            return False
        if any(int(src) == 25 or int(dst) == 25 for src, dst, _ in list(conversions or [])):
            return False

        prev_penalty = _block_c_penalty(before_diag)
        for k in _unique_k_values(int(desired_k)):
            ordinary_max = int(before_diag.get('ordinary_aliphatic_max_total', 10**9))
            ordinary_total = int(before_diag.get('ordinary_aliphatic_total', 0))
            if int(ordinary_total) >= int(ordinary_max) and int(_ordinary_delta(conversions, int(k))) > 0:
                continue
            prev_rot_state = int(getattr(adjuster, '_h_rotation_state', 0))
            new_nodes = copy.deepcopy(tmp_nodes)
            new_H = H_work.detach().clone()

            if not _apply_conversion_bundle(new_nodes, new_H, conversions, int(k)):
                adjuster._h_rotation_state = int(prev_rot_state)
                continue

            h_moves: List[str] = []
            h_tol = float(getattr(adjuster, '_h_tolerance', 0.08))
            if abs(float(_get_h_ratio(adjuster, new_H, E_target))) > float(h_tol):
                h_moves = _apply_node_h_rotation(new_nodes, new_H)
                if abs(float(_get_h_ratio(adjuster, new_H, E_target))) > float(h_tol):
                    adjuster._h_rotation_state = int(prev_rot_state)
                    continue

            adjuster._refresh_node_counters(new_nodes)
            new_diag = _block_c_eval(adjuster, new_nodes, S_target, E_target)
            if not bool(new_diag.get('aromatic_balance_ok', True)):
                adjuster._h_rotation_state = int(prev_rot_state)
                continue
            if int(new_diag.get('ordinary_aliphatic_total', 0)) > int(new_diag.get('ordinary_aliphatic_max_total', 10**9)):
                if int(new_diag.get('ordinary_aliphatic_total', 0)) > int(before_diag.get('ordinary_aliphatic_total', 0)):
                    adjuster._h_rotation_state = int(prev_rot_state)
                    continue
            new_penalty = _block_c_penalty(new_diag)
            if tuple(new_penalty) >= tuple(prev_penalty):
                adjuster._h_rotation_state = int(prev_rot_state)
                continue

            tmp_nodes = new_nodes
            H_work = new_H.detach().clone().cpu()
            _refresh()
            delta_counts: Dict[int, int] = {}
            for src, dst, mult in list(conversions or []):
                delta_counts[int(src)] = int(delta_counts.get(int(src), 0) - int(mult) * int(k))
                delta_counts[int(dst)] = int(delta_counts.get(int(dst), 0) + int(mult) * int(k))
            payload = {
                'k': int(k),
                'delta_counts': dict(sorted(delta_counts.items())),
                'penalty_before': tuple(int(x) for x in prev_penalty),
                'penalty_after': tuple(int(x) for x in new_penalty),
                'h_moves': list(h_moves),
            }
            _log(str(op), str(stage), before_diag, extra_payload=payload)
            return True

        return False

    def _add_guided_candidate(cands: List[Tuple[str, str, int, List[Tuple[int, int, int]]]],
                              op: str,
                              stage: str,
                              desired: int,
                              conversions: List[Tuple[int, int, int]]) -> None:
        desired_i = int(max(0, desired))
        if desired_i <= 0:
            return
        if any(int(src) == 25 or int(dst) == 25 for src, dst, _ in list(conversions or [])):
            return
        h13_now = max(0, int(H_work[13].item())) if int(H_work.numel()) > 13 else 0
        su13_min_now = int(_block_c_eval(adjuster, tmp_nodes, S_target, E_target).get('su13_min', 0))
        consumes_13 = sum(int(mult) for src, _dst, mult in list(conversions or []) if int(src) == 13)
        if int(consumes_13) > 0 and int(h13_now - int(consumes_13) * desired_i) < int(su13_min_now):
            desired_i = max(0, (int(h13_now) - int(su13_min_now)) // max(1, int(consumes_13)))
            if int(desired_i) <= 0:
                return
        diag_now = _block_c_eval(adjuster, tmp_nodes, S_target, E_target)
        ordinary_total = int(diag_now.get('ordinary_aliphatic_total', 0))
        ordinary_max = int(diag_now.get('ordinary_aliphatic_max_total', 10**9))
        if int(ordinary_total) >= int(ordinary_max) and int(_ordinary_delta(conversions, 1)) > 0:
            return
        cands.append((str(op), str(stage), int(desired_i), list(conversions)))

    def _run_guided_extra_step(diag: Dict[str, Any]) -> bool:
        req11 = max(0, int(diag.get('required_extra_11', 0)))
        req22 = max(0, int(diag.get('required_extra_22', 0)))
        req23 = max(0, int(diag.get('required_extra_23', 0)))
        rem11 = max(0, int(diag.get('remaining_11', 0)))
        rem22 = max(0, int(diag.get('remaining_22', 0)))
        rem23 = max(0, int(diag.get('remaining_23', 0)))
        rem24 = max(0, int(diag.get('remaining_24', 0)))
        flex_count = int(diag.get('flex_count', 0))
        flex_lower = int(diag.get('flex_lower', 0))
        flex_upper = int(diag.get('flex_upper', 0))
        flex_short = max(0, int(flex_lower - flex_count))
        flex_excess = max(0, int(flex_count - flex_upper))
        unallocated_branch = max(0, int(diag.get('unallocated_branch', 0)))
        fixed_flex_count = int(diag.get('fixed_flexible_bridge_count', 0))
        aliphatic_total = int(diag.get('aliphatic_total', 0))
        aliphatic_min = int(diag.get('aliphatic_min_total', 0))
        preserve_24 = bool(tail_bias.get('prefer_preserve_24', False))

        def _hc(su: int) -> int:
            try:
                return max(0, int(H_work[int(su)].item()))
            except Exception:
                return 0

        h11 = _hc(11)
        h12 = _hc(12)
        h13 = _hc(13)
        h22 = _hc(22)
        h23 = _hc(23)
        h24 = _hc(24)
        cands: List[Tuple[str, str, int, List[Tuple[int, int, int]]]] = []

        if unallocated_branch > 0:
            _add_guided_candidate(
                cands, 'S2_C3_branch_23+11->24+13', 'S2_guided_C3',
                min(unallocated_branch, h23, h11, 6),
                [(23, 24, 1), (11, 13, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_branch_2x23->24+22', 'S2_guided_C3',
                min(unallocated_branch, h23 // 2, 6),
                [(23, 24, 1), (23, 22, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_req23_22+12->23+13', 'S2_guided_C3',
                min(max(req23, 0), h22, h12, 6),
                [(22, 23, 1), (12, 13, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_req11_12->11', 'S2_guided_C3',
                min(max(req11, 0), h12, 6),
                [(12, 11, 1)],
            )

        if req11 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_req11_12->11', 'S2_guided_C3',
                min(req11, h12, 8),
                [(12, 11, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_req11_req22_13+23->11+22', 'S2_guided_C3',
                min(max(req11, req22), h13, h23, 8),
                [(13, 11, 1), (23, 22, 1)],
            )

        if req22 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_req22_13+23->11+22', 'S2_guided_C3',
                min(req22, h13, h23, 8),
                [(13, 11, 1), (23, 22, 1)],
            )

        if req23 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_req23_22+12->23+13', 'S2_guided_C3',
                min(req23, h22, h12, 8),
                [(22, 23, 1), (12, 13, 1)],
            )
            if not bool(preserve_24):
                _add_guided_candidate(
                    cands, 'S2_C3_req23_24+13->23+11', 'S2_guided_C3',
                    min(req23 + rem24, h24, h13, 6),
                    [(24, 23, 1), (13, 11, 1)],
                )

        if rem24 > 0 and not bool(preserve_24):
            _add_guided_candidate(
                cands, 'S2_C3_residual24_24+13->23+11', 'S2_guided_C3',
                min(rem24, h24, h13, 6),
                [(24, 23, 1), (13, 11, 1)],
            )

        if flex_excess > 0:
            if fixed_flex_count > flex_upper:
                _add_guided_candidate(
                    cands, 'S2_C4_unlock_2x12+23->2x11+24', 'S2_guided_C4',
                    min(max(1, flex_excess), h12 // 2, h23, 4),
                    [(12, 11, 2), (23, 24, 1)],
                )
            _add_guided_candidate(
                cands, 'S2_C4_flex_excess_11+23->13+22', 'S2_guided_C4',
                min(flex_excess, h11, h23, 8),
                [(11, 13, 1), (23, 22, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C4_flex_excess_2x23->24+22', 'S2_guided_C4',
                min(flex_excess, h23 // 2, 6),
                [(23, 24, 1), (23, 22, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C4_flex_excess_12->11', 'S2_guided_C4',
                min(flex_excess + req11 + rem22, h12, 8),
                [(12, 11, 1)],
            )

        if flex_short > 0:
            _add_guided_candidate(
                cands, 'S2_C4_flex_short_2x13->11+23', 'S2_guided_C4',
                min(flex_short, h13 // 2, 8),
                [(13, 11, 1), (13, 23, 1)],
            )
            if not bool(preserve_24):
                _add_guided_candidate(
                    cands, 'S2_C4_flex_short_24+22->2x23', 'S2_guided_C4',
                    min(flex_short, h24, h22, 6),
                    [(24, 23, 1), (22, 23, 1)],
                )
            _add_guided_candidate(
                cands, 'S2_C4_flex_short_12->11', 'S2_guided_C4',
                min(flex_short + req11, h12, 6),
                [(12, 11, 1)],
            )

        if rem11 > 0 and rem23 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_residual_11+23->13+22', 'S2_guided_C3',
                min(rem11, rem23, max(1, flex_excess), h11, h23, 8),
                [(11, 13, 1), (23, 22, 1)],
            )
        if rem22 > 0 and req23 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_residual22_22+12->23+13', 'S2_guided_C3',
                min(rem22, req23, h22, h12, 8),
                [(22, 23, 1), (12, 13, 1)],
            )
        if rem22 > 0:
            _add_guided_candidate(
                cands, 'S2_residual_22+12->23+13', 'S2_guided_residual',
                min(rem22, h22, h12, 6),
                [(22, 23, 1), (12, 13, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_residual_22->23', 'S2_guided_residual',
                min(rem22, h22, 4),
                [(22, 23, 1)],
            )
        if rem24 > 0 and not bool(preserve_24):
            _add_guided_candidate(
                cands, 'S2_residual_24->23', 'S2_guided_residual',
                min(rem24, h24, 4),
                [(24, 23, 1)],
            )
        if rem11 > 0:
            _add_guided_candidate(
                cands, 'S2_residual_11->13', 'S2_guided_residual',
                min(rem11, h11, 6),
                [(11, 13, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_residual_11->12', 'S2_guided_residual',
                min(rem11, h11, 4),
                [(11, 12, 1)],
            )
        if rem23 > 0 and int(aliphatic_total - 1) >= int(aliphatic_min):
            _add_guided_candidate(
                cands, 'S2_residual_23->13', 'S2_guided_residual',
                min(rem23, h23, 6),
                [(23, 13, 1)],
            )

        for op, stage, desired, conversions in cands:
            if _try_guided_bundle(op, stage, int(desired), conversions, diag):
                return True
        return False

    diag = _block_c_eval(adjuster, tmp_nodes, S_target, E_target)
    initial_extra_diag = dict(diag)
    max_total_steps = max(1, int(guided_max_steps))

    def _run_guided_bounds_step(diag_local: Dict[str, Any]) -> bool:
        gap = max(0, int(diag_local.get('flex_lower', 0)) - int(diag_local.get('flex_upper', 0)))
        if int(gap) <= 0:
            return False
        h12 = max(0, int(H_work[12].item()))
        bounds_cands: List[Tuple[str, str, int, List[Tuple[int, int, int]]]] = []
        _add_guided_candidate(
            bounds_cands, 'S2_bounds_12->13', 'S2_guided_bounds',
            min(max(1, gap), h12, 6),
            [(12, 13, 1)],
        )
        _add_guided_candidate(
            bounds_cands, 'S2_bounds_2x12->2x10', 'S2_guided_bounds',
            min(max(1, gap), h12 // 2, 3),
            [(12, 10, 2)],
        )
        for op, stage, desired, conversions in bounds_cands:
            if _try_guided_bundle(op, stage, int(desired), conversions, diag_local):
                return True
        return False

    def _extra_stage_done(diag_local: Dict[str, Any]) -> bool:
        bounds_ok = bool(int(diag_local.get('flex_lower', 0)) <= int(diag_local.get('flex_upper', 0)))
        flex_count = int(diag_local.get('flex_count', 0))
        flex_lower = int(diag_local.get('flex_lower', 0))
        flex_upper = int(diag_local.get('flex_upper', 0))
        flex_ok = bool(int(flex_lower) <= int(flex_count) <= int(flex_upper))
        alloc_ok = bool(diag_local.get('ok', False))
        residual_ok = _block_c_residual_11_23_done(diag_local)
        hist_proxy_ok = bool(diag_local.get('hist_proxy_ok', True))
        return bool(bounds_ok and flex_ok and alloc_ok and residual_ok and hist_proxy_ok)

    step = 0
    while step < int(max_total_steps):
        diag = _block_c_eval(adjuster, tmp_nodes, S_target, E_target)
        fixed_flex_count = int(diag.get('fixed_flexible_bridge_count', 0))
        print(
            f"    [Step 2评估] X={diag.get('X', 0)} Y={diag.get('Y', 0)} Z={diag.get('Z', 0)} "
            f"flex={diag.get('flex_count', 0)}/[{diag.get('flex_lower', 0)},{diag.get('flex_upper', 0)}] "
            f"fixed_flex={fixed_flex_count} "
            f"native11={diag.get('native_remaining_11', 0)} proxy11={diag.get('proxy_remaining_11', 0)} "
            f"native22={diag.get('native_remaining_22', 0)} proxy22={diag.get('proxy_remaining_22', 0)} "
            f"native23={diag.get('native_remaining_23', 0)} proxy23={diag.get('proxy_remaining_23', 0)} "
            f"req11={diag.get('required_extra_11', 0)} req22={diag.get('required_extra_22', 0)} req23={diag.get('required_extra_23', 0)} "
            f"ali={diag.get('aliphatic_total', 0)}/[{diag.get('aliphatic_min_total', 0)},{diag.get('aliphatic_max_total', 0)}] "
            f"ordAli={diag.get('ordinary_aliphatic_total', 0)}/[{diag.get('ordinary_aliphatic_min_total', 0)},{diag.get('ordinary_aliphatic_max_total', 0)}] "
            f"oxyAli={diag.get('oxygenated_aliphatic_total', 0)}/[{diag.get('oxygenated_aliphatic_min_total', 0)},{diag.get('oxygenated_aliphatic_max_total', 0)}] "
            f"su11miss={diag.get('su11_missing_external', 0)} specialGap={diag.get('special_degree_gap', 0)} "
            f"histProxy={'ok' if bool(diag.get('hist_proxy_ok', True)) else 'bad'}"
        )
        unexpected = _block_c_unexpected_residual(diag)
        if int(sum(int(x) for x in unexpected)) > 0:
            print(
                "    [Step 2残余清理] 检测到非11/23残余: "
                f"22={int(unexpected[0])} 24={int(unexpected[1])} 25={int(unexpected[2])}"
            )
        if _extra_stage_done(diag):
            print("    [Step 2] 第三阶段资源分配、柔性链与残余约束通过")
            break
        acted = False
        if not bool(diag.get('bounds_ok', True)):
            acted = _run_guided_bounds_step(diag)
        if not acted:
            acted = _run_guided_extra_step(diag)
        if not acted:
            break
        step += 1

    extra_meta = _block_c_eval(adjuster, tmp_nodes, S_target, E_target)
    H_after_extra = H_work.detach().clone().cpu()

    align_meta: Dict[str, Any] = {
        'applied': False,
        'reason': 'not_run',
    }
    alignment_allowed = _extra_stage_done(extra_meta) or (
        tuple(_block_c_penalty(extra_meta)) < tuple(_block_c_penalty(initial_extra_diag))
    )
    try:
        if alignment_allowed:
            H_aligned, align_moves, align_meta = adjuster._apply_aromatic_cluster_alignment(
                tmp_nodes,
                H_work,
                protect_11=bool(int((extra_meta.get('required_extra_11', 0))) > 0),
            )
            H_work = H_aligned.detach().clone().cpu()
            for mv in align_moves:
                moves.append(dict(mv))
                phase_moves['align'].append(dict(mv))
            if align_moves:
                print(f"  [BlockC-Align] 12/13 芳香预对齐: {len(align_moves)} 次转换")
        else:
            align_meta = {
                'applied': False,
                'reason': 'skip_until_allocation_improves',
            }
    except Exception as e:
        align_meta = {
            'applied': False,
            'reason': 'error',
            'error': str(e),
        }
    H_after_align = H_work.detach().clone().cpu()

    post_meta: Dict[str, Any] = {}
    post_changed = False
    post_moves: List[Dict[str, Any]] = []
    try:
        H_post, final_moves, final_meta = adjuster._apply_final_structure_constraints(H_work)
        if final_moves:
            post_changed = True
            H_work = H_post.detach().clone().cpu()
            for mv in final_moves:
                tagged = dict(mv)
                tagged['stage'] = 'block_c_post_constraints'
                moves.append(dict(tagged))
                post_moves.append(dict(tagged))
                phase_moves['post'].append(dict(tagged))
        post_meta['final_structure_constraints'] = final_meta
    except Exception as e:
        post_meta['error'] = str(e)

    try:
        synced_post_moves = adjuster._apply_post_moves_to_nodes(tmp_nodes, post_moves)
        post_meta['node_sync_post_moves'] = list(synced_post_moves)
    except Exception as e:
        post_meta['node_sync_error'] = str(e)
    try:
        post_h_ops = _apply_node_h_rotation(tmp_nodes, H_work)
        if post_h_ops:
            for op_name in post_h_ops:
                tagged = {'stage': 'block_c_post_h_rotation', 'op': str(op_name)}
                moves.append(dict(tagged))
                phase_moves['post'].append(dict(tagged))
            adjuster._refresh_node_counters(tmp_nodes)
            post_meta['post_h_rotation_ops'] = list(post_h_ops)
    except Exception as e:
        post_meta['post_h_rotation_error'] = str(e)
    H_after_post = H_work.detach().clone().cpu()

    final_alloc_diag = {}
    recheck_completed = False
    try:
        strict_balance_diag = adjuster._evaluate_full_allocation_balance(
            tmp_nodes,
            flex_ratio=0.80,
            flex_lower_extra=1,
            S_target=S_target,
            E_target=E_target,
        )
        relaxed_balance_diag = adjuster._evaluate_full_allocation_balance(
            tmp_nodes,
            flex_ratio=float(relaxed_flexible_ratio),
            flex_lower_extra=1,
            S_target=S_target,
            E_target=E_target,
        )
        balance_diag = relaxed_balance_diag if bool(extra_meta.get('flex_upper', 0) < extra_meta.get('flex_lower', 0)) else strict_balance_diag
        final_alloc_diag = {
            'ok': bool(balance_diag.get('ok', False)),
            'reason': str(balance_diag.get('reason', 'unknown')),
            'selected_mode': 'relaxed' if bool(extra_meta.get('flex_upper', 0) < extra_meta.get('flex_lower', 0)) else 'strict',
            'strict_ok': bool(strict_balance_diag.get('ok', False)),
            'strict_reason': str(strict_balance_diag.get('reason', 'unknown')),
            'relaxed_ok': bool(relaxed_balance_diag.get('ok', False)),
            'relaxed_reason': str(relaxed_balance_diag.get('reason', 'unknown')),
            'warnings': list(balance_diag.get('warnings', []) or []),
            'cluster_count': int(balance_diag.get('cluster_count', 0)),
            'effective_cluster_count': int(balance_diag.get('effective_cluster_count', 0)),
            'rigid_cluster_count': int(balance_diag.get('rigid_cluster_count', 0)),
            'flexible_bridge_count': int(balance_diag.get('flexible_bridge_count', 0)),
            'flexible_bridge_min': int(balance_diag.get('flexible_bridge_min', 0)),
            'flexible_bridge_limit': int(balance_diag.get('flexible_bridge_limit', 0)),
            'rigid_pairs': int(balance_diag.get('rigid_pairs', 0)),
            'side_to_22_count': int(balance_diag.get('side_to_22_count', 0)),
            'aliphatic_total': int(balance_diag.get('aliphatic_total', 0)),
            'aliphatic_min_total': int(balance_diag.get('aliphatic_min_total', 0)),
            'aliphatic_max_total': int(balance_diag.get('aliphatic_max_total', 0)),
            'unallocated_bridge': int(balance_diag.get('unallocated_bridge', 0)),
            'unallocated_branch': int(balance_diag.get('unallocated_branch', 0)),
            'required_extra_11': int(balance_diag.get('required_extra_11', 0)),
            'required_extra_22': int(balance_diag.get('required_extra_22', 0)),
            'required_extra_23': int(balance_diag.get('required_extra_23', 0)),
            'remaining_11': int((balance_diag.get('remaining', {}) or {}).get('11', 0)),
            'remaining_22': int((balance_diag.get('remaining', {}) or {}).get('22', 0)),
            'remaining_23': int((balance_diag.get('remaining', {}) or {}).get('23', 0)),
            'remaining_24': int((balance_diag.get('remaining', {}) or {}).get('24', 0)),
            'remaining_25': int((balance_diag.get('remaining', {}) or {}).get('25', 0)),
            'native_remaining_11': int((balance_diag.get('native_remaining', {}) or {}).get('11', 0)),
            'native_remaining_22': int((balance_diag.get('native_remaining', {}) or {}).get('22', 0)),
            'native_remaining_23': int((balance_diag.get('native_remaining', {}) or {}).get('23', 0)),
            'native_remaining_24': int((balance_diag.get('native_remaining', {}) or {}).get('24', 0)),
            'native_remaining_25': int((balance_diag.get('native_remaining', {}) or {}).get('25', 0)),
            'proxy_remaining_11': int((balance_diag.get('proxy_remaining', {}) or {}).get('11', 0)),
            'proxy_remaining_22': int((balance_diag.get('proxy_remaining', {}) or {}).get('22', 0)),
            'proxy_remaining_23': int((balance_diag.get('proxy_remaining', {}) or {}).get('23', 0)),
            'proxy_remaining_24': int((balance_diag.get('proxy_remaining', {}) or {}).get('24', 0)),
            'proxy_remaining_25': int((balance_diag.get('proxy_remaining', {}) or {}).get('25', 0)),
            'allocation_details': dict(balance_diag.get('allocation_details', {}) or {}),
            'resource_ledger': dict(balance_diag.get('resource_ledger', {}) or {}),
            'cluster_meta': dict(balance_diag.get('cluster_meta', {}) or {}),
        }
        recheck_completed = True
    except Exception as e:
        final_alloc_diag = {'ok': False, 'reason': f'final_allocation_error:{e}'}

    try:
        adjuster._refresh_node_counters(tmp_nodes)
    except Exception:
        pass
    for i, tn in enumerate(tmp_nodes):
        nodes[i].su_type = tn.su_type
        nodes[i].hop1_su = Counter(tn.hop1_su)
        nodes[i].hop2_su = Counter(tn.hop2_su)
        nodes[i].target_hop1_degree = getattr(tn, 'target_hop1_degree', None)
        nodes[i].special_anchor_partition = getattr(tn, 'special_anchor_partition', None)
        nodes[i].target_fixed_anchor_count = getattr(tn, 'target_fixed_anchor_count', None)
        nodes[i].special_anchor_mode = getattr(tn, 'special_anchor_mode', None)

    final_h_ratio = float(_get_h_ratio(adjuster, H_work, E_target))
    post_meta['post_changed'] = bool(post_changed)
    post_meta['recheck_completed'] = bool(recheck_completed)

    overall_ok = bool(final_alloc_diag.get('ok', False))
    return H_work, moves, {
        'n_moves': len(moves),
        'ok': bool(overall_ok),
        'final_h_ratio': float(final_h_ratio),
        'records': moves,
        'final_diag': extra_meta,
        'phase': 'extra',
        'block_c_phase': 'extra_global',
        'relaxed_mode': bool(final_alloc_diag.get('selected_mode') == 'relaxed'),
        'align_meta': align_meta,
        'post_meta': post_meta,
        'final_allocation': final_alloc_diag,
        'phase_hists': {
            'after_extra': H_after_extra,
            'after_align': H_after_align,
            'after_post': H_after_post,
        },
        'phase_moves': phase_moves,
        'recheck_required': bool(post_changed and not recheck_completed),
        'post_changed': bool(post_changed),
        'final_scenario': 'ok' if bool(overall_ok) else str(final_alloc_diag.get('reason', 'extra_not_ok')),
    }
