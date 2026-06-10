import copy
import io
import math
from collections import Counter
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...shared.inverse_common import _NodeV3, get_effective_hist_element_vector
from RL_MTCS.RL_allocator import FlexAllocator


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


def _dominant_shortage_from_reqs(
    req11: Any,
    req22: Any,
    req23: Any,
    req24: Any = 0,
    req25: Any = 0,
) -> Optional[str]:
    reqs = {
        '11_shortage': max(0, int(req11 or 0)),
        '22_shortage': max(0, int(req22 or 0)),
        '23_shortage': max(0, int(req23 or 0)),
        '24_shortage': max(0, int(req24 or 0)),
        '25_shortage': max(0, int(req25 or 0)),
    }
    priority = {
        '23_shortage': 0,
        '24_shortage': 1,
        '22_shortage': 2,
        '11_shortage': 3,
        '25_shortage': 4,
    }
    ranked = sorted(
        reqs.items(),
        key=lambda kv: (-int(kv[1]), int(priority[str(kv[0])])),
    )
    if not ranked or int(ranked[0][1]) <= 0:
        return None
    return str(ranked[0][0])


def _pick_unit_branch_conversion_op(
    shortage: str,
    req_11: int,
    req_22: int,
    req_23: int,
    req_24: int,
    req_25: int,
    counts: Dict[int, int],
    remaining: Optional[Dict[str, int]] = None,
    preserve_24: bool = False,
) -> str:
    """Pick exactly one adjacent conversion for the branch topology loop."""
    req_11 = max(0, int(req_11))
    req_22 = max(0, int(req_22))
    req_23 = max(0, int(req_23))
    req_24 = max(0, int(req_24))
    req_25 = max(0, int(req_25))
    counts = {int(k): max(0, int(v)) for k, v in dict(counts or {}).items()}
    reqs = {
        '11': int(req_11),
        '22': int(req_22),
        '23': int(req_23),
        '24': int(req_24),
        '25': int(req_25),
    }
    rem = {
        str(k): max(0, int(v))
        for k, v in dict(remaining or {}).items()
        if str(k) in {'11', '22', '23', '24', '25'}
    }

    def has(su: int) -> bool:
        return int(counts.get(int(su), 0)) > 0

    def available(su: int) -> bool:
        key = str(int(su))
        if rem:
            return int(rem.get(key, 0)) > 0 and int(reqs.get(key, 0)) <= 0
        return has(int(su)) and int(reqs.get(key, 0)) <= 0

    def surplus(su: int) -> bool:
        key = str(int(su))
        if rem:
            return int(rem.get(key, 0)) > 0 and int(reqs.get(key, 0)) <= 0
        return has(int(su)) and int(reqs.get(key, 0)) <= 0

    def can_use_aromatic() -> bool:
        return has(12) or has(13)

    shortage = str(shortage or 'none')
    if shortage in {'unsupported_special_topology', 'abstract_topology_gap', 'pending_closure', 'none'}:
        shortage = _dominant_shortage_from_reqs(req_11, req_22, req_23, req_24, req_25) or shortage

    prefer_24_repair = bool(
        shortage == '24_shortage'
        or (int(req_24) > 0 and int(req_24) >= int(req_23))
    )

    if prefer_24_repair:
        if available(23):
            return '23->24'
        if surplus(25):
            return '25->24'
        # Only walk one edge per outer Layer1/2 rebuild.
        if available(22) or has(22):
            return '22->23'
        if available(11) or has(11):
            return '11->23'
        if can_use_aromatic():
            return '12/13->11'

    if shortage == '23_shortage' or req_23 > 0:
        if available(22):
            return '22->23'
        if available(24) and not bool(preserve_24):
            return '24->23'
        if available(11):
            return '11->23'
        if can_use_aromatic():
            return '12/13->11'

    if shortage == '22_shortage' or req_22 > 0:
        if available(23):
            return '23->22'
        if available(24) and not bool(preserve_24):
            return '24->23'
        if available(11):
            return '11->23'
        if can_use_aromatic():
            return '12/13->11'

    if shortage == '11_shortage' or req_11 > 0:
        if available(23):
            return '23->11'
        if available(22):
            return '22->23'
        if available(24) and not bool(preserve_24):
            return '24->23'
        if can_use_aromatic():
            return '12/13->11'

    if shortage == '25_shortage' or req_25 > 0:
        return ''

    return ''


def _unit_conversion_delta(op_name: str) -> Dict[str, int]:
    path_map = {
        '22->23': [22, 23],
        '24->23': [24, 23],
        '11->23': [11, 23],
        '23->24': [23, 24],
        '25->24': [25, 24],
        '23->22': [23, 22],
        '23->11': [23, 11],
        '22->23->24': [22, 23, 24],
        '11->23->24': [11, 23, 24],
        '24->23->22': [24, 23, 22],
        '11->23->22': [11, 23, 22],
        '22->23->11': [22, 23, 11],
        '24->23->11': [24, 23, 11],
        '12/13->11': [12, 11],
        '12/13->11->23': [12, 11, 23],
        '12/13->11->23->24': [12, 11, 23, 24],
        '12/13->11->23->22': [12, 11, 23, 22],
    }
    path = list(path_map.get(str(op_name), []) or [])
    if not path:
        return {}
    delta = {key: 0 for key in ('11', '22', '23', '24', '25')}
    if len(path) < 2:
        return delta
    src = int(path[0])
    dst = int(path[-1])
    if str(src) in delta:
        delta[str(src)] -= 1
    if str(dst) in delta:
        delta[str(dst)] += 1
    return delta


def _unit_op_allowed_by_shortage(
    op_name: str,
    reqs: Dict[str, int],
    remaining: Optional[Dict[str, int]] = None,
) -> bool:
    delta = _unit_conversion_delta(str(op_name))
    if not delta:
        return False
    for key in ('11', '22', '23', '24', '25'):
        if int(reqs.get(key, 0)) > 0 and int(delta.get(key, 0)) < 0:
            return False
    rem = {str(k): max(0, int(v)) for k, v in dict(remaining or {}).items()}
    src_key = next((key for key, value in delta.items() if int(value) < 0), '')
    if src_key and rem and int(rem.get(src_key, 0)) <= 0:
        return False
    return True


def _first_unit_op_from_recommendation(
    rec: Dict[str, Any],
    reqs: Dict[str, int],
    remaining: Optional[Dict[str, int]] = None,
) -> str:
    rec = dict(rec or {})
    preferred = [str(x) for x in list(rec.get('preferred_ops', []) or []) if str(x)]
    candidates = [str(rec.get('op', '') or '')] + preferred
    for op_name in candidates:
        if not op_name or op_name.startswith('increase_') or op_name.startswith('enable_'):
            continue
        if _unit_op_allowed_by_shortage(op_name, reqs, remaining=remaining):
            return str(op_name)
    return ''


def _sanitize_branch_shortage_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(diag or {})
    req11 = max(0, int(out.get('req_11', 0)))
    req22 = max(0, int(out.get('req_22', 0)))
    req23 = max(0, int(out.get('req_23', 0)))
    req24 = max(0, int(out.get('req_24', 0)))
    req25 = max(0, int(out.get('req_25', 0)))
    shortage = str(out.get('shortage_type', 'none') or 'none')
    abstract_gap = sum(
        max(0, int(v))
        for v in dict(out.get('abstract_gap_by_kind', {}) or {}).values()
    )
    dominant = _dominant_shortage_from_reqs(req11, req22, req23, req24, req25)
    warnings = list(out.get('sanity_warnings', []) or [])
    if int(abstract_gap) > 0:
        shortage = 'abstract_topology_gap'
        out['ok'] = False
        out['abstract_gap_total'] = int(abstract_gap)
    if shortage == 'unsupported_special_topology' and dominant is not None:
        warnings.append(
            f"unsupported_special_topology masked resource shortage: using {dominant} "
            f"from req11={req11}, req22={req22}, req23={req23}, req24={req24}, req25={req25}"
        )
        out['resource_shortage_hint'] = str(dominant)
    elif shortage in {'11_shortage', '22_shortage', '23_shortage', '24_shortage', '25_shortage'} and dominant is not None and str(shortage) != str(dominant):
        warnings.append(
            f"shortage_type={shortage} inconsistent with req11={req11}, req22={req22}, "
            f"req23={req23}, req24={req24}, req25={req25}; overriding to {dominant}"
        )
        shortage = str(dominant)
    elif shortage in {'none', 'general_shortage', 'error'} and dominant is not None:
        warnings.append(
            f"shortage_type={shortage} lacked resource direction; using {dominant} "
            f"from req11={req11}, req22={req22}, req23={req23}, req24={req24}, req25={req25}"
        )
        shortage = str(dominant)
    out['req_11'] = int(req11)
    out['req_22'] = int(req22)
    out['req_23'] = int(req23)
    out['req_24'] = int(req24)
    out['req_25'] = int(req25)
    out['shortage_type'] = str(shortage)
    out['sanity_warnings'] = list(warnings)
    return out


def _sanitize_extra_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(diag or {})
    req11 = max(0, int(out.get('required_extra_11', 0)))
    req22 = max(0, int(out.get('required_extra_22', 0)))
    req23 = max(0, int(out.get('required_extra_23', 0)))
    req24 = max(0, int(out.get('required_extra_24', 0)))
    req25 = max(0, int(out.get('required_extra_25', 0)))
    warnings = list(out.get('sanity_warnings', []) or [])
    reason = str(out.get('reason', '') or '')
    abstract_gap_by_kind = dict(out.get('abstract_gap_by_kind', {}) or {})
    gap11 = max(0, int(abstract_gap_by_kind.get('11', 0)))
    gap22 = max(0, int(abstract_gap_by_kind.get('22', 0)))
    gap23 = max(0, int(abstract_gap_by_kind.get('23', 0)))
    gap24 = max(0, int(abstract_gap_by_kind.get('24', 0)))
    gap25 = max(0, int(abstract_gap_by_kind.get('25', 0)))
    gap_total = sum(max(0, int(v)) for v in abstract_gap_by_kind.values())
    if reason.startswith('final_allocation_error:') or reason.startswith('alloc_eval_error:') or 'underflow' in reason:
        parsed = FlexAllocator._parse_underflow_requirements(reason)
        req11 = max(int(req11), int(parsed.get('req_11', 0)))
        req22 = max(int(req22), int(parsed.get('req_22', 0)))
        req23 = max(int(req23), int(parsed.get('req_23', 0)))
        if any(int(parsed.get(k, 0)) > 0 for k in ('req_11', 'req_22', 'req_23')):
            warnings.append(
                f"parsed underflow from allocation error: req11={parsed.get('req_11', 0)} req22={parsed.get('req_22', 0)} req23={parsed.get('req_23', 0)}"
            )
    if reason.startswith('alloc_eval_error:'):
        out['alloc_eval_failed'] = True
        out['flex_count_unknown'] = True
    resource_shortage = dict(out.get('resource_shortage', {}) or {})
    req24 = max(int(req24), int(resource_shortage.get('24', 0)))
    req25 = max(int(req25), int(resource_shortage.get('25', 0)))
    before_merge = (int(req11), int(req22), int(req23), int(req24), int(req25))
    req11 = max(int(req11), int(gap11))
    req22 = max(int(req22), int(gap22))
    req23 = max(int(req23), int(gap23))
    req24 = max(int(req24), int(gap24))
    req25 = max(int(req25), int(gap25))
    if (int(req11), int(req22), int(req23), int(req24), int(req25)) != before_merge:
        warnings.append(
            "merged abstract topology gap into required_extra: "
            f"gap11={gap11} gap22={gap22} gap23={gap23} gap24={gap24} gap25={gap25}"
        )
    out['abstract_gap_total'] = int(gap_total)
    out['required_extra_11'] = int(req11)
    out['required_extra_22'] = int(req22)
    out['required_extra_23'] = int(req23)
    out['required_extra_24'] = int(req24)
    out['required_extra_25'] = int(req25)
    out['sanity_warnings'] = list(warnings)
    return out


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
    H_base = H.cpu().clone()
    H_work = H_base.clone()
    adjuster.E_target = E_target
    tmp_nodes = copy.deepcopy(nodes)
    adjuster._refresh_node_counters(tmp_nodes)
    unit_step = bool(kwargs.get('unit_step', kwargs.get('single_step', False)))
    units_per_step = max(1, int(kwargs.get('units_per_step', kwargs.get('block_c_units_per_step', 1)) or 1))
    apply_recommendation_count = bool(
        kwargs.get(
            'apply_recommendation_count',
            kwargs.get('block_c_apply_recommendation_count', False),
        )
    )
    if bool(unit_step):
        max_steps = 1

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
    conversion_log: List[Dict[str, int]] = []
    last_recommendation_apply_meta: Dict[str, Any] = {}

    def _node_su_counts(nodes_local: List[_NodeV3]) -> Dict[int, int]:
        return dict(Counter(int(getattr(n, 'su_type', -1)) for n in list(nodes_local or [])))

    def _node_su_count_delta(before: Dict[int, int], after: Dict[int, int]) -> Dict[int, int]:
        keys = sorted(set(int(k) for k in dict(before or {}).keys()) | set(int(k) for k in dict(after or {}).keys()))
        return {
            int(k): int(dict(after or {}).get(int(k), 0)) - int(dict(before or {}).get(int(k), 0))
            for k in keys
            if int(dict(after or {}).get(int(k), 0)) - int(dict(before or {}).get(int(k), 0)) != 0
        }

    def _remaining_deficit_from_diag(diag: Dict[str, Any]) -> Dict[str, int]:
        return {
            '11': max(0, int((diag or {}).get('req_11', (diag or {}).get('required_extra_11', 0)) or 0)),
            '22': max(0, int((diag or {}).get('req_22', (diag or {}).get('required_extra_22', 0)) or 0)),
            '23': max(0, int((diag or {}).get('req_23', (diag or {}).get('required_extra_23', 0)) or 0)),
            '24': max(0, int((diag or {}).get('req_24', (diag or {}).get('required_extra_24', 0)) or 0)),
            '25': max(0, int((diag or {}).get('req_25', (diag or {}).get('required_extra_25', 0)) or 0)),
        }

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
        meta = dict(getattr(ch, 'metadata', {}) or {})
        actual_vals = list(meta.get('actual_topology_su') or meta.get('actual_used_su') or [])
        actual = '-'.join(str(int(x)) for x in actual_vals) if actual_vals else ''
        ctype = getattr(ch, 'chain_type', '?')
        origin = getattr(ch, 'origin_type', '?')
        src = getattr(ch, 'source_ids', [])
        actual_txt = f" actual={actual}" if actual else ""
        return f"{ctype}/{origin}: {comp}{actual_txt} src={src}"

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

    def _refresh_after_su_conversion() -> None:
        adjuster._refresh_node_counters(tmp_nodes)

    def _detach_node_from_current_topology(node: _NodeV3, old_su: int) -> None:
        """Drop stale edges before changing a node's structural-unit type."""
        try:
            node_id = int(getattr(node, 'global_id', -1))
        except Exception:
            node_id = -1
        for nb_id in list(getattr(node, 'hop1_ids', []) or []):
            try:
                nb_i = int(nb_id)
            except Exception:
                continue
            if nb_i < 0 or nb_i >= len(tmp_nodes):
                continue
            nb = tmp_nodes[nb_i]
            try:
                nb.hop1_ids = [int(x) for x in list(getattr(nb, 'hop1_ids', []) or []) if int(x) != int(node_id)]
            except Exception:
                nb.hop1_ids = []
            try:
                nb.fixed_hop1_ids = set(
                    int(x)
                    for x in set(getattr(nb, 'fixed_hop1_ids', set()) or set())
                    if int(x) != int(node_id)
                )
            except Exception:
                nb.fixed_hop1_ids = set()
            try:
                if int(old_su) in nb.hop1_su and int(nb.hop1_su[int(old_su)]) > 0:
                    nb.hop1_su[int(old_su)] -= 1
                    if int(nb.hop1_su[int(old_su)]) <= 0:
                        del nb.hop1_su[int(old_su)]
            except Exception:
                nb.hop1_su = Counter()
            try:
                nb.hop2_su = Counter()
            except Exception:
                pass
        try:
            node.hop1_ids = []
            node.fixed_hop1_ids = set()
            node.hop1_su = Counter()
            node.hop2_su = Counter()
            node.constraint_violations = set()
        except Exception:
            pass

    def _convert_node(node: _NodeV3, dst_su: int) -> None:
        src_su = int(node.su_type)
        _detach_node_from_current_topology(node, old_su=int(src_su))
        H_work[src_su] -= 1
        node.su_type = int(dst_su)
        H_work[int(dst_su)] += 1
        conversion_log.append({
            'global_id': int(getattr(node, 'global_id', -1)),
            'from': int(src_su),
            'to': int(dst_su),
        })
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
        _refresh_after_su_conversion()

    def _sync_hist_from_tmp_nodes() -> None:
        nonlocal H_work
        synced = torch.zeros_like(H_work)
        for n in list(tmp_nodes or []):
            try:
                su_i = int(getattr(n, 'su_type', -1))
            except Exception:
                continue
            if 0 <= int(su_i) < int(synced.numel()):
                synced[int(su_i)] += 1
        H_work = synced.detach().clone()

    aromatic_to_11_cursor = 0
    topology_covered_cache_key: Optional[Tuple[Tuple[int, int, Tuple[int, ...]], ...]] = None
    topology_covered_cache: set[int] = set()

    def _current_topology_covered_ids() -> set[int]:
        nonlocal topology_covered_cache_key, topology_covered_cache
        sig = _block_c_node_signature(tmp_nodes)
        if topology_covered_cache_key == sig:
            return set(topology_covered_cache)
        covered: set[int] = set()
        try:
            with redirect_stdout(io.StringIO()):
                result = FlexAllocator(nodes=copy.deepcopy(list(tmp_nodes or []))).allocate()
            for chain_group in (
                getattr(result, 'bridge_chains', []) or [],
                getattr(result, 'side_chains', []) or [],
                getattr(result, 'branch_chains', []) or [],
                getattr(result, 'pending_closure_topologies', []) or [],
            ):
                for chain in list(chain_group or []):
                    for gid in list(getattr(chain, 'source_ids', []) or []):
                        try:
                            if int(gid) >= 0:
                                covered.add(int(gid))
                        except Exception:
                            continue
        except Exception:
            covered = set()
        topology_covered_cache_key = sig
        topology_covered_cache = set(covered)
        return set(covered)

    def _pick_node(src: int) -> Optional[_NodeV3]:
        covered = _current_topology_covered_ids()
        picked = [n for n in tmp_nodes if int(n.su_type) == int(src)]
        picked.sort(key=lambda n: (
            1 if int(getattr(n, 'global_id', -1)) in covered else 0,
            int(getattr(n, 'global_id', 0)),
        ))
        return picked[0] if picked else None

    def _pick_aromatic_to_11_source() -> Optional[int]:
        nonlocal aromatic_to_11_cursor
        cycle = [12, 12, 13]
        for offset in range(len(cycle)):
            idx = (int(aromatic_to_11_cursor) + int(offset)) % len(cycle)
            src = int(cycle[idx])
            if _pick_node(src) is not None:
                aromatic_to_11_cursor = (idx + 1) % len(cycle)
                return int(src)
        return None

    def _apply_aromatic_to_11(max_count: int) -> int:
        applied = 0
        for _ in range(max(0, int(max_count))):
            src = _pick_aromatic_to_11_source()
            if src is None:
                break
            node = _pick_node(int(src))
            if node is None:
                break
            _convert_node(node, 11)
            applied += 1
        return int(applied)

    def _apply_path_once(path: List[int]) -> bool:
        if len(path) < 2:
            return False
        start = int(path[0])
        end = int(path[-1])
        ordinary = {22, 23, 24, 25}
        if start not in ordinary and end in ordinary and not _can_grow_aliphatic(1):
            return False
        node = _pick_node(start)
        if node is None:
            return False
        for dst in list(path[1:]):
            _convert_node(node, int(dst))
        _sync_hist_from_tmp_nodes()
        return True

    def _apply_path(path: List[int], max_count: int) -> int:
        applied = 0
        for _ in range(max(0, int(max_count))):
            if not _apply_path_once([int(x) for x in path]):
                break
            applied += 1
        return int(applied)

    def _apply_aromatic_chain(tail_path: List[int], max_count: int) -> int:
        applied = 0
        ordinary = {22, 23, 24, 25}
        final_su = int(tail_path[-1]) if tail_path else 11
        for _ in range(max(0, int(max_count))):
            if final_su in ordinary and not _can_grow_aliphatic(1):
                break
            src = _pick_aromatic_to_11_source()
            if src is None:
                break
            node = _pick_node(int(src))
            if node is None:
                break
            _convert_node(node, 11)
            for dst in list(tail_path):
                _convert_node(node, int(dst))
            _sync_hist_from_tmp_nodes()
            applied += 1
        return int(applied)

    def _apply_chain_op(op_name: str, count: int) -> Tuple[str, int]:
        count_i = max(0, int(count))
        if count_i <= 0:
            return '', 0

        def _apply_first(label: str, plans: List[Tuple[str, Any]]) -> Tuple[str, int]:
            left = int(count_i)
            applied_total = 0
            parts: List[str] = []
            for part_label, fn in list(plans):
                if left <= 0:
                    break
                applied = int(fn(left))
                if applied <= 0:
                    continue
                applied_total += applied
                left -= applied
                parts.append(f"{part_label}x{applied}")
            if applied_total <= 0:
                return '', 0
            return f"{label}({'+'.join(parts)})", int(applied_total)

        aromatic_chain_map = {
            '12/13->11': [],
            '12/13->11->23': [23],
            '12/13->11->23->24': [23, 24],
            '12/13->11->23->22': [23, 22],
        }
        if op_name in aromatic_chain_map:
            tail = list(aromatic_chain_map[str(op_name)])
            applied = _apply_aromatic_to_11(count_i) if not tail else _apply_aromatic_chain(tail, count_i)
            return (f"{op_name}x{applied}" if applied > 0 else '', applied)

        path_map = {
            '12->11': [12, 11],
            '13->11': [13, 11],
            '22->23': [22, 23],
            '24->23': [24, 23],
            '11->23': [11, 23],
            '23->24': [23, 24],
            '25->24': [25, 24],
            '23->22': [23, 22],
            '23->11': [23, 11],
            '22->23->24': [22, 23, 24],
            '11->23->24': [11, 23, 24],
            '24->23->22': [24, 23, 22],
            '11->23->22': [11, 23, 22],
            '22->23->11': [22, 23, 11],
            '24->23->11': [24, 23, 11],
        }
        if op_name in path_map:
            applied = _apply_path(path_map[str(op_name)], count_i)
            return (f"{op_name}x{applied}" if applied > 0 else '', applied)

        if op_name == 'flex_short_make_F1':
            applied_11 = _apply_aromatic_to_11(count_i)
            _, applied_23 = _apply_first('flex_short_internal_23', [
                ('22->23', lambda n: _apply_path([22, 23], n)),
                ('24->23', lambda n: _apply_path([24, 23], n)),
                ('11->23', lambda n: _apply_path([11, 23], n)),
                ('12/13->11->23', lambda n: _apply_aromatic_chain([23], n)),
            ])
            applied = int(applied_11 + applied_23)
            return (f"flex_short_F1_11x{applied_11}_23x{applied_23}" if applied > 0 else '', applied)
        if op_name == 'ordinary_pending_F1_resources':
            applied_11 = _apply_aromatic_to_11(count_i)
            _, applied_23 = _apply_first('ordinary_F1_internal_23', [
                ('22->23', lambda n: _apply_path([22, 23], n)),
                ('24->23', lambda n: _apply_path([24, 23], n)),
                ('11->23', lambda n: _apply_path([11, 23], n)),
                ('12/13->11->23', lambda n: _apply_aromatic_chain([23], n)),
            ])
            applied = int(applied_11 + applied_23)
            return (f"ordinary_F1_11x{applied_11}_23x{applied_23}" if applied > 0 else '', applied)
        if op_name == 'ordinary_pending_F2_resources':
            _, applied_23 = _apply_first('ordinary_F2_internal_23', [
                ('22->23', lambda n: _apply_path([22, 23], n)),
                ('24->23', lambda n: _apply_path([24, 23], n)),
                ('11->23', lambda n: _apply_path([11, 23], n)),
                ('12/13->11->23', lambda n: _apply_aromatic_chain([23], n)),
            ])
            _, applied_22 = _apply_first('ordinary_F2_side_22', [
                ('23->22', lambda n: _apply_path([23, 22], n)),
                ('24->23->22', lambda n: _apply_path([24, 23, 22], n)),
            ])
            applied = int(applied_23 + applied_22)
            return (f"ordinary_F2_23x{applied_23}_22x{applied_22}" if applied > 0 else '', applied)
        if op_name == 'like24_unclosed_rebalance':
            return _apply_first('like24_rebalance', [
                ('23->24', lambda n: _apply_path([23, 24], n)),
                ('25->24', lambda n: _apply_path([25, 24], n)),
                ('22->23->24', lambda n: _apply_path([22, 23, 24], n)),
                ('11->23->24', lambda n: _apply_path([11, 23, 24], n)),
                ('12/13->11->23->24', lambda n: _apply_aromatic_chain([23, 24], n)),
            ])
        if op_name == 'reduce_flex_prefer_side':
            return _apply_first('reduce_flex_prefer_side', [
                ('23->22', lambda n: _apply_path([23, 22], n)),
                ('24->23->22', lambda n: _apply_path([24, 23, 22], n)),
            ])
        return '', 0

    def _apply_planned_conversions(
        planned_conversions: List[Dict[str, Any]],
        allowed_phases: Optional[set[str]] = None,
    ) -> str:
        nonlocal last_recommendation_apply_meta
        last_recommendation_apply_meta = {}
        allowed = set(str(x) for x in (allowed_phases or {"closure", "residual", "closure_repair", "residual_cleanup"}))
        step_cap = max(1, int(units_per_step))
        applied_parts: List[str] = []
        planned_total = 0
        applied_total = 0
        for row in list(planned_conversions or []):
            rec = dict(row or {})
            phase = str(rec.get('phase', 'closure') or 'closure')
            if phase not in allowed:
                continue
            op_name = str(rec.get('op', '') or '')
            if not op_name:
                continue
            count_i = max(1, int(rec.get('count', 1) or 1))
            planned = count_i if bool(apply_recommendation_count) else min(count_i, max(1, int(step_cap - applied_total)))
            if int(planned) <= 0:
                break
            op_text, applied = _apply_chain_op(op_name, int(planned))
            planned_total += int(planned)
            applied_total += int(applied)
            if applied > 0:
                applied_parts.append(str(op_text))
            if int(applied_total) >= int(step_cap) and not bool(apply_recommendation_count):
                break
        if int(applied_total) <= 0:
            return ''
        last_recommendation_apply_meta = {
            'source_op': 'solver_planned_conversions',
            'recommendation_count': int(planned_total),
            'planned_count': int(planned_total),
            'applied_count': int(applied_total),
            'unit_op': '+'.join(applied_parts),
            'phase': 'solver_plan',
            'reason': 'topology_solver_planned_conversions',
            'units_per_step': int(step_cap),
            'apply_recommendation_count': bool(apply_recommendation_count),
        }
        return f"S0_solver_plan_{'+'.join(applied_parts)}"

    def _apply_topology_recommendations(
        recommendations: List[Dict[str, Any]],
        allowed_phases: Optional[set[str]] = None,
        unit_step_mode: bool = False,
        branch_res: Optional[Dict[str, Any]] = None,
    ) -> str:
        nonlocal last_recommendation_apply_meta
        last_recommendation_apply_meta = {}
        allowed = set(str(x) for x in (allowed_phases or {"closure_repair", "flex_repair"}))
        step_cap = max(1, int(units_per_step))

        def _planned_count(rec_count: int) -> int:
            rec_count_i = max(1, int(rec_count or 1))
            if bool(apply_recommendation_count):
                return int(rec_count_i)
            return max(1, min(int(rec_count_i), int(step_cap)))

        def _remember_apply(
            source_op: str,
            planned_count: int,
            applied_count: int,
            recommendation_count: int,
            unit_op: str = "",
            phase: str = "",
            reason: str = "",
        ) -> None:
            nonlocal last_recommendation_apply_meta
            last_recommendation_apply_meta = {
                'source_op': str(source_op),
                'recommendation_count': int(max(0, recommendation_count)),
                'planned_count': int(max(0, planned_count)),
                'applied_count': int(max(0, applied_count)),
                'unit_op': str(unit_op),
                'phase': str(phase or ''),
                'reason': str(reason or ''),
                'units_per_step': int(step_cap),
                'apply_recommendation_count': bool(apply_recommendation_count),
            }

        if bool(unit_step_mode):
            counts = Counter(int(getattr(n, 'su_type', -1)) for n in list(tmp_nodes or []))
            diag = dict(branch_res or {})
            fallback_shortage = str(diag.get('shortage_type', 'none') or 'none')
            req_11 = max(0, int(diag.get('req_11', 0) or 0))
            req_22 = max(0, int(diag.get('req_22', 0) or 0))
            req_23 = max(0, int(diag.get('req_23', 0) or 0))
            req_24 = max(0, int(diag.get('req_24', 0) or 0))
            req_25 = max(0, int(diag.get('req_25', 0) or 0))
            reqs = {
                '11': int(req_11),
                '22': int(req_22),
                '23': int(req_23),
                '24': int(req_24),
                '25': int(req_25),
            }
            fallback_remaining = {
                str(k): max(0, int(v))
                for k, v in dict(diag.get('remaining', {}) or {}).items()
                if str(k) in {'11', '22', '23', '24', '25'}
            }
            global_unit_op = _pick_unit_branch_conversion_op(
                fallback_shortage,
                req_11=req_11,
                req_22=req_22,
                req_23=req_23,
                req_24=req_24,
                req_25=req_25,
                counts=counts,
                remaining=fallback_remaining,
                preserve_24=bool(tail_bias.get('prefer_preserve_24', False)),
            )
            if global_unit_op:
                planned = int(step_cap)
                op_text, applied = _apply_chain_op(global_unit_op, planned)
                if applied > 0:
                    _remember_apply(
                        source_op='global_unit_shortage',
                        planned_count=planned,
                        applied_count=applied,
                        recommendation_count=planned,
                        unit_op=global_unit_op,
                        phase='closure_repair',
                        reason=str(fallback_shortage),
                    )
                    return f"S0_topology_unit_{op_text}"
            for rec in list(recommendations or []):
                if not bool(rec.get('actionable', True)):
                    continue
                phase = str(rec.get('phase', 'closure_repair') or 'closure_repair')
                if phase not in allowed:
                    continue
                op_name = str(rec.get('op', '') or '')
                rec_remaining = {
                    str(k): max(0, int(v))
                    for k, v in dict(rec.get('resource_surplus', fallback_remaining) or {}).items()
                    if str(k) in {'11', '22', '23', '24', '25'}
                }
                unit_op = _first_unit_op_from_recommendation(
                    rec,
                    reqs=reqs,
                    remaining=rec_remaining or fallback_remaining,
                )
                if not unit_op:
                    rec_shortage = str(rec.get('shortage_type', rec.get('shortage_key', fallback_shortage)) or fallback_shortage)
                    if rec_shortage in {'11', '22', '23', '24', '25'}:
                        rec_shortage = f"{rec_shortage}_shortage"
                    unit_op = _pick_unit_branch_conversion_op(
                        rec_shortage,
                        req_11=req_11,
                        req_22=req_22,
                        req_23=req_23,
                        req_24=req_24,
                        req_25=req_25,
                        counts=counts,
                        remaining=rec_remaining or fallback_remaining,
                        preserve_24=bool(tail_bias.get('prefer_preserve_24', False)),
                    )
                rec_shortage = str(rec.get('shortage_type', fallback_shortage) or fallback_shortage)
                if not rec_shortage or rec_shortage == 'none':
                    rec_shortage = fallback_shortage
                if not unit_op and op_name:
                    unit_op = _pick_unit_branch_conversion_op(
                        fallback_shortage,
                        req_11=req_11,
                        req_22=req_22,
                        req_23=req_23,
                        req_24=req_24,
                        req_25=req_25,
                        counts=counts,
                        remaining=rec_remaining or fallback_remaining,
                        preserve_24=bool(tail_bias.get('prefer_preserve_24', False)),
                    )
                if not unit_op:
                    continue
                rec_count = max(1, int(rec.get('count', 1) or 1))
                planned = _planned_count(rec_count)
                op_text, applied = _apply_chain_op(unit_op, planned)
                if applied > 0:
                    _remember_apply(
                        source_op=op_name or str(rec.get('shortage_type', fallback_shortage) or fallback_shortage),
                        planned_count=planned,
                        applied_count=applied,
                        recommendation_count=rec_count,
                        unit_op=unit_op,
                        phase=phase,
                        reason=str(rec.get('reason', '') or ''),
                    )
                    return f"S0_topology_unit_{op_text}"
            return ''
        for rec in list(recommendations or []):
            phase = str(rec.get('phase', 'closure_repair') or 'closure_repair')
            if phase not in allowed:
                continue
            op_name = str(rec.get('op', '') or '')
            rec_count = max(1, int(rec.get('count', 1) or 1))
            count = _planned_count(rec_count)
            op_text, applied = _apply_chain_op(op_name, count)
            if applied > 0:
                _remember_apply(
                    source_op=op_name,
                    planned_count=count,
                    applied_count=applied,
                    recommendation_count=rec_count,
                    unit_op=op_name,
                    phase=phase,
                    reason=str(rec.get('reason', '') or ''),
                )
                return f"S0_topology_{op_text}"
        return ''

    def _evaluate_branch_state(nodes_local: List[_NodeV3]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        eval_nodes = copy.deepcopy(nodes_local)
        alloc = FlexAllocator(nodes=eval_nodes)
        branch_res = _sanitize_branch_shortage_diag(
            alloc.evaluate_su24_branches(eval_nodes, quiet=True)
        )
        proxy_res = _block_c_eval(adjuster, copy.deepcopy(nodes_local), S_target, E_target)
        return branch_res, proxy_res

    res_24, proxy_diag_24 = _evaluate_branch_state(tmp_nodes)
    initial_branch_penalty = _branch_phase_penalty(res_24, proxy_diag_24)
    best_branch_penalty = tuple(initial_branch_penalty)
    best_H_work = H_work.detach().clone().cpu()
    best_tmp_nodes = copy.deepcopy(tmp_nodes)
    best_moves: List[Dict[str, Any]] = []
    best_res_24 = dict(res_24)
    best_proxy_diag = dict(proxy_diag_24)
    best_rot_idx = int(rot_idx)
    best_iter = 0

    allocator = FlexAllocator(nodes=tmp_nodes)

    print(f"\n  [Step 0] 完整分支调度评估")
    s1_iter = 0

    while s1_iter < max_steps:
        s1_iter += 1
        print(f"    [Step 0] 评估轮次 {s1_iter}/{max_steps}")
        res_24, proxy_diag_24 = _evaluate_branch_state(tmp_nodes)
        current_penalty = _branch_phase_penalty(res_24, proxy_diag_24)
        if tuple(current_penalty) < tuple(best_branch_penalty):
            best_branch_penalty = tuple(current_penalty)
            best_H_work = H_work.detach().clone().cpu()
            best_tmp_nodes = copy.deepcopy(tmp_nodes)
            best_moves = copy.deepcopy(moves)
            best_res_24 = dict(res_24)
            best_proxy_diag = dict(proxy_diag_24)
            best_rot_idx = int(rot_idx)
            best_iter = int(s1_iter - 1)
        for warn in list(res_24.get('sanity_warnings', []) or []):
            print(f"    [Step 0诊断修正] {warn}")
        print(
            f"    [Step 0诊断] shortage={res_24.get('shortage_type', 'none')} "
            f"req22={res_24.get('req_22', 0)} req11={res_24.get('req_11', 0)} "
            f"req23={res_24.get('req_23', 0)} req24={res_24.get('req_24', 0)} req25={res_24.get('req_25', 0)} "
            f"traceGap={int(res_24.get('abstract_gap_total', 0))} "
            f"auditGap={int(res_24.get('abstract_gap_audit_total', 0))} "
            f"su11miss={proxy_diag_24.get('su11_missing_external', 0)} "
            f"specialGap={proxy_diag_24.get('special_degree_gap', 0)}"
        )
        _print_alloc_snapshot('Step 0', res_24)

        if res_24['ok']:
            print("    [Step 0] 分支调度通过")
            break

        shortage = str(res_24.get('shortage_type', 'none'))
        req_22 = max(0, int(res_24.get('req_22', 0)))
        req_11 = max(0, int(res_24.get('req_11', 0)))
        req_23 = max(0, int(res_24.get('req_23', 0)))
        req_24 = max(0, int(res_24.get('req_24', 0)))
        req_25 = max(0, int(res_24.get('req_25', 0)))
        op = ''
        conversion_mark = int(len(conversion_log))
        before_apply_counts = _node_su_counts(tmp_nodes)
        op = _apply_planned_conversions(
            list(res_24.get('planned_conversions', []) or []),
            allowed_phases={"closure", "residual", "closure_repair", "residual_cleanup"},
        )
        if not op:
            op = _apply_topology_recommendations(
                list(res_24.get('recommended_adjustments', []) or []),
                allowed_phases={"closure_repair", "flex_repair"},
                unit_step_mode=bool(unit_step),
                branch_res=res_24,
            )

        if not op:
            if shortage == 'unsupported_special_topology' and req_23 > 0:
                shortage = '23_shortage'
            elif shortage == 'unsupported_special_topology' and req_24 > 0:
                shortage = '24_shortage'
            elif shortage == 'unsupported_special_topology' and req_22 > 0:
                shortage = '22_shortage'
            elif shortage == 'unsupported_special_topology' and req_11 > 0:
                shortage = '11_shortage'
            elif shortage == 'unsupported_special_topology' and req_25 > 0:
                shortage = '25_shortage'
            elif shortage == 'abstract_topology_gap':
                dominant = _dominant_shortage_from_reqs(req_11, req_22, req_23, req_24, req_25)
                if dominant is not None:
                    shortage = str(dominant)
                else:
                    print(
                        "    [Step 0] 抽象拓扑缺口不能通过H计数直接修复: "
                        f"gap={int(res_24.get('abstract_gap_total', 0))} "
                        f"by_kind={dict(res_24.get('abstract_gap_by_kind', {}) or {})}"
                    )
                    break

            safe_unit_op = _pick_unit_branch_conversion_op(
                shortage,
                req_11=req_11,
                req_22=req_22,
                req_23=req_23,
                req_24=req_24,
                req_25=req_25,
                counts=Counter(int(getattr(n, 'su_type', -1)) for n in list(tmp_nodes or [])),
                remaining=dict(res_24.get('remaining', {}) or {}),
                preserve_24=bool(tail_bias.get('prefer_preserve_24', False)),
            )
            if safe_unit_op:
                planned = max(1, int(units_per_step))
                op_text, applied = _apply_chain_op(safe_unit_op, planned)
                if applied > 0:
                    op = f'S0_{op_text}'
                    last_recommendation_apply_meta = {
                        'source_op': str(shortage),
                        'recommendation_count': int(planned),
                        'planned_count': int(planned),
                        'applied_count': int(applied),
                        'unit_op': str(safe_unit_op),
                        'phase': 'closure_repair',
                        'reason': 'fallback_safe_unit_op',
                        'units_per_step': int(max(1, int(units_per_step))),
                        'apply_recommendation_count': bool(apply_recommendation_count),
                    }

            elif shortage == '25_shortage' or req_25 > 0:
                print("    [Step 0] 25号短缺需要特定四价局部motif，跳过普通资源池互转")

        if not op:
            print(f"    [Step 0] 无法处理短缺: {shortage}")
            break

        _log_move(op, 'S0')
        after_apply_counts = _node_su_counts(tmp_nodes)
        moves[-1].update(dict(last_recommendation_apply_meta or {}))
        if 'planned_count' not in moves[-1]:
            moves[-1]['planned_count'] = int(max(1, int(units_per_step)))
        if 'applied_count' not in moves[-1]:
            moves[-1]['applied_count'] = int(max(0, len(conversion_log) - int(conversion_mark)))
        moves[-1]['delta_counts'] = _node_su_count_delta(before_apply_counts, after_apply_counts)
        moves[-1]['node_conversions'] = [
            dict(x) for x in list(conversion_log[conversion_mark:])
        ]
        moves[-1]['diagnostic_before'] = {
            'shortage_type': str(shortage),
            'req_22': int(res_24.get('req_22', 0)),
            'req_11': int(res_24.get('req_11', 0)),
            'req_23': int(res_24.get('req_23', 0)),
            'req_24': int(res_24.get('req_24', 0)),
            'req_25': int(res_24.get('req_25', 0)),
            'pending_closure_count': int(res_24.get('pending_closure_count', 0)),
            'pending_closure_total_count': int(
                res_24.get('pending_closure_total_count', res_24.get('pending_closure_count', 0))
            ),
        }
        print("      [Step 0] 开始H修正与重评估准备")
        if bool(unit_step):
            h_ops = []
        else:
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
        after_res_24, after_proxy_diag = _evaluate_branch_state(tmp_nodes)
        after_penalty = _branch_phase_penalty(after_res_24, after_proxy_diag)
        moves[-1]['penalty_before'] = tuple(int(x) for x in current_penalty)
        moves[-1]['penalty_after'] = tuple(int(x) for x in after_penalty)
        moves[-1]['remaining_deficit_after_step'] = _remaining_deficit_from_diag(after_res_24)
        if tuple(after_penalty) < tuple(best_branch_penalty):
            best_branch_penalty = tuple(after_penalty)
            best_H_work = H_work.detach().clone().cpu()
            best_tmp_nodes = copy.deepcopy(tmp_nodes)
            best_moves = copy.deepcopy(moves)
            best_res_24 = dict(after_res_24)
            best_proxy_diag = dict(after_proxy_diag)
            best_rot_idx = int(rot_idx)
            best_iter = int(s1_iter)

        if not res_24['ok']:
            print(
                f"  [Skeleton-Alloc] 分支仍未全部分配: "
                f"unallocated_branch={res_24.get('unallocated_branch', 0)} "
                f"req22={res_24.get('req_22', 0)} req11={res_24.get('req_11', 0)} "
                f"req23={res_24.get('req_23', 0)} req24={res_24.get('req_24', 0)} req25={res_24.get('req_25', 0)}"
            )
    res_24, final_proxy_diag = _evaluate_branch_state(tmp_nodes)
    final_branch_penalty = _branch_phase_penalty(res_24, final_proxy_diag)
    if tuple(final_branch_penalty) < tuple(best_branch_penalty):
        best_branch_penalty = tuple(final_branch_penalty)
        best_H_work = H_work.detach().clone().cpu()
        best_tmp_nodes = copy.deepcopy(tmp_nodes)
        best_moves = copy.deepcopy(moves)
        best_res_24 = dict(res_24)
        best_proxy_diag = dict(final_proxy_diag)
        best_rot_idx = int(rot_idx)
        best_iter = int(s1_iter)
    partial_progress = bool(
        bool(best_moves)
        and tuple(best_branch_penalty) < tuple(initial_branch_penalty)
    )
    if bool(partial_progress) and not bool(best_res_24.get('ok', False)):
        tmp_nodes = copy.deepcopy(best_tmp_nodes)
        H_work = best_H_work.detach().clone().cpu()
        moves = copy.deepcopy(best_moves)
        res_24 = dict(best_res_24)
        final_proxy_diag = dict(best_proxy_diag)
        rot_idx = int(best_rot_idx)
        print(
            "  [Skeleton-Alloc] 本轮未完全闭合，但保留最佳局部改善: "
            f"iter={int(best_iter)} penalty {tuple(int(x) for x in initial_branch_penalty)}"
            f" -> {tuple(int(x) for x in best_branch_penalty)}"
        )
    hist_reasons = [
        str(x)
        for x in list(final_proxy_diag.get('hist_proxy_reasons', []) or [])
        if str(x)
    ]
    if not hist_reasons and not bool(final_proxy_diag.get('hist_proxy_ok', True)):
        hist_reasons = [str(final_proxy_diag.get('hist_proxy_reason', final_proxy_diag.get('reason', 'unknown')))]
    hist_proxy_deferred = bool(
        bool(res_24['ok'])
        and not bool(final_proxy_diag.get('hist_proxy_ok', True))
        and bool(hist_reasons)
        and all(str(reason).startswith('ether_fixed_pool_mismatch') for reason in hist_reasons)
    )
    branch_ok = bool(
        bool(res_24['ok'])
        and (
            bool(final_proxy_diag.get('hist_proxy_ok', True))
            or bool(hist_proxy_deferred)
        )
    )
    branch_fail_parts: List[str] = []
    if not bool(res_24['ok']):
        gap = int(res_24.get('abstract_gap_total', 0))
        if gap > 0:
            branch_fail_parts.append(
                "abstract_topology_gap"
                f"(gap={gap},by_kind={dict(res_24.get('abstract_gap_by_kind', {}) or {})})"
            )
        branch_fail_parts.append(
            "allocation"
                f"(unallocated={int(res_24.get('unallocated_branch', 0))},"
                f"req22={int(res_24.get('req_22', 0))},"
                f"req11={int(res_24.get('req_11', 0))},"
                f"req23={int(res_24.get('req_23', 0))},"
                f"req24={int(res_24.get('req_24', 0))},"
                f"req25={int(res_24.get('req_25', 0))})"
            )
    if bool(hist_proxy_deferred):
        print(
            "  [Skeleton-Alloc] 分支候选的 hist proxy 仅剩 Ether 固定池差异，"
            "保留拓扑闭合写回，固定池约束留给后续 Layer1/Ether 阶段校验: "
            f"reason={';'.join(hist_reasons)}"
        )
    elif not bool(final_proxy_diag.get('hist_proxy_ok', True)):
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
    commit_branch_state = bool(branch_ok or partial_progress or (bool(unit_step) and bool(moves)))
    if bool(commit_branch_state):
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
        if not bool(branch_ok):
            fail_txt = "; ".join(branch_fail_parts) if branch_fail_parts else "not_fully_closed"
            if bool(unit_step):
                print(f"  [Skeleton-Alloc] 写回单步候选 seed topology，等待 Layer1/2 重建评估: {fail_txt}")
            else:
                print(f"  [Skeleton-Alloc] 写回局部改善 seed topology，下一轮继续闭合: {fail_txt}")
    else:
        fail_txt = "; ".join(branch_fail_parts) if branch_fail_parts else "unknown"
        print(f"  [Skeleton-Alloc] 分支候选未通过分支资源/直方图约束，跳过写回 seed topology: {fail_txt}")
    print(f"  [Skeleton-Alloc] 最终H偏差: {_h_ratio(H_work)*100:.2f}%")
    final_scenario = 'ok' if bool(branch_ok) else 'branch_not_ok'
    if not bool(branch_ok) and branch_fail_parts:
        final_scenario = f"{final_scenario}: {'; '.join(branch_fail_parts)}"
    result_meta = {
        'n_moves': len(moves),
        'ok': bool(branch_ok),
        'final_h_ratio': float(_h_ratio(H_work)),
        'records': moves,
        'final_diag': final_proxy_diag,
        'fail_reasons': list(branch_fail_parts),
        'hist_proxy_deferred': bool(hist_proxy_deferred),
        'hist_proxy_deferred_reasons': list(hist_reasons) if bool(hist_proxy_deferred) else [],
        'partial_progress': bool(partial_progress and not branch_ok),
        'unit_step': bool(unit_step),
        'units_per_step': int(units_per_step),
        'apply_recommendation_count': bool(apply_recommendation_count),
        'candidate_pending_layer12': bool(bool(unit_step) and bool(moves) and not bool(branch_ok)),
        'initial_penalty': tuple(int(x) for x in initial_branch_penalty),
        'final_penalty': tuple(int(x) for x in _branch_phase_penalty(res_24, final_proxy_diag)),
        'best_penalty': tuple(int(x) for x in best_branch_penalty),
        'best_iter': int(best_iter),
        'phase': 'branch',
        'block_c_phase': 'branch_topology',
        'final_scenario': str(final_scenario),
    }
    if not bool(commit_branch_state):
        # A failed branch candidate has only changed the temporary seed H; the
        # seed topology/fixed special metadata were deliberately not committed.
        # Returning that H would desynchronize Layer1's fixed-owner demand from
        # the actual node metadata and can trigger Ether Fixed Assign failures.
        result_meta['rejected_records'] = list(moves)
        result_meta['n_rejected_moves'] = int(len(moves))
        result_meta['returned_original_h'] = True
        return H_base, [], result_meta
    return H_work, moves, result_meta


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
    def _safe_bounds() -> Dict[str, int]:
        try:
            return _block_c_bounds(adjuster, nodes)
        except Exception:
            return {
                'X': 0,
                'Y': 0,
                'Z': 0,
                'flex_lower_raw': 0,
                'flex_upper_raw': 0,
            }

    try:
        diag = adjuster._evaluate_full_allocation_balance(
            nodes,
            flex_ratio=0.80,
            flex_lower_extra=1,
            S_target=S_target,
            E_target=E_target,
        )
    except Exception as e:
        bounds_fallback = _safe_bounds()
        try:
            aliphatic_bounds = adjuster._estimate_aliphatic_region_bounds(S_target, E_target)
        except Exception:
            aliphatic_bounds = {}
        aliphatic_total = int(sum(1 for n in list(nodes or []) if 19 <= int(getattr(n, 'su_type', -1)) <= 25))
        oxygenated_aliphatic_total = int(sum(1 for n in list(nodes or []) if int(getattr(n, 'su_type', -1)) in {19, 20, 21}))
        ordinary_aliphatic_total = int(sum(1 for n in list(nodes or []) if int(getattr(n, 'su_type', -1)) in {22, 23, 24, 25}))
        underflow_req = FlexAllocator._parse_underflow_requirements(str(e))
        diag = {
            'ok': False,
            'reason': f'alloc_eval_error:{e}',
            'warnings': ['allocation_eval_failed'],
            'cluster_count': int(bounds_fallback.get('X', 0)),
            'effective_cluster_count': max(1, int(bounds_fallback.get('X', 0))),
            'rigid_pairs': 0,
            'rigid_cluster_count': int(bounds_fallback.get('Z', 0)),
            'flexible_bridge_count': int(bounds_fallback.get('flex_lower_raw', 0)),
            'flexible_bridge_min': int(bounds_fallback.get('flex_lower_raw', 0)),
            'flexible_bridge_limit': int(bounds_fallback.get('flex_upper_raw', 0)),
            'fixed_flexible_bridge_count': 0,
            'extra_flexible_bridge_count': 0,
            'side_to_22_count': 0,
            'aliphatic_total': int(aliphatic_total),
            'aliphatic_min_total': int(aliphatic_bounds.get('total_min', 0)),
            'aliphatic_max_total': int(aliphatic_bounds.get('total_max', 10**9)),
            'ordinary_aliphatic_total': int(ordinary_aliphatic_total),
            'ordinary_aliphatic_min_total': int(aliphatic_bounds.get('ordinary_min', 0)),
            'ordinary_aliphatic_max_total': int(aliphatic_bounds.get('ordinary_max', 10**9)),
            'oxygenated_aliphatic_total': int(oxygenated_aliphatic_total),
            'oxygenated_aliphatic_min_total': int(aliphatic_bounds.get('oxygenated_min', 0)),
            'oxygenated_aliphatic_max_total': int(aliphatic_bounds.get('oxygenated_max', 10**9)),
            'unallocated_bridge': 0,
            'unallocated_branch': 0,
            'required_extra_11': int(underflow_req.get('req_11', 0)),
            'required_extra_22': int(underflow_req.get('req_22', 0)),
            'required_extra_23': int(underflow_req.get('req_23', 0)),
            'required_extra_24': 0,
            'required_extra_25': 0,
            'remaining': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'native_remaining': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'proxy_remaining': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'native_total': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'proxy_total': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'native_consumed': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'proxy_consumed': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'alloc_eval_error': str(e),
            'alloc_eval_failed': True,
            'flex_count_unknown': True,
        }
    bounds = _safe_bounds()
    diag = dict(diag)
    diag.update(bounds)
    topology_diag = dict(diag.get('topology_assembly_diagnostics', {}) or {})
    diag['topology_assembly_ok'] = bool(diag.get('topology_assembly_ok', topology_diag.get('ok', False)))
    diag['topology_plan_feasible'] = bool(diag.get('topology_plan_feasible', topology_diag.get('topology_plan_feasible', False)))
    diag['topology_materialized_ok'] = bool(diag.get('topology_materialized_ok', topology_diag.get('topology_materialized_ok', topology_diag.get('ok', False))))
    diag['materialization_required'] = bool(diag.get('materialization_required', topology_diag.get('materialization_required', False)))
    diag['planned_conversions'] = list(diag.get('planned_conversions', topology_diag.get('planned_conversions', []) or []) or [])
    diag['planned_conversion_count'] = int(diag.get('planned_conversion_count', topology_diag.get('planned_conversion_count', 0)) or 0)
    diag['residual_growth_actions'] = list(diag.get('residual_growth_actions', topology_diag.get('residual_growth_actions', []) or []) or [])
    diag['residual_chain_actions'] = list(diag.get('residual_chain_actions', topology_diag.get('residual_chain_actions', []) or []) or [])
    diag['residual_remaining'] = dict(diag.get('residual_remaining', topology_diag.get('residual_remaining', {}) or {}) or {})
    diag['chain_23_limit_violations'] = list(diag.get('chain_23_limit_violations', topology_diag.get('chain_23_limit_violations', []) or []) or [])
    diag['closed_topology_count'] = int(diag.get('closed_topology_count', topology_diag.get('closed_topology_count', 0)))
    diag['unclosed_like24_count'] = int(diag.get('unclosed_like24_count', topology_diag.get('unclosed_like24_count', 0)))
    diag['resource_shortage'] = dict(topology_diag.get('resource_shortage', diag.get('resource_shortage', {}) or {}) or {})
    diag['resource_surplus'] = dict(topology_diag.get('resource_surplus', diag.get('resource_surplus', {}) or {}) or {})
    diag['recommended_adjustments'] = list(
        diag.get('recommended_adjustments', topology_diag.get('recommended_adjustments', []) or []) or []
    )
    pending_blocking = int(diag.get('pending_closure_count', 0))
    pending_total = int(diag.get('pending_closure_total_count', pending_blocking))
    pending_deferred = int(diag.get('pending_closure_deferred_count', max(0, pending_total - pending_blocking)))
    diag['pending_closure_count'] = int(pending_blocking)
    diag['pending_closure_blocking_count'] = int(diag.get('pending_closure_blocking_count', pending_blocking))
    diag['pending_closure_total_count'] = int(pending_total)
    diag['pending_closure_deferred_count'] = int(pending_deferred)
    diag['backbone_residual'] = int(diag.get('backbone_residual', topology_diag.get('backbone_residual', 0)))
    diag['backbone_residual_ok'] = bool(diag.get('backbone_residual_ok', topology_diag.get('backbone_residual_ok', True)))
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
    shortage_local = dict(diag.get('resource_shortage', {}) or {})
    diag['required_extra_11'] = max(0, int(diag.get('required_extra_11', 0)), int(shortage_local.get('11', 0)))
    diag['required_extra_22'] = max(0, int(diag.get('required_extra_22', 0)), int(shortage_local.get('22', 0)))
    diag['required_extra_23'] = max(0, int(diag.get('required_extra_23', 0)), int(shortage_local.get('23', 0)))
    diag['required_extra_24'] = max(0, int(diag.get('required_extra_24', 0)), int(shortage_local.get('24', 0)))
    diag['required_extra_25'] = max(0, int(diag.get('required_extra_25', 0)), int(shortage_local.get('25', 0)))
    diag['bounds_ok'] = bool(int(diag['flex_lower']) <= int(diag['flex_upper']))
    diag['flex_short'] = max(0, int(diag['flex_lower']) - int(diag['flex_count']))
    diag['flex_excess'] = max(0, int(diag['flex_count']) - int(diag['flex_upper']))
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
    alloc_failed = bool(diag.get('alloc_eval_failed', False))
    req11 = max(0, int(diag.get('required_extra_11', 0)))
    req22 = max(0, int(diag.get('required_extra_22', 0)))
    req23 = max(0, int(diag.get('required_extra_23', 0)))
    req24 = max(0, int(diag.get('required_extra_24', 0)))
    req25 = max(0, int(diag.get('required_extra_25', 0)))
    abstract_gap = sum(
        max(0, int(v))
        for v in dict(diag.get('abstract_gap_by_kind', {}) or {}).values()
    )
    materialization_required = bool(diag.get('materialization_required', False))
    planned_conversion_count = max(0, int(diag.get('planned_conversion_count', 0) or 0))
    chain_23_limit_gap = int(len(list(diag.get('chain_23_limit_violations', []) or [])))
    residual_remaining = dict(diag.get('residual_remaining', {}) or {})
    residual_unplaced = int(sum(max(0, int(residual_remaining.get(str(k), 0))) for k in ('11', '22', '23', '24', '25')))
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
        1 if bool(alloc_failed) else 0,
        1 if bool(materialization_required) else 0,
        int(planned_conversion_count),
        int(chain_23_limit_gap),
        int(residual_unplaced),
        max(0, int(diag.get('unallocated_branch', 0))),
        max(0, int(diag.get('unallocated_bridge', 0))),
        int(abstract_gap),
        0 if bool(diag.get('hist_proxy_ok', True)) else 1,
        int(req11 + req22 + req23 + req24 + req25),
        int(req11),
        int(req22),
        int(req23),
        int(req24),
        int(req25),
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


def _branch_phase_penalty(res: Dict[str, Any], proxy_diag: Optional[Dict[str, Any]] = None) -> Tuple[int, ...]:
    res_local = dict(res or {})
    proxy_local = dict(proxy_diag or {})
    topology_diag = dict(res_local.get('topology_assembly_diagnostics', {}) or {})
    shortage = {
        '11': max(0, int(res_local.get('req_11', 0))),
        '22': max(0, int(res_local.get('req_22', 0))),
        '23': max(0, int(res_local.get('req_23', 0))),
        '24': max(0, int(res_local.get('req_24', 0))),
        '25': max(0, int(res_local.get('req_25', 0))),
    }
    shortage_total = sum(int(v) for v in shortage.values())
    pending_total = int(
        res_local.get(
            'pending_closure_total_count',
            res_local.get('pending_closure_count', 0),
        )
        or 0
    )
    abstract_gap = int(
        res_local.get(
            'abstract_gap_total',
            sum(max(0, int(v)) for v in dict(res_local.get('abstract_gap_by_kind', {}) or {}).values()),
        )
        or 0
    )
    flex_short = max(0, int(topology_diag.get('flex_short', proxy_local.get('flex_short', 0)) or 0))
    flex_excess = max(0, int(topology_diag.get('flex_excess', proxy_local.get('flex_excess', 0)) or 0))
    topology_ok = bool(res_local.get('topology_assembly_ok', topology_diag.get('ok', False)))
    materialization_required = bool(res_local.get('materialization_required', topology_diag.get('materialization_required', False)))
    planned_conversion_count = max(0, int(res_local.get('planned_conversion_count', topology_diag.get('planned_conversion_count', 0)) or 0))
    chain_23_limit_gap = int(len(list(res_local.get('chain_23_limit_violations', topology_diag.get('chain_23_limit_violations', [])) or [])))
    residual_remaining = dict(res_local.get('residual_remaining', topology_diag.get('residual_remaining', {}) or {}) or {})
    residual_unplaced = int(sum(max(0, int(residual_remaining.get(str(k), 0))) for k in ('11', '22', '23', '24', '25')))
    hist_ok = bool(proxy_local.get('hist_proxy_ok', True))
    return (
        0 if bool(res_local.get('ok', False)) else 1,
        1 if bool(materialization_required) else 0,
        int(planned_conversion_count),
        int(chain_23_limit_gap),
        int(residual_unplaced),
        max(0, int(pending_total)),
        max(0, int(res_local.get('unallocated_branch', 0))),
        max(0, int(abstract_gap)),
        int(shortage_total),
        int(shortage['23']),
        int(shortage['24']),
        int(shortage['22']),
        int(shortage['11']),
        int(shortage['25']),
        int(flex_short + flex_excess),
        int(flex_excess),
        int(flex_short),
        0 if bool(topology_ok) else 1,
        0 if bool(hist_ok) else 1,
    )


def _histogram_from_block_c_nodes(nodes: List[_NodeV3]) -> torch.Tensor:
    H_nodes = torch.zeros(33, dtype=torch.long)
    for node in list(nodes or []):
        try:
            su_i = int(getattr(node, 'su_type', -1))
        except Exception:
            su_i = -1
        if 0 <= int(su_i) < int(H_nodes.numel()):
            H_nodes[int(su_i)] += 1
    return H_nodes


def _branch_diag_from_nodes(adjuster: Any, nodes: List[_NodeV3]) -> Dict[str, Any]:
    eval_nodes = copy.deepcopy(list(nodes or []))
    alloc = FlexAllocator(nodes=eval_nodes)
    return _sanitize_branch_shortage_diag(
        alloc.evaluate_su24_branches(eval_nodes, quiet=True)
    )


def _branch_penalty_from_nodes(
    adjuster: Any,
    nodes: List[_NodeV3],
    S_target: Optional[torch.Tensor],
    E_target: Optional[torch.Tensor],
) -> Tuple[Tuple[int, ...], Dict[str, Any], Dict[str, Any]]:
    branch_diag = _branch_diag_from_nodes(adjuster, nodes)
    proxy_diag = _block_c_eval(adjuster, copy.deepcopy(list(nodes or [])), S_target, E_target)
    return _branch_phase_penalty(branch_diag, proxy_diag), branch_diag, proxy_diag


def _topology_result_summary(result: Any, nodes: List[_NodeV3]) -> Dict[str, Any]:
    diag = dict(getattr(result, 'topology_assembly_diagnostics', {}) or {})
    shortage = dict(diag.get('resource_shortage', {}) or {})
    alternative_reasons = {
        'candidate_group_already_selected',
        'exclusive_group_conflict',
        'source_id_conflict_with_ordinary_chain',
        'source_id_conflict_with_selected_option',
    }

    def _source_key(values: Any) -> Tuple[int, ...]:
        return tuple(sorted(int(x) for x in list(values or []) if int(x) >= 0))

    reason_lookup: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
    for row in list(getattr(result, 'topology_unclosed_candidates', []) or []):
        if not isinstance(row, dict):
            continue
        group_id = str(row.get('candidate_group_id') or '')
        src_key = _source_key(row.get('source_ids', []) or [])
        reason_lookup[(group_id, src_key)] = dict(row)
        reason_lookup[('', src_key)] = dict(row)

    effective_pending = []
    seen_pending: set[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = set()
    for chain in list(getattr(result, 'pending_closure_topologies', []) or []):
        meta = dict(getattr(chain, 'metadata', {}) or {})
        group_id = str(meta.get('candidate_group_id') or meta.get('source_group_id') or '')
        src_key = _source_key(getattr(chain, 'source_ids', []) or [])
        reason_row = reason_lookup.get((group_id, src_key)) or reason_lookup.get(('', src_key)) or {}
        if str(reason_row.get('reason', '') or '') in alternative_reasons:
            continue
        key = (
            str(getattr(chain, 'chain_type', '')),
            tuple(int(x) for x in list(getattr(chain, 'composition', []) or [])),
            src_key,
        )
        if key in seen_pending:
            continue
        seen_pending.add(key)
        effective_pending.append(chain)

    pending_total = int(len(effective_pending))
    pending_blocking = int(len(effective_pending))
    unallocated_branch = int(
        getattr(
            result,
            'unallocated_branch',
            diag.get('unclosed_like24_count', 0),
        )
        or 0
    )
    flex_short = max(0, int(diag.get('flex_short', 0) or 0))
    flex_excess = max(0, int(diag.get('flex_excess', 0) or 0))
    shortage_by_kind = {
        str(key): max(0, int(shortage.get(str(key), 0) or 0))
        for key in ('11', '22', '23', '24', '25')
    }
    topology_materialized_ok = bool(diag.get('topology_materialized_ok', diag.get('ok', False)))
    materialization_required = bool(diag.get('materialization_required', False))
    planned_conversion_count = max(0, int(diag.get('planned_conversion_count', 0) or 0))
    residual_remaining = dict(diag.get('residual_remaining', {}) or {})
    residual_unplaced = int(sum(max(0, int(residual_remaining.get(str(k), 0))) for k in ('11', '22', '23', '24', '25')))
    chain_23_limit_violations = list(diag.get('chain_23_limit_violations', []) or [])
    ok = bool(
        bool(topology_materialized_ok)
        and not bool(materialization_required)
        and int(pending_total) == 0
        and int(unallocated_branch) == 0
        and all(int(v) == 0 for v in shortage_by_kind.values())
        and int(residual_unplaced) == 0
        and not chain_23_limit_violations
        and int(flex_short) == 0
        and int(flex_excess) == 0
    )
    closed_count = (
        len(getattr(result, 'bridge_chains', []) or [])
        + len(getattr(result, 'side_chains', []) or [])
        + len(getattr(result, 'branch_chains', []) or [])
    )
    covered_ids = set()
    for chain_group in (
        getattr(result, 'bridge_chains', []) or [],
        getattr(result, 'side_chains', []) or [],
        getattr(result, 'branch_chains', []) or [],
        getattr(result, 'pending_closure_topologies', []) or [],
    ):
        for chain in list(chain_group or []):
            for gid in list(getattr(chain, 'source_ids', []) or []):
                try:
                    if int(gid) >= 0:
                        covered_ids.add(int(gid))
                except Exception:
                    continue
    ignored_topology_su = {10, 11, 12, 13, 22, 23, 26, 30}
    unrecognized = 0
    for node in list(nodes or []):
        try:
            gid = int(getattr(node, 'global_id', -1))
            su_i = int(getattr(node, 'su_type', -1))
        except Exception:
            continue
        if su_i in ignored_topology_su:
            continue
        if gid >= 0 and gid not in covered_ids:
            unrecognized += 1
    return {
        'ok': bool(ok),
        'topology_assembly_ok': bool(topology_materialized_ok),
        'topology_plan_feasible': bool(diag.get('topology_plan_feasible', False)),
        'topology_materialized_ok': bool(topology_materialized_ok),
        'materialization_required': bool(materialization_required),
        'planned_conversions': list(diag.get('planned_conversions', []) or []),
        'planned_conversion_count': int(planned_conversion_count),
        'residual_remaining': dict(residual_remaining),
        'chain_23_limit_violations': list(chain_23_limit_violations),
        'closed_topology_count': int(closed_count),
        'pending_closure_count': int(pending_blocking),
        'pending_closure_total_count': int(pending_total),
        'unallocated_branch': int(unallocated_branch),
        'resource_shortage': dict(shortage_by_kind),
        'req_11': int(shortage_by_kind.get('11', 0)),
        'req_22': int(shortage_by_kind.get('22', 0)),
        'req_23': int(shortage_by_kind.get('23', 0)),
        'req_24': int(shortage_by_kind.get('24', 0)),
        'req_25': int(shortage_by_kind.get('25', 0)),
        'flex_count': int(diag.get('flex_count', 0) or 0),
        'flex_lower': int(diag.get('flex_lower', 0) or 0),
        'flex_upper': int(diag.get('flex_upper', 0) or 0),
        'flex_short': int(flex_short),
        'flex_excess': int(flex_excess),
        'unrecognized_topology_node_count': int(unrecognized),
        'recommended_adjustments': list(diag.get('recommended_adjustments', []) or []),
    }


def _evaluate_full_topology_from_nodes(nodes: List[_NodeV3]) -> Dict[str, Any]:
    try:
        with redirect_stdout(io.StringIO()):
            result = FlexAllocator(nodes=copy.deepcopy(list(nodes or []))).allocate()
        return _topology_result_summary(result, list(nodes or []))
    except Exception as exc:
        return {
            'ok': False,
            'topology_assembly_ok': False,
            'error': str(exc),
            'closed_topology_count': 0,
            'pending_closure_count': 10**9,
            'pending_closure_total_count': 10**9,
            'unallocated_branch': 10**9,
            'resource_shortage': {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0},
            'req_11': 0,
            'req_22': 0,
            'req_23': 0,
            'req_24': 0,
            'req_25': 0,
            'flex_short': 0,
            'flex_excess': 0,
            'unrecognized_topology_node_count': 10**9,
        }


def _topology_penalty(summary: Dict[str, Any]) -> Tuple[int, ...]:
    shortage = dict(summary.get('resource_shortage', {}) or {})
    shortage_total = sum(
        max(0, int(shortage.get(str(key), summary.get(f'req_{key}', 0)) or 0))
        for key in ('11', '22', '23', '24', '25')
    )
    return (
        0 if bool(summary.get('ok', False)) else 1,
        1 if bool(summary.get('materialization_required', False)) else 0,
        max(0, int(summary.get('planned_conversion_count', 0) or 0)),
        int(len(list(summary.get('chain_23_limit_violations', []) or []))),
        int(sum(max(0, int(dict(summary.get('residual_remaining', {}) or {}).get(str(k), 0))) for k in ('11', '22', '23', '24', '25'))),
        max(0, int(summary.get('pending_closure_total_count', 0) or 0)),
        max(0, int(summary.get('pending_closure_count', 0) or 0)),
        max(0, int(summary.get('unallocated_branch', 0) or 0)),
        int(shortage_total),
        max(0, int(shortage.get('23', summary.get('req_23', 0)) or 0)),
        max(0, int(shortage.get('24', summary.get('req_24', 0)) or 0)),
        max(0, int(shortage.get('22', summary.get('req_22', 0)) or 0)),
        max(0, int(shortage.get('11', summary.get('req_11', 0)) or 0)),
        max(0, int(shortage.get('25', summary.get('req_25', 0)) or 0)),
        max(0, int(summary.get('flex_short', 0) or 0) + int(summary.get('flex_excess', 0) or 0)),
        max(0, int(summary.get('flex_excess', 0) or 0)),
        max(0, int(summary.get('flex_short', 0) or 0)),
        max(0, int(summary.get('unrecognized_topology_node_count', 0) or 0)),
        -max(0, int(summary.get('closed_topology_count', 0) or 0)),
    )


def _copy_block_c_node_state(dst_nodes: List[_NodeV3], src_nodes: List[_NodeV3]) -> None:
    dst_by_gid = {
        int(getattr(node, 'global_id', idx)): node
        for idx, node in enumerate(list(dst_nodes or []))
    }
    for idx, src in enumerate(list(src_nodes or [])):
        try:
            gid = int(getattr(src, 'global_id', idx))
        except Exception:
            gid = int(idx)
        dst = dst_by_gid.get(int(gid))
        if dst is None and 0 <= int(idx) < len(dst_nodes):
            dst = dst_nodes[int(idx)]
        if dst is None:
            continue
        for attr_name in (
            'su_type',
            'hop1_su',
            'hop2_su',
            'hop1_ids',
            'fixed_hop1_ids',
            'target_hop1_degree',
            'init_target_hop1_degree',
            'special_degree_source',
            'special_anchor_partition',
            'target_fixed_anchor_count',
            'init_target_fixed_anchor_count',
            'special_anchor_mode',
            'z_vec',
            'mu',
            'pi',
            'z_history',
            'constraint_violations',
            'score_components',
            'template_key',
        ):
            try:
                setattr(dst, attr_name, copy.deepcopy(getattr(src, attr_name)))
            except Exception:
                continue


def _block_c_node_signature(nodes: List[_NodeV3]) -> Tuple[Tuple[int, int, Tuple[int, ...]], ...]:
    sig: List[Tuple[int, int, Tuple[int, ...]]] = []
    for idx, node in enumerate(list(nodes or [])):
        try:
            gid = int(getattr(node, 'global_id', idx))
        except Exception:
            gid = int(idx)
        try:
            su_i = int(getattr(node, 'su_type', -1))
        except Exception:
            su_i = -1
        hop1 = tuple(sorted(int(x) for x in list(getattr(node, 'hop1_ids', []) or [])))
        sig.append((int(gid), int(su_i), hop1))
    sig.sort(key=lambda x: int(x[0]))
    return tuple(sig)


def _summarize_branch_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'ok': bool(diag.get('ok', False)),
        'shortage_type': str(diag.get('shortage_type', 'none') or 'none'),
        'pending_closure_count': int(diag.get('pending_closure_count', 0) or 0),
        'pending_closure_total_count': int(
            diag.get('pending_closure_total_count', diag.get('pending_closure_count', 0)) or 0
        ),
        'unallocated_branch': int(diag.get('unallocated_branch', 0) or 0),
        'req_11': int(diag.get('req_11', 0) or 0),
        'req_22': int(diag.get('req_22', 0) or 0),
        'req_23': int(diag.get('req_23', 0) or 0),
        'req_24': int(diag.get('req_24', 0) or 0),
        'req_25': int(diag.get('req_25', 0) or 0),
        'abstract_gap_total': int(diag.get('abstract_gap_total', 0) or 0),
        'abstract_gap_by_kind': dict(diag.get('abstract_gap_by_kind', {}) or {}),
        'remaining': dict(diag.get('remaining', {}) or {}),
    }


def run_block_c_topology_closure_loop_impl(
    adjuster: Any,
    nodes: List[_NodeV3],
    E_target: Optional[torch.Tensor],
    S_target: Optional[torch.Tensor] = None,
    ppm: Optional[np.ndarray] = None,
    diff: Optional[np.ndarray] = None,
    max_steps: int = 8,
    phase: str = 'branch',
    rebuild_callback: Optional[Callable[[List[_NodeV3], torch.Tensor, int, Dict[str, Any]], Tuple[List[_NodeV3], torch.Tensor, Dict[str, Any], float]]] = None,
    keep_unimproved: bool = True,
    stop_on_cycle: bool = True,
    show_log: bool = False,
    return_best_on_exhaustion: bool = True,
    units_per_step: int = 1,
    apply_recommendation_count: bool = False,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """Run the real Block-C topology loop one unit conversion at a time.

    The loop deliberately accepts every produced SU conversion by default.  The
    rebuild callback is responsible for the Layer1/Layer2 rebuild; when no
    callback is supplied the converted seed nodes are evaluated directly by the
    topology allocator.  This keeps the skeleton decision logic in Layer4 while
    letting callers decide how expensive the rebuild should be.
    """
    if nodes is None:
        return torch.zeros(33, dtype=torch.long), [], {
            'ok': False,
            'n_moves': 0,
            'records': [],
            'stop_reason': 'missing_nodes',
            'phase': str(phase or 'branch'),
            'backend': 'layer4_adjuster',
        }

    n_steps = max(0, int(max_steps))
    units_per_step_i = max(1, int(units_per_step or 1))
    apply_recommendation_count_i = bool(apply_recommendation_count)
    phase_name = str(phase or 'branch').strip().lower()
    current_nodes = copy.deepcopy(list(nodes or []))
    adjuster._refresh_node_counters(current_nodes)
    current_H = _histogram_from_block_c_nodes(current_nodes)
    if E_target is None:
        try:
            E_target = torch.matmul(current_H.float(), adjuster.E_SU.cpu())
        except Exception:
            E_target = None

    initial_penalty, initial_branch_diag, initial_proxy_diag = _branch_penalty_from_nodes(
        adjuster,
        current_nodes,
        S_target,
        E_target,
    )
    initial_topology_summary = _evaluate_full_topology_from_nodes(current_nodes)
    initial_topology_penalty = _topology_penalty(initial_topology_summary)
    current_penalty = tuple(initial_topology_penalty)
    current_branch_penalty = tuple(initial_penalty)
    best_penalty = tuple(initial_topology_penalty)
    best_branch_penalty = tuple(initial_penalty)
    best_nodes = copy.deepcopy(current_nodes)
    best_H = current_H.detach().clone()
    best_diag = dict(initial_branch_diag)
    best_topology_summary = dict(initial_topology_summary)
    current_topology_summary = dict(initial_topology_summary)
    all_records: List[Dict[str, Any]] = []
    seen: Dict[Tuple[Tuple[int, int, Tuple[int, ...]], ...], int] = {
        _block_c_node_signature(current_nodes): 0,
    }
    stop_reason = 'max_steps_exhausted'
    final_branch_diag = dict(initial_branch_diag)
    final_proxy_diag = dict(initial_proxy_diag)
    chain_promote_to_24_gid: Optional[int] = None
    returned_best_state = False

    for step_idx in range(int(n_steps)):
        if bool(current_topology_summary.get('ok', False)) or bool(final_branch_diag.get('ok', False)):
            stop_reason = 'closed'
            break

        seed_nodes = copy.deepcopy(current_nodes)
        seed_H = current_H.detach().clone()
        forced_conversion: Optional[Dict[str, Any]] = None
        if chain_promote_to_24_gid is not None:
            try:
                req24_now = max(0, int(final_branch_diag.get('req_24', 0) or 0))
                req25_now = max(0, int(final_branch_diag.get('req_25', 0) or 0))
                has_25_candidate = any(
                    int(getattr(node, 'su_type', -1)) == 25
                    for node in list(seed_nodes or [])
                )
                gid_i = int(chain_promote_to_24_gid)
                if (
                    not (bool(has_25_candidate) and int(req25_now) <= 0) and
                    int(req24_now) > 0 and
                    0 <= int(gid_i) < len(seed_nodes) and
                    int(getattr(seed_nodes[int(gid_i)], 'su_type', -1)) == 23
                ):
                    forced_H = seed_H.detach().clone().cpu()
                    forced_H[23] -= 1
                    forced_H[24] += 1
                    forced_conversion = {
                        'gid': int(gid_i),
                        'from': 23,
                        'to': 24,
                        'H': forced_H.detach().clone(),
                    }
                else:
                    chain_promote_to_24_gid = None
            except Exception:
                chain_promote_to_24_gid = None
        try:
            if forced_conversion is not None:
                gid_i = int(forced_conversion['gid'])
                seed_nodes[int(gid_i)].su_type = 24
                step_H = forced_conversion['H'].detach().clone()
                step_moves = [{
                    'op': 'S0_chain_promote_23->24x1',
                    'stage': 'S0_chain',
                    'node_conversions': [{
                        'global_id': int(gid_i),
                        'from': 23,
                        'to': 24,
                    }],
                }]
                step_meta = {
                    'forced_chain_promotion': True,
                    'promoted_global_id': int(gid_i),
                    'phase': str(phase_name),
                }
                chain_promote_to_24_gid = None
            else:
                with redirect_stdout(io.StringIO()):
                    step_H, step_moves, step_meta = adjuster._adjust_skeleton_by_allocation(
                        seed_H.detach().clone(),
                        E_target,
                        S_target=S_target,
                        ppm=ppm,
                        diff=diff,
                        max_steps=1,
                        nodes=seed_nodes,
                        phase=phase_name,
                        unit_step=True,
                        units_per_step=units_per_step_i,
                        apply_recommendation_count=apply_recommendation_count_i,
                    )
        except Exception as exc:
            stop_reason = f'unit_step_exception:{exc}'
            break

        step_moves = list(step_moves or [])
        step_meta = dict(step_meta or {})
        if not step_moves and bool(torch.equal(step_H.detach().cpu(), current_H.detach().cpu())):
            stop_reason = 'no_unit_step_candidate'
            break

        layer12_meta: Dict[str, Any] = {
            'ok': True,
            'mode': 'topology_only',
            'reason': 'no_rebuild_callback',
        }
        rebuilt_nodes = seed_nodes
        rebuilt_H = _histogram_from_block_c_nodes(rebuilt_nodes)
        rebuilt_diff: Dict[str, Any] = {}
        layer2_r2 = float('nan')
        if callable(rebuild_callback):
            try:
                rebuilt_nodes, rebuilt_H, rebuilt_diff, layer2_r2 = rebuild_callback(
                    copy.deepcopy(seed_nodes),
                    step_H.detach().clone(),
                    int(step_idx + 1),
                    dict(step_meta),
                )
                layer12_meta = {
                    'ok': True,
                    'mode': str(dict(rebuilt_diff or {}).get('mode', 'layer1_layer2')),
                    'r2': float(layer2_r2) if np.isfinite(float(layer2_r2)) else None,
                }
            except Exception as exc:
                # The SU conversion remains the current seed.  This prevents the
                # same conversion being proposed forever, while making the failed
                # rebuild visible in the record.
                rebuilt_nodes = seed_nodes
                rebuilt_H = _histogram_from_block_c_nodes(rebuilt_nodes)
                rebuilt_diff = {'mode': 'layer12_failed', 'error': str(exc)}
                layer2_r2 = float('nan')
                layer12_meta = {
                    'ok': False,
                    'mode': 'layer12_failed',
                    'error': str(exc),
                }
                if bool(show_log):
                    print(f"  [BlockC loop] Layer1/2 rebuild failed at step {step_idx + 1}: {exc}")

        adjuster._refresh_node_counters(rebuilt_nodes)
        after_branch_penalty, after_branch_diag, after_proxy_diag = _branch_penalty_from_nodes(
            adjuster,
            rebuilt_nodes,
            S_target,
            E_target,
        )
        after_topology_summary = _evaluate_full_topology_from_nodes(rebuilt_nodes)
        after_penalty = _topology_penalty(after_topology_summary)
        improved = bool(tuple(after_penalty) < tuple(current_penalty))
        non_worse = bool(tuple(after_penalty) <= tuple(current_penalty))
        closed = bool(after_topology_summary.get('ok', False) or after_branch_diag.get('ok', False))
        layer12_ok = bool(layer12_meta.get('ok', True))
        accept = bool(layer12_ok and (closed or improved or non_worse or keep_unimproved))
        record = {
            'loop_step': int(step_idx + 1),
            'accepted': bool(accept),
            'accepted_policy': (
                'closed' if bool(closed) else (
                    'improved' if bool(improved) else (
                        'non_worse' if bool(non_worse) else 'kept_unimproved'
                    )
                )
            ),
            'layer12': dict(layer12_meta),
            'penalty_before': tuple(int(x) for x in tuple(current_penalty)),
            'penalty_after': tuple(int(x) for x in tuple(after_penalty)),
            'branch_penalty_before': tuple(int(x) for x in tuple(current_branch_penalty)),
            'branch_penalty_after': tuple(int(x) for x in tuple(after_branch_penalty)),
            'diagnostic_before': _summarize_branch_diag(final_branch_diag),
            'diagnostic_after': _summarize_branch_diag(after_branch_diag),
            'topology_before': dict(current_topology_summary),
            'topology_after': dict(after_topology_summary),
            'block_c_step_meta': dict(step_meta),
        }
        if np.isfinite(float(layer2_r2)):
            record['layer2_r2'] = float(layer2_r2)
        for move in step_moves:
            move_rec = dict(move or {})
            move_rec.update(record)
            all_records.append(move_rec)
        if not step_moves:
            all_records.append(dict(record))

        if bool(show_log):
            conversions: List[str] = []
            for move in list(step_moves or []):
                for conv in list(dict(move or {}).get('node_conversions', []) or []):
                    try:
                        conversions.append(
                            f"gid={int(conv.get('global_id', -1))}:"
                            f"{int(conv.get('from', -1))}->{int(conv.get('to', -1))}"
                        )
                    except Exception:
                        continue
            if not conversions and forced_conversion is not None:
                try:
                    conversions.append(
                        f"gid={int(forced_conversion.get('gid', -1))}:"
                        f"{int(forced_conversion.get('from', -1))}->{int(forced_conversion.get('to', -1))}"
                    )
                except Exception:
                    pass
            topo_after = dict(after_topology_summary or {})
            print(
                f"  [BlockC loop] step={step_idx + 1} "
                f"accepted={bool(accept)} policy={record['accepted_policy']} "
                f"move={';'.join(conversions) if conversions else 'none'} "
                f"layer12={layer12_meta.get('mode', '')}:{'ok' if layer12_meta.get('ok', False) else 'fail'} "
                f"topology_penalty {tuple(current_penalty)} -> {tuple(after_penalty)} "
                f"pending={int(topo_after.get('pending_closure_total_count', 0) or 0)} "
                f"unallocated={int(topo_after.get('unallocated_branch', 0) or 0)} "
                f"req23={int(topo_after.get('req_23', 0) or 0)} "
                f"req24={int(topo_after.get('req_24', 0) or 0)} "
                f"req22={int(topo_after.get('req_22', 0) or 0)} "
                f"req11={int(topo_after.get('req_11', 0) or 0)} "
                f"req25={int(topo_after.get('req_25', 0) or 0)}"
            )

        if not bool(accept):
            stop_reason = 'candidate_rejected'
            break

        current_nodes = copy.deepcopy(rebuilt_nodes)
        current_H = rebuilt_H.detach().clone().cpu()
        current_penalty = tuple(after_penalty)
        current_branch_penalty = tuple(after_branch_penalty)
        current_topology_summary = dict(after_topology_summary)
        final_branch_diag = dict(after_branch_diag)
        final_proxy_diag = dict(after_proxy_diag)
        chain_promote_to_24_gid = None
        try:
            req24_after = max(0, int(final_branch_diag.get('req_24', 0) or 0))
            req23_after = max(0, int(final_branch_diag.get('req_23', 0) or 0))
            if int(req24_after) > 0 and int(req24_after) >= int(req23_after):
                for move in list(step_moves or []):
                    for conv in list(dict(move or {}).get('node_conversions', []) or []):
                        if int(conv.get('from', -1)) in {22, 11} and int(conv.get('to', -1)) == 23:
                            chain_promote_to_24_gid = int(conv.get('global_id'))
                            break
                    if chain_promote_to_24_gid is not None:
                        break
        except Exception:
            chain_promote_to_24_gid = None
        if tuple(after_penalty) < tuple(best_penalty):
            best_penalty = tuple(after_penalty)
            best_branch_penalty = tuple(after_branch_penalty)
            best_nodes = copy.deepcopy(current_nodes)
            best_H = current_H.detach().clone()
            best_diag = dict(after_branch_diag)
            best_topology_summary = dict(after_topology_summary)

        if bool(closed):
            stop_reason = 'closed'
            break

        sig = _block_c_node_signature(current_nodes)
        if bool(stop_on_cycle) and sig in seen:
            stop_reason = f'cycle_detected:first_seen_step={int(seen[sig])}'
            break
        seen[sig] = int(step_idx + 1)
    else:
        stop_reason = 'max_steps_exhausted'

    explored_final_penalty = tuple(current_penalty)
    explored_final_topology_summary = dict(current_topology_summary)
    if (
        bool(return_best_on_exhaustion)
        and str(stop_reason) == 'max_steps_exhausted'
        and tuple(best_penalty) < tuple(current_penalty)
    ):
        current_nodes = copy.deepcopy(best_nodes)
        current_H = best_H.detach().clone().cpu()
        current_penalty = tuple(best_penalty)
        current_branch_penalty = tuple(best_branch_penalty)
        current_topology_summary = dict(best_topology_summary)
        _bp, final_branch_diag, final_proxy_diag = _branch_penalty_from_nodes(
            adjuster,
            current_nodes,
            S_target,
            E_target,
        )
        current_branch_penalty = tuple(_bp)
        stop_reason = 'max_steps_exhausted_returned_best_checkpoint'
        returned_best_state = True

    _copy_block_c_node_state(nodes, current_nodes)
    adjuster._refresh_node_counters(nodes)
    final_H = _histogram_from_block_c_nodes(list(nodes or []))
    partial_progress = bool(tuple(best_penalty) < tuple(initial_topology_penalty))
    final_ok = bool(current_topology_summary.get('ok', False) or final_branch_diag.get('ok', False))
    if not bool(final_ok) and bool(partial_progress):
        best_diag = dict(best_diag or {})
    meta = {
        'backend': 'layer4_adjuster',
        'loop_kind': 'block_c_topology_closure',
        'phase': str(phase_name),
        'ok': bool(final_ok),
        'branch_ok': bool(final_ok) if phase_name in {'branch', 'full'} else True,
        'extra_ok': True if phase_name == 'branch' else bool(final_ok),
        'partial_progress': bool(partial_progress and not final_ok),
        'n_moves': int(len(all_records)),
        'records': list(all_records),
        'initial_penalty': tuple(int(x) for x in tuple(initial_topology_penalty)),
        'initial_topology_penalty': tuple(int(x) for x in tuple(initial_topology_penalty)),
        'initial_branch_penalty': tuple(int(x) for x in tuple(initial_penalty)),
        'best_penalty': tuple(int(x) for x in tuple(best_penalty)),
        'best_branch_penalty': tuple(int(x) for x in tuple(best_branch_penalty)),
        'final_penalty': tuple(int(x) for x in tuple(current_penalty)),
        'final_branch_penalty': tuple(int(x) for x in tuple(current_branch_penalty)),
        'initial_diag': _summarize_branch_diag(initial_branch_diag),
        'best_diag': _summarize_branch_diag(best_diag),
        'final_diag': dict(final_proxy_diag),
        'final_branch_diag': dict(final_branch_diag),
        'initial_topology_summary': dict(initial_topology_summary),
        'best_topology_summary': dict(best_topology_summary),
        'final_topology_summary': dict(current_topology_summary),
        'stop_reason': str(stop_reason),
        'requested_unit_steps': int(n_steps),
        'units_per_step': int(units_per_step_i),
        'apply_recommendation_count': bool(apply_recommendation_count_i),
        'keep_unimproved': bool(keep_unimproved),
        'return_best_on_exhaustion': bool(return_best_on_exhaustion),
        'returned_best_state': bool(returned_best_state),
        'explored_final_penalty': tuple(int(x) for x in tuple(explored_final_penalty)),
        'explored_final_topology_summary': dict(explored_final_topology_summary),
        'layer12_rebuild_enabled': bool(callable(rebuild_callback)),
        'final_scenario': (
            'ok'
            if bool(final_ok)
            else (
                "branch_not_ok: "
                f"abstract_topology_gap(gap={int(final_branch_diag.get('abstract_gap_total', 0) or 0)},"
                f"by_kind={dict(final_branch_diag.get('abstract_gap_by_kind', {}) or {})}); "
                f"allocation(unallocated={int(final_branch_diag.get('unallocated_branch', 0) or 0)},"
                f"req22={int(final_branch_diag.get('req_22', 0) or 0)},"
                f"req11={int(final_branch_diag.get('req_11', 0) or 0)},"
                f"req23={int(final_branch_diag.get('req_23', 0) or 0)},"
                f"req24={int(final_branch_diag.get('req_24', 0) or 0)},"
                f"req25={int(final_branch_diag.get('req_25', 0) or 0)})"
            )
        ),
        'branch_meta': {
            'ok': bool(final_ok),
            'n_moves': int(len(all_records)),
            'partial_progress': bool(partial_progress and not final_ok),
            'units_per_step': int(units_per_step_i),
            'apply_recommendation_count': bool(apply_recommendation_count_i),
            'final_scenario': (
                'ok'
                if bool(final_ok)
                else str(final_branch_diag.get('shortage_type', 'branch_not_ok'))
            ),
        },
        'extra_meta': {
            'ok': True,
            'n_moves': 0,
            'reason': 'phase_branch_only' if phase_name == 'branch' else 'not_run',
            'final_scenario': '',
        },
    }
    if best_H is not None:
        meta['best_hist'] = best_H
    return final_H, list(all_records), meta


def _block_c_unexpected_residual(diag: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        max(0, int(diag.get('remaining_22', 0))),
        max(0, int(diag.get('remaining_24', 0))),
        max(0, int(diag.get('remaining_25', 0))),
    )


def _block_c_residual_11_23_done(diag: Dict[str, Any]) -> bool:
    if 'topology_materialized_ok' in dict(diag or {}) or 'materialization_required' in dict(diag or {}):
        residual = dict(diag.get('residual_remaining', {}) or {})
        if not residual:
            residual = {
                '11': int(diag.get('remaining_11', 0)),
                '22': int(diag.get('remaining_22', 0)),
                '23': int(diag.get('remaining_23', 0)),
                '24': int(diag.get('remaining_24', 0)),
                '25': int(diag.get('remaining_25', 0)),
            }
        return bool(
            bool(diag.get('topology_materialized_ok', diag.get('topology_assembly_ok', False)))
            and not bool(diag.get('materialization_required', False))
            and all(max(0, int(residual.get(str(k), 0))) == 0 for k in ('11', '22', '23', '24', '25'))
            and not list(diag.get('chain_23_limit_violations', []) or [])
        )
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
                'required_extra_24': int(before_diag.get('required_extra_24', 0)),
                'required_extra_25': int(before_diag.get('required_extra_25', 0)),
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
            f"req24={before_diag.get('required_extra_24', 0)} "
            f"req25={before_diag.get('required_extra_25', 0)} "
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
        adjuster._refresh_node_counters(nodes_local)
        return True

    aromatic_to_11_cursor_extra = 0

    def _pick_aromatic_to_11_source_extra(nodes_local: List[_NodeV3]) -> Optional[int]:
        nonlocal aromatic_to_11_cursor_extra
        cycle = [12, 12, 13]
        for offset in range(len(cycle)):
            idx = (int(aromatic_to_11_cursor_extra) + int(offset)) % len(cycle)
            src = int(cycle[idx])
            if _count_su(nodes_local, src) > 0:
                aromatic_to_11_cursor_extra = (idx + 1) % len(cycle)
                return int(src)
        return None

    def _apply_aromatic_to_11_extra(max_count: int) -> int:
        applied = 0
        for _ in range(max(0, int(max_count))):
            src = _pick_aromatic_to_11_source_extra(tmp_nodes)
            if src is None:
                break
            if not _convert_n(tmp_nodes, H_work, int(src), 11, 1):
                break
            applied += 1
        return int(applied)

    def _apply_path_extra(path: List[int], max_count: int) -> int:
        applied = 0
        for _ in range(max(0, int(max_count))):
            if len(path) < 2:
                break
            start = int(path[0])
            if _count_su(tmp_nodes, start) <= 0:
                break
            ok = True
            for dst in list(path[1:]):
                if not _convert_n(tmp_nodes, H_work, start, int(dst), 1):
                    ok = False
                    break
                start = int(dst)
            if not ok:
                break
            applied += 1
        return int(applied)

    def _apply_aromatic_chain_extra(tail_path: List[int], max_count: int) -> int:
        applied = 0
        for _ in range(max(0, int(max_count))):
            src = _pick_aromatic_to_11_source_extra(tmp_nodes)
            if src is None:
                break
            if not _convert_n(tmp_nodes, H_work, int(src), 11, 1):
                break
            start = 11
            ok = True
            for dst in list(tail_path):
                if not _convert_n(tmp_nodes, H_work, int(start), int(dst), 1):
                    ok = False
                    break
                start = int(dst)
            if not ok:
                break
            applied += 1
        return int(applied)

    def _apply_chain_op_extra(op_name: str, count: int) -> Tuple[str, int]:
        count_i = max(0, int(count))
        if count_i <= 0:
            return '', 0

        def _apply_first(label: str, plans: List[Tuple[str, Any]]) -> Tuple[str, int]:
            left = int(count_i)
            applied_total = 0
            parts: List[str] = []
            for part_label, fn in list(plans):
                if left <= 0:
                    break
                applied = int(fn(left))
                if applied <= 0:
                    continue
                applied_total += applied
                left -= applied
                parts.append(f"{part_label}x{applied}")
            if applied_total <= 0:
                return '', 0
            return f"{label}({'+'.join(parts)})", int(applied_total)

        aromatic_chain_map = {
            '12/13->11': [],
            '12/13->11->23': [23],
            '12/13->11->23->24': [23, 24],
            '12/13->11->23->22': [23, 22],
        }
        if op_name in aromatic_chain_map:
            tail = list(aromatic_chain_map[str(op_name)])
            applied = _apply_aromatic_to_11_extra(count_i) if not tail else _apply_aromatic_chain_extra(tail, count_i)
            return (f"{op_name}x{applied}" if applied > 0 else '', applied)

        path_map = {
            '12->11': [12, 11],
            '13->11': [13, 11],
            '22->23': [22, 23],
            '24->23': [24, 23],
            '11->23': [11, 23],
            '23->24': [23, 24],
            '25->24': [25, 24],
            '23->22': [23, 22],
            '23->11': [23, 11],
            '22->23->24': [22, 23, 24],
            '11->23->24': [11, 23, 24],
            '24->23->22': [24, 23, 22],
            '11->23->22': [11, 23, 22],
            '22->23->11': [22, 23, 11],
            '24->23->11': [24, 23, 11],
        }
        if op_name in path_map:
            applied = _apply_path_extra(path_map[str(op_name)], count_i)
            return (f"{op_name}x{applied}" if applied > 0 else '', applied)

        if op_name == 'flex_short_make_F1':
            applied_11 = _apply_aromatic_to_11_extra(count_i)
            _, applied_23 = _apply_first('flex_short_internal_23', [
                ('22->23', lambda n: _apply_path_extra([22, 23], n)),
                ('24->23', lambda n: _apply_path_extra([24, 23], n)),
                ('11->23', lambda n: _apply_path_extra([11, 23], n)),
                ('12/13->11->23', lambda n: _apply_aromatic_chain_extra([23], n)),
            ])
            applied = int(applied_11 + applied_23)
            return (f"flex_short_F1_11x{applied_11}_23x{applied_23}" if applied > 0 else '', applied)
        if op_name == 'ordinary_pending_F1_resources':
            applied_11 = _apply_aromatic_to_11_extra(count_i)
            _, applied_23 = _apply_first('ordinary_F1_internal_23', [
                ('22->23', lambda n: _apply_path_extra([22, 23], n)),
                ('24->23', lambda n: _apply_path_extra([24, 23], n)),
                ('11->23', lambda n: _apply_path_extra([11, 23], n)),
                ('12/13->11->23', lambda n: _apply_aromatic_chain_extra([23], n)),
            ])
            applied = int(applied_11 + applied_23)
            return (f"ordinary_F1_11x{applied_11}_23x{applied_23}" if applied > 0 else '', applied)
        if op_name == 'ordinary_pending_F2_resources':
            _, applied_23 = _apply_first('ordinary_F2_internal_23', [
                ('22->23', lambda n: _apply_path_extra([22, 23], n)),
                ('24->23', lambda n: _apply_path_extra([24, 23], n)),
                ('11->23', lambda n: _apply_path_extra([11, 23], n)),
                ('12/13->11->23', lambda n: _apply_aromatic_chain_extra([23], n)),
            ])
            _, applied_22 = _apply_first('ordinary_F2_side_22', [
                ('23->22', lambda n: _apply_path_extra([23, 22], n)),
                ('24->23->22', lambda n: _apply_path_extra([24, 23, 22], n)),
            ])
            applied = int(applied_23 + applied_22)
            return (f"ordinary_F2_23x{applied_23}_22x{applied_22}" if applied > 0 else '', applied)
        if op_name == 'like24_unclosed_rebalance':
            return _apply_first('like24_rebalance', [
                ('23->24', lambda n: _apply_path_extra([23, 24], n)),
                ('25->24', lambda n: _apply_path_extra([25, 24], n)),
                ('22->23->24', lambda n: _apply_path_extra([22, 23, 24], n)),
                ('11->23->24', lambda n: _apply_path_extra([11, 23, 24], n)),
                ('12/13->11->23->24', lambda n: _apply_aromatic_chain_extra([23, 24], n)),
            ])
        if op_name == 'reduce_flex_prefer_side':
            return _apply_first('reduce_flex_prefer_side', [
                ('23->22', lambda n: _apply_path_extra([23, 22], n)),
                ('24->23->22', lambda n: _apply_path_extra([24, 23, 22], n)),
            ])
        return '', 0

    def _apply_topology_recommendations_extra(
        recommendations: List[Dict[str, Any]],
        allowed_phases: set[str],
    ) -> str:
        allowed = set(str(x) for x in list(allowed_phases or []))
        for rec in list(recommendations or []):
            phase = str(rec.get('phase', 'closure_repair') or 'closure_repair')
            if phase not in allowed:
                continue
            op_name = str(rec.get('op', '') or '')
            count = max(1, int(rec.get('count', 1) or 1))
            op_text, applied = _apply_chain_op_extra(op_name, count)
            if applied > 0:
                return f"S2_topology_{op_text}"
        return ''

    def _apply_planned_conversions_extra(
        planned_conversions: List[Dict[str, Any]],
        allowed_phases: set[str],
    ) -> str:
        allowed = set(str(x) for x in list(allowed_phases or []))
        applied_parts: List[str] = []
        for rec in list(planned_conversions or []):
            row = dict(rec or {})
            phase = str(row.get('phase', 'closure') or 'closure')
            if phase not in allowed:
                continue
            op_name = str(row.get('op', '') or '')
            if not op_name:
                continue
            count = max(1, int(row.get('count', 1) or 1))
            op_text, applied = _apply_chain_op_extra(op_name, count)
            if applied > 0:
                applied_parts.append(str(op_text))
        if not applied_parts:
            return ''
        return f"S2_solver_plan_{'+'.join(applied_parts)}"

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
            new_diag = _sanitize_extra_diag(_block_c_eval(adjuster, new_nodes, S_target, E_target))
            if bool(new_diag.get('alloc_eval_failed', False)):
                adjuster._h_rotation_state = int(prev_rot_state)
                continue
            if not bool(new_diag.get('aromatic_balance_ok', True)):
                adjuster._h_rotation_state = int(prev_rot_state)
                continue
            if int(new_diag.get('ordinary_aliphatic_total', 0)) > int(new_diag.get('ordinary_aliphatic_max_total', 10**9)):
                if int(new_diag.get('ordinary_aliphatic_total', 0)) > int(before_diag.get('ordinary_aliphatic_total', 0)):
                    adjuster._h_rotation_state = int(prev_rot_state)
                    continue
            prev_abstract_gap = sum(
                max(0, int(v))
                for v in dict(before_diag.get('abstract_gap_by_kind', {}) or {}).values()
            )
            new_abstract_gap = sum(
                max(0, int(v))
                for v in dict(new_diag.get('abstract_gap_by_kind', {}) or {}).values()
            )
            if int(new_abstract_gap) > int(prev_abstract_gap):
                adjuster._h_rotation_state = int(prev_rot_state)
                continue
            prev_flex_gap = max(
                0,
                int(before_diag.get('flex_lower', 0)) - int(before_diag.get('flex_count', 0)),
            )
            new_flex_gap = max(
                0,
                int(new_diag.get('flex_lower', 0)) - int(new_diag.get('flex_count', 0)),
            )
            if int(prev_flex_gap) > 0 and int(new_flex_gap) > int(prev_flex_gap):
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
        h13_now = max(0, int(H_work[13].item())) if int(H_work.numel()) > 13 else 0
        arom_now = adjuster._evaluate_aromatic_balance_constraints(H_work, S_target=S_target, E_target=E_target)
        su13_min_now = int(arom_now.get('su13_min', 0))
        consumes_13 = sum(int(mult) for src, _dst, mult in list(conversions or []) if int(src) == 13)
        if int(consumes_13) > 0 and int(h13_now - int(consumes_13) * desired_i) < int(su13_min_now):
            desired_i = max(0, (int(h13_now) - int(su13_min_now)) // max(1, int(consumes_13)))
            if int(desired_i) <= 0:
                return
        ordinary_total = int(sum(
            int(H_work[int(su)].item())
            for su in [22, 23, 24, 25]
            if int(su) < int(H_work.numel())
        ))
        ordinary_max = int(adjuster._estimate_aliphatic_region_bounds(S_target, E_target).get('ordinary_max', 10**9))
        if int(ordinary_total) >= int(ordinary_max) and int(_ordinary_delta(conversions, 1)) > 0:
            return
        cands.append((str(op), str(stage), int(desired_i), list(conversions)))

    def _resource_delta(conversions: List[Tuple[int, int, int]]) -> Dict[int, int]:
        out: Dict[int, int] = {11: 0, 22: 0, 23: 0, 24: 0, 25: 0}
        for src, dst, mult in list(conversions or []):
            amount = int(mult)
            if int(src) in out:
                out[int(src)] = int(out.get(int(src), 0)) - int(amount)
            if int(dst) in out:
                out[int(dst)] = int(out.get(int(dst), 0)) + int(amount)
        return out

    def _candidate_allowed_by_shortage(conversions: List[Tuple[int, int, int]],
                                       req11: int,
                                       req22: int,
                                       req23: int,
                                       req24: int = 0,
                                       req25: int = 0) -> bool:
        delta = _resource_delta(conversions)
        if int(req11) > 0 and int(delta.get(11, 0)) < 0:
            return False
        if int(req22) > 0 and int(delta.get(22, 0)) < 0:
            return False
        if int(req23) > 0 and int(delta.get(23, 0)) < 0:
            return False
        if int(req24) > 0 and int(delta.get(24, 0)) < 0:
            return False
        if int(req25) > 0 and int(delta.get(25, 0)) < 0:
            return False
        return True

    def _run_guided_extra_step(diag: Dict[str, Any]) -> bool:
        req11 = max(0, int(diag.get('required_extra_11', 0)))
        req22 = max(0, int(diag.get('required_extra_22', 0)))
        req23 = max(0, int(diag.get('required_extra_23', 0)))
        req24 = max(0, int(diag.get('required_extra_24', 0)))
        req25 = max(0, int(diag.get('required_extra_25', 0)))
        flex_count = int(diag.get('flex_count', 0))
        flex_lower = int(diag.get('flex_lower', 0))
        flex_upper = int(diag.get('flex_upper', 0))
        flex_short = max(0, int(flex_lower - flex_count))
        flex_excess = max(0, int(flex_count - flex_upper))
        closure_done = bool(
            bool(diag.get('topology_assembly_ok', False))
            and int(diag.get('pending_closure_total_count', diag.get('pending_closure_count', 0))) == 0
            and int(req11) == 0
            and int(req22) == 0
            and int(req23) == 0
            and int(req24) == 0
            and int(req25) == 0
            and int(flex_short) == 0
            and int(flex_excess) == 0
        )
        allowed_phases = {"residual"} if bool(closure_done) else {"closure", "residual"}
        topo_op = _apply_planned_conversions_extra(
            list(diag.get('planned_conversions', []) or []),
            allowed_phases=allowed_phases,
        )
        if not topo_op:
            topo_op = _apply_topology_recommendations_extra(
                list(diag.get('recommended_adjustments', []) or []),
                allowed_phases={"residual_cleanup"} if bool(closure_done) else {"closure_repair", "flex_repair"},
            )
        if topo_op:
            _log(str(topo_op), 'S2_topology_recommendation', diag)
            return True
        rem11 = max(0, int(diag.get('remaining_11', 0)))
        rem22 = max(0, int(diag.get('remaining_22', 0)))
        rem23 = max(0, int(diag.get('remaining_23', 0)))
        rem24 = max(0, int(diag.get('remaining_24', 0)))
        rem25 = max(0, int(diag.get('remaining_25', 0)))
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
        h25 = _hc(25)
        cands: List[Tuple[str, str, int, List[Tuple[int, int, int]]]] = []

        if unallocated_branch > 0:
            if int(req11) <= 0 and int(req23) <= 0:
                _add_guided_candidate(
                    cands, 'S2_C3_branch_23+11->24+13', 'S2_guided_C3',
                    min(unallocated_branch, h23, h11, 6),
                    [(23, 24, 1), (11, 13, 1)],
                )
            if int(req23) <= 0:
                _add_guided_candidate(
                    cands, 'S2_C3_branch_2x23->24+22', 'S2_guided_C3',
                    min(unallocated_branch, h23 // 2, 6),
                    [(23, 24, 1), (23, 22, 1)],
                )
            _add_guided_candidate(
                cands, 'S2_C3_req23_22->23', 'S2_guided_C3',
                min(max(req23, 0), h22, 6),
                [(22, 23, 1)],
            )

        if req11 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_req11_23->11', 'S2_guided_C3',
                min(req11, h23, 8),
                [(23, 11, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_req11_22->23->11', 'S2_guided_C3',
                min(req11, h22, 6),
                [(22, 23, 1), (23, 11, 1)],
            )

        if req22 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_req22_23->22', 'S2_guided_C3',
                min(req22, h23, 8),
                [(23, 22, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_req22_24->23->22', 'S2_guided_C3',
                min(req22, h24, 6),
                [(24, 23, 1), (23, 22, 1)],
            )

        if req23 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_req23_22->23', 'S2_guided_C3',
                min(req23, h22, 8),
                [(22, 23, 1)],
            )
            if not bool(preserve_24):
                _add_guided_candidate(
                    cands, 'S2_C3_req23_24->23', 'S2_guided_C3',
                    min(req23 + rem24, h24, 6),
                    [(24, 23, 1)],
                )
            _add_guided_candidate(
                cands, 'S2_C3_req23_11->23', 'S2_guided_C3',
                min(req23, h11, 6),
                [(11, 23, 1)],
            )

        if req24 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_req24_23->24', 'S2_guided_C3',
                min(req24, h23, 8),
                [(23, 24, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_req24_25->24', 'S2_guided_C3',
                min(req24, h25, 8),
                [(25, 24, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_req24_22->23->24', 'S2_guided_C3',
                min(req24, h22, 6),
                [(22, 23, 1), (23, 24, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_C3_req24_11->23->24', 'S2_guided_C3',
                min(req24, h11, 6),
                [(11, 23, 1), (23, 24, 1)],
            )

        if bool(closure_done) and rem24 > 0 and not bool(preserve_24):
            _add_guided_candidate(
                cands, 'S2_C3_residual24_24->23', 'S2_guided_C3',
                min(rem24, h24, 6),
                [(24, 23, 1)],
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

        if flex_short > 0:
            _add_guided_candidate(
                cands, 'S2_C4_flex_short_22->23', 'S2_guided_C4',
                min(flex_short, h22, 8),
                [(22, 23, 1)],
            )
            if not bool(preserve_24):
                _add_guided_candidate(
                    cands, 'S2_C4_flex_short_24+22->2x23', 'S2_guided_C4',
                    min(flex_short, h24, h22, 6),
                    [(24, 23, 1), (22, 23, 1)],
                )

        if bool(closure_done) and rem11 > 0 and rem23 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_residual_11+23->13+22', 'S2_guided_C3',
                min(rem11, rem23, max(1, flex_excess), h11, h23, 8),
                [(11, 13, 1), (23, 22, 1)],
            )
        if rem22 > 0 and req23 > 0:
            _add_guided_candidate(
                cands, 'S2_C3_residual22_22->23', 'S2_guided_C3',
                min(rem22, req23, h22, 8),
                [(22, 23, 1)],
            )
        if bool(closure_done) and rem22 > 0:
            _add_guided_candidate(
                cands, 'S2_residual_22->23', 'S2_guided_residual',
                min(rem22, h22, 6),
                [(22, 23, 1)],
            )
            _add_guided_candidate(
                cands, 'S2_residual_22->23', 'S2_guided_residual',
                min(rem22, h22, 4),
                [(22, 23, 1)],
            )
        if bool(closure_done) and rem24 > 0 and not bool(preserve_24):
            _add_guided_candidate(
                cands, 'S2_residual_24->23', 'S2_guided_residual',
                min(rem24, h24, 4),
                [(24, 23, 1)],
            )
        if bool(closure_done) and rem11 > 0:
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
        if bool(closure_done) and rem23 > 0 and int(aliphatic_total - 1) >= int(aliphatic_min):
            _add_guided_candidate(
                cands, 'S2_residual_23->13', 'S2_guided_residual',
                min(rem23, h23, 6),
                [(23, 13, 1)],
            )

        for op, stage, desired, conversions in cands:
            if not _candidate_allowed_by_shortage(conversions, req11, req22, req23, req24, req25):
                continue
            if _try_guided_bundle(op, stage, int(desired), conversions, diag):
                return True
        return False

    diag = _sanitize_extra_diag(_block_c_eval(adjuster, tmp_nodes, S_target, E_target))
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
        diag = _sanitize_extra_diag(_block_c_eval(adjuster, tmp_nodes, S_target, E_target))
        fixed_flex_count = int(diag.get('fixed_flexible_bridge_count', 0))
        flex_display = '?' if bool(diag.get('flex_count_unknown', False)) else str(diag.get('flex_count', 0))
        fixed_flex_display = '?' if bool(diag.get('flex_count_unknown', False)) else str(fixed_flex_count)
        for warn in list(diag.get('sanity_warnings', []) or []):
            print(f"    [Step 2诊断修正] {warn}")
        if bool(diag.get('alloc_eval_failed', False)):
            print(f"    [Step 2诊断修正] allocation balance failed: {diag.get('alloc_eval_error', diag.get('reason', 'unknown'))}")
        trace_gap_total = int(
            diag.get(
                'abstract_gap_total',
                sum(int(v) for v in dict(diag.get('abstract_gap_by_kind', {}) or {}).values()),
            )
        )
        print(
            f"    [Step 2评估] X={diag.get('X', 0)} Y={diag.get('Y', 0)} Z={diag.get('Z', 0)} "
            f"flex={flex_display}/[{diag.get('flex_lower', 0)},{diag.get('flex_upper', 0)}] "
            f"fixed_flex={fixed_flex_display} "
            f"native11={diag.get('native_remaining_11', 0)} proxy11={diag.get('proxy_remaining_11', 0)} "
            f"native22={diag.get('native_remaining_22', 0)} proxy22={diag.get('proxy_remaining_22', 0)} "
            f"native23={diag.get('native_remaining_23', 0)} proxy23={diag.get('proxy_remaining_23', 0)} "
            f"req11={diag.get('required_extra_11', 0)} req22={diag.get('required_extra_22', 0)} req23={diag.get('required_extra_23', 0)} "
            f"traceGap={trace_gap_total} "
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

    extra_meta = _sanitize_extra_diag(_block_c_eval(adjuster, tmp_nodes, S_target, E_target))
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
            'trace_exact_ok': bool(balance_diag.get('trace_exact_ok', False)),
            'trace_exact_chain_count': int(balance_diag.get('trace_exact_chain_count', 0)),
            'trace_inexact_chain_count': int(balance_diag.get('trace_inexact_chain_count', 0)),
            'abstract_gap_by_kind': dict(balance_diag.get('abstract_gap_by_kind', {}) or {}),
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
            'required_extra_24': int(balance_diag.get('required_extra_24', 0)),
            'required_extra_25': int(balance_diag.get('required_extra_25', 0)),
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
