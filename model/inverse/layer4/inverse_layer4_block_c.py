"""Layer4 Block-C workflow and Block-B tail-spectrum helpers.

This file owns the Block-B tail-spectrum helper and the authoritative Block-C
Stage3 workflow loop. Detailed topology rules still live under
``model.inverse.topology``.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...shared.inverse_common import get_effective_hist_element_vector
from ..topology import Stage3WorkflowCycleRecord, Stage3WorkflowResult

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


def _tail_h_total(adjuster: Any, H: torch.Tensor) -> float:
    return float(_tail_element_vector(adjuster, H)[1].item())


def _tail_element_vector(adjuster: Any, H: torch.Tensor) -> torch.Tensor:
    H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
    special_degree_meta = adjuster._get_special_degree_meta(H_cpu)
    return get_effective_hist_element_vector(
        H_cpu,
        special_degree_meta=special_degree_meta,
        E_SU_tensor=adjuster.E_SU.detach().cpu(),
    )


def _tail_bundle_conservation_reason(adjuster: Any,
                                     H_before: torch.Tensor,
                                     H_after: torch.Tensor) -> str:
    before = torch.clamp(H_before.detach().cpu(), min=0).long()
    after = torch.clamp(H_after.detach().cpu(), min=0).long()
    if bool(torch.any(H_after.detach().cpu() < 0)):
        return 'negative_histogram'
    if int(before.sum().item()) != int(after.sum().item()):
        return 'su_total_not_conserved'
    before_elements = _tail_element_vector(adjuster, before).detach().cpu().float()
    after_elements = _tail_element_vector(adjuster, after).detach().cpu().float()
    copied = min(int(before_elements.numel()), int(after_elements.numel()))
    fixed_indices = [idx for idx in range(copied) if int(idx) != 1]
    if any(abs(float(before_elements[idx].item()) - float(after_elements[idx].item())) > 1e-6 for idx in fixed_indices):
        return 'fixed_elements_not_conserved'
    return ''


TAIL_H_COMPENSATION_CYCLE: Tuple[Tuple[int, int], ...] = (
    (23, 11),
    (23, 11),
    (22, 11),
)


def _tail_h_compensation_at(cursor: int) -> Tuple[int, int]:
    return TAIL_H_COMPENSATION_CYCLE[int(cursor) % len(TAIL_H_COMPENSATION_CYCLE)]


def _print_tail_spectrum_audit(meta: Dict[str, Any], moves: List[Dict[str, Any]]) -> None:
    print("  [Block B/tail_spectrum] 差谱窗口")
    for su_type in (22, 23, 24):
        core = dict(dict(meta.get('windows', {}) or {}).get(f'{su_type}_core', {}) or {})
        wide = dict(dict(meta.get('windows', {}) or {}).get(str(su_type), {}) or {})
        need = float(dict(meta.get('tail_needs', {}) or {}).get(str(su_type), 0.0) or 0.0)
        print(
            f"    SU{su_type}: core[pos={float(core.get('pos', 0.0)):.3f} "
            f"neg={float(core.get('neg', 0.0)):.3f}] "
            f"wide[pos={float(wide.get('pos', 0.0)):.3f} "
            f"neg={float(wide.get('neg', 0.0)):.3f}] need={need:.3f}"
        )
    for move in list(moves or []):
        compensation = dict(move.get('compensation', {}) or {})
        comp_text = str(compensation.get('op', 'none')) if compensation else 'none'
        print(
            f"    bundle primary={move.get('op', '')} compensation={comp_text} "
            f"H={float(move.get('h_before', 0.0)):.3f}"
            f"->{float(move.get('h_after_primary', 0.0)):.3f}"
            f"->{float(move.get('h_after_final', 0.0)):.3f}"
        )
    for rejected in list(meta.get('rejected_bundles', []) or []):
        print(
            f"    rejected primary={rejected.get('primary', '')} "
            f"compensation={rejected.get('compensation', 'none')} "
            f"reason={rejected.get('reason', '')}"
        )


def adjust_block_b_tail_spectrum_impl(
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
    aromatic_couple: bool = True,
    h_tolerance: float = 0.08,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[Block B/tail_spectrum] 脂肪尾部 22/23/24 与 13/23 联合调整")
    max_moves = max(0, int(max_moves))

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
    move_order: List[Tuple[str, Dict[int, int]]] = []
    if float(need24) > float(thr) and float(need23) < -float(thr):
        move_order.append(('B_tail_23to24', {23: -1, 24: +1}))
    if float(need24) > float(thr) and float(need22) < -float(thr):
        move_order.append(('B_tail_22to24', {22: -1, 24: +1}))
    if float(need23) > float(thr) and float(need22) < -float(thr):
        move_order.append(('B_tail_22to23', {22: -1, 23: +1}))
    if float(s23_core['neg']) > float(thr) and float(s12_13['pos']) > float(thr):
        move_order.append(('B_tail_23to13', {23: -1, 13: +1}))
    if float(need22) > float(thr) and float(need23) < -float(thr):
        move_order.append(('B_tail_23to22', {23: -1, 22: +1}))
    if float(s12_13['neg']) > float(thr) and float(s23_wide['pos']) > float(thr):
        move_order.append(('B_tail_13to23', {13: -1, 23: +1}))
    if float(need23) > float(thr) and float(need24) < -float(thr):
        move_order.append(('B_tail_24to23', {24: -1, 23: +1}))
    if float(need22) > float(thr) and float(need24) < -float(thr):
        move_order.append(('B_tail_24to22', {24: -1, 22: +1}))

    if bool(aromatic_couple) and float(mid['pos']) > float(thr) and float(low['neg']) > float(thr):
        coupled = []
        if float(need24) > float(thr) and float(need23) < -float(thr):
            coupled.append(('B_tail_couple_23to24', {23: -1, 24: +1}))
        if float(need24) > float(thr) and float(need22) < -float(thr):
            coupled.append(('B_tail_couple_22to24', {22: -1, 24: +1}))
        if float(need23) > float(thr) and float(need22) < -float(thr):
            coupled.append(('B_tail_couple_22to23', {22: -1, 23: +1}))
        if float(s12_13['neg']) > float(thr) and float(s23_wide['pos']) > float(thr):
            coupled.append(('B_tail_couple_13to23', {13: -1, 23: +1}))
        move_order = coupled + move_order

    H_work = torch.clamp(H, min=0).long().clone()
    all_moves: List[Dict[str, Any]] = []
    rejected_bundles: List[Dict[str, Any]] = []
    keep = {22: int(min_keep_22), 23: int(min_keep_23), 24: int(min_keep_24), 13: 0}
    compensation_cursor = 0
    target_h = None
    if E_target is not None:
        try:
            target_h = float(E_target.detach().cpu().float()[1].item())
        except Exception:
            target_h = None

    for _ in range(max(0, int(max_moves))):
        applied = False
        for name, delta in move_order:
            H_primary = adjuster._apply_count_delta(H_work, delta, min_keep=keep)
            if H_primary is None:
                continue
            h_before = _tail_h_total(adjuster, H_work)
            h_after_primary = _tail_h_total(adjuster, H_primary)
            H_final = H_primary.detach().clone().cpu()
            compensation = None
            h_after_final = float(h_after_primary)
            primary_h_rel = 0.0
            if target_h is not None and target_h > 0:
                primary_h_rel = abs(float(h_after_primary) - float(target_h)) / float(target_h)
            if target_h is not None and target_h > 0 and primary_h_rel > float(h_tolerance):
                if int(delta.get(13, 0)) != -1 or int(delta.get(23, 0)) != 1:
                    rejected_bundles.append({
                        'primary': str(name),
                        'reason': 'tail_primary_h_out_of_tolerance_without_compensation',
                        'h_before': float(h_before),
                        'h_after_primary': float(h_after_primary),
                        'target_h': float(target_h),
                    })
                    continue
                comp_from, comp_to = _tail_h_compensation_at(compensation_cursor)
                comp_keep = dict(keep)
                comp_delta = {int(comp_from): -1, int(comp_to): 1}
                H_comp = adjuster._apply_count_delta(H_primary, comp_delta, min_keep=comp_keep)
                if H_comp is None:
                    rejected_bundles.append({
                        'primary': str(name),
                        'compensation': f'{int(comp_from)}->{int(comp_to)}',
                        'reason': 'tail_h_compensation_donor_unavailable',
                    })
                    continue
                h_after_comp = _tail_h_total(adjuster, H_comp)
                comp_h_rel = abs(float(h_after_comp) - float(target_h)) / float(target_h)
                if not (
                    abs(float(h_after_comp) - float(target_h)) + 1e-9 < abs(float(h_after_primary) - float(target_h))
                    and float(comp_h_rel) <= float(h_tolerance)
                ):
                    rejected_bundles.append({
                        'primary': str(name),
                        'compensation': f'{int(comp_from)}->{int(comp_to)}',
                        'reason': 'tail_h_compensation_not_acceptable',
                        'h_after_primary': float(h_after_primary),
                        'h_after_compensation': float(h_after_comp),
                        'target_h': float(target_h),
                    })
                    continue
                H_final = H_comp.detach().clone().cpu()
                h_after_final = float(h_after_comp)
                compensation = {
                    'op': f'B_tail_h_comp_{int(comp_from)}to{int(comp_to)}',
                    'delta': dict(comp_delta),
                    'from_su': int(comp_from),
                    'to_su': int(comp_to),
                }
                compensation_cursor = (int(compensation_cursor) + 1) % len(TAIL_H_COMPENSATION_CYCLE)
            conservation_reason = _tail_bundle_conservation_reason(adjuster, H_work, H_final)
            if conservation_reason:
                rejected_bundles.append({
                    'primary': str(name),
                    'compensation': str(dict(compensation or {}).get('op', 'none')),
                    'reason': str(conservation_reason),
                })
                if compensation is not None:
                    compensation_cursor = (int(compensation_cursor) - 1) % len(TAIL_H_COMPENSATION_CYCLE)
                continue
            H_work = H_final
            combined_delta = dict(delta)
            if compensation is not None:
                for su_type, value in dict(compensation['delta']).items():
                    combined_delta[int(su_type)] = int(combined_delta.get(int(su_type), 0)) + int(value)
                combined_delta = {int(k): int(v) for k, v in combined_delta.items() if int(v) != 0}
            move_rec = {
                'block': 'B',
                'substage': 'tail_spectrum',
                'op': name,
                'primary_delta': dict(delta),
                'compensation': compensation,
                'delta': combined_delta,
                'h_before': float(h_before),
                'h_after_primary': float(h_after_primary),
                'h_after_final': float(h_after_final),
                'target_h': target_h,
            }
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
        'phase': 'block_b_tail_spectrum',
        'compensation_cycle': ['23->11', '23->11', '22->11'],
        'compensation_cursor_after': int(compensation_cursor),
        'rejected_bundles': rejected_bundles,
        'reason': (
            'ok' if all_moves else
            'tail_h_compensation_unavailable' if any('compensation' in item or 'h_' in str(item.get('reason', '')) for item in rejected_bundles) else
            'no_tail_move'
        ),
    }
    _print_tail_spectrum_audit(meta, all_moves)
    return H_work, all_moves, meta


def _stage3_metrics_from_plan(plan: Any) -> Dict[str, Any]:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    if phase_eval is not None:
        return dict(getattr(phase_eval, "topology_metrics", {}) or {})
    phase_eval_dict = dict(runtime.get("phase_evaluation_dict", {}) or {})
    return dict(phase_eval_dict.get("topology_metrics", {}) or {})


def _stage3_signature_from_plan(plan: Any) -> str:
    runtime = dict(getattr(plan, "runtime", {}) or {})
    phase_eval = runtime.get("phase_evaluation")
    if phase_eval is not None:
        return str(getattr(phase_eval, "topology_signature", "") or "")
    phase_eval_dict = dict(runtime.get("phase_evaluation_dict", {}) or {})
    return str(phase_eval_dict.get("topology_signature", "") or "")


def _stage3_plan_fingerprint(plan: Any) -> Tuple[Any, ...]:
    plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
    concrete_ops: List[Tuple[Any, ...]] = []
    for bundle in list(plan_dict.get("adjustment_bundles", []) or []):
        for op in list(dict(bundle or {}).get("bundle_ops", []) or []):
            op_dict = dict(op or {})
            if op_dict.get("node_id") is None and op_dict.get("delta") is None:
                continue
            concrete_ops.append((
                str(op_dict.get("op", "")),
                op_dict.get("from_su"),
                op_dict.get("to_su"),
                op_dict.get("node_id"),
                tuple(sorted((str(key), int(value)) for key, value in dict(op_dict.get("delta", {}) or {}).items())),
            ))
    return (
        str(plan_dict.get("phase", "")),
        str(plan_dict.get("active_subphase", "")),
        _stage3_signature_from_plan(plan),
        tuple(concrete_ops),
    )


def _histogram_dict(H: torch.Tensor) -> Dict[str, int]:
    values = torch.clamp(H.detach().cpu(), min=0).long().tolist()
    return {str(index): int(value) for index, value in enumerate(values) if int(value) != 0}


def _resource_gap(inventory: Dict[int, int], demand: Dict[int, int]) -> Dict[str, int]:
    keys = set(int(key) for key in inventory) | set(int(key) for key in demand)
    return {
        str(key): int(demand.get(int(key), 0)) - int(inventory.get(int(key), 0))
        for key in sorted(keys)
        if int(demand.get(int(key), 0)) - int(inventory.get(int(key), 0)) != 0
    }


def _print_stage3_histogram_adjustments(cycle_number: int, histogram_plan: Any) -> None:
    groups = list(getattr(histogram_plan, "groups", []) or [])
    if not groups:
        print(f"[Block C:resource #{cycle_number}] no physical histogram adjustment", flush=True)
        return
    print(
        f"[Block C:resource #{cycle_number}] structure-unit adjustments: {len(groups)} primary bundle(s)",
        flush=True,
    )
    for group_index, group in enumerate(groups, start=1):
        primary = getattr(group, "primary_op", None)
        compensation = list(getattr(group, "h_compensation_ops", []) or [])
        primary_text = (
            f"SU{int(getattr(primary, 'from_su', -1))}->"
            + (
                f"SU{int(getattr(primary, 'via_su'))}->"
                if getattr(primary, "via_su", None) is not None
                else ""
            )
            + f"SU{int(getattr(primary, 'to_su', -1))}"
            f" x{int(getattr(primary, 'count', 1) or 1)}"
        )
        compensation_text = ", ".join(
            f"SU{int(getattr(op, 'from_su', -1))}->SU{int(getattr(op, 'to_su', -1))}"
            f" x{int(getattr(op, 'count', 1) or 1)}"
            for op in compensation
        ) or "none"
        print(
            f"  [Adjustment {group_index}] primary={primary_text} "
            f"H_compensation={compensation_text} "
            f"delta={dict(getattr(group, 'combined_delta', {}) or {})} "
            f"H={float(getattr(group, 'h_before', 0.0)):.1f}"
            f"->{float(getattr(group, 'h_after_primary', 0.0)):.1f}"
            f"->{float(getattr(group, 'h_after_final', 0.0)):.1f}",
            flush=True,
        )


def _histogram_h_compensation_count(histogram_plan: Any) -> int:
    return sum(
        len(list(getattr(group, "h_compensation_ops", []) or []))
        for group in list(getattr(histogram_plan, "groups", []) or [])
    )


def _topology_strictly_improved(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[bool, str]:
    if not before or not after:
        return False, "topology_metrics_missing"
    if int(after.get("subphase_rank", 0)) > int(before.get("subphase_rank", 0)):
        return True, "subphase_advanced"
    if int(after.get("pending_count", 0)) < int(before.get("pending_count", 0)):
        return True, "pending_reduced"
    if int(after.get("unresolved_total", 0)) < int(before.get("unresolved_total", 0)):
        return True, "unresolved_shortage_reduced"
    if int(after.get("shortage_total", 0)) < int(before.get("shortage_total", 0)):
        return True, "resource_shortage_reduced"
    if (
        int(after.get("flex_gap", 0)) < int(before.get("flex_gap", 0))
        and int(after.get("unresolved_total", 0)) <= int(before.get("unresolved_total", 0))
    ):
        return True, "flex_gap_reduced"
    if (
        int(after.get("product_count", 0)) > int(before.get("product_count", 0))
        and int(after.get("pending_count", 0)) <= int(before.get("pending_count", 0))
        and int(after.get("shortage_total", 0)) <= int(before.get("shortage_total", 0))
    ):
        return True, "product_count_increased"
    return False, "topology_not_improved"


def _rollback_stage3_plan_transactions(plan: Any, reason: str) -> Tuple[Dict[str, Any], List[int]]:
    phase_result = getattr(plan, "phase_result", None)
    rolled_back_contract_ids: List[int] = []
    for transaction in list(getattr(phase_result, "transactions", []) or []):
        has_rebuild_effect = bool(
            list(getattr(transaction, "converted_node_ids", []) or [])
            or list(getattr(transaction, "conversion_path", []) or [])
            or list(getattr(transaction, "compensation_node_ids", []) or [])
        )
        if not has_rebuild_effect:
            continue
        attempted = {
            "converted_node_ids": list(getattr(transaction, "converted_node_ids", []) or []),
            "conversion_path": list(getattr(transaction, "conversion_path", []) or []),
            "compensation_node_ids": list(getattr(transaction, "compensation_node_ids", []) or []),
            "claimed_resource_ids": dict(getattr(transaction, "claimed_resource_ids", {}) or {}),
            "product_node_ids": list(getattr(transaction, "product_node_ids", []) or []),
            "slot_assignments": list(getattr(transaction, "slot_assignments", []) or []),
            "edge_plan": list(getattr(transaction, "edge_plan", []) or []),
        }
        transaction.committed = False
        transaction.rollback_reason = str(reason)
        transaction.infeasibility_proof = {
            **dict(getattr(transaction, "infeasibility_proof", {}) or {}),
            "reason": str(reason),
            "rolled_back": True,
            "attempted_transaction": attempted,
        }
        transaction.converted_node_ids = []
        transaction.conversion_path = []
        transaction.compensation_node_ids = []
        transaction.claimed_resource_ids = {}
        transaction.product_node_ids = []
        transaction.slot_assignments = []
        transaction.edge_plan = []
        rolled_back_contract_ids.append(int(getattr(transaction, "contract_id", -1)))
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
    diagnostics["phase_result"] = phase_result.to_dict() if hasattr(phase_result, "to_dict") else {}
    diagnostics["rolled_back_contract_ids"] = list(rolled_back_contract_ids)
    diagnostics["rollback_reason"] = str(reason)
    plan.diagnostics = diagnostics
    return plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {}), rolled_back_contract_ids


def _restore_stage3_rotation_cursor(adjuster: Any, cursor_before: Optional[int]) -> None:
    if cursor_before is not None and hasattr(adjuster, "_stage3_su11_rotation_index"):
        adjuster._stage3_su11_rotation_index = int(cursor_before)


def run_block_c_resource_adjustment_loop_impl(
    adjuster: Any,
    H: torch.Tensor,
    nodes: Optional[List[Any]],
    E_target: Optional[torch.Tensor],
    S_target: Optional[torch.Tensor] = None,
    rebuild_fn: Any = None,
    validate_fn: Any = None,
    output_dir: Optional[str] = None,
    max_stage3_iterations: int = 20,
    max_primary_adjustments_per_stage: int = 10,
    **kwargs: Any,
) -> Stage3WorkflowResult:
    primary_limit = max(0, int(max_primary_adjustments_per_stage))
    compensation_limit = int(kwargs.pop("max_h_compensation_ops_per_adjustment", 8))
    chain_23_limit = int(kwargs.pop("chain_23_limit", 6))
    h_tolerance = float(kwargs.pop("h_tolerance", 0.06))
    outer_cycle = int(kwargs.pop("outer_cycle", 0))
    iteration_offset = max(0, int(kwargs.pop("iteration_offset", 0)))
    nodes_current = list(nodes or [])
    H_current = torch.clamp(H.detach().clone(), min=0).long()
    diff_current: Dict[str, Any] = {}
    cycles: List[Stage3WorkflowCycleRecord] = []
    accepted_cycles = 0
    stop_reason = "max_stage3_iterations_reached"
    accepted_state_signatures = set()

    for cycle_idx in range(max(0, int(max_stage3_iterations))):
        cycle_number = int(iteration_offset) + int(cycle_idx) + 1
        rotation_cursor_before = (
            int(getattr(adjuster, "_stage3_su11_rotation_index", 0) or 0)
            if hasattr(adjuster, "_stage3_su11_rotation_index")
            else None
        )
        plan_started = time.perf_counter()
        print(f"[Block C:resource #{cycle_number}] topology plan start", flush=True)
        plan = adjuster.plan_topology_resource_adjustments(
            H=H_current.detach().clone(),
            nodes=list(nodes_current or []),
            S_target=S_target,
            E_target=E_target,
            max_primary_adjustments_per_stage=primary_limit,
            max_h_compensation_ops_per_adjustment=compensation_limit,
            chain_23_limit=chain_23_limit,
            h_tolerance=h_tolerance,
        )
        phase_result = getattr(plan, "phase_result", None)
        for transaction in list(getattr(phase_result, "transactions", []) or []):
            transaction.outer_cycle = int(outer_cycle)
            transaction.iteration = int(cycle_number)
        print(
            f"[Block C:resource #{cycle_number}] topology plan done "
            f"({time.perf_counter() - plan_started:.2f}s)",
            flush=True,
        )
        phase = str(getattr(plan, "phase", "") or dict(getattr(plan, "diagnostics", {}) or {}).get("stage3_phase", ""))
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
        before_metrics = _stage3_metrics_from_plan(plan)
        before_signature = _stage3_signature_from_plan(plan)
        histogram_plan = getattr(plan, "histogram_adjustment_plan", None)
        phase_inventory = dict(getattr(histogram_plan, "inventory_before", {}) or {})
        phase_demand = dict(getattr(histogram_plan, "demand_before", {}) or {})
        histogram_before = _histogram_dict(H_current)
        planned_conversions = list(plan_dict.get('planned_conversions', []) or [])
        adjustment_bundles = list(plan_dict.get('adjustment_bundles', []) or [])
        state_signature = repr((
            tuple(int(value) for value in H_current.detach().cpu().long().tolist()),
            phase,
            str(getattr(plan, "active_subphase", "") or ""),
            before_signature,
            tuple(sorted((int(key), int(value)) for key, value in phase_demand.items())),
            tuple(sorted((int(key), int(value)) for key, value in phase_inventory.items())),
        ))
        phase_eval_dict = dict(dict(getattr(plan, "runtime", {}) or {}).get("phase_evaluation_dict", {}) or {})
        flex_audit = dict(phase_eval_dict.get("flex_audit", {}) or {})
        assembly_view = dict(phase_eval_dict.get("assembly_view", {}) or {})
        flex_target_plan = dict(assembly_view.get("flex_target_plan", {}) or {})
        protected_counts = dict(assembly_view.get("protected_resource_counts", {}) or {})
        print(
            "[Flex Target] "
            f"M={int(flex_audit.get('flex_lower', 0) or 0)} "
            f"N={int(flex_audit.get('flex_upper', 0) or 0)} "
            f"T={int(flex_audit.get('flex_target', flex_audit.get('target_flex_count', 0)) or 0)} "
            f"current={int(dict(phase_eval_dict.get('flex_counts', {}) or {}).get('final', 0) or 0)} "
            f"external_capacity={int(flex_target_plan.get('external_candidate_capacity', 0) or 0)}"
            f" X/Y/Z={int(flex_audit.get('X', 0) or 0)}/{int(flex_audit.get('Y', 0) or 0)}/{int(flex_audit.get('Z', 0) or 0)}"
        )
        if phase in {
            "external_flex_closure",
            "external_fatty_ring_closure",
            "external_side_chain_closure",
        }:
            preferred_demand = dict(plan_dict.get("preferred_required_counts", {}) or {})
            print(
                f"[External Demand] hard={phase_demand} preferred={preferred_demand} "
                f"available={phase_inventory} gap={_resource_gap(phase_inventory, phase_demand)}"
            )
            resource_breakdown = dict(plan_dict.get("resource_request_breakdown", {}) or {})
            selected_contracts = list(resource_breakdown.get("selected_active_contracts", []) or [])
            print(
                "[External Contracts] "
                f"before={int(resource_breakdown.get('active_contract_count_before_overlap_filter', 0) or 0)} "
                f"after={int(resource_breakdown.get('active_contract_count_after_overlap_filter', 0) or 0)}"
            )
            for contract in selected_contracts:
                print(
                    "  [External Contract] "
                    f"contract={int(dict(contract or {}).get('contract_id', 0) or 0)} "
                    f"kind={str(dict(contract or {}).get('contract_kind', '') or '')} "
                    f"fragments={list(dict(contract or {}).get('source_fragment_ids', []) or [])} "
                    f"nodes={list(dict(contract or {}).get('claimed_source_node_ids', []) or [])} "
                    f"required={dict(dict(contract or {}).get('required_counts', {}) or {})}"
                )
        elif phase == "residual_resource_self_closure":
            print(
                "[Residual Flex] "
                f"minimum_gap={int(flex_target_plan.get('remaining_minimum_gap', 0) or 0)} "
                f"preferred_gap={int(flex_target_plan.get('remaining_preferred_gap', 0) or 0)}"
            )
        if str(getattr(plan, "reason", "")) == "stage3_complete":
            stop_reason = "stage3_complete"
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase="complete",
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                topology_signature=before_signature,
                accepted=True,
                acceptance_reason=stop_reason,
                state_signature=state_signature,
            ))
            return Stage3WorkflowResult(
                nodes=nodes_current,
                H=H_current,
                diff=diff_current,
                ok=True,
                stop_reason=stop_reason,
                accepted_cycles=accepted_cycles,
                cycles=cycles,
                diagnostics={"stage": "topology_resource_adjustment"},
            )
        if state_signature in accepted_state_signatures:
            stop_reason = "repeated_histogram_phase_state"
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                topology_signature=before_signature,
                accepted=False,
                acceptance_reason=stop_reason,
                state_signature=state_signature,
            ))
            break

        _print_stage3_histogram_adjustments(cycle_number, histogram_plan)
        apply_started = time.perf_counter()
        print(f"[Block C:resource #{cycle_number}] histogram apply start", flush=True)
        apply_result = adjuster.apply_topology_resource_adjustments(
            nodes=list(nodes_current or []),
            H=H_current.detach().clone(),
            plan=plan,
        )
        print(
            f"[Block C:resource #{cycle_number}] histogram apply done "
            f"({time.perf_counter() - apply_started:.2f}s)",
            flush=True,
        )
        requested_counts = dict(getattr(apply_result, "requested_counts", {}) or {})
        unresolved_counts = dict(getattr(apply_result, "unresolved_counts", {}) or {})
        applied_ops = list(getattr(apply_result, "applied_ops", []) or [])
        request_ops = list(getattr(apply_result, "request_ops", []) or [])
        rejected_ops = list(getattr(apply_result, "rejected_ops", []) or [])
        histogram_changed = not torch.equal(
            torch.clamp(getattr(apply_result, "H").detach().cpu(), min=0).long(),
            torch.clamp(H_current.detach().cpu(), min=0).long(),
        )
        if not bool(getattr(apply_result, "changed", False)):
            plan_blocked_reason = str(
                getattr(plan, "blocked_reason", "")
                or getattr(histogram_plan, "blocked_reason", "")
                or getattr(plan, "reason", "")
                or ""
            )
            stop_reason = (
                "phase_resource_infeasible"
                if request_ops or unresolved_counts or str(getattr(plan, "reason", "")) == "phase_resource_infeasible"
                else plan_blocked_reason or "histogram_adjustment_impossible"
            )
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=False,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                topology_signature=before_signature,
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=str(getattr(plan, "blocked_reason", "") or stop_reason),
                accepted=False,
                acceptance_reason=stop_reason,
                diagnostics={"applied_ops": applied_ops, "request_ops": request_ops, "rejected_ops": rejected_ops},
            ))
            break

        if rebuild_fn is None or validate_fn is None:
            stop_reason = "layer12_rebuild_invalid"
            plan_dict, rolled_back_contract_ids = _rollback_stage3_plan_transactions(plan, stop_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                plan=plan_dict,
                before_metrics=before_metrics,
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=stop_reason,
                accepted=False,
                acceptance_reason=stop_reason,
                diagnostics={"rolled_back_contract_ids": list(rolled_back_contract_ids)},
            ))
            break

        try:
            rebuild_started = time.perf_counter()
            print(f"[Block C:resource #{cycle_number}] cold Layer1/2 rebuild start", flush=True)
            H_seed = torch.clamp(getattr(apply_result, "H").detach().cpu(), min=0).long()
            cand_nodes, cand_H, cand_diff, cand_r2 = rebuild_fn(
                H_seed=getattr(apply_result, "H"),
                phase=phase,
                cycle_idx=cycle_number - 1,
                identity_seed_nodes=list(getattr(apply_result, "nodes", []) or []),
            )
            print(
                f"[Block C:resource #{cycle_number}] cold Layer1/2 rebuild done "
                f"({time.perf_counter() - rebuild_started:.2f}s, r2={float(cand_r2):.4f})",
                flush=True,
            )
        except Exception as exc:
            stop_reason = "layer12_rebuild_invalid"
            rollback_reason = f"{stop_reason}:{exc}"
            plan_dict, rolled_back_contract_ids = _rollback_stage3_plan_transactions(plan, rollback_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=stop_reason,
                accepted_state_retained=True,
                rebuild_valid=False,
                plan=plan_dict,
                before_metrics=before_metrics,
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=str(exc),
                accepted=False,
                acceptance_reason=stop_reason,
                primary_adjustment_count=int(getattr(histogram_plan, "primary_adjustment_count", 0) or 0),
                physical_conversion_count=int(getattr(histogram_plan, "physical_conversion_count", 0) or 0),
                h_compensation_count=_histogram_h_compensation_count(histogram_plan),
                phase_inventory={str(key): int(value) for key, value in phase_inventory.items()},
                phase_demand={str(key): int(value) for key, value in phase_demand.items()},
                histogram_before=histogram_before,
                histogram_after=_histogram_dict(getattr(apply_result, 'H')),
                gap_before=_resource_gap(phase_inventory, phase_demand),
                state_signature=state_signature,
                cold_rebuild=True,
                diagnostics={
                    "applied_ops": applied_ops,
                    "request_ops": request_ops,
                    "rejected_ops": rejected_ops,
                    "rebuild_exception": repr(exc),
                    "planned_conversions": planned_conversions,
                    "adjustment_bundles": adjustment_bundles,
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                },
            ))
            break
        candidate_histogram = torch.clamp(cand_H.detach().cpu(), min=0).long()
        if not torch.equal(candidate_histogram, H_seed):
            valid_reason = "cold_rebuild_histogram_mismatch"
            plan_dict, rolled_back_contract_ids = _rollback_stage3_plan_transactions(plan, valid_reason)
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=f"layer12_rebuild_invalid:{valid_reason}",
                accepted_state_retained=True,
                rebuild_valid=False,
                plan=plan_dict,
                before_metrics=before_metrics,
                layer2_r2=float(cand_r2),
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=valid_reason,
                accepted=False,
                acceptance_reason="layer12_rebuild_invalid",
                cold_rebuild=True,
                diagnostics={
                    "expected_histogram": _histogram_dict(H_seed),
                    "rebuilt_histogram": _histogram_dict(candidate_histogram),
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                },
            ))
            stop_reason = "all_histogram_adjustment_variants_invalid"
            break
        validate_started = time.perf_counter()
        print(f"[Block C:resource #{cycle_number}] rebuilt state validation start", flush=True)
        valid, valid_reason = validate_fn(nodes=cand_nodes, H=cand_H, tag=f"BlockC:resource #{cycle_number}")
        print(
            f"[Block C:resource #{cycle_number}] rebuilt state validation done "
            f"({time.perf_counter() - validate_started:.2f}s, valid={bool(valid)})",
            flush=True,
        )
        if not valid:
            plan_dict, rolled_back_contract_ids = _rollback_stage3_plan_transactions(plan, str(valid_reason))
            _restore_stage3_rotation_cursor(adjuster, rotation_cursor_before)
            cycles.append(Stage3WorkflowCycleRecord(
                cycle=cycle_number,
                phase=phase,
                applied=True,
                stop_reason=f"layer12_rebuild_invalid:{valid_reason}",
                accepted_state_retained=True,
                rebuild_valid=False,
                plan=plan_dict,
                before_metrics=before_metrics,
                layer2_r2=float(cand_r2),
                requested_counts=requested_counts,
                unresolved_counts=unresolved_counts,
                blocked_reason=str(valid_reason),
                accepted=False,
                acceptance_reason="layer12_rebuild_invalid",
                cold_rebuild=True,
                diagnostics={
                    "applied_ops": applied_ops,
                    "request_ops": request_ops,
                    "rejected_ops": rejected_ops,
                    "planned_conversions": planned_conversions,
                    "adjustment_bundles": adjustment_bundles,
                    "rolled_back_contract_ids": list(rolled_back_contract_ids),
                },
            ))
            stop_reason = "all_histogram_adjustment_variants_invalid"
            break

        replan_started = time.perf_counter()
        print(f"[Block C:resource #{cycle_number}] rebuilt topology replan start", flush=True)
        after_plan = adjuster.plan_topology_resource_adjustments(
            H=torch.clamp(cand_H.detach().clone(), min=0).long(),
            nodes=list(cand_nodes or []),
            S_target=S_target,
            E_target=E_target,
            max_primary_adjustments_per_stage=primary_limit,
            max_h_compensation_ops_per_adjustment=compensation_limit,
            chain_23_limit=chain_23_limit,
            h_tolerance=h_tolerance,
        )
        print(
            f"[Block C:resource #{cycle_number}] rebuilt topology replan done "
            f"({time.perf_counter() - replan_started:.2f}s)",
            flush=True,
        )
        after_metrics = _stage3_metrics_from_plan(after_plan)
        after_signature = _stage3_signature_from_plan(after_plan)
        after_histogram_plan = getattr(after_plan, 'histogram_adjustment_plan', None)
        after_inventory = dict(getattr(after_histogram_plan, 'inventory_before', {}) or {})
        after_demand = dict(getattr(after_histogram_plan, 'demand_before', {}) or {})
        topology_improved, topology_improvement_reason = _topology_strictly_improved(
            before_metrics,
            after_metrics,
        )
        acceptance_reason = "valid_histogram_cold_rebuild"

        nodes_current = list(cand_nodes or [])
        H_current = torch.clamp(cand_H.detach().clone(), min=0).long()
        diff_current = dict(cand_diff or {})
        accepted_cycles += 1
        accepted_state_signatures.add(state_signature)
        cycles.append(Stage3WorkflowCycleRecord(
            cycle=cycle_number,
            phase=phase,
            applied=True,
            stop_reason="accepted",
            accepted_state_retained=False,
            rebuild_valid=True,
            plan=plan_dict,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            layer2_r2=float(cand_r2),
            topology_signature=after_signature,
            requested_counts=requested_counts,
            unresolved_counts=unresolved_counts,
            accepted=True,
            acceptance_reason=acceptance_reason,
            primary_adjustment_count=int(getattr(histogram_plan, "primary_adjustment_count", 0) or 0),
            physical_conversion_count=int(getattr(histogram_plan, "physical_conversion_count", 0) or 0),
            h_compensation_count=_histogram_h_compensation_count(histogram_plan),
            phase_inventory={str(key): int(value) for key, value in phase_inventory.items()},
            phase_demand={str(key): int(value) for key, value in phase_demand.items()},
            histogram_before=histogram_before,
            histogram_after=_histogram_dict(cand_H),
            gap_before=_resource_gap(phase_inventory, phase_demand),
            gap_after=_resource_gap(after_inventory, after_demand),
            state_signature=state_signature,
            cold_rebuild=True,
            diagnostics={
                "applied_ops": applied_ops,
                "request_ops": request_ops,
                "rejected_ops": rejected_ops,
                "planned_conversions": planned_conversions,
                "adjustment_bundles": adjustment_bundles,
                "before_requirements": dict(plan_dict.get('resource_deficit', {}) or {}),
                "conversion_pattern_key": str(plan_dict.get('plan_kind', '') or ''),
                "flex_target_plan": dict(flex_target_plan),
                "protected_resource_counts": dict(protected_counts),
            },
        ))

    return Stage3WorkflowResult(
        nodes=nodes_current,
        H=H_current,
        diff=diff_current,
        ok=stop_reason == "stage3_complete",
        stop_reason=stop_reason,
        accepted_cycles=accepted_cycles,
        cycles=cycles,
        diagnostics={"stage": "topology_resource_adjustment", "output_dir": output_dir},
    )
