import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple


def adjust_carbonyl_by_difference_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    window_12: float = 5.0,
    window_3: float = 10.0,
    score_rel_threshold: float = 0.15,
    max_moves: int = 5,
    min_keep: int = 1,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """
    基于差谱调整羰基类型（1/2/3 互转）

    策略：
    - 1号（羧酸）~174.8ppm ±window_12
    - 2号（酯）~169.6ppm ±window_12
    - 3号（醛酮）~195.8ppm ±window_3
    - 正峰 -> 增加该类型
    - 负峰 -> 减少该类型
    - 守恒互转：优先 3↔1/2，允许 1↔2
    """
    print("\n[羰基调整] 基于差谱分析")

    if ppm is None or diff is None:
        print("  无差谱数据，跳过调整")
        return H, [], {}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        print("  差谱为空，跳过调整")
        return H, [], {}

    lo_0, hi_0, mu_0 = adjuster._get_su_common_window(0, fallback_mu=167.125, pad=0.25 * float(window_12), min_half_width=float(window_12))
    lo_1, hi_1, mu_1 = adjuster._get_su_common_window(1, fallback_mu=174.8, pad=0.25 * float(window_12), min_half_width=float(window_12))
    lo_2, hi_2, mu_2 = adjuster._get_su_common_window(2, fallback_mu=169.6, pad=0.25 * float(window_12), min_half_width=float(window_12))
    lo_3, hi_3, mu_3 = adjuster._get_su_common_window(3, fallback_mu=195.8, pad=0.25 * float(window_3), min_half_width=float(window_3))

    carbonyl_mask = (ppm_arr >= 160.0) & (ppm_arr <= 240.0)
    carbonyl_abs = float(np.sum(np.abs(diff_arr[carbonyl_mask]))) if bool(carbonyl_mask.any()) else float(np.sum(np.abs(diff_arr)))
    thr = float(score_rel_threshold) * max(1e-9, float(carbonyl_abs))

    s0 = adjuster._window_stats(ppm_arr, diff_arr, lo_0, hi_0)
    s1 = adjuster._window_stats(ppm_arr, diff_arr, lo_1, hi_1)
    s2 = adjuster._window_stats(ppm_arr, diff_arr, lo_2, hi_2)
    s3 = adjuster._window_stats(ppm_arr, diff_arr, lo_3, hi_3)

    print(f"  0号@{mu_0:.3f} [{lo_0:.3f},{hi_0:.3f}] pos={float(s0['pos']):.3f}, neg={float(s0['neg']):.3f}, net={float(s0['net']):.3f} (固定不调整)")
    print(f"  1号@{mu_1:.3f} [{lo_1:.3f},{hi_1:.3f}] pos={float(s1['pos']):.3f}, neg={float(s1['neg']):.3f}, net={float(s1['net']):.3f}")
    print(f"  2号@{mu_2:.3f} [{lo_2:.3f},{hi_2:.3f}] pos={float(s2['pos']):.3f}, neg={float(s2['neg']):.3f}, net={float(s2['net']):.3f}")
    print(f"  3号@{mu_3:.3f} [{lo_3:.3f},{hi_3:.3f}] pos={float(s3['pos']):.3f}, neg={float(s3['neg']):.3f}, net={float(s3['net']):.3f}")
    print(f"  threshold={thr:.3f} (score_rel_threshold={float(score_rel_threshold):.4f}, carbonyl_abs={carbonyl_abs:.3f})")

    def _need(stats: Dict[str, float]) -> int:
        pos = float(stats.get("pos", 0.0))
        neg_abs = abs(float(stats.get("neg", 0.0)))
        net_abs = abs(float(stats.get("net", 0.0)))
        if pos > thr and neg_abs > thr and net_abs < 0.25 * (pos + neg_abs):
            return 0
        dom = float(stats.get("dom", 0.0))
        if dom > thr:
            return 1
        if dom < -thr:
            return -1
        return 0

    stats_map = {1: s1, 2: s2, 3: s3}
    needs = {k: _need(v) for k, v in stats_map.items()}
    print(f"  需求判断(正=缺乏/需增加, 负=过量/需减少): {needs}")

    H_new = H.clone()
    moves: List[Dict[str, Any]] = []

    def _count(k: int) -> int:
        return int(H_new[k].item())

    for _ in range(int(max_moves)):
        inc_candidates = [k for k, v in needs.items() if int(v) > 0]
        if not inc_candidates:
            break

        receiver = max(inc_candidates, key=lambda k: abs(float(stats_map[k].get("dom", 0.0))))

        donor = None
        if int(receiver) in (1, 2) and _count(3) > int(min_keep) and int(receiver) != 3:
            if int(needs.get(3, 0)) <= 0:
                if int(needs.get(3, 0)) < 0 or not any(int(needs.get(k, 0)) < 0 for k in (1, 2, 3)):
                    donor = 3

        if donor is None:
            dec_candidates = [k for k, v in needs.items() if int(v) < 0 and int(k) != int(receiver) and _count(int(k)) > int(min_keep)]
            if dec_candidates:
                donor = min(dec_candidates, key=lambda k: float(stats_map[k].get("dom", 0.0)))

        if donor is None:
            fallback = [k for k in (3, 1, 2) if int(k) != int(receiver) and _count(int(k)) > int(min_keep)]
            if fallback:
                donor = min(fallback, key=lambda k: float(stats_map[int(k)].get("dom", 0.0)))

        if donor is None:
            break

        if _count(int(donor)) <= int(min_keep):
            break

        H_new[int(donor)] -= 1
        H_new[int(receiver)] += 1
        moves.append({"from": int(donor), "to": int(receiver)})
        print(f"    {int(donor)} -> {int(receiver)}")

    meta = {
        "n_moves": int(len(moves)),
        "threshold": float(thr),
        "carbonyl_abs": float(carbonyl_abs),
        "scores": {"0": s0, "1": s1, "2": s2, "3": s3},
        "needs": needs,
    }

    print(f"  完成 {len(moves)} 次羰基互转")
    return H_new, moves, meta


def adjust_block_a_carbonyl_anchor_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    max_moves: int = 6,
    carbonyl_max_moves: int = 2,
    score_rel_threshold: float = 0.02,
    peak_rel_threshold: float = 0.01,
    min_keep: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[Block A] 羰基-锚点联合调整")

    H_work = torch.clamp(H, min=0).long().clone()
    all_moves: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    max_moves = min(int(max_moves), 3)
    carbonyl_max_moves = min(int(carbonyl_max_moves), 3)

    def _rebalance_oxygen_related(reason: str) -> None:
        nonlocal H_work, all_moves, meta
        layer0_estimator = getattr(adjuster, 'layer0_estimator', None)
        E_target = getattr(adjuster, 'E_target', None)
        if layer0_estimator is None or E_target is None:
            return
        try:
            H_before = H_work.clone()
            H_target, rebalance_meta = layer0_estimator._rebalance_oxygen_linked_units_after_carbonyl_adjust(
                H_work,
                E_target,
            )
            H_new = H_work.clone()
            conversion_records: List[Dict[str, Any]] = []
            unmet: List[Dict[str, Any]] = []

            def _ival(hh: torch.Tensor, idx: int) -> int:
                return int(hh[int(idx)].item())

            def _carbon_count(hh: torch.Tensor) -> Optional[int]:
                try:
                    e_su = getattr(adjuster, 'E_SU', None)
                    if e_su is None:
                        return None
                    pred = torch.matmul(hh.detach().cpu().float(), e_su.detach().cpu().float())
                    return int(round(float(pred[0].item())))
                except Exception:
                    return None

            def _add_conversion(op: str, src: int, dst: int, count: int) -> None:
                if int(count) <= 0:
                    return
                conversion_records.append({
                    'op': str(op),
                    'from': int(src),
                    'to': int(dst),
                    'count': int(count),
                })

            def _convert(src: int, dst: int, count: int, op: str) -> int:
                take = min(max(0, int(count)), max(0, _ival(H_new, int(src))))
                if int(take) <= 0:
                    return 0
                H_new[int(src)] -= int(take)
                H_new[int(dst)] += int(take)
                _add_conversion(op, int(src), int(dst), int(take))
                return int(take)

            def _convert_anchor_to_target(anchor_su: int) -> None:
                target = max(0, _ival(H_target, int(anchor_su)))
                current = max(0, _ival(H_new, int(anchor_su)))
                delta = int(target - current)
                if int(delta) < 0:
                    _convert(int(anchor_su), 11, int(-delta), f'A_{anchor_su}->11_fixed_link')
                    return
                if int(delta) > 0:
                    done = _convert(11, int(anchor_su), int(delta), f'A_11->{anchor_su}_fixed_link')
                    if int(done) < int(delta):
                        unmet.append({
                            'kind': f'anchor_{int(anchor_su)}',
                            'target': int(target),
                            'applied': int(current + done),
                            'missing_11': int(delta - done),
                        })

            # 1/2/3 are carbonyl-type interconversions, while 28/29 are
            # hetero-atom units. These changes are allowed to follow the O
            # budget directly; carbon-bearing anchors below must be converted.
            for su_idx in (1, 2, 3, 28, 29):
                if 0 <= int(su_idx) < int(H_new.numel()):
                    H_new[int(su_idx)] = max(0, _ival(H_target, int(su_idx)))

            # First free 11 from anchors that became unnecessary, then consume
            # 11 for anchors whose fixed-link demand increased.
            for anchor in (9, 5, 7):
                if _ival(H_target, int(anchor)) < _ival(H_new, int(anchor)):
                    _convert_anchor_to_target(int(anchor))
            for anchor in (9, 5, 7):
                if _ival(H_target, int(anchor)) > _ival(H_new, int(anchor)):
                    _convert_anchor_to_target(int(anchor))

            def _clone_meta_19(src_meta: Dict[int, Dict[int, int]], total_19: int) -> Dict[int, int]:
                raw = dict(src_meta.get(19, src_meta.get('19', {})) or {})
                out = {
                    int(deg): max(0, int(raw.get(int(deg), raw.get(str(int(deg)), 0)) or 0))
                    for deg in (1, 2, 3)
                }
                if int(sum(out.values())) != int(total_19):
                    # Keep a simple CH3/CH2/CH preference if old metadata is
                    # stale; the exact partition is normalized by Layer4 below.
                    base = {1: 0, 2: int(total_19), 3: 0}
                    out = base
                return out

            try:
                special_meta = {
                    int(su): {int(deg): int(cnt) for deg, cnt in dict(parts).items()}
                    for su, parts in dict(adjuster._get_special_degree_meta(H_new)).items()
                }
            except Exception:
                special_meta = {19: {1: 0, 2: _ival(H_new, 19), 3: 0}}

            target_fixed_meta = dict((rebalance_meta or {}).get('fixed_partition_meta', {}) or {})
            target_special_meta = dict(target_fixed_meta.get('special_degree_meta', {}) or {})
            cur19_meta = _clone_meta_19(special_meta, _ival(H_new, 19))
            target19_meta = _clone_meta_19(target_special_meta, _ival(H_target, 19))
            tail_for_degree = {1: 22, 2: 23, 3: 24}

            def _dec_19_degree(degree_i: int, count: int) -> int:
                take = min(max(0, int(count)), max(0, int(cur19_meta.get(int(degree_i), 0))), max(0, _ival(H_new, 19)))
                if int(take) <= 0:
                    return 0
                dst = int(tail_for_degree[int(degree_i)])
                H_new[19] -= int(take)
                H_new[dst] += int(take)
                cur19_meta[int(degree_i)] = max(0, int(cur19_meta.get(int(degree_i), 0)) - int(take))
                _add_conversion(f'A_19d{int(degree_i)}->{dst}_fixed_link', 19, dst, int(take))
                return int(take)

            def _inc_19_degree(degree_i: int, count: int) -> int:
                src = int(tail_for_degree[int(degree_i)])
                take = min(max(0, int(count)), max(0, _ival(H_new, src)))
                if int(take) <= 0:
                    return 0
                H_new[src] -= int(take)
                H_new[19] += int(take)
                cur19_meta[int(degree_i)] = int(cur19_meta.get(int(degree_i), 0)) + int(take)
                _add_conversion(f'A_{src}->19d{int(degree_i)}_fixed_link', src, 19, int(take))
                return int(take)

            for degree_i in (1, 2, 3):
                surplus = int(cur19_meta.get(int(degree_i), 0)) - int(target19_meta.get(int(degree_i), 0))
                if int(surplus) > 0:
                    _dec_19_degree(int(degree_i), int(surplus))

            for degree_i in (1, 2, 3):
                deficit = int(target19_meta.get(int(degree_i), 0)) - int(cur19_meta.get(int(degree_i), 0))
                if int(deficit) > 0:
                    done = _inc_19_degree(int(degree_i), int(deficit))
                    if int(done) < int(deficit):
                        unmet.append({
                            'kind': f'19_degree_{int(degree_i)}',
                            'target': int(target19_meta.get(int(degree_i), 0)),
                            'applied': int(cur19_meta.get(int(degree_i), 0)),
                            'missing_tail': int(deficit - done),
                            'tail_su': int(tail_for_degree[int(degree_i)]),
                        })

            # If the target total changed but degree metadata was too stale to
            # express it, finish with degree-compatible fallbacks only.
            while _ival(H_new, 19) > _ival(H_target, 19):
                before19 = _ival(H_new, 19)
                for degree_i in (2, 1, 3):
                    if _ival(H_new, 19) <= _ival(H_target, 19):
                        break
                    _dec_19_degree(int(degree_i), 1)
                if _ival(H_new, 19) == int(before19):
                    break

            while _ival(H_new, 19) < _ival(H_target, 19):
                before19 = _ival(H_new, 19)
                for degree_i in (2, 1, 3):
                    if _ival(H_new, 19) >= _ival(H_target, 19):
                        break
                    _inc_19_degree(int(degree_i), 1)
                if _ival(H_new, 19) == int(before19):
                    unmet.append({
                        'kind': '19_total',
                        'target': int(_ival(H_target, 19)),
                        'applied': int(_ival(H_new, 19)),
                    })
                    break

            special_meta[19] = {int(deg): int(cur19_meta.get(int(deg), 0)) for deg in (1, 2, 3)}

            if target_fixed_meta:
                fixed_meta_seed = dict(target_fixed_meta)
            else:
                fixed_meta_seed = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
            fixed_meta_seed['special_degree_meta'] = {
                int(su): {int(deg): int(cnt) for deg, cnt in dict(parts).items()}
                for su, parts in dict(special_meta).items()
            }
            fixed_meta_seed['n19_total'] = int(_ival(H_new, 19))
            fixed_meta_seed['n5_total'] = int(_ival(H_new, 5))
            fixed_meta_seed['n7_total'] = int(_ival(H_new, 7))
            fixed_meta_seed['n9_total'] = int(_ival(H_new, 9))
            adjuster.fixed_partition_meta = dict(fixed_meta_seed)
            try:
                adjuster._set_special_degree_meta(H_new, special_meta)
            except Exception:
                try:
                    layer0_estimator.fixed_partition_meta = dict(fixed_meta_seed)
                    layer0_estimator.special_degree_meta = dict(fixed_meta_seed.get('special_degree_meta', {}) or {})
                except Exception:
                    pass

            try:
                fixed_meta_after = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
            except Exception:
                fixed_meta_after = dict(fixed_meta_seed)

            delta_rebalance: Dict[int, int] = {}
            for su_idx in (1, 2, 3, 5, 7, 9, 11, 19, 22, 23, 24, 28, 29):
                dv = int(H_new[int(su_idx)].item()) - int(H_before[int(su_idx)].item())
                if int(dv) != 0:
                    delta_rebalance[int(su_idx)] = int(dv)
            H_work = H_new
            if delta_rebalance:
                all_moves.append({
                    "block": "A_rebalance",
                    "op": f"A_recompute_carbonyl_O_fixed_links_by_conversion[{reason}]",
                    "delta": dict(delta_rebalance),
                    "conversions": list(conversion_records),
                })
            meta["post_oxygen_rebalance"] = {
                "reason": str(reason),
                **dict(rebalance_meta or {}),
                "delta": dict(delta_rebalance),
                "target_delta_raw": {
                    int(su_idx): int(H_target[int(su_idx)].item()) - int(H_before[int(su_idx)].item())
                    for su_idx in (1, 2, 3, 5, 7, 9, 19, 28, 29)
                    if int(H_target[int(su_idx)].item()) - int(H_before[int(su_idx)].item()) != 0
                },
                "conversions": list(conversion_records),
                "unmet": list(unmet),
                "carbon_before": _carbon_count(H_before),
                "carbon_after": _carbon_count(H_work),
                "fixed_partition_meta": dict(fixed_meta_after),
            }
        except Exception as e:
            meta["post_oxygen_rebalance_error"] = str(e)

    # 先做少量羰基中心类型修正，再做 9/13 与 22/23/24 的联合迁移。
    # 注意：11 号与脂肪碳数量高度关联，Block A 不再把 11 作为可随意增减的缓冲池。
    center_moves: List[Dict[str, Any]] = []
    if int(max_moves) > 0 and int(carbonyl_max_moves) > 0:
        H_work, center_moves, center_meta = adjust_carbonyl_by_difference_impl(
            adjuster,
            H_work,
            ppm,
            diff,
            score_rel_threshold=float(score_rel_threshold),
            max_moves=min(int(max_moves), int(carbonyl_max_moves)),
            min_keep=int(min_keep),
        )
        for mv in center_moves:
            tagged = dict(mv)
            tagged["block"] = "A_center"
            all_moves.append(tagged)
        meta["center_meta"] = center_meta
        if center_moves:
            _rebalance_oxygen_related(reason='after_center_moves')

    if ppm is None or diff is None:
        meta["joint_direction"] = None
        return H_work, all_moves, meta

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        meta["joint_direction"] = None
        return H_work, all_moves, meta

    carb_windows = adjuster._get_carb_joint_windows()
    low_lo, low_hi = carb_windows["low"]
    mid_lo, mid_hi = carb_windows["mid"]
    high_lo, high_hi = carb_windows["high"]
    overall_lo, overall_hi = carb_windows["overall"]
    low = adjuster._window_stats(ppm_arr, diff_arr, low_lo, low_hi)
    mid = adjuster._window_stats(ppm_arr, diff_arr, mid_lo, mid_hi)
    high = adjuster._window_stats(ppm_arr, diff_arr, high_lo, high_hi)
    carbonyl_mask = (ppm_arr >= overall_lo) & (ppm_arr <= overall_hi)
    carbonyl_abs = float(np.sum(np.abs(diff_arr[carbonyl_mask]))) if np.any(carbonyl_mask) else float(np.sum(np.abs(diff_arr)))
    thr = float(peak_rel_threshold) * max(1e-8, carbonyl_abs)

    direction = None
    if float(low["neg"]) > float(thr) and float(mid["pos"]) > float(thr):
        direction = "to_aliphatic"
    elif float(low["pos"]) > float(thr) and float(mid["neg"]) > float(thr):
        direction = "to_aryl9"

    meta["joint_direction"] = direction
    meta["joint_windows"] = {
        f"{low_lo:.1f}_{low_hi:.1f}": low,
        f"{mid_lo:.1f}_{mid_hi:.1f}": mid,
        f"{high_lo:.1f}_{high_hi:.1f}": high,
    }
    meta["joint_threshold"] = float(thr)

    remain_moves = min(3, max(0, int(max_moves) - len(center_moves)))
    if direction is None or int(remain_moves) <= 0:
        return H_work, all_moves, meta

    tail_rank = adjuster._rank_tail_targets(ppm_arr, diff_arr)
    joint_candidates: List[Tuple[str, Dict[int, int]]] = []
    if direction == "to_aliphatic":
        for su in tail_rank:
            if su == 23:
                joint_candidates.append(("A_9to13__22to23", {9: -1, 13: +1, 22: -1, 23: +1}))
                joint_candidates.append(("A_9to23_via13", {9: -1, 23: +1}))
            elif su == 24:
                joint_candidates.append(("A_9to13__23to24", {9: -1, 13: +1, 23: -1, 24: +1}))
        joint_candidates.extend([
            ("A_9to13", {9: -1, 13: +1}),
            ("A_2to1__9to13__22to23", {2: -1, 1: +1, 9: -1, 13: +1, 22: -1, 23: +1}),
            ("A_3to2__9to13__23to24", {3: -1, 2: +1, 9: -1, 13: +1, 23: -1, 24: +1}),
        ])
    else:
        joint_candidates.extend([
            ("A_13to9__23to22", {13: -1, 9: +1, 23: -1, 22: +1}),
            ("A_13to9__24to23", {13: -1, 9: +1, 24: -1, 23: +1}),
            ("A_23to9_via13", {23: -1, 9: +1}),
            ("A_1to2__13to9__23to22", {1: -1, 2: +1, 13: -1, 9: +1, 23: -1, 22: +1}),
            ("A_2to3__13to9__24to23", {2: -1, 3: +1, 13: -1, 9: +1, 24: -1, 23: +1}),
        ])

    keep = {1: int(min_keep), 2: int(min_keep), 3: int(min_keep), 22: 1}
    for _ in range(int(remain_moves)):
        applied = False
        for name, delta in joint_candidates:
            H_try = adjuster._apply_count_delta(H_work, delta, min_keep=keep)
            if H_try is None:
                continue
            H_work = H_try
            all_moves.append({"block": "A_joint", "op": name, "delta": dict(delta)})
            if any(int(k) in {1, 2, 3} for k in delta.keys()):
                _rebalance_oxygen_related(reason=f'after_joint_move:{name}')
            applied = True
            break
        if not applied:
            break

    if all_moves:
        _rebalance_oxygen_related(reason='final_block_a_sync')

    return H_work, all_moves, meta
