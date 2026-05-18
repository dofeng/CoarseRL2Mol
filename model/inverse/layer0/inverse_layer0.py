"""
Inverse Pipeline Layer0 Module - Layer0: SU直方图估计器
"""
import math
import torch
from typing import Tuple, Dict, Any, Optional

from ...shared.inverse_common import (
    SPECIAL_DEGREE_PRIORS,
    allocate_special_degree_counts,
    get_effective_hist_element_vector,
    estimate_region_carbon_budgets,
)

LAYER0_SPECIAL_MODE_WINDOWS: Dict[int, Dict[str, Tuple[float, float]]] = {
    19: {
        'single_d1': (50.0, 60.0),
        'single_d2': (60.0, 70.0),
        'single_d3': (70.0, 90.0),
        'double_d2': (90.0, 100.0),
        'double_d3': (90.0, 100.0),
    },
    20: {
        'single_d1': (38.0, 45.0),
        'single_d2': (45.0, 52.0),
        'single_d3': (52.0, 65.0),
        'double_d2': (40.0, 50.0),
        'double_d3': (50.0, 60.0),
    },
    21: {
        'single_d2': (35.0, 42.0),
        'single_d3': (55.0, 64.0),
    },
}


class Layer0Estimator:
    """Layer0直方图估计器"""
    def __init__(self, s2n_model, E_SU_tensor: torch.Tensor, device: str = 'cpu'):
        self.s2n = s2n_model
        self.E_SU = E_SU_tensor.to(device)
        self.device = device
        self.fixed_partition_meta: Dict[str, Any] = {}
        self.special_degree_meta: Dict[int, Dict[int, int]] = {}
        self.special_partition_meta: Dict[int, Dict[str, Dict[int, int]]] = {}
        self.special_anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]] = {}

    def _record_fixed_partition_meta(self,
                                     H: torch.Tensor,
                                     ether_meta: Optional[Dict[str, Any]] = None,
                                     thio_meta: Optional[Dict[str, Any]] = None,
                                     special_degree_meta: Optional[Dict[int, Dict[int, int]]] = None,
                                     special_anchor_mode_meta: Optional[Dict[int, Dict[str, Dict[int, int]]]] = None) -> Dict[str, Any]:
        meta = {
            'o_base_19': int((ether_meta or {}).get('o_base_19', 0)),
            'o_fixed_edges_19': int((ether_meta or {}).get('o_fixed_edges_19', (ether_meta or {}).get('o_base_19', 0))),
            's_reserved_19': int((thio_meta or {}).get('s_reserved_19', 0)),
            's_fixed_edges_19': int((thio_meta or {}).get('s_fixed_edges_19', (thio_meta or {}).get('s_reserved_19', 0))),
            'n19_total': int(H[19].item()) if int(H.numel()) > 19 else 0,
            'n5_total': int(H[5].item()) if int(H.numel()) > 5 else 0,
            'n7_total': int(H[7].item()) if int(H.numel()) > 7 else 0,
            'n9_total': int(H[9].item()) if int(H.numel()) > 9 else 0,
            'special_degree_meta': {
                int(su): {int(deg): int(cnt) for deg, cnt in dict(parts).items()}
                for su, parts in dict(special_degree_meta or self.special_degree_meta or {}).items()
            },
            'special_partition_meta': {
                int(su): {
                    str(part): {int(deg): int(cnt) for deg, cnt in dict(part_counts).items()}
                    for part, part_counts in dict(parts).items()
                }
                for su, parts in dict(getattr(self, 'special_partition_meta', {}) or {}).items()
            },
            'special_anchor_mode_meta': {
                int(su): {
                    str(bucket): {int(deg): int(cnt) for deg, cnt in dict(bucket_counts).items()}
                    for bucket, bucket_counts in dict(parts).items()
                }
                for su, parts in dict(special_anchor_mode_meta or self.special_anchor_mode_meta or {}).items()
            },
        }
        self.fixed_partition_meta = dict(meta)
        self.special_degree_meta = dict(meta.get('special_degree_meta', {}) or {})
        self.special_partition_meta = dict(meta.get('special_partition_meta', {}) or {})
        self.special_anchor_mode_meta = dict(meta.get('special_anchor_mode_meta', {}) or {})
        return dict(meta)

    def _allocate_special_degree_meta(self,
                                      H: torch.Tensor,
                                      o_base_19: Optional[int] = None,
                                      s_reserved_19: Optional[int] = None,
                                      o_fixed_edges_19: Optional[int] = None,
                                      s_fixed_edges_19: Optional[int] = None,
                                      n_fixed_edges_20: Optional[int] = None,
                                      x_fixed_edges_21: Optional[int] = None,
                                      S_target: Optional[torch.Tensor] = None) -> Dict[int, Dict[int, int]]:
        H_cpu = H.detach().cpu().long()
        meta: Dict[int, Dict[int, int]] = {}
        partition_meta: Dict[int, Dict[str, Dict[int, int]]] = {}
        anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]] = {}
        S_hint = S_target if S_target is not None else getattr(self, '_current_S_target', None)
        o_fixed_edges_i = int(o_fixed_edges_19) if o_fixed_edges_19 is not None else None
        s_fixed_edges_i = int(s_fixed_edges_19) if s_fixed_edges_19 is not None else None

        for su_type, ratio_map in SPECIAL_DEGREE_PRIORS.items():
            count = int(H_cpu[int(su_type)].item()) if int(H_cpu.numel()) > int(su_type) else 0
            if int(su_type) == 19:
                total_19 = int(count)
                o_part = int(o_base_19) if o_base_19 is not None else int(total_19)
                s_part = int(s_reserved_19) if s_reserved_19 is not None else max(0, int(total_19 - o_part))
                o_part = max(0, min(int(o_part), int(total_19)))
                s_part = max(0, min(int(s_part), int(total_19 - o_part)))
                if int(o_part + s_part) < int(total_19):
                    o_part += int(total_19 - (o_part + s_part))

                mode_o, meta_o, o_fixed_edges_actual = self._special_mode_counts_for_nodes(
                    19,
                    int(o_part),
                    S_hint,
                    max_fixed_edges=o_fixed_edges_i,
                )
                # Ether-partition SU19(d1) nodes can be consumed by:
                #   - one SU2 slot, and
                #   - at most one slot on each SU29 owner.
                # SU28 cannot attach to SU19(d1), and one SU29 may not consume
                # two d1-SU19 nodes simultaneously.
                ether_d1_cap = int(H_cpu[2].item()) + int(H_cpu[29].item())
                if int(meta_o.get(1, 0)) > int(ether_d1_cap):
                    excess = int(meta_o.get(1, 0)) - int(ether_d1_cap)
                    meta_o[1] = int(ether_d1_cap)
                    meta_o[2] = int(meta_o.get(2, 0)) + int(excess)
                    take = min(int(mode_o.get('single_d1', 0)), int(excess))
                    if int(take) > 0:
                        mode_o['single_d1'] = int(mode_o.get('single_d1', 0)) - int(take)
                        mode_o['single_d2'] = int(mode_o.get('single_d2', 0)) + int(take)
                        excess -= int(take)
                    if int(excess) > 0 and int(mode_o.get('double_d2', 0)) > 0:
                        take = min(int(mode_o.get('double_d2', 0)), int(excess))
                        mode_o['double_d2'] = int(mode_o.get('double_d2', 0)) - int(take)
                        mode_o['single_d2'] = int(mode_o.get('single_d2', 0)) + int(take)
                mode_o = self._enforce_layer1_mode_pool_caps(H_cpu, 19, mode_o, partition='ether')
                meta_o = {1: 0, 2: 0, 3: 0}
                for mode_name, cnt in dict(mode_o).items():
                    deg_i = int(self._mode_degree(str(mode_name)))
                    if int(deg_i) > 0:
                        meta_o[int(deg_i)] = int(meta_o.get(int(deg_i), 0)) + int(cnt)
                o_fixed_edges_actual = int(sum(
                    int(cnt) * int(self._mode_fixed_edges(str(mode)))
                    for mode, cnt in dict(mode_o).items()
                ))

                mode_s, meta_s, s_fixed_edges_actual = self._special_mode_counts_for_nodes(
                    19,
                    int(s_part),
                    S_hint,
                    max_fixed_edges=s_fixed_edges_i,
                )
                mode_s = self._enforce_layer1_mode_pool_caps(H_cpu, 19, mode_s, partition='thio')
                meta_s = {1: 0, 2: 0, 3: 0}
                for mode_name, cnt in dict(mode_s).items():
                    deg_i = int(self._mode_degree(str(mode_name)))
                    if int(deg_i) > 0:
                        meta_s[int(deg_i)] = int(meta_s.get(int(deg_i), 0)) + int(cnt)
                s_fixed_edges_actual = int(sum(
                    int(cnt) * int(self._mode_fixed_edges(str(mode)))
                    for mode, cnt in dict(mode_s).items()
                ))
                merged = {
                    int(deg): int(meta_o.get(int(deg), 0)) + int(meta_s.get(int(deg), 0))
                    for deg in ratio_map.keys()
                }
                meta[int(su_type)] = merged
                partition_meta[int(su_type)] = {
                    'ether': {int(deg): int(meta_o.get(int(deg), 0)) for deg in ratio_map.keys()},
                    'thio': {int(deg): int(meta_s.get(int(deg), 0)) for deg in ratio_map.keys()},
                }
                anchor_mode_meta[19] = {
                    'ether_single': {
                        int(deg): int(mode_o.get(f'single_d{int(deg)}', 0))
                        for deg in [1, 2, 3]
                    },
                    'ether_double': {
                        int(deg): int(mode_o.get(f'double_d{int(deg)}', 0))
                        for deg in [2, 3]
                    },
                    'thio_single': {
                        int(deg): int(mode_s.get(f'single_d{int(deg)}', 0))
                        for deg in [1, 2, 3]
                    },
                    'thio_double': {
                        int(deg): int(mode_s.get(f'double_d{int(deg)}', 0))
                        for deg in [2, 3]
                    },
                }
                if o_fixed_edges_i is None:
                    o_fixed_edges_i = int(o_fixed_edges_actual)
                if s_fixed_edges_i is None:
                    s_fixed_edges_i = int(s_fixed_edges_actual)
            else:
                allowed_modes = None
                if int(su_type) == 21:
                    allowed_modes = ('single_d2', 'single_d3')
                mode_counts, degree_counts, _fixed_edges = self._special_mode_counts_for_nodes(
                    int(su_type),
                    int(count),
                    S_hint,
                    max_fixed_edges=(
                        int(n_fixed_edges_20)
                        if int(su_type) == 20 and n_fixed_edges_20 is not None
                        else (
                            int(x_fixed_edges_21)
                            if int(su_type) == 21 and x_fixed_edges_21 is not None
                            else None
                        )
                    ),
                    allowed_modes=allowed_modes,
                )
                mode_counts = self._enforce_layer1_mode_pool_caps(H_cpu, int(su_type), mode_counts)
                degree_counts = {
                    int(deg): 0 for deg in dict(SPECIAL_DEGREE_PRIORS.get(int(su_type), {})).keys()
                }
                for mode_name, cnt in dict(mode_counts).items():
                    deg_i = int(self._mode_degree(str(mode_name)))
                    if int(deg_i) > 0:
                        degree_counts[int(deg_i)] = int(degree_counts.get(int(deg_i), 0)) + int(cnt)
                meta[int(su_type)] = {
                    int(deg): int(degree_counts.get(int(deg), 0))
                    for deg in ratio_map.keys()
                }
                if int(su_type) in {20, 21}:
                    degrees = [1, 2, 3] if int(su_type) == 20 else [2, 3]
                    anchor_mode_meta[int(su_type)] = {
                        'single': {
                            int(deg): int(mode_counts.get(f'single_d{int(deg)}', 0))
                            for deg in degrees
                        },
                        'double': {
                            int(deg): int(mode_counts.get(f'double_d{int(deg)}', 0))
                            for deg in [2, 3]
                        },
                    }
        self.special_degree_meta = {
            int(su): {int(deg): int(cnt) for deg, cnt in degree_map.items()}
            for su, degree_map in meta.items()
        }
        self.special_partition_meta = {
            int(su): {
                str(part): {int(deg): int(cnt) for deg, cnt in dict(part_counts).items()}
                for part, part_counts in dict(parts).items()
            }
            for su, parts in partition_meta.items()
        }
        self.special_anchor_mode_meta = {
            int(su): {
                str(bucket): {int(deg): int(cnt) for deg, cnt in dict(bucket_counts).items()}
                for bucket, bucket_counts in dict(parts).items()
            }
            for su, parts in anchor_mode_meta.items()
        }
        return {
            int(su): {int(deg): int(cnt) for deg, cnt in degree_map.items()}
            for su, degree_map in meta.items()
        }

    @staticmethod
    def _nearest_even_int(value: float) -> int:
        value = max(0.0, float(value))
        lower = int(math.floor(value))
        if lower % 2 != 0:
            lower -= 1
        upper = int(math.ceil(value))
        if upper % 2 != 0:
            upper += 1
        lower = max(0, lower)
        upper = max(0, upper)
        if abs(float(value) - float(lower)) <= abs(float(upper) - float(value)):
            return int(lower)
        return int(upper)

    @staticmethod
    def _allocate_ratio_counts(total: int, ratios: Tuple[float, ...]) -> Tuple[int, ...]:
        total = max(0, int(total))
        if total <= 0:
            return tuple(0 for _ in ratios)
        raw = [max(0.0, float(r)) * float(total) for r in ratios]
        base = [int(math.floor(v)) for v in raw]
        remainder = int(total - sum(base))
        if remainder > 0:
            order = sorted(
                range(len(raw)),
                key=lambda i: (raw[i] - float(base[i]), raw[i]),
                reverse=True,
            )
            for i in range(remainder):
                base[order[i % len(order)]] += 1
        return tuple(int(v) for v in base)

    @staticmethod
    def _allocate_count_map(total: int, weights: Dict[str, float]) -> Dict[str, int]:
        total_i = max(0, int(total))
        keys = [str(k) for k in weights.keys()]
        if total_i <= 0 or not keys:
            return {str(k): 0 for k in keys}
        vals = [max(0.0, float(weights.get(str(k), 0.0))) for k in keys]
        if float(sum(vals)) <= 1e-12:
            vals = [1.0 for _ in keys]
        weight_sum = float(sum(vals))
        raw = [float(v) * float(total_i) / float(weight_sum) for v in vals]
        base = [int(math.floor(v)) for v in raw]
        remainder = int(total_i - sum(base))
        if remainder > 0:
            order = sorted(
                range(len(keys)),
                key=lambda i: (raw[i] - float(base[i]), vals[i], -i),
                reverse=True,
            )
            for i in range(remainder):
                base[order[i % len(order)]] += 1
        return {str(k): int(v) for k, v in zip(keys, base)}

    @staticmethod
    def _mode_degree(mode_name: str) -> int:
        try:
            return int(str(mode_name).rsplit('_d', 1)[1])
        except Exception:
            return 0

    @staticmethod
    def _mode_fixed_edges(mode_name: str) -> int:
        return 2 if str(mode_name).startswith('double_') else 1

    @staticmethod
    def _nonterminal_fixed_anchor_slot_capacity(H_cpu: torch.Tensor,
                                                anchor_types: Tuple[int, ...]) -> int:
        terminal_like = {1, 22, 28, 32}
        slot_multipliers = {27: 2, 29: 2, 31: 2}
        total = 0
        for su_i in tuple(int(x) for x in anchor_types):
            if int(su_i) in terminal_like or int(su_i) < 0 or int(su_i) >= int(H_cpu.numel()):
                continue
            total += int(H_cpu[int(su_i)].item()) * int(slot_multipliers.get(int(su_i), 1))
        return int(total)

    def _enforce_layer1_mode_pool_caps(self,
                                       H_cpu: torch.Tensor,
                                       su_type: int,
                                       mode_counts: Dict[str, int],
                                       partition: Optional[str] = None) -> Dict[str, int]:
        """Keep Layer0's special-anchor mode metadata feasible for Layer1."""
        counts = {str(k): max(0, int(v)) for k, v in dict(mode_counts or {}).items()}
        su_i = int(su_type)

        def _move(src: str, dst: str, amount: int) -> None:
            take = min(max(0, int(amount)), int(counts.get(str(src), 0)))
            if int(take) <= 0:
                return
            counts[str(src)] = int(counts.get(str(src), 0)) - int(take)
            counts[str(dst)] = int(counts.get(str(dst), 0)) + int(take)

        if int(su_i) == 19 and str(partition or 'ether') == 'ether':
            d1_cap = int(H_cpu[2].item()) + int(H_cpu[29].item())
            _move('single_d1', 'single_d2', max(0, int(counts.get('single_d1', 0)) - int(d1_cap)))
            d3_cap = self._nonterminal_fixed_anchor_slot_capacity(H_cpu, (2, 28, 29))
            _move('double_d3', 'single_d3', max(0, int(counts.get('double_d3', 0)) - int(d3_cap)))
        elif int(su_i) == 19 and str(partition or '') == 'thio':
            d3_cap = self._nonterminal_fixed_anchor_slot_capacity(H_cpu, (31,))
            _move('double_d3', 'single_d3', max(0, int(counts.get('double_d3', 0)) - int(d3_cap)))
        elif int(su_i) == 20:
            d1_cap = int(H_cpu[0].item()) + 2 * int(H_cpu[27].item())
            _move('single_d1', 'single_d2', max(0, int(counts.get('single_d1', 0)) - int(d1_cap)))
            d3_cap = self._nonterminal_fixed_anchor_slot_capacity(H_cpu, (0, 27))
            _move('double_d3', 'single_d3', max(0, int(counts.get('double_d3', 0)) - int(d3_cap)))
        elif int(su_i) == 21:
            _move('double_d2', 'single_d2', int(counts.get('double_d2', 0)))
            _move('double_d3', 'single_d3', int(counts.get('double_d3', 0)))
        return counts

    def _special_mode_node_weights(self,
                                   su_type: int,
                                   S_target: Optional[torch.Tensor],
                                   allowed_modes: Optional[Tuple[str, ...]] = None) -> Dict[str, float]:
        su_i = int(su_type)
        windows = dict(LAYER0_SPECIAL_MODE_WINDOWS.get(int(su_i), {}) or {})
        if allowed_modes is None:
            allowed = tuple(windows.keys())
        else:
            allowed = tuple(str(m) for m in allowed_modes if str(m) in windows)
        weights: Dict[str, float] = {}

        if int(su_i) == 19:
            # 19 double_d2/d3 are both observed in the 90-100 ppm band; split
            # that band instead of double-counting it.
            for mode_name in ('single_d1', 'single_d2', 'single_d3'):
                if str(mode_name) in allowed:
                    lo, hi = windows[str(mode_name)]
                    weights[str(mode_name)] = self._window_area_from_spectrum(S_target, float(lo), float(hi))
            double_area = self._window_area_from_spectrum(S_target, 90.0, 100.0)
            if 'double_d2' in allowed:
                weights['double_d2'] = 0.5 * float(double_area)
            if 'double_d3' in allowed:
                weights['double_d3'] = 0.5 * float(double_area)
        else:
            for mode_name in allowed:
                lo, hi = windows[str(mode_name)]
                area = self._window_area_from_spectrum(S_target, float(lo), float(hi))
                # SU20/SU21 single/double windows partly overlap, so keep
                # double as a weaker initial prior and let Block B count stages
                # promote it when the graph/diff evidence supports it.
                if str(mode_name).startswith('double_'):
                    area *= 0.50
                weights[str(mode_name)] = float(area)

        if float(sum(float(v) for v in weights.values())) <= 1e-8:
            priors = dict(SPECIAL_DEGREE_PRIORS.get(int(su_i), {}) or {})
            weights = {}
            for degree_i, ratio in priors.items():
                mode_name = f"single_d{int(degree_i)}"
                if mode_name in allowed:
                    weights[str(mode_name)] = float(ratio)
            for mode_name in allowed:
                weights.setdefault(str(mode_name), 0.0)
        return {str(k): max(0.0, float(v)) for k, v in weights.items()}

    def _average_mode_fixed_edges(self,
                                  su_type: int,
                                  S_target: Optional[torch.Tensor],
                                  allowed_modes: Optional[Tuple[str, ...]] = None) -> float:
        weights = self._special_mode_node_weights(int(su_type), S_target, allowed_modes=allowed_modes)
        total = float(sum(float(v) for v in weights.values()))
        if total <= 1e-8:
            return 1.0
        return float(sum(float(v) * float(self._mode_fixed_edges(str(k))) for k, v in weights.items()) / total)

    def _special_mode_counts_for_nodes(self,
                                       su_type: int,
                                       total_nodes: int,
                                       S_target: Optional[torch.Tensor],
                                       max_fixed_edges: Optional[int] = None,
                                       allowed_modes: Optional[Tuple[str, ...]] = None) -> Tuple[Dict[str, int], Dict[int, int], int]:
        su_i = int(su_type)
        total_i = max(0, int(total_nodes))
        weights = self._special_mode_node_weights(int(su_i), S_target, allowed_modes=allowed_modes)
        mode_counts = self._allocate_count_map(int(total_i), weights)

        cap = None if max_fixed_edges is None else max(int(max_fixed_edges), int(total_i))
        if cap is not None:
            # Downgrade double nodes to same-degree single nodes first; this
            # preserves the NMR carbon count and only changes fixed-anchor count.
            # Prefer preserving double_d3; only downgrade it when the total
            # fixed-edge budget is still impossible after double_d2 cleanup.
            for mode_name in ('double_d2', 'double_d3'):
                single_name = str(mode_name).replace('double_', 'single_')
                if single_name not in mode_counts:
                    continue
                while (
                    int(mode_counts.get(str(mode_name), 0)) > 0
                    and int(sum(int(v) * self._mode_fixed_edges(str(k)) for k, v in mode_counts.items())) > int(cap)
                ):
                    mode_counts[str(mode_name)] -= 1
                    mode_counts[str(single_name)] = int(mode_counts.get(str(single_name), 0)) + 1

        degree_counts: Dict[int, int] = {
            int(deg): 0 for deg in dict(SPECIAL_DEGREE_PRIORS.get(int(su_i), {})).keys()
        }
        for mode_name, count in mode_counts.items():
            deg_i = int(self._mode_degree(str(mode_name)))
            if int(deg_i) <= 0:
                continue
            degree_counts[int(deg_i)] = int(degree_counts.get(int(deg_i), 0)) + int(count)
        fixed_edges = int(sum(int(v) * int(self._mode_fixed_edges(str(k))) for k, v in mode_counts.items()))
        return (
            {str(k): int(v) for k, v in mode_counts.items()},
            {int(k): int(v) for k, v in degree_counts.items()},
            int(fixed_edges),
        )

    def _split_aromatic_special_fixed_edges(self,
                                            total_edges: int,
                                            aromatic_su: int,
                                            special_su: int,
                                            aromatic_window: Tuple[float, float],
                                            S_target: Optional[torch.Tensor],
                                            fallback_special_ratio: float) -> Tuple[int, int, int, Dict[str, Any]]:
        edges_i = max(0, int(total_edges))
        if edges_i <= 0:
            return 0, 0, 0, {
                'ratio_source': 'empty_edges',
                'special_node_ratio': 0.0,
                'special_avg_fixed_edges': 1.0,
            }
        area_arom = self._window_area_from_spectrum(S_target, float(aromatic_window[0]), float(aromatic_window[1]))
        mode_weights = self._special_mode_node_weights(int(special_su), S_target)
        area_special = float(sum(float(v) for v in mode_weights.values()))
        if float(area_arom + area_special) <= 1e-8:
            ratio_special = float(fallback_special_ratio)
            ratio_source = 'fallback'
        else:
            ratio_special = float(area_special / float(area_arom + area_special))
            ratio_source = 'spectrum'
        ratio_special = max(0.0, min(0.95, float(ratio_special)))
        avg_edges = self._average_mode_fixed_edges(int(special_su), S_target)
        denom = float(1.0 + float(ratio_special) * max(0.0, float(avg_edges) - 1.0))
        target_special_nodes = int(round(float(ratio_special) * float(edges_i) / max(1e-8, denom)))
        target_special_nodes = max(0, min(int(edges_i), int(target_special_nodes)))

        while target_special_nodes >= 0:
            _, _, fixed_edges = self._special_mode_counts_for_nodes(
                int(special_su),
                int(target_special_nodes),
                S_target,
                max_fixed_edges=int(edges_i),
            )
            if int(fixed_edges) <= int(edges_i):
                aromatic_nodes = int(edges_i - fixed_edges)
                return int(aromatic_nodes), int(target_special_nodes), int(fixed_edges), {
                    'ratio_source': str(ratio_source),
                    'aromatic_su': int(aromatic_su),
                    'special_su': int(special_su),
                    'area_aromatic': float(area_arom),
                    'area_special_modes': float(area_special),
                    'special_node_ratio': float(ratio_special),
                    'special_avg_fixed_edges': float(avg_edges),
                    'special_fixed_edges': int(fixed_edges),
                }
            target_special_nodes -= 1

        return int(edges_i), 0, 0, {
            'ratio_source': 'fallback_all_aromatic',
            'special_node_ratio': 0.0,
            'special_avg_fixed_edges': 1.0,
        }

    def _reconcile_carbon_total(self, H: torch.Tensor, S_target: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """最终修正C总量，只在纯碳结构单元中调节，避免破坏 O/N/S/X 守恒。"""
        H_new = H.clone()
        try:
            E_curr = torch.matmul(H_new.float(), self.E_SU.to(H_new.device))
            target_C = int(E_target[0].item())
            current_C = int(round(float(E_curr[0].item())))
        except Exception:
            return H_new

        delta_C = int(target_C - current_C)
        if delta_C == 0:
            return H_new

        spectrum = S_target.detach().cpu().numpy()
        total_area = float(spectrum.sum() * 0.1)
        aliphatic_area = float(spectrum[:900].sum() * 0.1)
        aromatic_area = float(spectrum[900:1600].sum() * 0.1)
        if total_area > 1e-6:
            x = float(aliphatic_area / total_area)
            y = float(aromatic_area / total_area)
        else:
            x, y = 0.33, 0.33

        if delta_C > 0:
            add_aro = int(round(float(delta_C) * y / max(1e-6, x + y)))
            add_ali = int(round(float(delta_C) * x / max(1e-6, x + y)))
            add_uns = int(delta_C - add_aro - add_ali)
            H_new[13] += max(0, add_aro)
            H_new[23] += max(0, add_ali)
            H_new[15] += max(0, add_uns)
        else:
            deficit = int(-delta_C)
            removal_order = [13, 23, 15, 12, 10, 24, 22, 14, 16, 17, 18, 25, 11]
            for su in removal_order:
                if deficit <= 0:
                    break
                available = int(H_new[su].item())
                if available <= 0:
                    continue
                take = min(available, deficit)
                H_new[su] -= int(take)
                deficit -= int(take)

        return torch.clamp(H_new, min=0).long()

    @staticmethod
    def _enforce_even_su10(H: torch.Tensor) -> torch.Tensor:
        """
        Enforce the same SU10 parity rule expected later by Layer1/Layer4.

        SU10's external anchor port is satisfied by SU4 or SU10. When SU10 is odd,
        Layer1 can leave one SU10 unmatched. We therefore convert one SU10 to SU11
        at Layer0 closure time so the histogram enters Layer1 in a feasible state.
        This conversion is element-neutral.
        """
        H_new = torch.clamp(H.clone(), min=0).long()
        try:
            if int(H_new[10].item()) % 2 != 0 and int(H_new[10].item()) > 0:
                H_new[10] -= 1
                H_new[11] += 1
        except Exception:
            return H
        return H_new

    @staticmethod
    def _rebalance_pair_counts(count_a: int,
                               count_b: int,
                               total_target: int,
                               ratio_a: float,
                               ratio_b: float,
                               tie_inc_prefer: str,
                               tie_dec_prefer: str) -> Tuple[int, int]:
        """将两个计数按目标总量与比例重分配。"""
        a = max(0, int(count_a))
        b = max(0, int(count_b))
        total_target = max(0, int(total_target))
        target_a = float(total_target) * float(ratio_a)
        target_b = float(total_target) * float(ratio_b)

        while int(a + b) < int(total_target):
            delta_a = float(a) - float(target_a)
            delta_b = float(b) - float(target_b)
            if delta_a < delta_b:
                a += 1
            elif delta_b < delta_a:
                b += 1
            elif str(tie_inc_prefer) == 'a':
                a += 1
            else:
                b += 1

        while int(a + b) > int(total_target):
            delta_a = float(a) - float(target_a)
            delta_b = float(b) - float(target_b)
            can_del_a = int(a) > 0
            can_del_b = int(b) > 0
            if can_del_a and (not can_del_b or delta_a > delta_b):
                a -= 1
            elif can_del_b and (not can_del_a or delta_b > delta_a):
                b -= 1
            elif can_del_a and can_del_b:
                if str(tie_dec_prefer) == 'a':
                    a -= 1
                else:
                    b -= 1
            elif can_del_a:
                a -= 1
            elif can_del_b:
                b -= 1
            else:
                break

        return max(0, int(a)), max(0, int(b))

    def _audit_fixed_connection_histogram(self,
                                          H: torch.Tensor,
                                          o_base_19: int = 0,
                                          s_reserved_19: int = 0,
                                          o_fixed_edges_19: Optional[int] = None,
                                          s_fixed_edges_19: Optional[int] = None) -> Dict[str, int]:
        h = torch.clamp(H.detach().cpu().long(), min=0)
        w_carb = int(h[0].item()) + int(h[1].item()) + int(h[2].item()) + 2 * int(h[3].item())
        w_ether = int(h[2].item()) + int(h[28].item()) + 2 * int(h[29].item())
        w_thio = 2 * int(h[31].item())
        o_edges = int(o_fixed_edges_19) if o_fixed_edges_19 is not None else int(o_base_19)
        s_edges = int(s_fixed_edges_19) if s_fixed_edges_19 is not None else int(s_reserved_19)
        return {
            'carbonyl_edges': int(w_carb),
            'target_9_min': int(round(0.5 * float(w_carb))),
            'n9': int(h[9].item()),
            'ether_required': int(w_ether),
            'ether_supply': int(h[5].item()) + int(o_edges),
            'thio_required': int(w_thio),
            'thio_supply': int(h[7].item()) + int(s_edges),
            'o_base_19': int(o_base_19),
            'o_fixed_edges_19': int(o_edges),
            's_reserved_19': int(s_reserved_19),
            's_fixed_edges_19': int(s_edges),
            'n19_total': int(h[19].item()),
        }

    @staticmethod
    def _compute_region_area_ratios(S_target: torch.Tensor) -> Tuple[float, float, float]:
        budgets = estimate_region_carbon_budgets(
            S_target,
            torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float),
        )
        return float(budgets.get('x', 0.33)), float(budgets.get('y', 0.33)), float(budgets.get('z', 0.34))

    def _estimate_region_carbon_budgets(self,
                                        S_target: torch.Tensor,
                                        E_target: torch.Tensor) -> Dict[str, float]:
        """
        根据谱图区域面积比例估计 C 元素预算。

        约定:
          ordinary_x: 0-50 ppm 普通脂肪区面积占比
          oxygenated_x: 50-100 ppm 含氧脂肪区面积占比
          x: 0-90 ppm 历史脂肪区面积占比
          y: 90-160 ppm 芳香/非饱和区面积占比
          z: 160-240 ppm 羰基区面积占比

        碳预算:
          aliphatic_C = scale(H/C) * x * N
          carbonyl_C = z * N
          aromatic_C = N - aliphatic_C - carbonyl_C
        """
        return estimate_region_carbon_budgets(S_target, E_target)

    def estimate_su_histogram(self, S_target: torch.Tensor, 
                               E_target: torch.Tensor) -> torch.Tensor:
        """
        Layer0: 从谱图和元素推断初始SU直方图
        
        改进点：
        1. 修正顺序优化：X → S → N → C=O羰基 → O → 连接匹配
        2. 基于光谱区域比例修正羰基分布
        3. 多样性惩罚：防止某些SU过度集中
        """
        device = self.device
        S_target = S_target.to(device)
        E_target = E_target.to(device)
        self._current_S_target = S_target.detach().cpu()
        
        print("\n[Layer0] SU直方图估计开始")
        
        # Step 1: S2N模型预测初始SU分布
        with torch.no_grad():
            if hasattr(self.s2n, 'infer_su_hist'):
                H_pred = self.s2n.infer_su_hist(S_target.unsqueeze(0), E_target.unsqueeze(0)).squeeze(0)
            else:
                H_pred = torch.nn.functional.softplus(
                    self.s2n(S_target.unsqueeze(0), E_target.unsqueeze(0))
                ).squeeze(0)
        
        # 整数化（四舍五入）
        H_init = torch.round(H_pred).long()
        H_init = torch.clamp(H_init, min=0)
        
        # Step 2: 杂原子元素修正（X → S → N 顺序）
        H_corrected = H_init.clone()
        
        # 2.1 修正X元素（32号）
        H_corrected = self._correct_halogen_X(H_corrected, E_target)
        
        # 2.2 修正S元素（30, 31号）
        H_corrected = self._correct_sulfur_S(H_corrected, E_target)
        
        # 2.3 修正N元素（0, 4, 26, 27号）
        H_corrected = self._correct_nitrogen_N(H_corrected, E_target)
        
        # Step 3: C=O羰基分布修正（基于光谱区域比例）
        self._o_cap_triggered = False
        H_corrected = self._correct_carbonyl_distribution(H_corrected, S_target, E_target)
        
        # Step 4: O元素修正（只修正28, 29号）
        target_O = int(E_target[2].item())
        used_O_03 = (
            int(H_corrected[0].item())
            + 2 * int(H_corrected[1].item())
            + 2 * int(H_corrected[2].item())
            + int(H_corrected[3].item())
        )
        if bool(getattr(self, '_o_cap_triggered', False)) or used_O_03 >= target_O:
            H_corrected = H_corrected.clone()
            H_corrected[28] = 0
            H_corrected[29] = 0
        else:
            H_corrected = self._correct_oxygen_O(H_corrected, E_target, S_target=S_target)
        
        # Step 5: 含碳结构单元连接匹配修正
        
        # 3.1 修正C=O连接（9号）
        H_corrected = self._correct_carbonyl_connection(H_corrected)
        
        # 3.2 修正-O-连接（5号、19号）
        H_corrected, ether_meta = self._correct_ether_connection(H_corrected, S_target=S_target)
        
        # 3.3 修正-S-连接（7号），传入O基准
        o_base_19 = ether_meta.get('o_base_19', 0)
        H_corrected, thio_meta = self._correct_thioether_connection(
            H_corrected,
            o_base_19,
            S_target=S_target,
        )
        self._record_fixed_partition_meta(H_corrected, ether_meta=ether_meta, thio_meta=thio_meta)
        
        # 3.4 修正-NH-连接（6号、20号）
        H_corrected, amine_meta = self._correct_amine_connection(H_corrected, S_target=S_target)
        
        # 3.5 修正-X连接（8号、21号）
        H_corrected, halogen_meta = self._correct_halogen_connection(H_corrected, S_target=S_target)
        
        # Step 6: 脂肪碳结构修正（22, 23, 24, 25号）
        H_corrected = self._correct_aliphatic_carbons(H_corrected, S_target, E_target)
        
        # Step 7: 非饱和结构修正（14, 15, 16, 17, 18号）
        H_corrected = self._correct_unsaturated_carbons(H_corrected, S_target, E_target)
        
        # Step 8: 芳香结构修正（10, 11, 12, 13号）
        H_tmp = self._correct_aromatic_carbons(H_corrected, S_target, E_target)
        if H_tmp is not None:
            H_corrected = H_tmp

        # Step 8.5: 最终碳元素守恒修正
        H_tmp = self._reconcile_carbon_total(H_corrected, S_target, E_target)
        if H_tmp is not None:
            H_corrected = H_tmp

        self._allocate_special_degree_meta(
            H_corrected,
            o_base_19=int((ether_meta or {}).get('o_base_19', 0)),
            s_reserved_19=int((thio_meta or {}).get('s_reserved_19', 0)),
            o_fixed_edges_19=int((ether_meta or {}).get('o_fixed_edges_19', (ether_meta or {}).get('o_base_19', 0))),
            s_fixed_edges_19=int((thio_meta or {}).get('s_fixed_edges_19', (thio_meta or {}).get('s_reserved_19', 0))),
            n_fixed_edges_20=int((amine_meta or {}).get('n20_fixed_edges', H_corrected[20].item() if int(H_corrected.numel()) > 20 else 0)),
            x_fixed_edges_21=int((halogen_meta or {}).get('n21_fixed_edges', H_corrected[21].item() if int(H_corrected.numel()) > 21 else 0)),
            S_target=S_target,
        )
            
        # Step 9: H元素调整（三区域调整）
        H_tmp = self._adjust_hydrogen(H_corrected, E_target)
        if H_tmp is not None:
            H_corrected = H_tmp

        # Layer1 之前再做一次 SU10 偶数化收口。
        # Step 8 的约束可能被后续 C/H 修正重新打坏，这里保证进入 Layer1 时仍满足。
        H_corrected = self._enforce_even_su10(H_corrected)
            
        # 确保所有值为非负整数
        H_corrected = torch.clamp(H_corrected, min=0).long()
        special_degree_meta = self._allocate_special_degree_meta(
            H_corrected,
            o_base_19=int((ether_meta or {}).get('o_base_19', 0)),
            s_reserved_19=int((thio_meta or {}).get('s_reserved_19', 0)),
            o_fixed_edges_19=int((ether_meta or {}).get('o_fixed_edges_19', (ether_meta or {}).get('o_base_19', 0))),
            s_fixed_edges_19=int((thio_meta or {}).get('s_fixed_edges_19', (thio_meta or {}).get('s_reserved_19', 0))),
            n_fixed_edges_20=int((amine_meta or {}).get('n20_fixed_edges', H_corrected[20].item() if int(H_corrected.numel()) > 20 else 0)),
            x_fixed_edges_21=int((halogen_meta or {}).get('n21_fixed_edges', H_corrected[21].item() if int(H_corrected.numel()) > 21 else 0)),
            S_target=S_target,
        )
        fixed_partition_meta = self._record_fixed_partition_meta(
            H_corrected,
            ether_meta=ether_meta,
            thio_meta=thio_meta,
            special_degree_meta=special_degree_meta,
            special_anchor_mode_meta=getattr(self, 'special_anchor_mode_meta', {}) or {},
        )

        try:
            audit = self._audit_fixed_connection_histogram(
                H_corrected,
                o_base_19=int(ether_meta.get('o_base_19', 0)),
                s_reserved_19=int(thio_meta.get('s_reserved_19', 0)),
                o_fixed_edges_19=int(ether_meta.get('o_fixed_edges_19', ether_meta.get('o_base_19', 0))),
                s_fixed_edges_19=int(thio_meta.get('s_fixed_edges_19', thio_meta.get('s_reserved_19', 0))),
            )
            print(
                "[Layer0 Fixed Audit] "
                f"9={audit['n9']} target_min={audit['target_9_min']} | "
                f"ether={audit['ether_supply']}/{audit['ether_required']} | "
                f"thio={audit['thio_supply']}/{audit['thio_required']} | "
                f"19={audit['n19_total']} "
                f"(O={audit['o_base_19']}, O_edges={audit['o_fixed_edges_19']}, "
                f"S={audit['s_reserved_19']}, S_edges={audit['s_fixed_edges_19']}) | "
                f"partition={fixed_partition_meta}"
            )
        except Exception:
            pass
            
        print(f"[Layer0] 完成 - 总SU={int(H_corrected.sum().item())}")
        
        return H_corrected
    
    def _correct_halogen_X(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        修正X元素（32号）
        直接调整到目标X数量
        """
        target_X = int(E_target[5].item())
        current_X = int(H[32].item())

        if current_X == target_X:
            return H

        H_new = H.clone()
        H_new[32] = target_X
        return H_new
    
    def _correct_sulfur_S(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        修正S元素（30, 31号）
        - 总数 = 目标S
        - 优先级：31 > 30
        - 目标比例：30:31 = 0.4:0.6
        """
        target_S = int(E_target[4].item())
        m = int(H[30].item())  # 30号当前数量
        n = int(H[31].item())  # 31号当前数量
        current_S = m + n

        if current_S == target_S:
            return H
        
        H_new = H.clone()
        
        # 目标分布：30:31 = 0.4:0.6
        target_30 = target_S * 0.4
        target_31 = target_S * 0.6
        
        if current_S < target_S:
            # 需要补充
            diff = target_S - current_S
            
            for _ in range(diff):
                # 计算当前偏差
                delta_30 = m - target_30
                delta_31 = n - target_31
                
                # 补充负值最大的；偏差相同时优先补充优先级高的（31 > 30）
                if delta_30 < delta_31:
                    m += 1
                elif delta_31 < delta_30:
                    n += 1
                else:  # 偏差相同，优先补充31号
                    n += 1
        else:
            # 需要删除
            diff = current_S - target_S
            
            for _ in range(diff):
                # 计算当前偏差
                delta_30 = m - target_30
                delta_31 = n - target_31
                
                # 删除正值最大的；偏差相同时优先删除优先级低的（30 < 31）
                if delta_30 > delta_31 and m > 0:
                    m -= 1
                elif delta_31 > delta_30 and n > 0:
                    n -= 1
                elif delta_30 == delta_31:  # 偏差相同，优先删除30号
                    if m > 0:
                        m -= 1
                    elif n > 0:
                        n -= 1
                elif m > 0:  # 兜底
                    m -= 1
                elif n > 0:
                    n -= 1
        
        H_new[30] = m
        H_new[31] = n
        return H_new
    
    def _correct_nitrogen_N(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        修正N元素（0, 4, 26, 27号）
        """
        target_N = int(E_target[3].item())
        x = int(H[0].item())   
        y = int(H[4].item())   
        z = int(H[26].item())  
        w = int(H[27].item())  
        current_N = x + y + z + w

        if current_N == target_N:
            return H
        
        H_new = H.clone()
        
        # 目标分布：0:4:26:27 = 0.1:0.05:0.45:0.4
        target_0 = target_N * 0.1
        target_4 = target_N * 0.05
        target_26 = target_N * 0.45
        target_27 = target_N * 0.4
        
        if current_N < target_N:
            # 需要补充
            diff = target_N - current_N
            
            for _ in range(diff):
                # 计算当前偏差
                delta_0 = x - target_0
                delta_4 = y - target_4
                delta_26 = z - target_26
                delta_27 = w - target_27

                priority = {26: -4, 27: -3, 0: -2, 4: -1}
                deltas = [(delta_0, priority[0], 0, '0号'), 
                          (delta_4, priority[4], 4, '4号'), 
                          (delta_26, priority[26], 26, '26号'), 
                          (delta_27, priority[27], 27, '27号')]
                _, _, su_idx, _ = min(deltas, key=lambda t: (t[0], t[1]))
                
                # 补充该结构单元
                if su_idx == 0:
                    x += 1
                elif su_idx == 4:
                    y += 1
                elif su_idx == 26:
                    z += 1
                else:  # 27
                    w += 1
        else:
            # 需要删除
            diff = current_N - target_N
            
            for _ in range(diff):
                # 计算当前偏差
                delta_0 = x - target_0
                delta_4 = y - target_4
                delta_26 = z - target_26
                delta_27 = w - target_27

                priority = {26: 4, 27: 3, 0: 2, 4: 1}
                candidates = []
                if x > 0:
                    candidates.append((delta_0, -priority[0], 0, '0号'))
                if y > 0:
                    candidates.append((delta_4, -priority[4], 4, '4号'))
                if z > 0:
                    candidates.append((delta_26, -priority[26], 26, '26号'))
                if w > 0:
                    candidates.append((delta_27, -priority[27], 27, '27号'))
                
                if not candidates:
                    break
                
                _, _, su_idx, _ = max(candidates, key=lambda t: (t[0], t[1]))
                
                # 删除该结构单元
                if su_idx == 0:
                    x -= 1
                elif su_idx == 4:
                    y -= 1
                elif su_idx == 26:
                    z -= 1
                else:  # 27
                    w -= 1
                        
        H_new[0] = x
        H_new[4] = y
        H_new[26] = z
        H_new[27] = w
              
        return H_new
    
    def _correct_carbonyl_distribution(self, H: torch.Tensor, S_target: torch.Tensor,
                                       E_target: torch.Tensor) -> torch.Tensor:
        """
        修正C=O羰基分布（1, 2, 3号）
        """
        self._o_cap_triggered = False
        budgets = self._estimate_region_carbon_budgets(S_target, E_target)
        carbonyl_C = int(round(float(budgets['carbonyl_C'])))
        
        # 0号由N修正固定，保持不变
        n_0 = int(H[0].item())
        
        # 需要修正的羰基结构数量
        W = carbonyl_C - n_0
        W = max(0, W)  # 防止负数

        H_new = H.clone()
        
        # 当前1、2、3号数量
        m = int(H[1].item())  # 1号 (COOH)
        n = int(H[2].item())  # 2号 (-COO-)
        p = int(H[3].item())  # 3号 (-C=O-)
        
        current_total = m + n + p
        
        # 目标分布：1:2:3 = 0.35:0.25:0.4
        target_1 = W * 0.35
        target_2 = W * 0.25
        target_3 = W * 0.4
        
        if W == 0:
            m = 0
            n = 0
            p = 0
        elif current_total < W:
            # 需要补充
            diff = W - current_total
            
            for _ in range(diff):
                delta_1 = m - target_1
                delta_2 = n - target_2
                delta_3 = p - target_3
                
                # 补充负值最大的；偏差相同时优先补充优先级高的（3 > 1 > 2）
                priority = {3: -3, 1: -2, 2: -1}
                candidates = [
                    (delta_1, priority[1], 1, '1号'),
                    (delta_2, priority[2], 2, '2号'),
                    (delta_3, priority[3], 3, '3号')
                ]
                _, _, su_idx, _ = min(candidates, key=lambda t: (t[0], t[1]))
                
                if su_idx == 1:
                    m += 1
                elif su_idx == 2:
                    n += 1
                else:  # 3
                    p += 1
        elif current_total > W:
            # 需要删除
            diff = current_total - W
            
            for _ in range(diff):
                delta_1 = m - target_1
                delta_2 = n - target_2
                delta_3 = p - target_3
                
                # 删除正值最大的；偏差相同时优先删除优先级低的（2 < 1 < 3）
                priority = {3: 3, 1: 2, 2: 1}
                candidates = []
                if m > 0:
                    candidates.append((delta_1, -priority[1], 1, '1号'))
                if n > 0:
                    candidates.append((delta_2, -priority[2], 2, '2号'))
                if p > 0:
                    candidates.append((delta_3, -priority[3], 3, '3号'))
                
                if not candidates:
                    break
                
                _, _, su_idx, _ = max(candidates, key=lambda t: (t[0], t[1]))
                
                if su_idx == 1:
                    m -= 1
                elif su_idx == 2:
                    n -= 1
                else:  # 3
                    p -= 1
        
        H_new[1] = m
        H_new[2] = n
        H_new[3] = p

        target_O = int(E_target[2].item())
        used_O_03 = int(n_0 + 2 * m + 2 * n + p)
        if used_O_03 > target_O:

            self._o_cap_triggered = True
            Y = int(target_O - n_0)
            Y = max(0, Y)

            W_total = int(carbonyl_C)
            target_o1 = float(0.35 * W_total)
            target_o2 = float(0.25 * W_total)
            target_o3 = float(0.4 * W_total)

            delete_priority = {1: 0, 2: 1, 3: 2}
            max_iters = int(max(100, used_O_03 - target_O + 50))
            iters = 0
            while (2 * m + 2 * n + p) > Y and iters < max_iters:
                iters += 1
                cur_o = int(2 * m + 2 * n + p)
                need_remove = int(cur_o - Y)

                d1 = float(2 * m) - target_o1
                d2 = float(2 * n) - target_o2
                d3 = float(p) - target_o3

                candidates = []
                if m > 0 and (cur_o - 2) >= Y:
                    candidates.append((d1, delete_priority[1], 1))
                if n > 0 and (cur_o - 2) >= Y:
                    candidates.append((d2, delete_priority[2], 2))
                if p > 0 and (cur_o - 1) >= Y:
                    candidates.append((d3, delete_priority[3], 3))

                if not candidates:
                    break

                if need_remove == 1 and p > 0 and (cur_o - 1) >= Y:
                    su_idx = 3
                else:
                    _, _, su_idx = max(candidates, key=lambda t: (t[0], t[1]))

                if su_idx == 1 and m > 0 and (cur_o - 2) >= Y:
                    m -= 1
                elif su_idx == 2 and n > 0 and (cur_o - 2) >= Y:
                    n -= 1
                elif su_idx == 3 and p > 0 and (cur_o - 1) >= Y:
                    p -= 1
                else:
                    break

            H_new[1] = m
            H_new[2] = n
            H_new[3] = p
            used_O_03 = int(n_0 + 2 * m + 2 * n + p)

        return H_new
    
    def _oxygen_2829_ratio_from_spectrum(self,
                                         S_target: Optional[torch.Tensor],
                                         oxygen_atoms: int,
                                         n_2: int = 0) -> Tuple[float, float, Dict[str, float]]:
        """
        Estimate the initial 28/29 split from mode-specific oxygenated-carbon
        evidence. 28 contributes one O edge, while 29 contributes two; the
        90-100 ppm SU19 double/anomeric band therefore increases the 29 prior.
        """
        W = max(0, int(oxygen_atoms))
        if int(W) <= 0:
            return 1.0, 0.0, {'ratio_source': 'empty_oxygen'}

        area_single = (
            self._window_area_from_spectrum(S_target, 50.0, 60.0)
            + self._window_area_from_spectrum(S_target, 60.0, 70.0)
            + self._window_area_from_spectrum(S_target, 70.0, 90.0)
        )
        area_double = self._window_area_from_spectrum(S_target, 90.0, 100.0)
        area_aromatic_o = self._window_area_from_spectrum(S_target, 145.0, 165.0)
        total = float(area_single + area_double + area_aromatic_o)

        if total <= 1e-8:
            ratio_29 = 0.45
            ratio_source = 'fallback_55_45'
        else:
            # Single oxygenated aliphatic carbons can be alcohol/ether; double
            # and anomeric-like carbons need more two-port oxygen. Aromatic O
            # substitution sits between these two extremes.
            ratio_29 = (
                0.35 * float(area_single)
                + 0.75 * float(area_double)
                + 0.50 * float(area_aromatic_o)
            ) / float(total)
            ratio_29 = max(0.15, min(0.75, float(ratio_29)))
            ratio_source = 'su19_mode_weighted_28_29'
        ratio_28 = 1.0 - float(ratio_29)
        target_29 = int(round(float(W) * float(ratio_29)))
        target_29 = max(0, min(int(W), int(target_29)))
        target_28 = int(W) - int(target_29)
        return float(ratio_28), float(ratio_29), {
            'ratio_source': str(ratio_source),
            'oxygen_atoms': int(W),
            'n2_ether_edges': int(n_2),
            'area_su19_single_50_90': float(area_single),
            'area_su19_double_90_100': float(area_double),
            'area_su5_145_165': float(area_aromatic_o),
            'target_28': int(target_28),
            'target_29': int(target_29),
        }

    def _correct_oxygen_O(self,
                          H: torch.Tensor,
                          E_target: torch.Tensor,
                          S_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        修正O元素（28, 29号）
        """
        target_O = int(E_target[2].item())
        
        # 0,1,2,3号已被N修正和羰基修正处理
        n_0 = int(H[0].item())   # 0号 (1个O)
        n_1 = int(H[1].item())   # 1号 (2个O)
        n_2 = int(H[2].item())   # 2号 (2个O)
        n_3 = int(H[3].item())   # 3号 (1个O)
        
        # 0-3号已贡献的O
        used_O = n_0 * 1 + n_1 * 2 + n_2 * 2 + n_3 * 1
        
        # 需要修正的O数量
        W = target_O - used_O
        H_new = H.clone()
        if int(W) <= 0:
            H_new[28] = 0
            H_new[29] = 0
            return H_new
        
        W = max(0, W)  # 防止负数
        
        # 当前28、29号数量
        m = int(H[28].item())  # 28号 (OH)
        n = int(H[29].item())  # 29号 (-O-)
        
        current_total = m + n
        
        ratio_28, ratio_29, _ratio_meta = self._oxygen_2829_ratio_from_spectrum(
            S_target if S_target is not None else getattr(self, '_current_S_target', None),
            int(W),
            n_2=int(n_2),
        )
        target_28 = W * float(ratio_28)
        target_29 = W * float(ratio_29)
        
        if current_total < W:
            # 需要补充
            diff = W - current_total
            
            for _ in range(diff):
                delta_28 = m - target_28
                delta_29 = n - target_29
                
                # 补充负值最大的；偏差相同时优先补充优先级高的（28 > 29）
                if delta_28 < delta_29:
                    m += 1
                elif delta_29 < delta_28:
                    n += 1
                else:  # 偏差相同，优先补充28号
                    m += 1
        elif current_total > W:
            # 需要删除
            diff = current_total - W
            
            for _ in range(diff):
                delta_28 = m - target_28
                delta_29 = n - target_29
                
                # 删除正值最大的；偏差相同时优先删除优先级低的（29 < 28）
                if delta_28 > delta_29 and m > 0:
                    m -= 1
                elif delta_29 > delta_28 and n > 0:
                    n -= 1
                elif delta_28 == delta_29: 
                    if n > 0:
                        n -= 1
                    elif m > 0:
                        m -= 1
                elif m > 0:
                    m -= 1
                elif n > 0:
                    n -= 1
        else:
            target_28_i = int(round(float(target_28)))
            target_28_i = max(0, min(int(W), int(target_28_i)))
            target_29_i = int(W) - int(target_28_i)
            while int(m) > int(target_28_i) and int(n) < int(target_29_i):
                m -= 1
                n += 1
            while int(n) > int(target_29_i) and int(m) < int(target_28_i):
                n -= 1
                m += 1
        H_new[28] = m
        H_new[29] = n
        
        return H_new

    def _rebalance_oxygen_linked_units_after_carbonyl_adjust(
        self,
        H: torch.Tensor,
        E_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        在 Block A 调整 1/2/3 之后做最小氧守恒修正。

        逻辑：
        1. 若 0/1/2/3 自身已经超出 O 预算，则优先做 2->3、1->3 转换，
           每次减少 1 个 O，但保持羰基碳单元总数不变。
        2. 重算羰基固定连接 9。
        3. 再重算 28/29。
        4. 最后重算 5/19 与 7/19。
        """
        H_work = torch.clamp(H, min=0).long().clone()
        target_O = int(E_target[2].item())
        n0 = int(H_work[0].item())
        n1 = int(H_work[1].item())
        n2 = int(H_work[2].item())
        n3 = int(H_work[3].item())

        carbonyl_moves: List[Dict[str, int]] = []
        used_O_03 = int(n0 + 2 * n1 + 2 * n2 + n3)
        excess = int(max(0, used_O_03 - target_O))
        while excess > 0:
            if int(n2) >= int(n1) and int(n2) > 0:
                n2 -= 1
                n3 += 1
                carbonyl_moves.append({'from': 2, 'to': 3})
                excess -= 1
                continue
            if int(n1) > 0:
                n1 -= 1
                n3 += 1
                carbonyl_moves.append({'from': 1, 'to': 3})
                excess -= 1
                continue
            break

        H_work[1] = int(n1)
        H_work[2] = int(n2)
        H_work[3] = int(n3)

        H_before_carbonyl_anchor = H_work.clone()
        H_work = self._correct_carbonyl_connection(H_work)
        carbonyl_anchor_delta = int(H_work[9].item()) - int(H_before_carbonyl_anchor[9].item())

        H_work = self._correct_oxygen_O(H_work, E_target, S_target=getattr(self, '_current_S_target', None))
        S_hint = getattr(self, '_current_S_target', None)
        H_work, ether_meta = self._correct_ether_connection(H_work, S_target=S_hint)
        H_work, thio_meta = self._correct_thioether_connection(
            H_work,
            int((ether_meta or {}).get('o_base_19', 0)),
            S_target=S_hint,
        )
        special_degree_meta = self._allocate_special_degree_meta(
            H_work,
            o_base_19=int((ether_meta or {}).get('o_base_19', 0)),
            s_reserved_19=int((thio_meta or {}).get('s_reserved_19', 0)),
            o_fixed_edges_19=int((ether_meta or {}).get('o_fixed_edges_19', (ether_meta or {}).get('o_base_19', 0))),
            s_fixed_edges_19=int((thio_meta or {}).get('s_fixed_edges_19', (thio_meta or {}).get('s_reserved_19', 0))),
            n_fixed_edges_20=int(H_work[20].item()) if int(H_work.numel()) > 20 else 0,
            x_fixed_edges_21=int(H_work[21].item()) if int(H_work.numel()) > 21 else 0,
            S_target=S_hint,
        )
        fixed_partition_meta = self._record_fixed_partition_meta(
            H_work,
            ether_meta=ether_meta,
            thio_meta=thio_meta,
            special_degree_meta=special_degree_meta,
            special_anchor_mode_meta=getattr(self, 'special_anchor_mode_meta', {}) or {},
        )

        return H_work, {
            'carbonyl_o_repair_moves': list(carbonyl_moves),
            'carbonyl_anchor_delta': {'9': int(carbonyl_anchor_delta)},
            'ether_meta': dict(ether_meta or {}),
            'thio_meta': dict(thio_meta or {}),
            'fixed_partition_meta': dict(fixed_partition_meta or {}),
        }
    
    def _correct_carbonyl_connection(self, H: torch.Tensor) -> torch.Tensor:
        """
        修正C=O连接（9号芳香羰基取代碳）
        """
        n_0 = int(H[0].item())
        n_1 = int(H[1].item())
        n_2 = int(H[2].item())
        n_3 = int(H[3].item())
        
        # 计算C=O总连接量
        W = n_0 * 1 + n_1 * 1 + n_2 * 1 + n_3 * 2
        target_9 = int(round(float(W) * 0.50))

        H_new = H.clone()
        H_new[9] = target_9
        return H_new
    
    @staticmethod
    def _window_area_from_spectrum(S_target: Optional[torch.Tensor],
                                   lo_ppm: float,
                                   hi_ppm: float,
                                   ppm_step: float = 0.1) -> float:
        if S_target is None:
            return 0.0
        try:
            spec = S_target.detach().cpu().flatten().float()
        except Exception:
            return 0.0
        if int(spec.numel()) <= 0:
            return 0.0
        lo_i = max(0, int(math.floor(float(lo_ppm) / float(ppm_step))))
        hi_i = min(int(spec.numel()), int(math.ceil(float(hi_ppm) / float(ppm_step))))
        if hi_i <= lo_i:
            return 0.0
        return float(spec[lo_i:hi_i].sum().item()) * float(ppm_step)

    def _correct_ether_connection(self,
                                  H: torch.Tensor,
                                  S_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        修正-O-连接（5号芳香醚取代碳、19号醚接脂肪碳）
        
        Returns:
            Tuple[torch.Tensor, Dict[str, int]]: (修正后的直方图, 元数据字典)
        """
        n_2 = int(H[2].item())
        n_28 = int(H[28].item())
        n_29 = int(H[29].item())
        
        # 计算-O-总连接量
        W_ether = n_2 * 1 + n_28 * 1 + n_29 * 2
        
        H_new = H.clone()
        
        m_target, n_19_target, o_fixed_edges_19, ratio_meta = self._split_aromatic_special_fixed_edges(
            int(W_ether),
            aromatic_su=5,
            special_su=19,
            aromatic_window=(145.0, 165.0),
            S_target=S_target,
            fallback_special_ratio=0.35,
        )
        
        H_new[5] = int(m_target)
        H_new[19] = int(n_19_target)  
        
        # 返回元数据供后续S连接修正使用
        meta = {
            'o_base_19': int(n_19_target),
            'o_fixed_edges_19': int(o_fixed_edges_19),
            **dict(ratio_meta),
        }
        
        return H_new, meta
    
    def _correct_thioether_connection(self,
                                      H: torch.Tensor,
                                      o_base_19: int = 0,
                                      S_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        修正-S-连接（7号芳香硫醚取代碳、19号硫醚接脂肪碳）
        
        Args:
            H: SU直方图
            o_base_19: O专用的19号基准（从_correct_ether_connection获取）
            
        Returns:
            Tuple[torch.Tensor, Dict[str, int]]: (修正后的直方图, 元数据字典)
        """
        n_31 = int(H[31].item())
        
        # 计算-S-总连接量
        W_thioether = n_31 * 2
        
        H_new = H.clone()

        m, n_19_thioether, s_fixed_edges_19, split_meta = self._split_aromatic_special_fixed_edges(
            int(W_thioether),
            aromatic_su=7,
            special_su=19,
            aromatic_window=(145.0, 160.0),
            S_target=S_target,
            fallback_special_ratio=0.60,
        )
        
        H_new[7] = m
        H_new[19] = o_base_19 + n_19_thioether
  
        # 返回元数据
        meta = {
            'o_base_19': o_base_19,
            's_reserved_19': n_19_thioether,
            's_fixed_edges_19': int(s_fixed_edges_19),
            **dict(split_meta),
        }
        
        return H_new, meta
    
    def _correct_amine_connection(self,
                                  H: torch.Tensor,
                                  S_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        修正-NH-连接（6号芳香氨基取代碳、20号氨基接脂肪碳）
        """
        n_0 = int(H[0].item())
        n_27 = int(H[27].item())
        
        # 计算-NH-总连接量
        W = n_0 * 1 + n_27 * 2
        
        H_new = H.clone()
        m, n, fixed_edges_20, split_meta = self._split_aromatic_special_fixed_edges(
            int(W),
            aromatic_su=6,
            special_su=20,
            aromatic_window=(135.0, 155.0),
            S_target=S_target,
            fallback_special_ratio=0.40,
        )
        
        H_new[6] = m
        H_new[20] = n
        
        return H_new, {
            'n20_total': int(n),
            'n20_fixed_edges': int(fixed_edges_20),
            **dict(split_meta),
        }
    
    def _correct_halogen_connection(self,
                                    H: torch.Tensor,
                                    S_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        修正-X连接（8号芳香卤取代碳、21号卤接脂肪碳）
        """
        n_32 = int(H[32].item())
        
        # 计算-X总连接量
        W = n_32 * 1
        
        H_new = H.clone()
        m, n, fixed_edges_21, split_meta = self._split_aromatic_special_fixed_edges(
            int(W),
            aromatic_su=8,
            special_su=21,
            aromatic_window=(126.0, 135.0),
            S_target=S_target,
            fallback_special_ratio=0.40,
        )
        
        H_new[8] = m
        H_new[21] = n
        return H_new, {
            'n21_total': int(n),
            'n21_fixed_edges': int(fixed_edges_21),
            **dict(split_meta),
        }
    

    def _correct_aliphatic_carbons(self, H: torch.Tensor, S_target: torch.Tensor,
                                   E_target: torch.Tensor) -> torch.Tensor:
        """
        修正脂肪碳结构（22, 23, 24, 25号）
        """
        budgets = self._estimate_region_carbon_budgets(S_target, E_target)
        M_float = float(budgets.get('ordinary_aliphatic_C', budgets.get('aliphatic_C', 0.0)))
        M = int(round(M_float))
        
        if M == 0:
            return H
        
        H_new = H.clone()
        
        # M is now the ordinary 0-50 ppm aliphatic budget, so SU19/20/21
        # are handled by the oxygenated-aliphatic/hetero-anchor logic instead
        # of being subtracted from SU23.
        target_22 = int(round(0.20 * M_float))
        target_24 = int(round(0.10 * M_float))
        target_25 = int(round(0.02 * M_float))
        target_23 = int(round(0.68 * M_float))
        target_23 = max(0, target_23)
        
        H_new[22] = target_22
        H_new[23] = target_23
        H_new[24] = target_24
        H_new[25] = target_25 
        
        return H_new
    
    def _correct_unsaturated_carbons(self, H: torch.Tensor, S_target: torch.Tensor,
                                     E_target: torch.Tensor) -> torch.Tensor:
        """
        修正非饱和结构（14, 15, 16, 17, 18号）
        """
        budgets = self._estimate_region_carbon_budgets(S_target, E_target)
        aromatic_C = float(budgets['aromatic_C'])

        # 新规则：
        # W = 0.05 * aromatic_C，W 取最近偶数
        W = max(0, self._nearest_even_int(0.05 * float(aromatic_C)))
        
        if W == 0:
            return H
        
        H_new = H.clone()

        # 双键:三键 = 0.8:0.2，并要求两端都为偶数。
        double_bond = self._nearest_even_int(0.8 * float(W))
        double_bond = max(0, min(int(W), int(double_bond)))
        triple_bond = self._nearest_even_int(0.2 * float(W))
        triple_bond = max(0, int(triple_bond))

        # 若独立取偶数后不再守恒，则优先保持总W守恒。
        if int(double_bond + triple_bond) != int(W):
            triple_bond = max(0, int(W) - int(double_bond))
        if int(triple_bond) % 2 != 0:
            triple_bond = max(0, int(triple_bond) - 1)
            double_bond = max(0, int(W) - int(triple_bond))

        target_14, target_15, target_16 = self._allocate_ratio_counts(
            int(double_bond), (0.1, 0.65, 0.25)
        )

        # q:r = 0.5:0.5
        target_17 = int(triple_bond) // 2
        target_18 = int(triple_bond) // 2
        
        H_new[14] = target_14
        H_new[15] = target_15
        H_new[16] = target_16
        H_new[17] = target_17
        H_new[18] = target_18
        
        return H_new
    
    def _correct_aromatic_carbons(self, H: torch.Tensor, S_target: torch.Tensor,
                                  E_target: torch.Tensor) -> torch.Tensor:
        """
        修正芳香结构（10, 11, 12, 13号）
        """
        budgets = self._estimate_region_carbon_budgets(S_target, E_target)
        total_C = float(budgets['N'])
        xN = float(budgets['xN'])
        yN = float(budgets['yN'])
        aromatic_C = float(budgets['aromatic_C'])

        # 新规则：
        # W = 0.95 * aromatic_C - 4号
        fa = float(yN + 0.1 * xN) / max(1.0, float(total_C))
        n_4 = int(H[4].item())

        W_float = float(0.95 * float(aromatic_C) - float(n_4))
        W = max(0, int(round(W_float)))
        
        if W == 0:
            return H
        
        H_new = H.clone()
        
        n_5 = int(H[5].item())
        n_6 = int(H[6].item())
        n_7 = int(H[7].item())
        n_8 = int(H[8].item())
        n_9 = int(H[9].item())

        if fa <= 0.5:
            frac_10, frac_11, frac_12, frac_13 = 0.10, 0.22, 0.15, 0.53
        elif fa <= 0.6:
            frac_10, frac_11, frac_12, frac_13 = 0.10, 0.22, 0.18, 0.50
        elif fa <= 0.7:
            frac_10, frac_11, frac_12, frac_13 = 0.10, 0.20, 0.20, 0.50
        elif fa <= 0.75:
            frac_10, frac_11, frac_12, frac_13 = 0.09, 0.20, 0.225, 0.485
        elif fa <= 0.8:
            frac_10, frac_11, frac_12, frac_13 = 0.085, 0.19, 0.255, 0.47
        elif fa <= 0.85:
            frac_10, frac_11, frac_12, frac_13 = 0.08, 0.175, 0.285, 0.46
        elif fa <= 0.9:
            frac_10, frac_11, frac_12, frac_13 = 0.07, 0.17, 0.305, 0.445
        else:
            frac_10, frac_11, frac_12, frac_13 = 0.07, 0.165, 0.34, 0.425

        existing_5_9 = int(n_5 + n_6 + n_7 + n_8 + n_9)
        target_10 = int(round(float(frac_10) * float(W)))
        target_12 = int(round(float(frac_12) * float(W)))
        target_13 = int(round(float(frac_13) * float(W)))
        target_11 = int(round(float(frac_11) * float(W) - float(existing_5_9)))
        target_11 = max(0, target_11)

        aromatic_excess = int(existing_5_9 + target_10 + target_11 + target_12 + target_13 - W)
        if aromatic_excess > 0:
            reducible = min(int(target_13), int(aromatic_excess))
            target_13 -= reducible
            aromatic_excess -= reducible
        if aromatic_excess > 0:
            reducible = min(int(target_12), int(aromatic_excess))
            target_12 -= reducible
            aromatic_excess -= reducible
        if aromatic_excess > 0:
            reducible = min(int(target_10), int(aromatic_excess))
            target_10 -= reducible
            aromatic_excess -= reducible
        if aromatic_excess > 0:
            reducible = min(int(target_11), int(aromatic_excess))
            target_11 -= reducible
            aromatic_excess -= reducible

        aromatic_deficit = int(W - existing_5_9 - target_10 - target_11 - target_12 - target_13)
        if aromatic_deficit > 0:
            target_11 += int(aromatic_deficit)

        # 约束：10号数量必须为偶数；若为奇数，转一个到11号
        if int(target_10) % 2 != 0 and int(target_10) > 0:
            target_10 -= 1
            target_11 += 1
        
        H_new[10] = target_10
        H_new[11] = target_11
        H_new[12] = target_12
        H_new[13] = target_13
        
        return H_new

    def _adjust_hydrogen(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        H元素调整（三区域调整）
        """
        # TODO(common): 这组 H 平衡互转规则和 Layer4 的 H 校正存在重复语义，后续统一。
        # 将H转移到CPU上以避免频繁的GPU同步
        H_cpu = H.cpu()
        E_SU_cpu = self.E_SU.cpu()
        
        E_current = get_effective_hist_element_vector(
            H_cpu,
            special_degree_meta=getattr(self, 'special_degree_meta', {}) or None,
            E_SU_tensor=E_SU_cpu,
            device=torch.device('cpu'),
        )
        current_H = E_current[1].item()
        target_H = E_target[1].item()
        
        delta_H = current_H - target_H
        rel_error = abs(delta_H) / max(1.0, float(target_H))

        if rel_error < 0.08:
            return H
        
        W = abs(current_H - 1.04 * target_H) if delta_H > 0 else abs(current_H - 0.96 * target_H)
        W = int(W)
        if W <= 0:
            return H
        
        X = int(W * 0.4)
        Y = int(W * 0.3)
        Z = int(W * 0.3)

        H_new = H_cpu.clone()
        
        if delta_H > 0:
            H_new = self._reduce_hydrogen_aromatic(H_new, X)
            H_new = self._reduce_hydrogen_aliphatic(H_new, Y)
            H_new = self._reduce_hydrogen_unsaturated(H_new, Z)
        else:
            H_new = self._increase_hydrogen_aromatic(H_new, X)
            H_new = self._increase_hydrogen_aliphatic(H_new, Y)
            H_new = self._increase_hydrogen_unsaturated(H_new, Z)
            
        # 限制25号的数量不超过3%的脂肪碳总量！22号的数量不少于23号的10%！
        aliphatic_total = sum(H_new[i].item() for i in [19, 20, 21, 22, 23, 24, 25])
        max_25 = int(0.03 * aliphatic_total)
        if H_new[25] > max_25:
            diff = H_new[25] - max_25
            H_new[25] = max_25
            H_new[23] += diff
            
        min_22 = int(0.10 * H_new[23].item())
        if H_new[22] < min_22:
            diff = min_22 - H_new[22]
            H_new[22] = min_22
            H_new[23] -= diff
            if H_new[23] < 0:
                H_new[23] = 0

        # 返回时移回原来的设备
        return H_new.to(H.device)

    def _reduce_hydrogen_aromatic(self, H: torch.Tensor, X: int) -> torch.Tensor:
        """每次减少一个13号，按预设 cycle 在 12/11/10 间轮动补偿。"""
        if X <= 0: return H
        H_new = H.clone()
        reduced = 0
        cycle = [12, 12, 11, 12, 12, 11, 10]
        c_idx = 0

        while reduced < X:
            if H_new[13] > 0:
                H_new[13] -= 1
                H_new[cycle[c_idx]] += 1
                reduced += 1
                c_idx = (c_idx + 1) % len(cycle)
            else:
                break
        return H_new

    def _reduce_hydrogen_aliphatic(self, H: torch.Tensor, Y: int) -> torch.Tensor:
        """第一轮减22增23/23/23/24，第二轮减23增24"""
        if Y <= 0: return H
        H_new = H.clone()
        reduced = 0
        
        step = 0
        stuck = 0
        
        while reduced < Y:
            if step < 4:
                # 第一轮: 减22, 增23/23/23/24
                if H_new[22] > 0:
                    tgt = 24 if step == 3 else 23
                    H_new[22] -= 1
                    H_new[tgt] += 1
                    diff = 3 - (1 if tgt == 24 else 2)
                    reduced += diff
                    stuck = 0
                else:
                    stuck += 1
                step += 1
            else:
                # 第二轮: 减23, 增24
                if H_new[23] > 0:
                    H_new[23] -= 1
                    H_new[24] += 1
                    reduced += 1
                    stuck = 0
                else:
                    stuck += 1
                step = 0
                
            if stuck > 5:
                break
        return H_new
    
    def _reduce_hydrogen_unsaturated(self, H: torch.Tensor, Z: int) -> torch.Tensor:
        """第一轮减16增15/14，第二轮减15增14，必须保证15 > 14"""
        if Z <= 0: return H
        H_new = H.clone()
        reduced = 0
        
        step = 0
        stuck = 0
        
        while reduced < Z:
            if step == 0:
                # 第一轮, 步1: -16, +15
                if H_new[16] > 0:
                    H_new[16] -= 1
                    H_new[15] += 1
                    reduced += 1
                    stuck = 0
                else:
                    stuck += 1
                step = 1
            elif step == 1:
                # 第一轮, 步2: -16, +14 (需保证 15 > 14)
                if H_new[16] > 0 and H_new[15] > (H_new[14] + 1):
                    H_new[16] -= 1
                    H_new[14] += 1
                    reduced += 2
                    stuck = 0
                else:
                    stuck += 1
                step = 2
            else:
                # 第二轮: -15, +14 (需保证 15 > 14)
                if H_new[15] > 0 and (H_new[15] - 1) > (H_new[14] + 1):
                    H_new[15] -= 1
                    H_new[14] += 1
                    reduced += 1
                    stuck = 0
                else:
                    stuck += 1
                step = 0
                
            if stuck > 3:
                break
        return H_new

    def _increase_hydrogen_aromatic(self, H: torch.Tensor, X: int) -> torch.Tensor:
        """每次增加一个13号，按预设 cycle 在 11/10/12 间轮动扣减。"""
        if X <= 0: return H
        H_new = H.clone()
        increased = 0
        cycle = [11, 10, 12, 10]
        c_idx = 0
        stuck = 0
        
        while increased < X:
            tgt = cycle[c_idx]
            if H_new[tgt] > 0:
                H_new[tgt] -= 1
                H_new[13] += 1
                increased += 1
                stuck = 0
            else:
                stuck += 1
                if stuck >= 3:
                    break
            c_idx = (c_idx + 1) % len(cycle)
        return H_new

    def _increase_hydrogen_aliphatic(self, H: torch.Tensor, Y: int) -> torch.Tensor:
        """第一轮每次增加一个22号，轮流减少一个23/23/23/24/25；第二轮增加23，减少24"""
        if Y <= 0: return H
        H_new = H.clone()
        increased = 0
        
        r1_targets = [23, 23, 23, 24, 25]
        c_idx = 0
        stuck = 0
        
        while increased < Y:
            if c_idx < 5:
                tgt = r1_targets[c_idx]
                if H_new[tgt] > 0:
                    H_new[tgt] -= 1
                    H_new[22] += 1
                    diff = 3 - (2 if tgt == 23 else (1 if tgt == 24 else 0))
                    increased += diff
                    stuck = 0
                else:
                    stuck += 1
                c_idx += 1
            else:
                if H_new[24] > 0:
                    H_new[24] -= 1
                    H_new[23] += 1
                    increased += 1
                    stuck = 0
                else:
                    stuck += 1
                    
                if stuck > 6:
                    break
                c_idx = 0
        return H_new

    def _increase_hydrogen_unsaturated(self, H: torch.Tensor, Z: int) -> torch.Tensor:
        """第一轮每次增加16，减少15/14；第二轮增加15，减少14"""
        if Z <= 0: return H
        H_new = H.clone()
        increased = 0
        
        r1_targets = [15, 14]
        c_idx = 0
        stuck = 0
        
        while increased < Z:
            if c_idx < 2:
                tgt = r1_targets[c_idx]
                if H_new[tgt] > 0 and (H_new[16] + 1) <= (H_new[15] + H_new[14] - 1):
                    H_new[tgt] -= 1
                    H_new[16] += 1
                    diff = 2 - (1 if tgt == 15 else 0)
                    increased += diff
                    stuck = 0
                else:
                    stuck += 1
                c_idx += 1
            else:
                if H_new[14] > 0:
                    H_new[14] -= 1
                    H_new[15] += 1
                    increased += 1
                    stuck = 0
                else:
                    stuck += 1
                    
                if stuck > 4:
                    break
                c_idx = 0
        return H_new
    
