import copy
import random
import ast
import pandas as pd
from collections import Counter
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Callable, Set
from .RL_state import MCTSState, ConnectionGraph, compute_su_delta, compute_su_l1_delta
from .RL_allocator import STRUCTURAL_PLACEHOLDER_TO_23

# SU category tuples for pattern matching
ARO = tuple(range(5, 14))        # 5..13   aromatic
ALI = (19, 20, 21, 23, 24, 25)   # aliphatic (excl. terminal)
TERM = (22,)                      # terminal CH3

FIXED_HOP1_PORTS: Dict[int, List[Tuple[int, ...]]] = {
    0: [(9, 23, 24, 25, 22, 14, 15, 17), (6, 20)],
    1: [(9, 23, 24, 25, 19, 20, 21, 14, 15, 17)],
    2: [(9, 23, 24, 25, 22, 19, 20, 21, 14, 15, 17), (5, 19)],
    3: [(9, 23, 24, 25, 22, 19, 20, 21, 14, 15, 17), (9, 23, 24, 25, 19, 20, 21, 14, 15, 17)],
    4: [(23, 24, 25, 10)],
    27: [(6, 20), (6, 20)],
    28: [(5, 19)],
    29: [(5, 19), (5, 19)],
    31: [(7, 19), (7, 19)],
    32: [(8, 21)],
}


class StateSubstitutor:
    """
    Replaces generic placeholder SUs (11, 22, 23 etc.) with the original
    real SUs after the MCTS skeleton is built.

    Covers:
      - Carbonyls   : 0, 1, 2, 3
      - Oxygen      : 28 (degree-1), 29 (degree-2)
      - N-bridge    : 27
      - S-bridge    : 31
      - Halogen     : 32
      - Unsaturated : 14, 15, 16, 17, 18
      - N-heteroaro : 26 (pyridine / pyrrole, inside aromatic clusters)
      - S-heteroaro : 30 (thiophene, inside aromatic clusters)

    Args:
        original_su_counts: {su_type: count} from the original node CSV.
        nodes_csv:          Path to node CSV for 1-hop info extraction.
        nmr_eval_fn:        Optional callback  state -> float  used to
                            score trial replacements for 26/30 selection.
    """

    def __init__(self, original_su_counts: Dict[int, int],
                 nodes_csv: str = None,
                 nmr_eval_fn: Optional[Callable] = None,
                 verbose: bool = False):
        self.original_su_counts = copy.deepcopy(original_su_counts)
        self.nodes_csv = nodes_csv
        self.nmr_eval_fn = nmr_eval_fn
        self.verbose = bool(verbose)
        self.su_details = self._parse_nodes()
        self.randomize = False
        self.last_summary: Dict[str, Any] = {}

        # Remaining counts for every SU that needs substitution
        self.remaining_sus: Dict[int, int] = {}
        for su in [0, 1, 2, 3, 14, 15, 16, 17, 18,
                   26, 27, 28, 29, 30, 31, 32]:
            self.remaining_sus[su] = original_su_counts.get(su, 0)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ----------------------------------------------------------------
    # Node CSV parsing
    # ----------------------------------------------------------------
    def _parse_nodes(self) -> Dict[int, List[Dict]]:
        """Parse original node CSV for 1-hop info of specific SUs."""
        tracked = [0, 1, 2, 3, 14, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31, 32]
        su_details: Dict[int, List[Dict]] = {su: [] for su in tracked}
        if not self.nodes_csv:
            return su_details
        try:
            df = pd.read_csv(self.nodes_csv)
            for _, row in df.iterrows():
                su_type = int(row['su_type'])
                if su_type in su_details:
                    template = row.get('template_key', '')
                    if isinstance(template, str):
                        try:
                            parsed = ast.literal_eval(template)
                        except Exception:
                            parsed = None
                        if (isinstance(parsed, tuple) and len(parsed) >= 3
                                and int(parsed[0]) == su_type):
                            hop1_raw = parsed[1] if isinstance(parsed[1], (tuple, list)) else ()
                            hop2_raw = parsed[2] if isinstance(parsed[2], (tuple, list)) else ()
                            hop1 = [int(x) for x in hop1_raw]
                            hop2 = [int(x) for x in hop2_raw]
                            su_details[su_type].append({
                                'id': row['global_id'],
                                'hop1': hop1,
                                'hop2': hop2,
                            })
        except Exception as e:
            print(f"[StateSubstitutor] Warning parsing nodes: {e}")
        return su_details

    @staticmethod
    def _is_aromatic_su(su: int) -> bool:
        return 5 <= su <= 13

    @staticmethod
    def _pattern_token_for_endpoint(su: int):
        if 5 <= su <= 13:
            return ARO
        if su == 22:
            return TERM
        return ALI

    @staticmethod
    def _genericize_chain_neighbor(su: int) -> int:
        if su in STRUCTURAL_PLACEHOLDER_TO_23 or su == 23:
            return 23
        return su

    def _assign_fixed_ports(self, center_su: int, hop1: List[int]) -> Optional[List[int]]:
        """Assign original hop1 multiset to ordered port roles for fixed-pattern SUs."""
        port_sets = FIXED_HOP1_PORTS.get(int(center_su))
        if not port_sets:
            return None
        # 14/24/25 are already represented globally by the dedicated
        # branch/unsaturated placeholder pool built during allocation.
        # Re-introducing them here from local fixed-port matching would create
        # extra 14/24/25 nodes and break global SU conservation.
        values = [int(self._genericize_chain_neighbor(int(x))) for x in list(hop1 or [])]
        if len(values) != len(port_sets):
            return None

        used = [False] * len(values)
        ordered = sorted(range(len(port_sets)), key=lambda i: len(port_sets[i]))
        assignment: List[Optional[int]] = [None] * len(port_sets)

        def _dfs(pos: int) -> bool:
            if pos >= len(ordered):
                return True
            port_idx = ordered[pos]
            allowed = set(int(x) for x in port_sets[port_idx])
            for vi, val in enumerate(values):
                if used[vi] or int(val) not in allowed:
                    continue
                used[vi] = True
                assignment[port_idx] = int(val)
                if _dfs(pos + 1):
                    return True
                used[vi] = False
                assignment[port_idx] = None
            return False

        if not _dfs(0):
            return None
        return [int(x) for x in assignment if x is not None]

    def _build_fixed_specs(self, center_su: int, hop1: List[int]) -> List[Tuple[List[Any], List[int]]]:
        assigned = self._assign_fixed_ports(center_su, hop1)
        if not assigned:
            return []
        if len(assigned) == 1:
            endpoint = int(assigned[0])
            return self._build_single_terminal_specs(endpoint, int(center_su))
        if len(assigned) == 2:
            left, right = int(assigned[0]), int(assigned[1])
            return self._build_bridge_specs(left, int(center_su), right)
        return []

    @staticmethod
    def _score_exact_bridge_preference(path, start_idx: int, find_p, repl_p, is_rev: bool) -> int:
        score = 0
        end_idx = start_idx + len(find_p)
        if start_idx == 0:
            score += 40
        if end_idx == len(path):
            score += 40
        if len(path) == len(find_p):
            score += 120
        if len(find_p) == 3:
            center = path[start_idx + 1].su_type
            if center == 23:
                score += 20
        return score

    def _build_unsat_anchor_counts(self) -> Dict[int, Counter]:
        counts: Dict[int, Counter] = {su: Counter() for su in [0, 1, 2, 3]}
        for info in self.su_details.get(15, []):
            anchors = [h for h in info.get('hop1', []) if h in counts]
            if not anchors:
                continue
            anchor = anchors[0]
            hop1 = info.get('hop1', [])
            if 15 in hop1:
                counts[anchor]['15_15'] += 1
            elif 16 in hop1:
                counts[anchor]['15_16'] += 1
            elif 14 in hop1:
                counts[anchor]['15_14'] += 1
        for info in self.su_details.get(17, []):
            anchors = [h for h in info.get('hop1', []) if h in counts]
            if not anchors:
                continue
            anchor = anchors[0]
            if 18 in info.get('hop1', []):
                counts[anchor]['17_18'] += 1
        return counts

    def _pick_aromatic_endpoint(self, hop1: List[int], preferred: Tuple[int, ...]) -> int:
        for su in preferred:
            if su in hop1:
                return su
        aromatic = [su for su in hop1 if self._is_aromatic_su(su)]
        return aromatic[0] if aromatic else preferred[0]

    def _pick_aliphatic_endpoint(self, hop1: List[int], preferred: Tuple[int, ...], default: int) -> int:
        for su in preferred:
            if su in hop1:
                return su
        aliphatic = [su for su in hop1 if su in (19, 20, 21)]
        return aliphatic[0] if aliphatic else default

    def _apply_specs(self, paths, specs, score_fn=None) -> bool:
        matches = []
        for spec_idx, spec in enumerate(specs):
            find_p, repl_p = spec
            for path_idx, path in enumerate(paths):
                types = [n.su_type for n in path]
                for m in self._find_all_matches(types, find_p):
                    score = score_fn(path, m, find_p, repl_p, False) if score_fn else len(find_p)
                    matches.append((score, spec_idx, path_idx, m, False))
                rev = types[::-1]
                for m in self._find_all_matches(rev, find_p):
                    actual = len(types) - m - len(find_p)
                    score = score_fn(path, actual, find_p, repl_p, True) if score_fn else len(find_p)
                    matches.append((score, spec_idx, path_idx, actual, True))
        if not matches:
            return False
        matches.sort(key=lambda x: x[0], reverse=True)
        score, spec_idx, path_idx, start_idx, is_rev = matches[0]
        find_p, repl_p = specs[spec_idx]
        if is_rev:
            self._do_replace(paths[path_idx], start_idx, find_p[::-1], repl_p[::-1])
        else:
            self._do_replace(paths[path_idx], start_idx, find_p, repl_p)
        return True

    @staticmethod
    def _score_terminal_preference(path, start_idx: int, find_p, repl_p, is_rev: bool) -> int:
        score = len(find_p) * 10
        end_idx = start_idx + len(find_p)
        if end_idx == len(path):
            score += 10
        if start_idx == 0:
            score += 2
        return score

    def _iter_infos(self, su: int) -> List[Dict]:
        return list(self.su_details.get(su, []))

    def _consume_one(self, su: int, ok: bool):
        if ok:
            self.remaining_sus[su] = max(0, self.remaining_sus.get(su, 0) - 1)

    def _build_single_terminal_specs(self, endpoint: int, target_su: int) -> List[Tuple[List[Any], List[int]]]:
        return [([self._pattern_token_for_endpoint(endpoint), 22], [endpoint, target_su])]

    def _build_bridge_specs(self, left_endpoint: int, target_su: int, right_endpoint: int) -> List[Tuple[List[Any], List[int]]]:
        return [([
            self._pattern_token_for_endpoint(left_endpoint),
            23,
            self._pattern_token_for_endpoint(right_endpoint),
        ], [left_endpoint, target_su, right_endpoint])]

    def _build_bridge_to_terminal_specs(self, left_endpoint: int, target_su: int, terminal_endpoint: int = 22) -> List[Tuple[List[Any], List[int]]]:
        return [([
            self._pattern_token_for_endpoint(left_endpoint),
            23,
            22,
        ], [left_endpoint, target_su, terminal_endpoint])]

    def _build_unsat_bridge_specs(self, anchor_endpoint: int, target_su: int, unsat_pair: List[int]) -> List[Tuple[List[Any], List[int]]]:
        if unsat_pair[-1] == 22:
            return [([
                self._pattern_token_for_endpoint(anchor_endpoint),
                23,
                23,
                22,
            ], [anchor_endpoint, target_su, unsat_pair[0], unsat_pair[1]])]
        return [([
            23,
            23,
            23,
            self._pattern_token_for_endpoint(anchor_endpoint),
        ], [unsat_pair[0], unsat_pair[1], target_su, anchor_endpoint])]

    # ================================================================
    # Main entry
    # ================================================================
    def substitute_all(self, state: MCTSState,
                       randomize: bool = False) -> bool:
        """Run every substitution rule on *state* in-place.

        Execution order (CRITICAL):
        1. Unsaturated (14, 15, 16, 17, 18) - HIGHEST PRIORITY
        2. Carbonyls (0, 1, 2, 3)
        3. Heteroatom functional groups (27, 28, 29, 31, 32)
        4. Heteroaromatic rings (26, 30) - LOWEST PRIORITY
        """
        self.randomize = randomize
        before = copy.deepcopy(self.remaining_sus)
        self._unsat_anchor_counts = self._build_unsat_anchor_counts()
        paths = self._extract_paths(state.graph)

        self._log(f"\n[StateSubstitutor] Starting substitution")
        self._log(f"  Initial remaining: {dict(self.remaining_sus)}")

        # --- PRIORITY 1: Unsaturated structures (MUST BE FIRST) ---
        self._substitute_unsaturated(state)
        self._log(
            f"  After unsaturated: 14={self.remaining_sus.get(14, 0)}, "
            f"15={self.remaining_sus.get(15, 0)}, 16={self.remaining_sus.get(16, 0)}, "
            f"17={self.remaining_sus.get(17, 0)}, 18={self.remaining_sus.get(18, 0)}"
        )

        # --- PRIORITY 2: Chain-based substitutions ---
        self._substitute_carbonyls(paths)
        self._log(
            f"  After carbonyls: 0={self.remaining_sus.get(0, 0)}, "
            f"1={self.remaining_sus.get(1, 0)}, 2={self.remaining_sus.get(2, 0)}, "
            f"3={self.remaining_sus.get(3, 0)}"
        )

        self._substitute_oxygen(paths)
        self._substitute_nitrogen_bridge(paths)
        self._substitute_sulfur_bridge(paths)
        self._substitute_halogen(paths)
        self._log(
            f"  After heteroatoms: 27={self.remaining_sus.get(27, 0)}, "
            f"28={self.remaining_sus.get(28, 0)}, 29={self.remaining_sus.get(29, 0)}, "
            f"31={self.remaining_sus.get(31, 0)}, 32={self.remaining_sus.get(32, 0)}"
        )

        # --- PRIORITY 3: Cluster-internal heteroaromatic substitutions ---
        self._substitute_heteroaromatic_N(state)
        self._substitute_heteroaromatic_S(state)
        self._log(
            f"  After heteroaromatic: 26={self.remaining_sus.get(26, 0)}, "
            f"30={self.remaining_sus.get(30, 0)}"
        )

        applied = {
            su: before[su] - self.remaining_sus.get(su, 0)
            for su in before
            if before[su] - self.remaining_sus.get(su, 0) > 0
        }
        remaining = {
            su: cnt for su, cnt in self.remaining_sus.items()
            if cnt > 0
        }
        self.last_summary = {
            'applied': applied,
            'remaining': remaining,
            'complete': not remaining,
        }

        if remaining:
            self._log(f"  [WARNING] Incomplete substitution, remaining: {remaining}")
        else:
            self._log(f"  [SUCCESS] All substitutions completed")

        return bool(not remaining)

    # Backward-compatible alias
    def substitute_carbonyls(self, state: MCTSState,
                             randomize: bool = False) -> bool:
        return self.substitute_all(state, randomize)

    # ================================================================
    # Chain-based substitutions
    # ================================================================
    def _substitute_carbonyls(self, paths):
        """Carbonyl SUs 0, 1, 2, 3 driven by original node instances."""
        for info in self._iter_infos(3):
            if self.remaining_sus.get(3, 0) <= 0:
                break
            self._consume_one(3, self._substitute_su3_instance(paths, info))

        for info in self._iter_infos(2):
            if self.remaining_sus.get(2, 0) <= 0:
                break
            self._consume_one(2, self._substitute_su2_instance(paths, info))

        for info in self._iter_infos(0):
            if self.remaining_sus.get(0, 0) <= 0:
                break
            self._consume_one(0, self._substitute_su0_instance(paths, info))

        for info in self._iter_infos(1):
            if self.remaining_sus.get(1, 0) <= 0:
                break
            self._consume_one(1, self._substitute_su1_instance(paths, info))

    def _substitute_oxygen(self, paths):
        """Oxygen SUs: 28 (degree-1) and 29 (degree-2)."""
        for info in self._iter_infos(29):
            if self.remaining_sus.get(29, 0) <= 0:
                break
            hop1 = [h for h in info.get('hop1', []) if h != 29]
            fixed_specs = self._build_fixed_specs(29, hop1)
            ok = False
            if fixed_specs:
                ok = self._apply_specs(paths, fixed_specs, self._score_exact_bridge_preference)
            if not ok:
                left = self._pick_aromatic_endpoint(hop1, (5,)) if any(self._is_aromatic_su(h) for h in hop1) else self._pick_aliphatic_endpoint(hop1, (19,), 19)
                right = left
                if len(hop1) >= 2:
                    remaining = hop1.copy()
                    if left in remaining:
                        remaining.remove(left)
                    if any(self._is_aromatic_su(h) for h in remaining):
                        right = self._pick_aromatic_endpoint(remaining, (5,))
                    elif remaining:
                        right = self._pick_aliphatic_endpoint(remaining, (19,), 19)
                ok = self._apply_specs(paths, self._build_bridge_specs(left, 29, right), self._score_exact_bridge_preference)
            self._consume_one(29, ok)

        for info in self._iter_infos(28):
            if self.remaining_sus.get(28, 0) <= 0:
                break
            hop1 = [h for h in info.get('hop1', []) if h != 28]
            fixed_specs = self._build_fixed_specs(28, hop1)
            ok = False
            if fixed_specs:
                ok = self._apply_specs(paths, fixed_specs, self._score_terminal_preference)
            if not ok:
                endpoint = self._pick_aromatic_endpoint(hop1, (5,)) if any(self._is_aromatic_su(h) for h in hop1) else self._pick_aliphatic_endpoint(hop1, (19,), 19)
                ok = self._apply_specs(paths, self._build_single_terminal_specs(endpoint, 28), self._score_terminal_preference)
            self._consume_one(28, ok)

    def _substitute_nitrogen_bridge(self, paths):
        """Bridging nitrogen SU 27 (degree-2, 1-hop [6/20, 6/20])."""
        for info in self._iter_infos(27):
            if self.remaining_sus.get(27, 0) <= 0:
                break
            hop1 = [h for h in info.get('hop1', []) if h != 27]
            fixed_specs = self._build_fixed_specs(27, hop1)
            ok = False
            if fixed_specs:
                ok = self._apply_specs(paths, fixed_specs, self._score_exact_bridge_preference)
            if not ok:
                left = self._pick_aromatic_endpoint(hop1, (6,)) if any(self._is_aromatic_su(h) for h in hop1) else self._pick_aliphatic_endpoint(hop1, (20,), 20)
                right = left
                if len(hop1) >= 2:
                    remaining = hop1.copy()
                    if left in remaining:
                        remaining.remove(left)
                    if any(self._is_aromatic_su(h) for h in remaining):
                        right = self._pick_aromatic_endpoint(remaining, (6,))
                    elif 22 in remaining:
                        ok = self._apply_specs(paths, self._build_bridge_to_terminal_specs(left, 27, 22))
                        self._consume_one(27, ok)
                        continue
                    elif remaining:
                        right = self._pick_aliphatic_endpoint(remaining, (20,), 20)
                ok = self._apply_specs(paths, self._build_bridge_specs(left, 27, right), self._score_exact_bridge_preference)
            self._consume_one(27, ok)

    def _substitute_sulfur_bridge(self, paths):
        """Bridging sulfur SU 31 (degree-2, 1-hop [7/19, 7/19])."""
        for info in self._iter_infos(31):
            if self.remaining_sus.get(31, 0) <= 0:
                break
            hop1 = [h for h in info.get('hop1', []) if h != 31]
            fixed_specs = self._build_fixed_specs(31, hop1)
            ok = False
            if fixed_specs:
                ok = self._apply_specs(paths, fixed_specs, self._score_exact_bridge_preference)
            if not ok:
                left = self._pick_aromatic_endpoint(hop1, (7,)) if any(self._is_aromatic_su(h) for h in hop1) else self._pick_aliphatic_endpoint(hop1, (19,), 19)
                right = left
                if len(hop1) >= 2:
                    remaining = hop1.copy()
                    if left in remaining:
                        remaining.remove(left)
                    if any(self._is_aromatic_su(h) for h in remaining):
                        right = self._pick_aromatic_endpoint(remaining, (7,))
                    elif 22 in remaining:
                        ok = self._apply_specs(paths, self._build_bridge_to_terminal_specs(left, 31, 22))
                        self._consume_one(31, ok)
                        continue
                    elif remaining:
                        right = self._pick_aliphatic_endpoint(remaining, (19,), 19)
                ok = self._apply_specs(paths, self._build_bridge_specs(left, 31, right), self._score_exact_bridge_preference)
            self._consume_one(31, ok)

    def _substitute_halogen(self, paths):
        """Halogen SU 32 (terminal, 1-hop [8/21])."""
        for info in self._iter_infos(32):
            if self.remaining_sus.get(32, 0) <= 0:
                break
            hop1 = [h for h in info.get('hop1', []) if h != 32]
            fixed_specs = self._build_fixed_specs(32, hop1)
            ok = False
            if fixed_specs:
                ok = self._apply_specs(paths, fixed_specs, self._score_terminal_preference)
            if not ok:
                endpoint = self._pick_aromatic_endpoint(hop1, (8,)) if any(self._is_aromatic_su(h) for h in hop1) else self._pick_aliphatic_endpoint(hop1, (21,), 21)
                ok = self._apply_specs(paths, self._build_single_terminal_specs(endpoint, 32), self._score_terminal_preference)
            self._consume_one(32, ok)

    def _peek_unsat_kind(self, anchor_su: int, preferred: List[str]) -> Optional[str]:
        pool = getattr(self, '_unsat_anchor_counts', {}).get(anchor_su)
        if not pool:
            return None
        for kind in preferred:
            if pool.get(kind, 0) > 0:
                return kind
        return None

    def _consume_unsat_kind(self, anchor_su: int, kind: Optional[str]):
        if not kind:
            return
        pool = getattr(self, '_unsat_anchor_counts', {}).get(anchor_su)
        if pool and pool.get(kind, 0) > 0:
            pool[kind] -= 1
        if kind == '15_15':
            self.remaining_sus[15] = max(0, self.remaining_sus.get(15, 0) - 2)
        elif kind == '15_16':
            self.remaining_sus[15] = max(0, self.remaining_sus.get(15, 0) - 1)
            self.remaining_sus[16] = max(0, self.remaining_sus.get(16, 0) - 1)
        elif kind == '17_18':
            self.remaining_sus[17] = max(0, self.remaining_sus.get(17, 0) - 1)
            self.remaining_sus[18] = max(0, self.remaining_sus.get(18, 0) - 1)

    def _build_su0_generic_specs(self, hop1: List[int]) -> List[Tuple[List[Any], List[int]]]:
        aromatic = [h for h in hop1 if self._is_aromatic_su(h)]
        aliphatic = [self._genericize_chain_neighbor(h) for h in hop1 if h not in aromatic and h != 22]
        specs: List[Tuple[List[Any], List[int]]] = []
        if len(aromatic) >= 2:
            specs.extend(self._build_bridge_specs(self._pick_aromatic_endpoint(hop1, (9,)), 0, self._pick_aromatic_endpoint(hop1, (6,))))
        if 22 in hop1 and aromatic:
            specs.extend(self._build_bridge_to_terminal_specs(self._pick_aromatic_endpoint(hop1, (9,)), 0, 22))
        if 22 in hop1 and aliphatic:
            specs.extend(self._build_bridge_to_terminal_specs(23, 0, 22))
        if aromatic and aliphatic:
            specs.extend(self._build_bridge_specs(self._pick_aromatic_endpoint(hop1, (9,)), 0, self._pick_aliphatic_endpoint(hop1, (20,), 20)))
            specs.extend(self._build_bridge_specs(23, 0, self._pick_aromatic_endpoint(hop1, (6,))))
        if aliphatic:
            specs.extend(self._build_bridge_specs(23, 0, self._pick_aliphatic_endpoint(hop1, (20,), 20)))
        return specs

    def _substitute_su0_instance(self, paths, info: Dict[str, Any]) -> bool:
        hop1 = [h for h in info.get('hop1', []) if h != 0]
        unsat_kind = None
        if 17 in hop1:
            unsat_kind = self._peek_unsat_kind(0, ['17_18'])
        elif 15 in hop1:
            unsat_kind = self._peek_unsat_kind(0, ['15_15', '15_16', '15_14'])
        anchor_endpoint = self._pick_aromatic_endpoint(hop1, (6,)) if any(self._is_aromatic_su(h) for h in hop1) else self._pick_aliphatic_endpoint(hop1, (20,), 20)

        if unsat_kind == '15_15':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 0, [15, 15]))
            if ok:
                self._consume_unsat_kind(0, unsat_kind)
            return ok
        if unsat_kind == '15_16':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 0, [15, 16, 22]), self._score_terminal_preference)
            if ok:
                self._consume_unsat_kind(0, unsat_kind)
            return ok
        if unsat_kind == '17_18':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 0, [17, 18, 22]), self._score_terminal_preference)
            if ok:
                self._consume_unsat_kind(0, unsat_kind)
            return ok

        fixed_specs = self._build_fixed_specs(0, hop1)
        if fixed_specs:
            ok = self._apply_specs(paths, fixed_specs, self._score_exact_bridge_preference)
            if ok:
                return ok
        return self._apply_specs(paths, self._build_su0_generic_specs(hop1))

    def _build_su1_generic_specs(self, hop1: List[int]) -> List[Tuple[List[Any], List[int]]]:
        if any(self._is_aromatic_su(h) for h in hop1):
            return self._build_single_terminal_specs(self._pick_aromatic_endpoint(hop1, (9,)), 1)
        endpoint = self._pick_aliphatic_endpoint([self._genericize_chain_neighbor(h) for h in hop1], (21, 20, 19, 23), 23)
        if endpoint in (19, 20, 21):
            return self._build_single_terminal_specs(endpoint, 1)
        return self._build_single_terminal_specs(23, 1)

    def _substitute_su1_instance(self, paths, info: Dict[str, Any]) -> bool:
        hop1 = [h for h in info.get('hop1', []) if h != 1]
        unsat_kind = None
        if 17 in hop1:
            unsat_kind = self._peek_unsat_kind(1, ['17_18'])
        elif 15 in hop1:
            unsat_kind = self._peek_unsat_kind(1, ['15_15', '15_16', '15_14'])

        if unsat_kind == '15_16':
            return False
        if unsat_kind == '15_15':
            ok = self._apply_specs(paths, [([23, 23, 22], [15, 15, 1])], self._score_terminal_preference)
            if ok:
                self._consume_unsat_kind(1, unsat_kind)
            return ok
        if unsat_kind == '17_18':
            ok = self._apply_specs(paths, [([23, 23, 22], [17, 18, 1])], self._score_terminal_preference)
            if ok:
                self._consume_unsat_kind(1, unsat_kind)
            return ok

        fixed_specs = self._build_fixed_specs(1, hop1)
        if fixed_specs:
            ok = self._apply_specs(paths, fixed_specs, self._score_terminal_preference)
            if ok:
                return ok
        return self._apply_specs(paths, self._build_su1_generic_specs(hop1), self._score_terminal_preference)

    def _build_su2_generic_specs(self, hop1: List[int]) -> List[Tuple[List[Any], List[int]]]:
        normalized = [self._genericize_chain_neighbor(h) for h in hop1]
        if 5 in hop1:
            other = [h for h in hop1 if h != 5]
            left = int(other[0]) if other else 19
            return self._build_bridge_specs(left, 2, 5)
        left = self._pick_aromatic_endpoint(hop1, (9,)) if any(self._is_aromatic_su(h) for h in hop1) else 23
        if 22 in normalized:
            right = self._pick_aromatic_endpoint(hop1, (5,)) if any(self._is_aromatic_su(h) and h != 9 for h in hop1) else self._pick_aliphatic_endpoint(hop1, (19,), 19)
            return self._build_bridge_to_terminal_specs(right, 2, 22)
        if any(self._is_aromatic_su(h) and h != 9 for h in hop1):
            right = self._pick_aromatic_endpoint(hop1, (5,))
        else:
            right = self._pick_aliphatic_endpoint(normalized, (19,), 19)
        return self._build_bridge_specs(left, 2, right)

    def _substitute_su2_instance(self, paths, info: Dict[str, Any]) -> bool:
        hop1 = [h for h in info.get('hop1', []) if h != 2]
        unsat_kind = None
        if 17 in hop1:
            unsat_kind = self._peek_unsat_kind(2, ['17_18'])
        elif 15 in hop1:
            unsat_kind = self._peek_unsat_kind(2, ['15_15', '15_16', '15_14'])
        anchor_endpoint = self._pick_aromatic_endpoint(hop1, (5,)) if any(self._is_aromatic_su(h) and h != 9 for h in hop1) else self._pick_aliphatic_endpoint(hop1, (19,), 19)

        if unsat_kind == '15_15':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 2, [15, 15]))
            if ok:
                self._consume_unsat_kind(2, unsat_kind)
            return ok
        if unsat_kind == '15_16':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 2, [15, 16, 22]), self._score_terminal_preference)
            if ok:
                self._consume_unsat_kind(2, unsat_kind)
            return ok
        if unsat_kind == '17_18':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 2, [17, 18, 22]), self._score_terminal_preference)
            if ok:
                self._consume_unsat_kind(2, unsat_kind)
            return ok

        fixed_specs = self._build_fixed_specs(2, hop1)
        if fixed_specs:
            ok = self._apply_specs(paths, fixed_specs, self._score_exact_bridge_preference)
            if ok:
                return ok
        return self._apply_specs(paths, self._build_su2_generic_specs(hop1))

    def _build_su3_generic_specs(self, hop1: List[int]) -> List[Tuple[List[Any], List[int]]]:
        normalized = [self._genericize_chain_neighbor(h) for h in hop1]
        left = self._pick_aromatic_endpoint(hop1, (9,)) if any(self._is_aromatic_su(h) for h in hop1) else 23
        if 22 in normalized:
            return self._build_bridge_to_terminal_specs(left, 3, 22)
        right_aromatic = [h for h in hop1 if self._is_aromatic_su(h) and h != left]
        if right_aromatic:
            right = self._pick_aromatic_endpoint(right_aromatic, (9,))
        else:
            right = self._pick_aliphatic_endpoint(normalized, (21, 20, 19, 23), 23)
        return self._build_bridge_specs(left, 3, right)

    def _substitute_su3_instance(self, paths, info: Dict[str, Any]) -> bool:
        hop1 = [h for h in info.get('hop1', []) if h != 3]
        unsat_kind = None
        if 17 in hop1:
            unsat_kind = self._peek_unsat_kind(3, ['17_18'])
        elif 15 in hop1:
            unsat_kind = self._peek_unsat_kind(3, ['15_15', '15_16', '15_14'])
        anchor_endpoint = self._pick_aromatic_endpoint(hop1, (9,)) if any(self._is_aromatic_su(h) for h in hop1) else self._pick_aliphatic_endpoint(hop1, (21, 20, 19, 23), 23)

        if unsat_kind == '15_15':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 3, [15, 15]))
            if ok:
                self._consume_unsat_kind(3, unsat_kind)
            return ok
        if unsat_kind == '15_16':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 3, [15, 16, 22]), self._score_terminal_preference)
            if ok:
                self._consume_unsat_kind(3, unsat_kind)
            return ok
        if unsat_kind == '17_18':
            ok = self._apply_specs(paths, self._build_unsat_bridge_specs(anchor_endpoint, 3, [17, 18, 22]), self._score_terminal_preference)
            if ok:
                self._consume_unsat_kind(3, unsat_kind)
            return ok

        fixed_specs = self._build_fixed_specs(3, hop1)
        if fixed_specs:
            ok = self._apply_specs(paths, fixed_specs, self._score_exact_bridge_preference)
            if ok:
                return ok
        return self._apply_specs(paths, self._build_su3_generic_specs(hop1))

    def _substitute_unsaturated(self, state: MCTSState):
        """Unsaturated structure substitution with correct priority order.

        Priority order:
        1. 14-14: Fixed C-type 24-24 pairs in aliphatic fused rings
        2. 14-15: A/C-type 24 and branch-attached 23 (ring non-branch > branch attachment > branch tail)
        3. 14-16: B/D-type 24 and terminal 22
        4. 15-15: 23-23 pairs (ring priority)
        5. 15-16: 23-22 pairs
        6. 17-18: 23-22 pairs (triple bond)
        """
        plans = self._build_unsaturated_plans()
        nodes, adj, cluster_uids = self._build_graph_maps(state)
        pair_roles = self._build_pair_role_map(state, nodes)

        self._log(
            f"  [Unsaturated] Plans: 14-14={len(plans['14_14'])}, "
            f"14-15={len(plans['14_15'])}, 14-16={len(plans['14_16'])}, "
            f"15-15={plans['15_15']}, 15-16={plans['15_16']}, 17-18={plans['17_18']}"
        )

        # Priority 1: 14-15 (A/C-type 24 and branch-attached / ring 23)
        plan_14_15_by_type: Dict[str, int] = defaultdict(int)
        for hop1 in plans['14_15']:
            desired_type = '24_A' if self._hop1_has_aromatic(hop1) else '24_C'
            plan_14_15_by_type[str(desired_type)] += 1

        for desired_type, target_pairs in plan_14_15_by_type.items():
            if self.remaining_sus.get(14, 0) < 1 or self.remaining_sus.get(15, 0) < 1:
                break
            self._log(f"  [Unsaturated] Processing 14-15 type={desired_type}, target={target_pairs}")
            n_applied = self._apply_14_15_matches(
                nodes,
                adj,
                cluster_uids,
                str(desired_type),
                min(
                    int(target_pairs),
                    int(self.remaining_sus.get(14, 0)),
                    int(self.remaining_sus.get(15, 0)),
                ),
                pair_roles=pair_roles,
            )
            self.remaining_sus[14] -= int(n_applied)
            self.remaining_sus[15] -= int(n_applied)
            self._log(
                f"  [Unsaturated] Applied {n_applied} 14-15 pairs, "
                f"remaining 14={self.remaining_sus.get(14, 0)}, 15={self.remaining_sus.get(15, 0)}"
            )

        # Priority 2: 14-14 (adjacent C-type 24-24 ring positions)
        for hop1 in plans['14_14']:
            if self.remaining_sus.get(14, 0) < 2:
                break
            if not self._replace_14_14(nodes, adj, cluster_uids, pair_roles=pair_roles):
                self._log(f"  [Unsaturated] Warning: no pattern for 14-14, remaining={self.remaining_sus.get(14, 0)}")
                break
            self.remaining_sus[14] -= 2
            self._log(f"  [Unsaturated] Applied 14-14, remaining 14={self.remaining_sus.get(14, 0)}")

        # Priority 3: 14-16 (B/D-type 24 and terminal 22)
        for hop1 in plans['14_16']:
            if self.remaining_sus.get(14, 0) < 1 or self.remaining_sus.get(16, 0) < 1:
                break
            desired_type = '24_B' if self._hop1_has_aromatic(hop1) else '24_D'
            if not self._replace_14_16(nodes, adj, cluster_uids, desired_type, pair_roles=pair_roles):
                continue
            self.remaining_sus[14] -= 1
            self.remaining_sus[16] -= 1
            self._log(
                f"  [Unsaturated] Applied 14-16, remaining 14={self.remaining_sus.get(14, 0)}, "
                f"16={self.remaining_sus.get(16, 0)}"
            )

        # Saturation pass:
        # After honoring the original plan counts, keep consuming any remaining
        # feasible 14-based substitutions from aliphatic rings / ring branches
        # before dropping to lower-priority 15-15 / 15-16.
        while self.remaining_sus.get(14, 0) > 0:
            progressed = False

            if self.remaining_sus.get(15, 0) > 0 and self._apply_best_14_15(nodes, adj, cluster_uids, pair_roles=pair_roles):
                self.remaining_sus[14] -= 1
                self.remaining_sus[15] -= 1
                progressed = True
                self._log(
                    f"  [Unsaturated] Applied extra 14-15, remaining 14={self.remaining_sus.get(14, 0)}, "
                    f"15={self.remaining_sus.get(15, 0)}"
                )
                continue

            if self.remaining_sus.get(14, 0) >= 2 and self._replace_14_14(nodes, adj, cluster_uids, pair_roles=pair_roles):
                self.remaining_sus[14] -= 2
                progressed = True
                self._log(f"  [Unsaturated] Applied extra 14-14, remaining 14={self.remaining_sus.get(14, 0)}")
                continue

            if self.remaining_sus.get(16, 0) > 0 and self._apply_best_14_16(nodes, adj, cluster_uids, pair_roles=pair_roles):
                self.remaining_sus[14] -= 1
                self.remaining_sus[16] -= 1
                progressed = True
                self._log(
                    f"  [Unsaturated] Applied extra 14-16, remaining 14={self.remaining_sus.get(14, 0)}, "
                    f"16={self.remaining_sus.get(16, 0)}"
                )
                continue

            if not progressed:
                self._log(
                    f"  [Unsaturated] No more feasible 14-based replacements in current graph, "
                    f"remaining 14={self.remaining_sus.get(14, 0)}"
                )
                break

        # Priority 4: 15-15 (23-23 pairs, ring priority)
        for _ in range(plans['15_15']):
            if self.remaining_sus.get(15, 0) < 2:
                break
            if not self._replace_15_15_ring_priority(nodes, adj, pair_roles=pair_roles):
                self._log(f"  [Unsaturated] Warning: no pattern for 15-15, remaining={self.remaining_sus.get(15, 0)}")
                break
            self.remaining_sus[15] -= 2
            self._log(f"  [Unsaturated] Applied 15-15, remaining 15={self.remaining_sus.get(15, 0)}")

        # Priority 5: 15-16 (23-22 pairs)
        for _ in range(plans['15_16']):
            if self.remaining_sus.get(15, 0) < 1 or self.remaining_sus.get(16, 0) < 1:
                break
            if not self._replace_unsat_pair_by_adj(nodes, adj, '15_16', pair_roles=pair_roles):
                self._log(f"  [Unsaturated] Warning: no pattern for 15-16")
                break
            self.remaining_sus[15] -= 1
            self.remaining_sus[16] -= 1
            self._log(
                f"  [Unsaturated] Applied 15-16, remaining 15={self.remaining_sus.get(15, 0)}, "
                f"16={self.remaining_sus.get(16, 0)}"
            )

        # Priority 6: 17-18 (23-22 pairs, triple bond)
        for _ in range(plans['17_18']):
            if self.remaining_sus.get(17, 0) < 1 or self.remaining_sus.get(18, 0) < 1:
                break
            if not self._replace_unsat_pair_by_adj(nodes, adj, '17_18', pair_roles=pair_roles):
                self._log(f"  [Unsaturated] Warning: no pattern for 17-18")
                break
            self.remaining_sus[17] -= 1
            self.remaining_sus[18] -= 1
            self._log(
                f"  [Unsaturated] Applied 17-18, remaining 17={self.remaining_sus.get(17, 0)}, "
                f"18={self.remaining_sus.get(18, 0)}"
            )

    def _build_unsaturated_plans(self) -> Dict[str, Any]:
        infos_14 = list(self.su_details.get(14, []))
        infos_15 = list(self.su_details.get(15, []))
        infos_16 = list(self.su_details.get(16, []))
        infos_17 = list(self.su_details.get(17, []))
        has_info = bool(infos_14 or infos_15 or infos_16 or infos_17)

        if not has_info:
            rem14 = self.remaining_sus.get(14, 0)
            rem15 = self.remaining_sus.get(15, 0)
            rem16 = self.remaining_sus.get(16, 0)
            rem17 = self.remaining_sus.get(17, 0)
            rem18 = self.remaining_sus.get(18, 0)

            n14_15 = min(rem14, rem15)
            rem14 -= n14_15
            rem15 -= n14_15
            n14_14 = rem14 // 2
            rem14 -= n14_14 * 2
            n14_16 = min(rem14, rem16)
            rem14 -= n14_16
            rem16 -= n14_16
            n15_16 = min(rem15, rem16)
            rem15 -= n15_16
            rem16 -= n15_16
            n15_15 = rem15 // 2
            n17_18 = min(rem17, rem18)

            return {
                '14_14': [[] for _ in range(n14_14)],
                '14_15': [[] for _ in range(n14_15)],
                '14_16': [[] for _ in range(n14_16)],
                '15_15': n15_15,
                '15_16': n15_16,
                '17_18': n17_18,
            }

        rem14 = self.remaining_sus.get(14, 0)
        rem15 = self.remaining_sus.get(15, 0)
        rem16 = self.remaining_sus.get(16, 0)
        rem17 = self.remaining_sus.get(17, 0)
        rem18 = self.remaining_sus.get(18, 0)

        raw_14_14 = [info['hop1'] for info in infos_14 if 14 in info.get('hop1', [])]
        raw_14_15_a = [info['hop1'] for info in infos_14 if 15 in info.get('hop1', [])]
        raw_14_15_b = [info['hop1'] for info in infos_15 if 14 in info.get('hop1', [])]
        raw_14_15 = raw_14_15_a if len(raw_14_15_a) >= len(raw_14_15_b) else raw_14_15_b
        raw_14_16_a = [info['hop1'] for info in infos_14 if 16 in info.get('hop1', [])]
        raw_14_16_b = [info['hop1'] for info in infos_16 if 14 in info.get('hop1', [])]
        raw_14_16 = raw_14_16_a if len(raw_14_16_a) >= len(raw_14_16_b) else raw_14_16_b
        raw_15_15 = len([info for info in infos_15 if 15 in info.get('hop1', [])]) // 2
        raw_15_16 = max(
            len([info for info in infos_15 if 16 in info.get('hop1', [])]),
            len([info for info in infos_16 if 15 in info.get('hop1', [])]),
        )
        raw_17_18 = len([info for info in infos_17 if 18 in info.get('hop1', [])])

        take_14_15 = min(len(raw_14_15), rem14, rem15)
        plan_14_15 = raw_14_15[:take_14_15]
        rem14 -= take_14_15
        rem15 -= take_14_15

        take_14_14 = min(len(raw_14_14) // 2, rem14 // 2)
        plan_14_14 = raw_14_14[:take_14_14]
        rem14 -= take_14_14 * 2

        take_14_16 = min(len(raw_14_16), rem14, rem16)
        plan_14_16 = raw_14_16[:take_14_16]
        rem14 -= take_14_16
        rem16 -= take_14_16

        n15_16 = min(raw_15_16, rem15, rem16)
        rem15 -= n15_16
        rem16 -= n15_16

        n15_15 = min(raw_15_15, rem15 // 2)
        rem15 -= n15_15 * 2

        n17_18 = min(raw_17_18, rem17, rem18)

        if not (plan_14_14 or plan_14_15 or plan_14_16 or n15_15 or n15_16 or n17_18):
            rem14 = self.remaining_sus.get(14, 0)
            rem15 = self.remaining_sus.get(15, 0)
            rem16 = self.remaining_sus.get(16, 0)
            rem17 = self.remaining_sus.get(17, 0)
            rem18 = self.remaining_sus.get(18, 0)

            n14_15 = min(rem14, rem15)
            rem14 -= n14_15
            rem15 -= n14_15
            n14_14 = rem14 // 2
            rem14 -= n14_14 * 2
            n14_16 = min(rem14, rem16)
            rem14 -= n14_16
            rem16 -= n14_16
            n15_16 = min(rem15, rem16)
            rem15 -= n15_16
            rem16 -= n15_16
            n15_15 = rem15 // 2
            n17_18 = min(rem17, rem18)
            return {
                '14_14': [[] for _ in range(n14_14)],
                '14_15': [[] for _ in range(n14_15)],
                '14_16': [[] for _ in range(n14_16)],
                '15_15': n15_15,
                '15_16': n15_16,
                '17_18': n17_18,
            }

        return {
            '14_14': plan_14_14,
            '14_15': plan_14_15,
            '14_16': plan_14_16,
            '15_15': n15_15,
            '15_16': n15_16,
            '17_18': n17_18,
        }

    @staticmethod
    def _hop1_has_aromatic(hop1: List[int]) -> bool:
        return any(5 <= h <= 13 for h in hop1)

    def _build_graph_maps(self, state: MCTSState):
        nodes: Dict[str, Any] = {}
        cluster_uids: Set[str] = set()

        for c in state.graph.clusters:
            if not getattr(c, 'placed', False):
                continue
            for s in c.sites:
                nodes[s.uid] = s
                cluster_uids.add(s.uid)

        for cn in state.graph.chains:
            nodes[cn.uid] = cn

        adj: Dict[str, Set[str]] = {uid: set() for uid in nodes}

        def connect(u: Optional[str], v: Optional[str]):
            if not u or not v or u == v:
                return
            if u not in nodes or v not in nodes:
                return
            adj[u].add(v)
            adj[v].add(u)

        for c in state.graph.clusters:
            if not getattr(c, 'placed', False):
                continue
            for a, b in c.edges:
                if 0 <= a < len(c.sites) and 0 <= b < len(c.sites):
                    connect(c.sites[a].uid, c.sites[b].uid)

        for e in state.graph.flex:
            seq = [e.u] + [n.uid for n in e.chain] + [e.v]
            for i in range(len(seq) - 1):
                connect(seq[i], seq[i + 1])

        for e in getattr(state.graph, 'side', []):
            seq = [e.u] + [n.uid for n in e.chain]
            for i in range(len(seq) - 1):
                connect(seq[i], seq[i + 1])

        for e in getattr(state.graph, 'branch', []):
            seq = [e.base] + [n.uid for n in e.chain]
            if e.target:
                seq.append(e.target)
            for i in range(len(seq) - 1):
                connect(seq[i], seq[i + 1])

        return nodes, adj, cluster_uids

    def _build_pair_role_map(self, state: MCTSState, nodes: Dict[str, Any]) -> Dict[frozenset, Set[str]]:
        pair_roles: Dict[frozenset, Set[str]] = {}

        def _add_role(u: Optional[str], v: Optional[str], role: str, pos_tag: Optional[str] = None):
            if not u or not v or u == v:
                return
            if u not in nodes or v not in nodes:
                return
            key = frozenset((u, v))
            roles = pair_roles.setdefault(key, set())
            roles.add(role)
            if pos_tag:
                roles.add(f"{role}:{pos_tag}")
            su_u = int(nodes[u].su_type)
            su_v = int(nodes[v].su_type)
            if {su_u, su_v} == {23, 22} or {su_u, su_v} == {24, 22}:
                roles.add('terminal_pair')
            if su_u == 23 and su_v == 23:
                roles.add('bridge_pair')

        for e in getattr(state.graph, 'flex', []):
            seq = [e.u] + [n.uid for n in e.chain] + [e.v]
            for i in range(len(seq) - 1):
                pos = 'first' if i == 0 else ('last' if i == len(seq) - 2 else None)
                _add_role(seq[i], seq[i + 1], 'flex', pos)

        for e in getattr(state.graph, 'side', []):
            seq = [e.u] + [n.uid for n in e.chain]
            for i in range(len(seq) - 1):
                pos = 'first' if i == 0 else ('last' if i == len(seq) - 2 else None)
                _add_role(seq[i], seq[i + 1], 'side', pos)

        for e in getattr(state.graph, 'branch', []):
            seq = [e.base] + [n.uid for n in e.chain]
            role = 'branch_ring' if getattr(e, 'target', None) else 'branch_tail'
            if getattr(e, 'target', None):
                seq = seq + [e.target]
            for i in range(len(seq) - 1):
                pos = 'first' if i == 0 else ('last' if i == len(seq) - 2 else None)
                _add_role(seq[i], seq[i + 1], role, pos)

        return pair_roles

    def _get_branch_type_from_meta(self, uid: str, nodes) -> Optional[str]:
        """Get branch_type from node metadata (more reliable than current neighbor analysis)."""
        node = nodes.get(uid)
        if node is None:
            return None
        meta = getattr(node, 'meta', {}) or {}
        branch_type = meta.get('branch_type')
        if isinstance(branch_type, str) and branch_type in ('24_A', '24_B', '24_C', '24_D'):
            return branch_type
        return None

    def _is_aliphatic_ring_main_chain(self, uid_24: str, uid_23: str, nodes, pair_roles) -> bool:
        node_24 = nodes.get(uid_24)
        node_23 = nodes.get(uid_23)
        if node_24 is None or node_23 is None:
            return False
        if int(node_24.su_type) != 24 or int(node_23.su_type) != 23:
            return False

        meta_24 = getattr(node_24, 'meta', {}) or {}
        meta_23 = getattr(node_23, 'meta', {}) or {}

        role_24 = str(meta_24.get('ring_role') or '')
        role_23 = str(meta_23.get('ring_role') or '')
        if not role_24 or not role_23:
            return False
        if meta_23.get('branch_kind') == 'tail':
            return False

        roles = (pair_roles or {}).get(frozenset((uid_24, uid_23)), set())
        if 'branch_ring' not in roles:
            return False

        allowed_pairs = {
            frozenset(('first_24', 'right_23')),
            frozenset(('first_24', 'left_23')),
            frozenset(('inter_right', 'right_23')),
            frozenset(('inter_right', 'closing_23')),
            frozenset(('inter_left', 'closing_23')),
            frozenset(('inter_left', 'left_23')),
            frozenset(('upper_outer', 'upper_bridge')),
            frozenset(('lower_outer', 'lower_bridge')),
            frozenset(('bridge_upper', 'outer_upper_23')),
            frozenset(('bridge_lower', 'outer_lower_23')),
            frozenset(('outer_upper_24', 'outer_upper_23')),
            frozenset(('outer_lower_24', 'outer_lower_23')),
        }
        return frozenset((role_24, role_23)) in allowed_pairs

    def _classify_current_24(self, uid: str, nodes, adj, cluster_uids) -> Optional[str]:
        """Classify 24 node type. Prefer metadata over current neighbor analysis."""
        node = nodes.get(uid)
        if node is None or node.su_type != 24:
            return None

        # First try to get from metadata (more reliable)
        meta_type = self._get_branch_type_from_meta(uid, nodes)
        if meta_type is not None:
            return meta_type

        # Fallback to neighbor analysis
        has_aro = any(nbr in cluster_uids for nbr in adj.get(uid, ()))
        has_22 = any(nodes[nbr].su_type == 22 for nbr in adj.get(uid, ()) if nbr in nodes)
        if has_aro and not has_22:
            return '24_A'
        if has_aro and has_22:
            return '24_B'
        if not has_aro and not has_22:
            return '24_C'
        return '24_D'

    def _find_terminal_branch_path(self, start_uid: str, base_uid: str, nodes, adj):
        start = nodes.get(start_uid)
        if start is None or start.su_type != 23:
            return None

        path = [start_uid]
        prev = base_uid
        cur = start_uid
        seen = {base_uid, start_uid}

        for _ in range(4):
            nexts = [nbr for nbr in adj.get(cur, ()) if nbr != prev]
            terminal_22 = [nbr for nbr in nexts if nodes[nbr].su_type == 22]
            if terminal_22:
                return path + [terminal_22[0]]

            step_23 = [
                nbr for nbr in nexts
                if nbr not in seen and nodes[nbr].su_type == 23
            ]
            if len(step_23) != 1:
                return None

            prev, cur = cur, step_23[0]
            seen.add(cur)
            path.append(cur)

        return None

    def _build_14_14_candidates(self, nodes, adj, cluster_uids, pair_roles=None):
        candidates = []
        for uid, node in nodes.items():
            if node.su_type != 24:
                continue
            node_branch_type = self._get_branch_type_from_meta(uid, nodes)
            if node_branch_type is None:
                node_branch_type = self._classify_current_24(uid, nodes, adj, cluster_uids)
            if node_branch_type != '24_C':
                continue
            role_uid = str((getattr(node, 'meta', {}) or {}).get('ring_role') or '')
            if not role_uid:
                continue

            for nbr in adj.get(uid, ()):
                if uid >= nbr:
                    continue
                if nbr not in nodes:
                    continue
                if nodes[nbr].su_type != 24:
                    continue
                nbr_branch_type = self._get_branch_type_from_meta(nbr, nodes)
                if nbr_branch_type is None:
                    nbr_branch_type = self._classify_current_24(nbr, nodes, adj, cluster_uids)
                if nbr_branch_type != '24_C':
                    continue
                role_nbr = str((getattr(nodes[nbr], 'meta', {}) or {}).get('ring_role') or '')
                if not role_nbr:
                    continue

                roles = (pair_roles or {}).get(frozenset((uid, nbr)), set())
                if 'branch_ring' not in roles:
                    continue
                role_pair = frozenset((role_uid, role_nbr))
                score = 600
                if role_pair == frozenset(('bridge_upper', 'bridge_lower')):
                    score = 1200
                elif role_pair == frozenset(('outer_upper_24', 'outer_lower_24')):
                    score = 1100
                elif role_pair == frozenset(('upper_bridge', 'lower_bridge')):
                    score = 1000
                elif role_pair == frozenset(('inter_right', 'inter_left')):
                    score = 900
                if 'branch_ring:first' in roles or 'branch_ring:last' in roles:
                    score += 20
                candidates.append((score, uid, nbr))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return candidates

    def _replace_14_14(self, nodes, adj, cluster_uids, pair_roles=None) -> bool:
        candidates = self._build_14_14_candidates(nodes, adj, cluster_uids, pair_roles=pair_roles)
        if not candidates:
            return False

        chosen = random.choice(candidates[:min(5, len(candidates))]) if self.randomize else candidates[0]
        _, uid_a, uid_b = chosen
        nodes[uid_a].su_type = 14
        nodes[uid_b].su_type = 14
        return True

    def _is_direct_branch_head(self, uid_base: str, uid_child: str, child_su: int, nodes, pair_roles=None) -> bool:
        node = nodes.get(uid_child)
        if node is None or int(node.su_type) != int(child_su):
            return False
        meta = getattr(node, 'meta', {}) or {}
        if meta.get('branch_kind') == 'tail':
            pos = meta.get('position_idx')
            try:
                return int(pos) == 0
            except Exception:
                return True
        roles = (pair_roles or {}).get(frozenset((uid_base, uid_child)), set())
        return 'branch_tail:first' in roles

    def _score_14_15_candidate(self, uid_24: str, uid_23: str, branch_path, nodes, pair_roles=None) -> Optional[int]:
        roles = (pair_roles or {}).get(frozenset((uid_24, uid_23)), set())
        meta_24 = getattr(nodes[uid_24], 'meta', {}) or {}
        role_24 = str(meta_24.get('ring_role') or '')

        if self._is_aliphatic_ring_main_chain(uid_24, uid_23, nodes, pair_roles):
            score = 1000
            if role_24 in ('outer_upper_24', 'outer_lower_24'):
                score += 80
            elif role_24 in ('inter_right', 'inter_left', 'first_24'):
                score += 60
            else:
                score += 40
        elif self._is_direct_branch_head(uid_24, uid_23, 23, nodes, pair_roles=pair_roles):
            score = 800
            if 'branch_tail:first' in roles:
                score += 80
        elif branch_path:
            score = 600 + max(0, 10 - len(branch_path))
        elif 'branch_ring' in roles:
            score = 400
        elif 'side' in roles:
            score = 300
        elif 'flex' in roles:
            score = 250
        else:
            return None

        if 'branch_ring' in roles:
            score += 20
        if 'branch_tail' in roles:
            score += 15
        if 'side' in roles:
            score += 10
        return int(score)

    def _build_14_15_candidates(self, nodes, adj, cluster_uids, desired_type: str, pair_roles=None):
        candidates = []
        for uid, node in nodes.items():
            if node.su_type != 24:
                continue
            node_branch_type = self._get_branch_type_from_meta(uid, nodes)
            if node_branch_type is None:
                node_branch_type = self._classify_current_24(uid, nodes, adj, cluster_uids)
            if node_branch_type != desired_type:
                continue

            for nbr in adj.get(uid, ()):
                if nbr not in nodes or int(nodes[nbr].su_type) != 23:
                    continue
                branch_path = self._find_terminal_branch_path(nbr, uid, nodes, adj)
                score = self._score_14_15_candidate(uid, nbr, branch_path, nodes, pair_roles=pair_roles)
                if score is not None:
                    candidates.append((int(score), uid, nbr))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return candidates

    def _select_disjoint_14_15_pairs(self, pool: List[Tuple[int, str, str]], target_pairs: int, nodes) -> List[Tuple[str, str]]:
        if not pool or target_pairs <= 0:
            return []

        valid_pool = []
        for score, uid_24, uid_23 in pool:
            if uid_24 not in nodes or uid_23 not in nodes:
                continue
            if int(nodes[uid_24].su_type) != 24 or int(nodes[uid_23].su_type) != 23:
                continue
            valid_pool.append((score, uid_24, uid_23))

        if not valid_pool:
            return []

        by_24: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        all_23: Set[str] = set()
        for score, uid_24, uid_23 in valid_pool:
            by_24[str(uid_24)].append((int(score), str(uid_24), str(uid_23)))
            all_23.add(str(uid_23))

        for uid_24 in by_24:
            by_24[uid_24].sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        available_24 = tuple(sorted(by_24.keys()))
        best: Dict[str, Any] = {'count': -1, 'score': -10**18, 'pairs': []}

        def _search(avail_24: Tuple[str, ...], used_23: Set[str],
                    selected: List[Tuple[str, str]], total_score: int):
            current_count = len(selected)
            max_possible = current_count + min(len(avail_24), len(all_23) - len(used_23))
            if max_possible < best['count']:
                return
            if current_count >= target_pairs or not avail_24:
                if current_count > best['count'] or (
                    current_count == best['count'] and total_score > best['score']
                ):
                    best['count'] = current_count
                    best['score'] = total_score
                    best['pairs'] = list(selected)
                return

            uid_24 = min(
                avail_24,
                key=lambda uid: (
                    sum(1 for _score, _u24, uid_23 in by_24.get(uid, []) if uid_23 not in used_23),
                    uid,
                ),
            )
            next_avail = tuple(uid for uid in avail_24 if uid != uid_24)

            viable = [cand for cand in by_24.get(uid_24, []) if cand[2] not in used_23]
            if not viable:
                _search(next_avail, used_23, selected, total_score)
                return

            for score, _u24, uid_23 in viable:
                used_23.add(uid_23)
                selected.append((uid_24, uid_23))
                _search(next_avail, used_23, selected, total_score + int(score))
                selected.pop()
                used_23.remove(uid_23)

            _search(next_avail, used_23, selected, total_score)

        _search(available_24, set(), [], 0)
        return list(best['pairs'][:max(0, int(target_pairs))])

    def _apply_14_15_matches(self, nodes, adj, cluster_uids, desired_type: str,
                             target_pairs: int, pair_roles=None) -> int:
        candidates = self._build_14_15_candidates(
            nodes,
            adj,
            cluster_uids,
            desired_type,
            pair_roles=pair_roles,
        )
        selected = self._select_disjoint_14_15_pairs(candidates, target_pairs, nodes)
        applied = 0
        for uid_24, uid_23 in selected:
            if uid_24 not in nodes or uid_23 not in nodes:
                continue
            if int(nodes[uid_24].su_type) != 24 or int(nodes[uid_23].su_type) != 23:
                continue
            nodes[uid_24].su_type = 14
            nodes[uid_23].su_type = 15
            applied += 1
        return int(applied)

    def _apply_best_14_15(self, nodes, adj, cluster_uids, pair_roles=None) -> bool:
        pools = []
        for desired_type in ('24_A', '24_C'):
            pools.extend(self._build_14_15_candidates(
                nodes,
                adj,
                cluster_uids,
                desired_type,
                pair_roles=pair_roles,
            ))
        if not pools:
            return False
        pools.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        chosen = random.choice(pools[:min(5, len(pools))]) if self.randomize else pools[0]
        _score, uid_24, uid_23 = chosen
        if uid_24 not in nodes or uid_23 not in nodes:
            return False
        if int(nodes[uid_24].su_type) != 24 or int(nodes[uid_23].su_type) != 23:
            return False
        nodes[uid_24].su_type = 14
        nodes[uid_23].su_type = 15
        return True

    def _build_14_16_candidates(self, nodes, adj, cluster_uids, desired_type: str, pair_roles=None):
        candidates = []
        for uid, node in nodes.items():
            if node.su_type != 24:
                continue
            node_branch_type = self._get_branch_type_from_meta(uid, nodes)
            if node_branch_type is None:
                node_branch_type = self._classify_current_24(uid, nodes, adj, cluster_uids)
            if node_branch_type != desired_type:
                continue

            for nbr in adj.get(uid, ()):
                if nbr not in nodes or int(nodes[nbr].su_type) != 22:
                    continue
                roles = (pair_roles or {}).get(frozenset((uid, nbr)), set())
                meta_22 = getattr(nodes[nbr], 'meta', {}) or {}
                score = 0
                if self._is_direct_branch_head(uid, nbr, 22, nodes, pair_roles=pair_roles):
                    score += 1000
                elif 'branch_tail' in roles:
                    score += 800
                elif 'terminal_pair' in roles:
                    score += 500
                elif 'side' in roles:
                    score += 300
                else:
                    score += 100
                if 'branch_tail:first' in roles:
                    score += 80
                if meta_22.get('branch_kind') == 'tail':
                    score += 40
                candidates.append((score, uid, nbr))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return candidates

    def _replace_14_16(self, nodes, adj, cluster_uids, desired_type: str, pair_roles=None) -> bool:
        candidates = self._build_14_16_candidates(nodes, adj, cluster_uids, desired_type, pair_roles=pair_roles)
        if not candidates:
            return False

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        chosen = random.choice(candidates[:min(5, len(candidates))]) if self.randomize else candidates[0]
        _, uid_24, uid_22 = chosen
        nodes[uid_24].su_type = 14
        nodes[uid_22].su_type = 16
        return True

    def _apply_best_14_16(self, nodes, adj, cluster_uids, pair_roles=None) -> bool:
        pools = []
        for desired_type in ('24_B', '24_D'):
            pools.extend(self._build_14_16_candidates(
                nodes,
                adj,
                cluster_uids,
                desired_type,
                pair_roles=pair_roles,
            ))
        if not pools:
            return False
        pools.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        chosen = random.choice(pools[:min(5, len(pools))]) if self.randomize else pools[0]
        _score, uid_24, uid_22 = chosen
        if uid_24 not in nodes or uid_22 not in nodes:
            return False
        if int(nodes[uid_24].su_type) != 24 or int(nodes[uid_22].su_type) != 22:
            return False
        nodes[uid_24].su_type = 14
        nodes[uid_22].su_type = 16
        return True

    def _replace_15_15_ring_priority(self, nodes, adj, pair_roles=None) -> bool:
        """Replace 15-15 pairs with priority for ring 23-23 positions.

        Priority:
        1. Ring 23-23 (branch_ring role)
        2. Flex/side 23-23
        3. Other 23-23
        """
        candidates = []

        for uid, node in nodes.items():
            if int(node.su_type) != 23:
                continue
            for nbr in adj.get(uid, ()):
                if nbr not in nodes or int(nodes[nbr].su_type) != 23:
                    continue
                if uid >= nbr:  # Avoid duplicates
                    continue

                roles = (pair_roles or {}).get(frozenset((uid, nbr)), set())
                left_deg = len(adj.get(uid, ()))
                right_deg = len(adj.get(nbr, ()))
                left_nbr = self._neighbor_su_types(uid, nodes, adj)
                right_nbr = self._neighbor_su_types(nbr, nodes, adj)

                score = 0

                # HIGHEST PRIORITY: Ring 23-23
                if 'branch_ring' in roles:
                    score += 100

                # Both ends not terminal (internal position)
                if 22 not in left_nbr and 22 not in right_nbr:
                    score += 50

                # Both ends have degree >= 2 (ring position)
                if left_deg >= 2 and right_deg >= 2:
                    score += 30

                # Other roles
                if 'flex' in roles:
                    score += 25
                if 'side' in roles:
                    score += 20

                # Has aromatic or 24 neighbors
                if any(su in ARO or su == 24 or su == 11 for su in left_nbr + right_nbr):
                    score += 5

                candidates.append((score, uid, nbr))

        if not candidates:
            return False

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        _, uid_left, uid_right = (random.choice(candidates[:min(5, len(candidates))])
                                   if self.randomize else candidates[0])
        nodes[uid_left].su_type = 15
        nodes[uid_right].su_type = 15
        return True

    @staticmethod
    def _neighbor_su_types(uid: str, nodes, adj) -> List[int]:
        out = []
        for nbr in adj.get(uid, ()):
            if nbr in nodes:
                out.append(int(nodes[nbr].su_type))
        return out

    def _replace_unsat_pair_by_adj(self, nodes, adj, pair_kind: str, pair_roles=None) -> bool:
        if pair_kind == '15_15':
            find_pair = (23, 23)
            repl_pair = (15, 15)
        elif pair_kind == '15_16':
            find_pair = (23, 22)
            repl_pair = (15, 16)
        elif pair_kind == '17_18':
            find_pair = (23, 22)
            repl_pair = (17, 18)
        else:
            return False

        candidates = []
        for uid, node in nodes.items():
            if int(node.su_type) != int(find_pair[0]):
                continue
            for nbr in adj.get(uid, ()):
                if nbr not in nodes:
                    continue
                if int(nodes[nbr].su_type) != int(find_pair[1]):
                    continue
                if pair_kind == '15_15' and uid >= nbr:
                    continue

                left_deg = len(adj.get(uid, ()))
                right_deg = len(adj.get(nbr, ()))
                left_nbr = self._neighbor_su_types(uid, nodes, adj)
                right_nbr = self._neighbor_su_types(nbr, nodes, adj)
                score = 0

                if pair_kind == '15_15':
                    if 22 not in left_nbr and 22 not in right_nbr:
                        score += 20
                    if left_deg >= 2 and right_deg >= 2:
                        score += 10
                    roles = (pair_roles or {}).get(frozenset((uid, nbr)), set())
                    if 'flex' in roles:
                        score += 25
                    if 'side' in roles:
                        score += 20
                    if 'branch_ring' in roles:
                        score += 15
                    if any(su in ARO or su == 24 or su == 11 for su in left_nbr + right_nbr):
                        score += 5
                else:
                    roles = (pair_roles or {}).get(frozenset((uid, nbr)), set())
                    meta_l = getattr(nodes[uid], 'meta', {}) or {}
                    meta_r = getattr(nodes[nbr], 'meta', {}) or {}
                    origin_l = str(meta_l.get('origin_type', ''))
                    origin_r = str(meta_r.get('origin_type', ''))
                    if int(right_deg) == 1:
                        score += 20
                    if int(left_deg) >= 2:
                        score += 10
                    if 'branch_tail' in roles:
                        score += 25
                    if 'side' in roles:
                        score += 20
                    if pair_kind == '15_16' and (origin_l in ('B2', 'D2') or origin_r in ('B2', 'D2')):
                        score += 35
                    if pair_kind == '17_18' and (origin_l in ('B2', 'D2') or origin_r in ('B2', 'D2')):
                        score += 40
                    if any(su in (11, 23, 24) or su in ARO for su in left_nbr):
                        score += 5

                candidates.append((score, uid, nbr))

        if not candidates:
            return False

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        _, uid_left, uid_right = random.choice(candidates[:min(5, len(candidates))]) if self.randomize else candidates[0]
        nodes[uid_left].su_type = int(repl_pair[0])
        nodes[uid_right].su_type = int(repl_pair[1])
        return True

    # ================================================================
    # Heteroaromatic ring substitution (within aromatic clusters)
    # ================================================================
    def _substitute_heteroaromatic_N(self, state: MCTSState):
        """
        SU 26 — nitrogen heteroaromatic ring.

        Two modes, split 50/50 among the total count:
          * Pyridine — replace an adjacent 13-13 pair inside the same
            aromatic cluster with a single 26 (averaged coordinates,
            re-route edges).
          * Pyrrole  — replace a single 13 with 26.

        Selection priority:
          1. Candidates whose cluster neighbours match the 1-hop info.
          2. Among those (or all, if none match), pick the one with the
             highest NMR score via trial evaluation; fall back to random
             if no NMR evaluator is available.
        """
        total = self.remaining_sus.get(26, 0)
        if total == 0:
            return

        info_list = list(self.su_details.get(26, []))
        n_pyridine = total // 2
        n_pyrrole = total - n_pyridine

        # Pyridine (replace 13-13 pair)
        for _ in range(n_pyridine):
            hop1 = info_list.pop(0)['hop1'] if info_list else []
            self._replace_in_cluster(state, 26, hop1, replace_two=True)

        # Pyrrole (replace single 13)
        for _ in range(n_pyrrole):
            hop1 = info_list.pop(0)['hop1'] if info_list else []
            self._replace_in_cluster(state, 26, hop1, replace_two=False)

    def _substitute_heteroaromatic_S(self, state: MCTSState):
        """
        SU 30 — sulfur heteroaromatic ring (thiophene).
        Always replaces an adjacent 13-13 pair inside the same cluster.
        """
        while self.remaining_sus.get(30, 0) > 0:
            info_list = self.su_details.get(30, [])
            hop1 = info_list.pop(0)['hop1'] if info_list else []
            self._replace_in_cluster(state, 30, hop1, replace_two=True)

    # ================================================================
    # Core: replace SU-13 sites inside an aromatic cluster
    # ================================================================
    @staticmethod
    def _are_directly_bonded(cluster, idx1: int, idx2: int) -> bool:
        """True if site idx1 and idx2 share a direct intra-ring bond."""
        edge = (min(idx1, idx2), max(idx1, idx2))
        return edge in set(tuple(sorted(e)) for e in cluster.edges)

    def _edge_neighbors(self, cluster, idx: int) -> List[int]:
        """Return site indices directly bonded to *idx* via cluster.edges."""
        nbrs = []
        for a, b in cluster.edges:
            if a == idx:
                nbrs.append(b)
            elif b == idx:
                nbrs.append(a)
        return nbrs

    def _get_pair_neighbor_types(self, cluster, idx1: int, idx2: int) -> List[int]:
        """
        Return su_types of aromatic sites (SU 5-13) directly bonded to
        the {idx1, idx2} pair, excluding the pair itself.
        """
        exclude = {idx1, idx2}
        nbr_indices = set()
        for idx in (idx1, idx2):
            for n in self._edge_neighbors(cluster, idx):
                if n not in exclude:
                    nbr_indices.add(n)
        return [cluster.sites[i].su_type for i in nbr_indices
                if 5 <= cluster.sites[i].su_type <= 13]

    def _get_single_neighbor_types(self, cluster, idx: int) -> List[int]:
        """
        Return su_types of aromatic sites (SU 5-13) directly bonded to
        site *idx*.
        """
        return [cluster.sites[n].su_type
                for n in self._edge_neighbors(cluster, idx)
                if 5 <= cluster.sites[n].su_type <= 13]

    @staticmethod
    def _hop1_matches(neighbor_types: List[int], hop1: List[int]) -> bool:
        """Check multiset containment: every element in *hop1* can be
        found (and consumed) in *neighbor_types*."""
        if not hop1:
            return True
        pool = list(neighbor_types)
        for h in hop1:
            if h in pool:
                pool.remove(h)
            else:
                return False
        return True

    def _replace_in_cluster(self, state: MCTSState, target_su: int,
                            hop1: List[int], replace_two: bool):
        """
        Find the best SU-13 site(s) inside placed aromatic clusters and
        replace with *target_su*.

        Constraints:
          - Both sites of a 13-13 pair MUST belong to the same aromatic
            cluster (SU 5-13 sites only); bridging chain nodes are excluded.
          - The two sites must share a direct intra-ring bond (checked
            via cluster.edges, NOT coordinate proximity).

        Selection strategy:
          1. Collect all valid candidates.
          2. Partition into hop1-matched vs unmatched.
          3. From the preferred pool (hop1-matched first), pick the best
             via NMR trial evaluation; random fallback if no evaluator.
        """
        candidates = []  # (cluster_idx, [site_idx, ...])

        for c_idx, c in enumerate(state.graph.clusters):
            if not getattr(c, 'placed', False):
                continue

            # Only consider SU-13 sites (aromatic CH)
            sites_13 = [i for i, s in enumerate(c.sites) if s.su_type == 13]
            if not sites_13:
                continue

            if replace_two:
                # Directly bonded 13-13 pairs within the same cluster
                for ii in range(len(sites_13)):
                    for jj in range(ii + 1, len(sites_13)):
                        si, sj = sites_13[ii], sites_13[jj]
                        if self._are_directly_bonded(c, si, sj):
                            candidates.append((c_idx, [si, sj]))
            else:
                for idx in sites_13:
                    candidates.append((c_idx, [idx]))

        if not candidates:
            self._log(f"[StateSubstitutor] Warning: no SU-13 candidate for "
                      f"SU={target_su} (replace_two={replace_two})")
            self.remaining_sus[target_su] -= 1
            return

        # Partition by 1-hop match
        hop1_matched, hop1_unmatched = [], []
        for cand in candidates:
            c_idx, s_indices = cand
            cluster = state.graph.clusters[c_idx]
            if replace_two:
                nbr = self._get_pair_neighbor_types(
                    cluster, s_indices[0], s_indices[1])
            else:
                nbr = self._get_single_neighbor_types(cluster, s_indices[0])

            if self._hop1_matches(nbr, hop1):
                hop1_matched.append(cand)
            else:
                hop1_unmatched.append(cand)

        pool = hop1_matched if hop1_matched else hop1_unmatched

        # Pick the best candidate
        chosen = self._pick_best_candidate(state, pool, target_su,
                                           replace_two)

        # Apply the replacement
        self._apply_cluster_replacement(state, chosen, target_su,
                                        replace_two)
        self.remaining_sus[target_su] -= 1

    def _pick_best_candidate(self, state, pool, target_su, replace_two):
        """
        Among *pool* candidates, pick the one yielding the highest NMR
        score after a trial replacement.  Falls back to random (or first)
        if no NMR evaluator is available or the pool has only one entry.
        """
        if len(pool) == 1:
            return pool[0]

        if self.nmr_eval_fn is None:
            return random.choice(pool) if self.randomize else pool[0]

        best_score = -1e9
        best_cand = pool[0]
        for cand in pool:
            trial = state.copy()
            self._apply_cluster_replacement(trial, cand, target_su,
                                            replace_two)
            score = self.nmr_eval_fn(trial)
            if score > best_score:
                best_score = score
                best_cand = cand
        return best_cand

    @staticmethod
    def _apply_cluster_replacement(state, cand, target_su, replace_two):
        """Execute the actual SU replacement on *state*.

        For a 13-13 pair merge:
          1. Set site1.su_type = target_su with averaged coordinates.
          2. Re-route all external edge references from site2 → site1.
          3. Delete site2 from cluster.sites.
          4. Fix cluster.edges indices (shift down indices > idx2,
             remove self-loops and duplicates).
        """
        c_idx, s_indices = cand
        cluster = state.graph.clusters[c_idx]

        if replace_two and len(s_indices) == 2:
            idx1, idx2 = sorted(s_indices)  # ensure idx1 < idx2
            s1 = cluster.sites[idx1]
            s2 = cluster.sites[idx2]
            s2_uid = s2.uid

            # Assign target SU to s1 with averaged position
            s1.su_type = target_su
            s1.axial = (
                round((s1.axial[0] + s2.axial[0]) / 2.0),
                round((s1.axial[1] + s2.axial[1]) / 2.0),
            )
            s1.pos2d = (
                (s1.pos2d[0] + s2.pos2d[0]) / 2.0,
                (s1.pos2d[1] + s2.pos2d[1]) / 2.0,
            )

            # Re-route all external edge references: s2 → s1
            all_edges = (state.graph.rigid
                         + state.graph.flex
                         + getattr(state.graph, 'side', [])
                         + getattr(state.graph, 'branch', []))
            for edge in all_edges:
                if hasattr(edge, 'u') and edge.u == s2_uid:
                    edge.u = s1.uid
                if hasattr(edge, 'v') and edge.v == s2_uid:
                    edge.v = s1.uid

            # Update cluster.edges: redirect idx2 → idx1, remove self-loops
            new_edges_set = set()
            for a, b in cluster.edges:
                if a == idx2:
                    a = idx1
                if b == idx2:
                    b = idx1
                if a != b:
                    new_edges_set.add((min(a, b), max(a, b)))

            # Delete site2 from the sites list
            del cluster.sites[idx2]

            # Shift all indices > idx2 down by 1
            fixed_edges = set()
            for a, b in new_edges_set:
                fa = a - 1 if a > idx2 else a
                fb = b - 1 if b > idx2 else b
                if fa != fb:
                    fixed_edges.add((min(fa, fb), max(fa, fb)))
            cluster.edges = sorted(fixed_edges)

            # Update remaining site uid/idx fields for consistency
            for new_idx, site in enumerate(cluster.sites):
                site.idx = new_idx
        else:
            cluster.sites[s_indices[0]].su_type = target_su

    # ================================================================
    # Path extraction
    # ================================================================
    def _extract_paths(self, graph: ConnectionGraph, include_branch: bool = False) -> List[List[Any]]:
        """Build site/chain-node sequences from flex and side edges."""
        site_map = {}
        for c in graph.clusters:
            if getattr(c, 'placed', False):
                for s in c.sites:
                    site_map[s.uid] = s

        chain_map = {n.uid: n for n in graph.chains}

        def resolve(node_or_uid):
            if node_or_uid is None:
                return None
            uid = node_or_uid if isinstance(node_or_uid, str) else getattr(node_or_uid, 'uid', None)
            if uid is None:
                return node_or_uid
            return site_map.get(uid) or chain_map.get(uid) or node_or_uid

        paths = []
        for e in graph.flex:
            u_site = resolve(e.u)
            v_site = resolve(e.v)
            chain_nodes = [resolve(n) for n in e.chain]
            if u_site and v_site:
                paths.append([u_site] + [n for n in chain_nodes if n] + [v_site])

        for e in getattr(graph, 'side', []):
            u_site = resolve(e.u)
            chain_nodes = [resolve(n) for n in e.chain]
            if u_site:
                paths.append([u_site] + [n for n in chain_nodes if n])

        if include_branch:
            for e in getattr(graph, 'branch', []):
                base = resolve(e.base)
                target = resolve(e.target) if e.target else None
                chain_nodes = [resolve(n) for n in e.chain]
                if base and target:
                    paths.append([base] + [n for n in chain_nodes if n] + [target])
                elif base:
                    paths.append([base] + [n for n in chain_nodes if n])

        return paths

    @staticmethod
    def _find_all_matches(types: List[int], pattern: List[Any]) -> List[int]:
        """Return all start indices where *pattern* matches in *types*."""
        L = len(pattern)
        matches = []
        for i in range(len(types) - L + 1):
            ok = True
            for j in range(L):
                p = pattern[j]
                if isinstance(p, tuple):
                    if types[i + j] not in p:
                        ok = False
                        break
                else:
                    if types[i + j] != p:
                        ok = False
                        break
            if ok:
                matches.append(i)
        return matches

    @staticmethod
    def _do_replace(path, start_idx: int, find_p, repl_p):
        """Write *repl_p* SU types over *path* at *start_idx*."""
        for j in range(len(repl_p)):
            rp = repl_p[j]
            fp = find_p[j]
            # Preserve original aliphatic sub-type when replacement is
            # the generic ALI placeholder 23
            if (rp == 23 and isinstance(fp, tuple)
                    and path[start_idx + j].su_type in fp):
                continue
            path[start_idx + j].su_type = rp


class SubstitutionStage:
    """Unified-search wrapper for the final substitution phase.

    The underlying substitution rules remain unchanged: each action is a
    randomized macro-substitution trial, and the MCTS engine chooses among
    several such trials using the stage evaluator.
    """

    def __init__(self,
                 state: MCTSState,
                 original_su_counts: Dict[int, int],
                 nodes_csv: Optional[str] = None,
                 n_variants: int = 6,
                 seed_base: int = 0,
                 nmr_eval_fn: Optional[Callable] = None):
        self.state = state
        self.original_su_counts = copy.deepcopy(original_su_counts)
        self.nodes_csv = nodes_csv
        self.n_variants = max(1, int(n_variants))
        self.seed_base = int(seed_base)
        self.nmr_eval_fn = nmr_eval_fn
        self.substitutor = StateSubstitutor(
            self.original_su_counts,
            nodes_csv=self.nodes_csv,
            nmr_eval_fn=self.nmr_eval_fn,
        )
        self._done = False
        self._result: Dict[str, Any] = {
            'applied': {},
            'remaining': {},
            'complete': False,
            'remaining_total': 0,
            'l1_delta': 0,
        }

    def clone(self) -> 'SubstitutionStage':
        new = SubstitutionStage(
            self.state.copy(),
            self.original_su_counts,
            nodes_csv=self.nodes_csv,
            n_variants=self.n_variants,
            seed_base=self.seed_base,
            nmr_eval_fn=self.nmr_eval_fn,
        )
        new.substitutor = copy.deepcopy(self.substitutor)
        new._done = bool(self._done)
        new._result = copy.deepcopy(self._result)
        return new

    def is_done(self) -> bool:
        return bool(self._done)

    def get_candidates(self, k: int = 8) -> List[Dict[str, Any]]:
        if self._done:
            return []
        out = []
        limit = min(int(max(1, k)), int(self.n_variants))
        for i in range(limit):
            out.append({
                'type': 'subst',
                'seed': int(self.seed_base + i),
                'score': float(i),
            })
        return out

    def step(self, action: Dict[str, Any]) -> bool:
        if self._done:
            return False
        try:
            seed = int(action.get('seed', self.seed_base))
        except Exception:
            seed = int(self.seed_base)
        before_dist = self.state.get_su_distribution()
        random.seed(seed)
        ok = bool(self.substitutor.substitute_all(self.state, randomize=True))
        actual = self.state.get_su_distribution()
        su_change = compute_su_delta(actual, before_dist)
        remaining = compute_su_delta(actual, self.original_su_counts)
        l1_delta = compute_su_l1_delta(actual, self.original_su_counts)
        tracked_remaining = dict(self.substitutor.last_summary.get('remaining', {}) or {})
        tracked_applied = dict(self.substitutor.last_summary.get('applied', {}) or {})
        self._result = {
            'seed': int(seed),
            'applied': su_change,
            'remaining': remaining,
            'complete': not remaining,
            'remaining_total': int(len(remaining)),
            'l1_delta': int(l1_delta),
            'success': bool(ok and not remaining),
            'tracked_applied': tracked_applied,
            'tracked_remaining': tracked_remaining,
            'target_distribution': dict(self.original_su_counts),
            'before_distribution': before_dist,
            'after_distribution': actual,
        }
        self._done = True
        self.state.stage_step += 1
        return True

    def get_result(self) -> Dict[str, Any]:
        return dict(self._result)
