import math
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any


# ==================== Direction Constants ====================
RU, RD, LU, LD, UP, DN = (1, 1), (1, 0), (-1, 0), (-1, -1), (0, 1), (0, -1)
OPPOSITE = {RU: LD, RD: LU, LU: RD, LD: RU, UP: DN, DN: UP}

# Hexagon vertex offsets (clockwise from right)
HEX_VERTEX_OFFSETS = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))

# ==================== Hex Grid ====================

class HexGrid:
    """Hex grid coordinate system"""
    
    @staticmethod
    def axial_to_cart(q: float, r: float, scale: float = 1.0) -> Tuple[float, float]:
        """Convert axial coordinates to Cartesian"""
        sqrt3_half = math.sqrt(3.0) / 2.0
        x = scale * (sqrt3_half * q)
        y = scale * (r - 0.5 * q)
        return x, y
    
    @staticmethod
    def neighbor(q: int, r: int, d: Tuple[int, int]) -> Tuple[int, int]:
        return (q + d[0], r + d[1])
    
    @staticmethod
    def distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        aq, ar = a
        bq, br = b
        return int((abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) / 2)


# ==================== Core Data Types ====================

@dataclass
class Site:
    """Aromatic cluster site"""
    uid: str
    idx: int
    cluster_id: int
    su_type: int
    axial: Tuple[int, int]
    pos2d: Tuple[float, float] = (0.0, 0.0)
    occupied: bool = False
    
    def __hash__(self): return hash(self.uid)
    def __eq__(self, o): return isinstance(o, Site) and self.uid == o.uid


@dataclass
class ChainNode:
    """Chain node"""
    uid: str
    su_type: int
    axial: Tuple[int, int]
    pos2d: Tuple[float, float] = (0.0, 0.0)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self): return hash(self.uid)
    def __eq__(self, o): return isinstance(o, ChainNode) and self.uid == o.uid


@dataclass
class AromaticCluster:
    """Polycyclic aromatic cluster"""
    id: int
    kind: str
    rings: int
    sites: List[Site] = field(default_factory=list)
    centers: List[Tuple[int, int]] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    placed: bool = False
    translation: Tuple[int, int] = (0, 0)
    connected: Set[int] = field(default_factory=set)
    used_directions: Set[str] = field(default_factory=set)
    
    def centroid(self) -> Tuple[float, float]:
        if not self.sites: return (0.0, 0.0)
        return (sum(s.pos2d[0] for s in self.sites) / len(self.sites),
                sum(s.pos2d[1] for s in self.sites) / len(self.sites))
    
    def translate(self, tq: int, tr: int):
        self.translation = (self.translation[0] + tq, self.translation[1] + tr)
        for s in self.sites:
            s.axial = (s.axial[0] + tq, s.axial[1] + tr)
            s.pos2d = HexGrid.axial_to_cart(*s.axial)
        self.centers = [(c[0] + tq, c[1] + tr) for c in self.centers]
        self.placed = True
    
    def free_sites(self, su: Optional[int] = None) -> List[Tuple[int, Site]]:
        return [(i, s) for i, s in enumerate(self.sites) 
                if not s.occupied and (su is None or s.su_type == su)]
    
    def __hash__(self): return hash(self.id)
    def __eq__(self, o): return isinstance(o, AromaticCluster) and self.id == o.id


@dataclass
class EdgeRigid:
    """Rigid connection edge (10-10)"""
    u: str  # site_a uid
    v: str  # site_b uid
    cluster_a: int = -1
    cluster_b: int = -1
    dir_a: str = ''  # direction label on cluster_a side
    dir_b: str = ''  # direction label on cluster_b side
    line: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class EdgeFlex:
    """Flexible connection edge"""
    u: str
    v: str
    chain: List[ChainNode] = field(default_factory=list)


@dataclass
class EdgeSide:
    """Sidechain edge"""
    u: str
    chain: List[ChainNode] = field(default_factory=list)


@dataclass
class EdgeBranch:
    """Branch substitution edge"""
    base: str
    chain: List[ChainNode] = field(default_factory=list)
    target: Optional[str] = None


@dataclass
class ConnectionGraph:
    """Molecular connection graph"""
    clusters: List[AromaticCluster] = field(default_factory=list)
    chains: List[ChainNode] = field(default_factory=list)
    rigid: List[EdgeRigid] = field(default_factory=list)
    flex: List[EdgeFlex] = field(default_factory=list)
    side: List[EdgeSide] = field(default_factory=list)
    branch: List[EdgeBranch] = field(default_factory=list)


def count_graph_su_distribution(graph: ConnectionGraph,
                                dedupe_chain_uids: bool = True,
                                dedupe_axials: bool = False) -> Dict[int, int]:
    """Count SU types from placed cluster sites and chain nodes.

    By default this counts every logical node in the graph. Axial-coordinate
    deduplication remains available as an opt-in diagnostic mode when callers
    want to suppress accidental overlaps.
    """
    su_dist: Dict[int, int] = {}
    seen_axials: Set[Tuple[int, int]] = set()

    for c in graph.clusters:
        if not c.placed:
            continue
        for s in c.sites:
            axial = (int(s.axial[0]), int(s.axial[1]))
            if dedupe_axials and axial in seen_axials:
                continue
            seen_axials.add(axial)
            su = int(s.su_type)
            su_dist[su] = su_dist.get(su, 0) + 1

    seen_chain_keys: Set[Any] = set()
    for cn in graph.chains:
        axial = (int(cn.axial[0]), int(cn.axial[1]))
        if dedupe_axials and axial in seen_axials:
            continue
        if dedupe_chain_uids:
            uid = getattr(cn, 'uid', None)
            chain_key: Any = ('uid', str(uid)) if uid is not None else ('obj', id(cn))
            if chain_key in seen_chain_keys:
                continue
            seen_chain_keys.add(chain_key)
        seen_axials.add(axial)
        su = int(cn.su_type)
        su_dist[su] = su_dist.get(su, 0) + 1

    return su_dist


def find_overlapping_axials(graph: ConnectionGraph) -> Dict[Tuple[int, int], List[Tuple[str, int]]]:
    """Return all axial coordinates occupied by multiple logical nodes."""
    slots: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
    for c in graph.clusters:
        if not c.placed:
            continue
        for s in c.sites:
            axial = (int(s.axial[0]), int(s.axial[1]))
            slots.setdefault(axial, []).append((str(s.uid), int(s.su_type)))
    for cn in graph.chains:
        axial = (int(cn.axial[0]), int(cn.axial[1]))
        slots.setdefault(axial, []).append((str(cn.uid), int(cn.su_type)))
    return {ax: vals for ax, vals in slots.items() if len(vals) > 1}


def compute_su_delta(actual: Dict[int, int], target: Dict[int, int]) -> Dict[int, int]:
    """Return signed count differences as actual - target for each SU."""
    delta: Dict[int, int] = {}
    for su in sorted(set(actual) | set(target)):
        diff = int(actual.get(su, 0)) - int(target.get(su, 0))
        if diff != 0:
            delta[int(su)] = diff
    return delta


def compute_su_l1_delta(actual: Dict[int, int], target: Dict[int, int]) -> int:
    """Return the L1 distance between two SU distributions."""
    return int(sum(abs(diff) for diff in compute_su_delta(actual, target).values()))


# ==================== SU Constants ====================

SU_NAMES = {
    0: "Amide", 1: "Carboxylic", 2: "Ester", 3: "Aldehyde_Ketone", 4: "Nitrile",
    5: "O_Aro_C", 6: "N_Aro_C", 7: "S_Aro_C", 8: "X_Aro_C", 9: "Keto_Aro_C",
    10: "Aryl_Aro_C", 11: "Alkyl_Aro_C", 12: "Bridgehead_C", 13: "Aro_CH",
    14: "Vinyl_Cq", 15: "Vinyl_CH", 16: "Vinyl_CH2", 17: "Alkynyl_Cq", 18: "Alkynyl_CH",
    19: "Alcohol_C", 20: "Amine_C", 21: "Halogen_C", 22: "CH3", 23: "CH2", 24: "CH", 25: "Cq",
    26: "Hetero_N", 27: "Amine_N", 28: "OH", 29: "Ether_O", 30: "Hetero_S", 31: "Thioether", 32: "Halogen"
}

AROMATIC_SU = set(range(5, 14))
ALIPHATIC_SU = set(range(19, 26))
HETERO_SU = set(range(26, 33))

# ==================== MCTS State ====================

@dataclass
class MCTSState:
    """MCTS search state snapshot"""
    graph: ConnectionGraph
    su_counts: Dict[int, int]           # Remaining SU counts
    reserved_su: Dict[int, int]         # Reserved SU (e.g., SU 10 for SU 4)
    stage: str = 'rigid'                # Current stage
    stage_mode: str = 'default'         # Current stage mode / fallback mode
    step_count: int = 0                 # Current step number
    stage_step: int = 0                 # Stage-local step counter
    stage_meta: Dict[str, Any] = field(default_factory=dict)
    
    # Union-Find for connectivity tracking
    _parent: Dict[int, int] = field(default_factory=dict)
    _rank: Dict[int, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize Union-Find for all clusters
        for c in self.graph.clusters:
            self._parent[c.id] = c.id
            self._rank[c.id] = 0
    
    def _find(self, x: int) -> int:
        """Find root with path compression"""
        if self._parent.get(x, x) != x:
            self._parent[x] = self._find(self._parent[x])
        return self._parent.get(x, x)
    
    def _union(self, a: int, b: int):
        """Union by rank"""
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if self._rank.get(ra, 0) < self._rank.get(rb, 0):
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank.get(ra, 0) == self._rank.get(rb, 0):
            self._rank[ra] = self._rank.get(ra, 0) + 1
    
    def get_available(self, su_type: int) -> int:
        """Get available SU count (total - reserved)"""
        total = self.su_counts.get(su_type, 0)
        reserved = self.reserved_su.get(su_type, 0)
        return max(0, total - reserved)
    
    def copy(self) -> 'MCTSState':
        """Deep copy state for MCTS simulation"""
        new_graph = ConnectionGraph(
            clusters=[copy.deepcopy(c) for c in self.graph.clusters],
            chains=[copy.deepcopy(n) for n in self.graph.chains],
            rigid=[copy.deepcopy(e) for e in self.graph.rigid],
            flex=[copy.deepcopy(e) for e in self.graph.flex],
            side=[copy.deepcopy(e) for e in self.graph.side],
            branch=[copy.deepcopy(e) for e in self.graph.branch],
        )
        chain_lookup = {n.uid: n for n in new_graph.chains}
        for edge in new_graph.flex:
            edge.chain = [chain_lookup.get(n.uid, n) for n in edge.chain]
        for edge in new_graph.side:
            edge.chain = [chain_lookup.get(n.uid, n) for n in edge.chain]
        for edge in new_graph.branch:
            edge.chain = [chain_lookup.get(n.uid, n) for n in edge.chain]
        new_state = MCTSState(
            graph=new_graph,
            su_counts=self.su_counts.copy(),
            reserved_su=self.reserved_su.copy(),
            stage=self.stage,
            stage_mode=self.stage_mode,
            step_count=self.step_count,
            stage_step=self.stage_step,
            stage_meta=copy.deepcopy(self.stage_meta),
        )
        new_state._parent = self._parent.copy()
        new_state._rank = self._rank.copy()
        return new_state
    
    def get_component_count(self) -> int:
        """Get number of connected components"""
        placed = [c.id for c in self.graph.clusters if c.placed]
        if not placed:
            return 0
        roots = set(self._find(cid) for cid in placed)
        return len(roots)
    
    def is_all_connected(self) -> bool:
        """Check if all placed clusters are in one component"""
        placed = [c.id for c in self.graph.clusters if c.placed]
        if len(placed) <= 1:
            return True
        root = self._find(placed[0])
        return all(self._find(cid) == root for cid in placed)
    
    def get_su_distribution(self, dedupe_axials: bool = False) -> Dict[int, int]:
        """Get the current structural-unit distribution over the graph."""
        return count_graph_su_distribution(
            self.graph,
            dedupe_chain_uids=True,
            dedupe_axials=bool(dedupe_axials),
        )
        
    def get_placed_vertices(self) -> Set[Tuple[int, int]]:
        placed = set()
        for c in self.graph.clusters:
            if c.placed:
                for site in c.sites:
                    placed.add(site.axial)
                for center in c.centers:
                    placed.add(center)
        for cn in self.graph.chains:
            placed.add(cn.axial)
        return placed

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box of all placed sites and chain nodes"""
        xs, ys = [], []
        for c in self.graph.clusters:
            if c.placed:
                for s in c.sites:
                    xs.append(s.pos2d[0])
                    ys.append(s.pos2d[1])
        for cn in self.graph.chains:
            xs.append(cn.pos2d[0])
            ys.append(cn.pos2d[1])
        if not xs:
            return (0, 0, 0, 0)
        return (min(xs), min(ys), max(xs), max(ys))

    def get_axial_bbox(self) -> Tuple[int, int, int, int]:
        """Get axial-coordinate bounding box over all placed vertices/centers."""
        qs, rs = [], []
        for c in self.graph.clusters:
            if c.placed:
                for s in c.sites:
                    qs.append(int(s.axial[0]))
                    rs.append(int(s.axial[1]))
                for center in c.centers:
                    qs.append(int(center[0]))
                    rs.append(int(center[1]))
        for cn in self.graph.chains:
            qs.append(int(cn.axial[0]))
            rs.append(int(cn.axial[1]))
        if not qs:
            return (0, 0, 0, 0)
        return (min(qs), min(rs), max(qs), max(rs))

    def get_axial_spans(self) -> Tuple[float, float]:
        q0, r0, q1, r1 = self.get_axial_bbox()
        return float(q1 - q0), float(r1 - r0)

    def get_qr_ratio(self) -> float:
        q_span, r_span = self.get_axial_spans()
        if q_span < 1e-6 and r_span < 1e-6:
            return 1.0
        if r_span < 1e-6:
            return float('inf')
        return float(q_span / r_span)

    def get_qr_shape_score(self,
                           min_ratio: float = 0.9,
                           max_ratio: float = 2.3) -> float:
        """Score in [-1, 1]: positive if q/r span ratio stays within range."""
        ratio = self.get_qr_ratio()
        if ratio == float('inf'):
            return -1.0
        lo = float(min_ratio)
        hi = float(max_ratio)
        if lo <= ratio <= hi:
            mid = math.sqrt(lo * hi)
            return max(0.0, 1.0 - abs(ratio - mid) / max(mid, 1e-6))
        if ratio < lo:
            return -min(1.0, (lo - ratio) / max(lo, 1e-6))
        return -min(1.0, (ratio - hi) / max(hi, 1e-6))

    def get_spatial_uniformity_score(self, bins: int = 3) -> float:
        """Score in [0, 1]: higher means less local concentration in q/r space."""
        q0, r0, q1, r1 = self.get_axial_bbox()
        pts: List[Tuple[int, int]] = list(self.get_placed_vertices())
        if len(pts) <= 1:
            return 1.0

        q_span = max(1.0, float(q1 - q0))
        r_span = max(1.0, float(r1 - r0))
        n_bins = max(2, int(bins))
        counts = [[0 for _ in range(n_bins)] for _ in range(n_bins)]

        for q, r in pts:
            qi = min(n_bins - 1, int(((float(q) - float(q0)) / q_span) * n_bins))
            ri = min(n_bins - 1, int(((float(r) - float(r0)) / r_span) * n_bins))
            counts[qi][ri] += 1

        flat = [c for row in counts for c in row]
        occupied = sum(1 for c in flat if c > 0)
        avg = float(sum(flat)) / float(len(flat))
        variance = float(sum((c - avg) ** 2 for c in flat)) / float(len(flat))
        occupied_score = float(occupied) / float(len(flat))
        variance_penalty = min(1.0, variance / max(avg * avg + 1e-6, 1.0))
        return max(0.0, min(1.0, 0.65 * occupied_score + 0.35 * (1.0 - variance_penalty)))
    
    def get_aspect_ratio(self) -> float:
        """Get aspect ratio of current structure"""
        x0, y0, x1, y1 = self.get_bbox()
        w, h = x1 - x0, y1 - y0
        if w < 1e-6 or h < 1e-6:
            return 1.0
        return max(w, h) / min(w, h)

    def state_signature(self) -> Tuple[Any, ...]:
        """Stable structural signature for caching/equality in search."""
        cluster_sig = []
        for c in sorted(self.graph.clusters, key=lambda item: int(item.id)):
            site_sig = tuple(sorted(
                (
                    str(s.uid),
                    int(s.idx),
                    int(s.su_type),
                    int(s.cluster_id),
                    int(s.axial[0]),
                    int(s.axial[1]),
                    int(bool(s.occupied)),
                )
                for s in c.sites
            ))
            cluster_sig.append((
                int(c.id),
                str(c.kind),
                int(c.rings),
                int(bool(c.placed)),
                tuple(sorted((int(q), int(r)) for q, r in c.centers)),
                tuple(sorted((int(a), int(b)) for a, b in c.edges)),
                tuple(sorted(int(x) for x in c.connected)),
                tuple(sorted(str(x) for x in c.used_directions)),
                site_sig,
            ))

        chain_sig = tuple(sorted(
            (
                str(n.uid),
                int(n.su_type),
                int(n.axial[0]),
                int(n.axial[1]),
            )
            for n in self.graph.chains
        ))
        rigid_sig = tuple(sorted(
            (
                str(e.u),
                str(e.v),
                int(e.cluster_a),
                int(e.cluster_b),
                str(e.dir_a),
                str(e.dir_b),
            )
            for e in self.graph.rigid
        ))
        flex_sig = tuple(sorted(
            (
                str(e.u),
                str(e.v),
                tuple(str(cn.uid) for cn in e.chain),
            )
            for e in self.graph.flex
        ))
        side_sig = tuple(sorted(
            (
                str(e.u),
                tuple(str(cn.uid) for cn in e.chain),
            )
            for e in self.graph.side
        ))
        branch_sig = tuple(sorted(
            (
                str(e.base),
                str(e.target) if e.target is not None else '',
                tuple(str(cn.uid) for cn in e.chain),
            )
            for e in self.graph.branch
        ))
        su_sig = tuple(sorted((int(k), int(v)) for k, v in self.su_counts.items()))
        reserved_sig = tuple(sorted((int(k), int(v)) for k, v in self.reserved_su.items()))
        stage_meta_sig = tuple(sorted((str(k), repr(v)) for k, v in self.stage_meta.items()))
        return (
            str(self.stage),
            str(self.stage_mode),
            int(self.step_count),
            int(self.stage_step),
            su_sig,
            reserved_sig,
            stage_meta_sig,
            tuple(cluster_sig),
            chain_sig,
            rigid_sig,
            flex_sig,
            side_sig,
            branch_sig,
        )


@dataclass
class MCTSHistory:
    """MCTS search history recording"""
    actions: List[Dict] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    states_info: List[Dict] = field(default_factory=list)  # Lightweight state info
    
    def record(self, action: Dict, reward: float, state: MCTSState):
        """Record one step"""
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_info.append({
            'stage': state.stage,
            'step': state.step_count,
            'components': state.get_component_count(),
            'aspect_ratio': state.get_aspect_ratio(),
            'rigid_edges': len(state.graph.rigid),
        })
    
    def total_reward(self) -> float:
        """Get total accumulated reward"""
        return sum(self.rewards)
    
    def summary(self) -> Dict:
        """Get history summary"""
        return {
            'total_steps': len(self.actions),
            'total_reward': self.total_reward(),
            'actions': self.actions,
            'rewards': self.rewards,
        }


class BaseStageEvaluator:
    def __init__(self,
                 nmr_score_fn: Optional[Any] = None,
                 nmr_weight: float = 20.0):
        self.nmr_score_fn = nmr_score_fn
        self.nmr_weight = float(nmr_weight)

    def prior(self, stage: Any, action: Dict[str, Any]) -> float:
        raw = float(action.get('score', 0.0))
        state = getattr(stage, 'state', None)
        if state is not None:
            try:
                raw -= 12.0 * float(state.get_qr_shape_score(0.9, 2.3))
                raw -= 4.0 * float(state.get_spatial_uniformity_score())
            except Exception:
                pass
        if self.nmr_score_fn is not None:
            try:
                raw -= 10.0 * float(self.nmr_score_fn(stage))
            except Exception:
                pass
        clipped = max(-200.0, min(200.0, raw))
        return 1.0 / (1.0 + math.exp(clipped / 25.0))

    def select_rollout_action(self, stage: Any, candidates):
        if not candidates:
            return None
        top_n = min(3, len(candidates))
        import random
        return random.choice(candidates[:top_n])

    def evaluate(self, stage: Any) -> float:
        return 0.0


class RigidStageEvaluator(BaseStageEvaluator):
    def __init__(self, max_cluster_size: int = 4):
        super().__init__(nmr_score_fn=None, nmr_weight=0.0)
        self.max_cluster_size = max(1, int(max_cluster_size))

    def evaluate(self, stage: Any) -> float:
        result = stage.get_result()
        rigid_clusters = result.get('rigid_clusters', []) or []
        n_rc = max(1, len(rigid_clusters))
        n_with_edges = sum(1 for rc in rigid_clusters if rc.get('has_rigid_edges', False))
        remaining_su10 = float(result.get('remaining_su10', 0))
        total_clusters = max(1, int(result.get('total_clusters', 1)))
        placed_clusters = int(result.get('placed_clusters', 0))
        aspect = float(result.get('aspect_ratio', 1.0))

        score = 0.0
        score += 25.0 * (float(n_with_edges) / float(n_rc))
        score += 8.0 * (float(placed_clusters) / float(total_clusters))
        score += 10.0 * (1.0 / (1.0 + max(0.0, remaining_su10)))
        score += 6.0 * (1.0 / (1.0 + abs(aspect - 1.6)))

        giant_penalty = 0.0
        for rc in rigid_clusters:
            total_rings = int(rc.get('total_rings', 0))
            if total_rings > int(self.max_cluster_size * 2):
                giant_penalty += float(total_rings - self.max_cluster_size * 2)
        score -= 5.0 * giant_penalty
        return score


class FlexStageEvaluator(BaseStageEvaluator):
    def evaluate(self, stage: Any) -> float:
        result = stage.get_result()
        bridges_total = max(1, int(result.get('bridges_total', 1)))
        bridges_done = int(result.get('bridges_done', 0))
        components = int(result.get('components', 0))
        all_connected = bool(result.get('all_connected', False))
        aspect = float(result.get('aspect_ratio', 1.0))
        state = getattr(stage, 'state', None)

        score = 0.0
        score += 30.0 * (float(bridges_done) / float(bridges_total))
        score += 15.0 * (1.0 / (1.0 + max(0, components - 1)))
        score += 5.0 * (1.0 / (1.0 + abs(aspect - 1.8)))
        if all_connected:
            score += 20.0
        if state is not None:
            score += 8.0 * float(state.get_qr_shape_score(0.9, 2.3))
            score += 6.0 * float(state.get_spatial_uniformity_score())

        if self.nmr_score_fn is not None:
            progress = float(bridges_done) / float(bridges_total)
            if all_connected or progress >= 0.5:
                try:
                    score += self.nmr_weight * float(self.nmr_score_fn(stage))
                except Exception:
                    pass
        return score


class BranchStageEvaluator(BaseStageEvaluator):
    def evaluate(self, stage: Any) -> float:
        result = stage.get_result()
        total = max(1, int(result.get('branches_total', 1)))
        done = int(result.get('branches_placed', 0))
        base_score = float(stage.score()) if hasattr(stage, 'score') else 0.0
        state = getattr(stage, 'state', None)
        score = 10.0 * (float(done) / float(total)) + base_score
        if state is not None:
            score += 8.0 * float(state.get_qr_shape_score(0.9, 2.3))
            score += 6.0 * float(state.get_spatial_uniformity_score())
        if self.nmr_score_fn is not None:
            try:
                score += self.nmr_weight * float(self.nmr_score_fn(stage))
            except Exception:
                pass
        return score


class SideStageEvaluator(BaseStageEvaluator):
    def evaluate(self, stage: Any) -> float:
        result = stage.get_result()
        total = max(1, int(result.get('sides_total', 1)))
        done = int(result.get('sides_placed', 0))
        base_score = float(stage.score()) if hasattr(stage, 'score') else 0.0
        state = getattr(stage, 'state', None)
        score = 10.0 * (float(done) / float(total)) + base_score
        if state is not None:
            score += 8.0 * float(state.get_qr_shape_score(0.9, 2.3))
            score += 6.0 * float(state.get_spatial_uniformity_score())
        if self.nmr_score_fn is not None:
            try:
                score += self.nmr_weight * float(self.nmr_score_fn(stage))
            except Exception:
                pass
        return score


class SubstitutionStageEvaluator(BaseStageEvaluator):
    def evaluate(self, stage: Any) -> float:
        result = stage.get_result()
        remaining_total = int(result.get('remaining_total', 0))
        l1_delta = int(result.get('l1_delta', 0))
        complete = bool(result.get('complete', False))

        score = 0.0
        if complete:
            score += 20.0
        score -= 4.0 * float(remaining_total)
        score -= 0.5 * float(l1_delta)
        if self.nmr_score_fn is not None:
            try:
                score += self.nmr_weight * float(self.nmr_score_fn(stage))
            except Exception:
                pass
        return score
