import os
import re
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
from .RL_state import HexGrid, Site, AromaticCluster, ConnectionGraph, SU_NAMES, HEX_VERTEX_OFFSETS


def load_su_counts(csv_path: str) -> Dict[int, int]:
    """Load SU counts from CSV, compatible with multiple formats"""
    df = pd.read_csv(csv_path)
    for col in ['center_su_idx', 'su_type', 'su_idx', 'type']:
        if col in df.columns:
            counts = Counter(df[col].values)
            return dict(counts)
    raise ValueError(f"Cannot find SU type column: {csv_path}")


def _parse_template_key(template_str: str) -> Tuple[int, List[int], List[int]]:
    """Parse template_key string: (su_type, (hop1,...), (hop2,...))"""
    match = re.match(r'\((\d+),\s*\(([^)]*)\),\s*\(([^)]*)\)\)', template_str)
    if not match:
        return -1, [], []
    su_type = int(match.group(1))
    hop1 = [int(x) for x in re.findall(r'\d+', match.group(2))]
    hop2 = [int(x) for x in re.findall(r'\d+', match.group(3))]
    return su_type, hop1, hop2


def analyze_bridgehead_from_csv(csv_path: str) -> Tuple[int, int, int, int, int, int, int]:
    """Classify SU12 bridgeheads for aromatic-cluster generation.

    Returns:
        x: coronene/pyrene bridgeheads (hop1 is exactly three 12s)
        y: benzo-pyrene/perylene bridgeheads
        z: chrysene/triphenylene bridgeheads
        m: tetracene bridgeheads
        n: phenanthrene bridgeheads
        p: anthracene bridgeheads
        q: naphthalene bridgeheads
    """
    df = pd.read_csv(csv_path)
    
    # Find SU type column
    su_col = None
    for col in ['center_su_idx', 'su_type', 'su_idx', 'type']:
        if col in df.columns:
            su_col = col
            break
    if su_col is None:
        return 0, 0, 0, 0, 0, 0, 0
    
    # Filter rows where SU type is 12
    df_12 = df[df[su_col] == 12]
    
    x = y = z = m = n = p = q = 0
    
    # Check if template_key column exists
    if 'template_key' in df.columns:
        for _, row in df_12.iterrows():
            template = row.get('template_key', '')
            if not isinstance(template, str):
                continue
            su_type, hop1, hop2 = _parse_template_key(template)
            count_12_in_hop1 = hop1.count(12)
            count_12_in_hop2 = hop2.count(12)
            if len(hop1) == 3 and count_12_in_hop1 == 3:
                x += 1
            elif count_12_in_hop1 == 2 and count_12_in_hop2 >= 3:
                y += 1
            elif count_12_in_hop1 == 2 and count_12_in_hop2 == 2:
                z += 1
            elif count_12_in_hop2 >= 2:
                m += 1
            elif count_12_in_hop1 == 2 and count_12_in_hop2 == 1:
                n += 1
            elif count_12_in_hop2 == 1:
                p += 1
            else:
                q += 1
    else:
        # Fallback: treat all 12 as 2-ring bridgehead
        q = len(df_12)
    
    return x, y, z, m, n, p, q


def load_spectrum(csv_path: str, ppm_range: Tuple[float, float] = (0, 240), 
                  num_points: int = 2400) -> np.ndarray:
    """从CSV加载NMR谱图，自动检测分隔符"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"谱图文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path, header=None, sep=None, engine='python')
    
    if df.shape[1] == 1:
        spectrum = df.iloc[:, 0].values.astype(float)
    else:
        spectrum = df.iloc[:, 1].values.astype(float)
    
    if len(spectrum) != num_points:
        x_old = np.linspace(ppm_range[0], ppm_range[1], len(spectrum))
        x_new = np.linspace(ppm_range[0], ppm_range[1], num_points)
        spectrum = np.interp(x_new, x_old, spectrum)
    
    max_val = spectrum.max()
    if max_val > 0:
        spectrum = spectrum / max_val
    return spectrum.astype(np.float32)


AROMATIC_CLUSTER_CENTERS: Dict[str, List[Tuple[int, int]]] = {
    'benzene': [(0, 0)],
    'naphthalene': [(0, 0), (2, 1)],
    'phenanthrene': [(-2, -1), (0, 0), (1, -1)],
    'anthracene': [(0, 0), (2, 1), (4, 2)],
    'tetracene': [(0, 0), (2, 1), (4, 2), (6, 3)],
    'pyrene': [(0, 0), (2, 1), (1, -1), (3, 0)],
    'benzo_pyrene': [(-2, -1), (0, 0), (2, 1), (1, -1), (3, 0)],
    'perylene': [(0, 0), (-1, 1), (1, 2), (1, -1), (-1, -2)],
    'chrysene': [(-2, -1), (0, 0), (1, -1), (3, 0)],
    'triphenylene': [(0, 0), (-2, -1), (1, 2), (1, -1)],
    'coronene': [(0, 0), (-1, 1), (1, 2), (2, 1), (1, -1), (-1, -2), (-2, -1)],
}


def generate_fused_ring_cluster(cluster_id: int, kind: str) -> AromaticCluster:
    """Generate aromatic polycyclic cluster from predefined ring centers."""
    ring_centers = list(AROMATIC_CLUSTER_CENTERS.get(kind, AROMATIC_CLUSTER_CENTERS['benzene']))
    ring_count = len(ring_centers)
    
    # Collect all vertices from all rings
    vertex_set: Dict[Tuple[int, int], List[int]] = {}
    for ring_idx, (qc, rc) in enumerate(ring_centers):
        for dq, dr in HEX_VERTEX_OFFSETS:
            coord = (qc + dq, rc + dr)
            vertex_set.setdefault(coord, []).append(ring_idx)
    
    vertex_list = sorted(vertex_set.keys())
    
    # Create sites
    sites: List[Site] = []
    for idx, axial in enumerate(vertex_list):
        sharing = vertex_set[axial]
        su_type = 12 if len(sharing) >= 2 else 13
        pos2d = HexGrid.axial_to_cart(*axial)
        sites.append(Site(
            uid=f"C{cluster_id}-V{idx}",
            idx=idx,
            cluster_id=cluster_id,
            su_type=su_type,
            axial=axial,
            pos2d=pos2d
        ))
    
    # Build intra-ring edges
    index_lookup = {ax: idx for idx, ax in enumerate(vertex_list)}
    intra_edges: List[Tuple[int, int]] = []
    for ring_idx, (qc, rc) in enumerate(ring_centers):
        ring_vertices = [(qc + dq, rc + dr) for dq, dr in HEX_VERTEX_OFFSETS]
        ring_indices = [index_lookup[v] for v in ring_vertices]
        for i in range(6):
            a, b = ring_indices[i], ring_indices[(i + 1) % 6]
            edge = (min(a, b), max(a, b))
            if edge not in intra_edges:
                intra_edges.append(edge)
    
    return AromaticCluster(
        id=cluster_id, kind=kind, rings=ring_count,
        sites=sites, centers=ring_centers, edges=intra_edges
    )


# Cluster requirements: (12_needed, 13_needed)
CLUSTER_COST = {
    'coronene': (12, 12),
    'pyrene': (6, 10),
    'benzo_pyrene': (8, 12),
    'perylene': (8, 12),
    'chrysene': (6, 12),
    'triphenylene': (6, 12),
    'tetracene': (6, 12),
    'phenanthrene': (4, 10),
    'anthracene': (4, 10),
    'naphthalene': (2, 8),
    'benzene': (0, 6),
}

class ClusterGenerator:
    """
    Aromatic cluster generator based on 1/2-hop bridgehead analysis.
    """

    PRIORITY_ORDER = (
        'coronene',
        'pyrene',
        'benzo_pyrene',
        'perylene',
        'chrysene',
        'triphenylene',
        'tetracene',
        'phenanthrene',
        'anthracene',
        'naphthalene',
        'benzene',
    )
    DOWNGRADE_ORDER = {
        'coronene': ['pyrene', 'benzo_pyrene', 'perylene', 'chrysene', 'triphenylene', 'tetracene', 'phenanthrene', 'anthracene', 'naphthalene', 'benzene'],
        'pyrene': ['benzo_pyrene', 'perylene', 'chrysene', 'triphenylene', 'tetracene', 'phenanthrene', 'anthracene', 'naphthalene', 'benzene'],
        'benzo_pyrene': ['perylene', 'chrysene', 'triphenylene', 'tetracene', 'phenanthrene', 'anthracene', 'naphthalene', 'benzene'],
        'perylene': ['chrysene', 'triphenylene', 'tetracene', 'phenanthrene', 'anthracene', 'naphthalene', 'benzene'],
        'chrysene': ['triphenylene', 'tetracene', 'phenanthrene', 'anthracene', 'naphthalene', 'benzene'],
        'triphenylene': ['tetracene', 'phenanthrene', 'anthracene', 'naphthalene', 'benzene'],
        'tetracene': ['phenanthrene', 'anthracene', 'naphthalene', 'benzene'],
        'phenanthrene': ['anthracene', 'naphthalene', 'benzene'],
        'anthracene': ['naphthalene', 'benzene'],
        'naphthalene': ['benzene'],
        'benzene': [],
    }
    MAX_INTERCONVERSION = 4
    MAX_12_REDUCTION = 4
    MAX_12_INCREASE = 2
    
    def __init__(self, su_counts: Dict[int, int], bridgehead_info: Tuple[int, ...] = None):
        self.original_counts = su_counts.copy()
        self.clusters: List[AromaticCluster] = []
        self._id = 0
        
        if bridgehead_info:
            if len(bridgehead_info) >= 7:
                self.x, self.y, self.z, self.m, self.n, self.p, self.q = [int(v) for v in bridgehead_info[:7]]
            elif len(bridgehead_info) >= 5:
                self.x, self.y = [int(v) for v in bridgehead_info[:2]]
                self.z = 0
                self.m, self.n, self.p = [int(v) for v in bridgehead_info[2:5]]
                self.q = 0
            elif len(bridgehead_info) == 3:
                self.x = self.y = self.z = 0
                self.m = int(bridgehead_info[0])
                self.n = int(bridgehead_info[1])
                self.p = 0
                self.q = int(bridgehead_info[2])
            else:
                self.x = self.y = self.z = self.m = self.n = self.p = self.q = 0
        else:
            self.x = self.y = self.z = self.m = self.n = self.p = 0
            self.q = su_counts.get(12, 0)
        
        # Convert aromatic SUs to 13-equivalent counts.
        # Rule:
        #   5,6,7,8,9,10,11,13 -> 1 x 13
        #   26 -> 1.5 x 13
        #   30 -> 2 x 13
        self.n13 = 0.0
        for su in [5, 6, 7, 8, 9, 10, 11, 13]:
            self.n13 += su_counts.get(su, 0)
        self.n13 += su_counts.get(26, 0) * 1.5
        self.n13 += su_counts.get(30, 0) * 2.0
        self.synthetic_13_topup_used = 0
        
        # Total 12 (round up to even)
        self.original_12 = int(su_counts.get(12, 0))
        total_12 = self.original_12
        self.n12 = total_12 + (total_12 % 2)  # Round up to even

        # Global conversion budgets relative to original SU12 count.
        rounded_gain = int(self.n12 - self.original_12)
        self.max_13_to_12 = max(0, int(self.MAX_12_INCREASE - rounded_gain))
        self.max_12_to_13 = int(self.MAX_12_REDUCTION)
        self.used_13_to_12 = 0
        self.used_12_to_13 = 0
        
        # Remaining after generation
        self.remaining_12 = self.n12
        self.remaining_13 = self.n13
    
    def _next_id(self) -> int:
        cid = self._id
        self._id += 1
        return cid
    
    def _make_cluster(self, kind: str) -> AromaticCluster:
        return generate_fused_ring_cluster(self._next_id(), kind)
    
    def _find_conversion(self, cost_12: int, cost_13: int,
                         max_convert: int = MAX_INTERCONVERSION) -> Optional[Tuple[int, int]]:
        """Find a feasible (12->13, 13->12) conversion plan within the limits.

        Returns:
            (to13, to12): convert `to13` units of 12->13 and `to12` units of 13->12
        """
        best: Optional[Tuple[int, int]] = None
        best_key = None
        max_to13 = min(int(max_convert), int(self.max_12_to_13 - self.used_12_to_13), int(self.remaining_12))
        max_to12 = min(int(max_convert), int(self.max_13_to_12 - self.used_13_to_12), int(self.remaining_13))
        for to13 in range(max_to13 + 1):
            for to12 in range(max_to12 + 1):
                if int(to13 + to12) > int(max_convert):
                    continue
                rem12 = self.remaining_12 - to13 + to12
                rem13 = self.remaining_13 + to13 - to12
                if rem12 < cost_12 or rem13 < cost_13:
                    continue
                key = (to13 + to12, abs(to13 - to12), to13, to12)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (to13, to12)
        return best

    def _apply_conversion(self, to13: int, to12: int):
        if to13 > 0:
            self.remaining_12 -= to13
            self.remaining_13 += to13
            self.used_12_to_13 += int(to13)
        if to12 > 0:
            self.remaining_13 -= to12
            self.remaining_12 += to12
            self.used_13_to_12 += int(to12)

    def _try_make_one(self, kind: str, max_convert: int = MAX_INTERCONVERSION) -> bool:
        cost_12, cost_13 = CLUSTER_COST[kind]
        conversion = self._find_conversion(cost_12, cost_13, max_convert=max_convert)
        if conversion is None:
            return False
        self._apply_conversion(*conversion)
        self.clusters.append(self._make_cluster(kind))
        self.remaining_12 -= cost_12
        self.remaining_13 -= cost_13
        return True

    def _make_one_benzene_from_remaining(self) -> bool:
        need13 = int(CLUSTER_COST['benzene'][1])
        if self.remaining_13 >= need13:
            self.clusters.append(self._make_cluster('benzene'))
            self.remaining_13 -= need13
            return True

        # Final rescue: if 13 count is 4 or 5, first try topping up with
        # 12->13 conversion within budget, then allow a final 1-2 unit
        # synthetic 13补足 to complete one benzene ring.
        missing = int(need13 - self.remaining_13)
        if 0 < missing <= 2 and self.remaining_13 >= 4:
            convert_take = min(
                int(missing),
                int(self.remaining_12),
                int(max(0, self.max_12_to_13 - self.used_12_to_13)),
            )
            if convert_take > 0:
                self._apply_conversion(convert_take, 0)
                missing = int(need13 - self.remaining_13)

            if missing > 0:
                synth_budget = max(0, 2 - int(self.synthetic_13_topup_used))
                if missing > synth_budget:
                    return False
                self.remaining_13 += float(missing)
                self.synthetic_13_topup_used += int(missing)

            if self.remaining_13 >= need13:
                self.clusters.append(self._make_cluster('benzene'))
                self.remaining_13 -= need13
                return True
        return False

    def _downgrade_one(self, failed_kind: str) -> bool:
        """Try to realize a lower-priority aromatic cluster instead."""
        for next_kind in self.DOWNGRADE_ORDER.get(failed_kind, []):
            if next_kind == 'benzene':
                return self._make_one_benzene_from_remaining()
            if self._try_make_one(next_kind):
                return True
        return False

    def _consume_remaining_resources(self):
        """Consume leftover resources by building lower-tier clusters, then benzene."""
        progress = True
        while progress:
            progress = False
            for kind in ('tetracene', 'chrysene', 'triphenylene', 'phenanthrene', 'anthracene', 'naphthalene'):
                if self._try_make_one(kind):
                    progress = True
                    break

        # Only convert additional 12 -> 13 if we still have budget and it helps reach benzene.
        convertible = min(
            int(self.remaining_12),
            int(max(0, self.max_12_to_13 - self.used_12_to_13))
        )
        if 0 < self.remaining_13 < CLUSTER_COST['benzene'][1] and convertible > 0:
            needed = int(CLUSTER_COST['benzene'][1] - self.remaining_13)
            take = min(int(convertible), int(needed))
            if take > 0:
                self._apply_conversion(take, 0)

        while self.remaining_13 >= CLUSTER_COST['benzene'][1]:
            self.clusters.append(self._make_cluster('benzene'))
            self.remaining_13 -= CLUSTER_COST['benzene'][1]
        if 4 <= self.remaining_13 < CLUSTER_COST['benzene'][1]:
            if self._make_one_benzene_from_remaining():
                self.remaining_13 = max(0.0, float(self.remaining_13))
    
    def generate(self) -> List[AromaticCluster]:
        """
        Generate aromatic clusters based on prioritized bridgehead analysis.
        """
        self.clusters = []

        target_coronene = self.x // 6
        remaining_x = max(0, self.x - target_coronene * 6)
        target_pyrene = (remaining_x + 1) // 2
        target_poly5_total = (self.y + 1) // 2
        target_benzo = (target_poly5_total + 1) // 2
        target_perylene = target_poly5_total // 2
        target_poly4_total = (self.z + 1) // 2
        target_chrysene = (2 * target_poly4_total + 2) // 3
        target_triphenylene = target_poly4_total - target_chrysene
        target_tetra = self.m // 2
        target_phen = self.n // 2
        target_anth = self.p // 2
        target_naph = self.q // 2

        targets = [
            ('coronene', target_coronene),
            ('pyrene', target_pyrene),
            ('benzo_pyrene', target_benzo),
            ('perylene', target_perylene),
            ('tetracene', target_tetra),
            ('chrysene', target_chrysene),
            ('triphenylene', target_triphenylene),
            ('phenanthrene', target_phen),
            ('anthracene', target_anth),
            ('naphthalene', target_naph),
        ]

        for kind, target_count in targets:
            for _ in range(target_count):
                if self._try_make_one(kind):
                    continue
                if not self._downgrade_one(kind):
                    break

        self._consume_remaining_resources()
        
        return self.clusters


def initialize(nodes_csv: str, spectrum_csv: Optional[str] = None) -> Dict:
    """
    Main initialization function.
    """
    # 1. Load SU counts
    su_counts = load_su_counts(nodes_csv)
    
    # 2. Analyze bridgehead carbons from 1/2-hop info
    x, y, z, m, n, p, q = analyze_bridgehead_from_csv(nodes_csv)
    
    # 3. Generate aromatic clusters
    gen = ClusterGenerator(su_counts, bridgehead_info=(x, y, z, m, n, p, q))
    clusters = gen.generate()
    
    # 4. Create connection graph
    graph = ConnectionGraph(clusters=clusters)
    
    # 5. Load spectrum (optional)
    spectrum = None
    if spectrum_csv:
        spectrum = load_spectrum(spectrum_csv)
    
    # 6. Compute statistics
    total_su = sum(su_counts.values())
    aromatic_su = sum(su_counts.get(i, 0) for i in range(5, 14))
    aromatic_su += su_counts.get(26, 0) + su_counts.get(30, 0)  # raw aromatic-related node count
    aromatic_equiv_13 = (
        sum(su_counts.get(i, 0) for i in [5, 6, 7, 8, 9, 10, 11, 13])
        + su_counts.get(26, 0) * 1.5
        + su_counts.get(30, 0) * 2.0
    )
    aliphatic_su = sum(su_counts.get(i, 0) for i in range(19, 26))
    hetero_su = sum(su_counts.get(i, 0) for i in range(26, 33))
    carbonyl_su = sum(su_counts.get(i, 0) for i in range(0, 5))
    
    # 7. Output statistics
    print(f"{'='*70}")
    print(f"Initialization Complete - {nodes_csv}")
    print(f"{'='*70}")
    
    print(f"\n[SU Statistics] Total: {total_su}")
    print(f"  Aromatic(5-13): {aromatic_su}, Aliphatic(19-25): {aliphatic_su}")
    print(f"  Heteroatom(26-32): {hetero_su}, Carbonyl(0-4): {carbonyl_su}")
    print(f"  Aromatic-equivalent 13 count: {aromatic_equiv_13:.1f}")
    print()
    for su, cnt in sorted(su_counts.items()):
        name = SU_NAMES.get(su, f"SU-{su}")
        print(f"  {su:2d}: {name:20s} x {cnt}")
    
    print(f"\n[Bridgehead Analysis] (from 1/2-hop info)")
    print(f"  Coronene/Pyrene bridgehead (hop1 = 12,12,12): x = {x}")
    print(f"  Benzo_pyrene/Perylene bridgehead (hop1 has 2×12 and hop2 ≥3×12): y = {y}")
    print(f"  Chrysene/Triphenylene bridgehead (hop1 has 2×12 and hop2 =2×12): z = {z}")
    print(f"  4-ring bridgehead (hop2 ≥2 of 12): m = {m}")
    print(f"  Phenanthrene bridgehead (hop1 has 2×12 and hop2 =1×12): n = {n}")
    print(f"  Anthracene bridgehead (hop2 =1 of 12): p = {p}")
    print(f"  2-ring bridgehead (hop2 =0 of 12): q = {q}")
    
    print(f"\n[Aromatic Conversion]")
    n13_str = f"{gen.n13:.1f}" if abs(gen.n13 - round(gen.n13)) > 1e-6 else f"{int(round(gen.n13))}"
    print(f"  Converted 13: {n13_str} (from 5-11,13 + 1.5×26 + 2×30)")
    print(f"  Converted 12: {gen.n12} (rounded up to even)")
    if int(getattr(gen, 'synthetic_13_topup_used', 0)) > 0:
        print(f"  Benzene top-up 13 used: {int(gen.synthetic_13_topup_used)}")
    
    print(f"\n[Aromatic Clusters] Total: {len(clusters)}")
    cluster_counts = Counter(c.kind for c in clusters)
    print(
        f"  coronene: {cluster_counts.get('coronene', 0)}, "
        f"pyrene: {cluster_counts.get('pyrene', 0)}, "
        f"benzo_pyrene: {cluster_counts.get('benzo_pyrene', 0)}, "
        f"perylene: {cluster_counts.get('perylene', 0)}, "
        f"chrysene: {cluster_counts.get('chrysene', 0)}, "
        f"triphenylene: {cluster_counts.get('triphenylene', 0)}, "
        f"tetracene: {cluster_counts.get('tetracene', 0)}, "
        f"phenanthrene: {cluster_counts.get('phenanthrene', 0)}, "
        f"anthracene: {cluster_counts.get('anthracene', 0)}, "
        f"naphthalene: {cluster_counts.get('naphthalene', 0)}, "
        f"benzene: {cluster_counts.get('benzene', 0)}"
    )
    for c in clusters:
        n12 = sum(1 for s in c.sites if s.su_type == 12)
        n13 = sum(1 for s in c.sites if s.su_type == 13)
        print(f"  {c.kind:12s} ID={c.id} | {c.rings} rings, {len(c.sites)} sites (12:{n12}, 13:{n13})")
    
    print(f"\n[Remaining Resources]")
    print(f"  Unused 12: {gen.remaining_12}")
    rem13_str = f"{gen.remaining_13:.1f}" if abs(gen.remaining_13 - round(gen.remaining_13)) > 1e-6 else f"{int(round(gen.remaining_13))}"
    print(f"  Unused 13: {rem13_str}")
    
    if spectrum_csv:
        print(f"\n[NMR Spectrum]")
        print(f"  Path: {spectrum_csv}")
        print(f"  Points: {len(spectrum)}")
    print(f"{'='*70}")
    
    return {
        'su_counts': su_counts,
        'bridgehead_info': (x, y, z, m, n, p, q),
        'clusters': clusters,
        'graph': graph,
        'remaining_12': gen.remaining_12,
        'remaining_13': gen.remaining_13,
        'synthetic_13_topup_used': int(getattr(gen, 'synthetic_13_topup_used', 0)),
        'spectrum': spectrum,
    }
