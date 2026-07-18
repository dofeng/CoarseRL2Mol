from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


AROMATIC_CLUSTER_RING_COUNTS = {
    "benzene": 1,
    "naphthalene": 2,
    "phenanthrene": 3,
    "anthracene": 3,
    "tetracene": 4,
    "pyrene": 4,
    "benzo_pyrene": 5,
    "perylene": 5,
    "chrysene": 4,
    "triphenylene": 4,
    "coronene": 7,
}

AROMATIC_CLUSTER_COST = {
    "coronene": (12, 12),
    "pyrene": (6, 10),
    "benzo_pyrene": (8, 12),
    "perylene": (8, 12),
    "chrysene": (6, 12),
    "triphenylene": (6, 12),
    "tetracene": (6, 12),
    "phenanthrene": (4, 10),
    "anthracene": (4, 10),
    "naphthalene": (2, 8),
    "benzene": (0, 6),
}


@dataclass(frozen=True)
class AromaticCluster:
    cluster_id: int
    kind: str
    rings: int


class AromaticClusterGenerator:
    DOWNGRADE_ORDER = {
        "coronene": ["pyrene", "benzo_pyrene", "perylene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "pyrene": ["benzo_pyrene", "perylene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "benzo_pyrene": ["perylene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "perylene": ["chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "chrysene": ["triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "triphenylene": ["tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "tetracene": ["phenanthrene", "anthracene", "naphthalene", "benzene"],
        "phenanthrene": ["anthracene", "naphthalene", "benzene"],
        "anthracene": ["naphthalene", "benzene"],
        "naphthalene": ["benzene"],
        "benzene": [],
    }
    MAX_INTERCONVERSION = 4
    MAX_12_REDUCTION = 4
    MAX_12_INCREASE = 2

    def __init__(self, su_counts: Dict[int, int], bridgehead_info: Optional[Tuple[int, ...]] = None):
        self.original_counts = {int(key): int(value) for key, value in dict(su_counts or {}).items()}
        self.clusters: List[AromaticCluster] = []
        self._next_cluster_id = 0
        self.x, self.y, self.z, self.m, self.n, self.p, self.q = self._normalize_bridgeheads(
            bridgehead_info,
            int(self.original_counts.get(12, 0)),
        )

        self.n13 = float(sum(self.original_counts.get(su, 0) for su in (5, 6, 7, 8, 9, 10, 11, 13)))
        self.n13 += 1.5 * float(self.original_counts.get(26, 0))
        self.n13 += 2.0 * float(self.original_counts.get(30, 0))
        self.synthetic_13_topup_used = 0

        self.original_12 = int(self.original_counts.get(12, 0))
        self.n12 = int(self.original_12 + self.original_12 % 2)
        rounded_gain = int(self.n12 - self.original_12)
        self.max_13_to_12 = max(0, int(self.MAX_12_INCREASE - rounded_gain))
        self.max_12_to_13 = int(self.MAX_12_REDUCTION)
        self.used_13_to_12 = 0
        self.used_12_to_13 = 0
        self.remaining_12 = int(self.n12)
        self.remaining_13 = float(self.n13)

    @staticmethod
    def _normalize_bridgeheads(
        bridgehead_info: Optional[Tuple[int, ...]],
        default_q: int,
    ) -> Tuple[int, int, int, int, int, int, int]:
        values = tuple(int(value) for value in tuple(bridgehead_info or ()))
        if len(values) >= 7:
            return values[:7]
        if len(values) >= 5:
            return values[0], values[1], 0, values[2], values[3], values[4], 0
        if len(values) == 3:
            return 0, 0, 0, values[0], values[1], 0, values[2]
        return 0, 0, 0, 0, 0, 0, int(default_q)

    def _append_cluster(self, kind: str) -> None:
        self.clusters.append(AromaticCluster(
            cluster_id=int(self._next_cluster_id),
            kind=str(kind),
            rings=int(AROMATIC_CLUSTER_RING_COUNTS[str(kind)]),
        ))
        self._next_cluster_id += 1

    def _find_conversion(self, cost_12: int, cost_13: int) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[int, int, int, int]] = None
        max_to13 = min(
            int(self.MAX_INTERCONVERSION),
            int(self.max_12_to_13 - self.used_12_to_13),
            int(self.remaining_12),
        )
        max_to12 = min(
            int(self.MAX_INTERCONVERSION),
            int(self.max_13_to_12 - self.used_13_to_12),
            int(self.remaining_13),
        )
        for to13 in range(max_to13 + 1):
            for to12 in range(max_to12 + 1):
                if int(to13 + to12) > int(self.MAX_INTERCONVERSION):
                    continue
                if self.remaining_12 - to13 + to12 < int(cost_12):
                    continue
                if self.remaining_13 + to13 - to12 < int(cost_13):
                    continue
                key = (to13 + to12, abs(to13 - to12), to13, to12)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (to13, to12)
        return best

    def _apply_conversion(self, to13: int, to12: int) -> None:
        self.remaining_12 -= int(to13)
        self.remaining_13 += int(to13)
        self.used_12_to_13 += int(to13)
        self.remaining_13 -= int(to12)
        self.remaining_12 += int(to12)
        self.used_13_to_12 += int(to12)

    def _try_make_one(self, kind: str) -> bool:
        cost_12, cost_13 = AROMATIC_CLUSTER_COST[str(kind)]
        conversion = self._find_conversion(cost_12, cost_13)
        if conversion is None:
            return False
        self._apply_conversion(*conversion)
        self._append_cluster(str(kind))
        self.remaining_12 -= int(cost_12)
        self.remaining_13 -= int(cost_13)
        return True

    def _make_one_benzene_from_remaining(self) -> bool:
        need_13 = int(AROMATIC_CLUSTER_COST["benzene"][1])
        if self.remaining_13 >= need_13:
            self._append_cluster("benzene")
            self.remaining_13 -= need_13
            return True
        missing = int(need_13 - self.remaining_13)
        if not (0 < missing <= 2 and self.remaining_13 >= 4):
            return False
        convert_take = min(
            missing,
            int(self.remaining_12),
            max(0, int(self.max_12_to_13 - self.used_12_to_13)),
        )
        if convert_take > 0:
            self._apply_conversion(convert_take, 0)
            missing = int(need_13 - self.remaining_13)
        if missing > 0:
            synthetic_budget = max(0, 2 - int(self.synthetic_13_topup_used))
            if missing > synthetic_budget:
                return False
            self.remaining_13 += float(missing)
            self.synthetic_13_topup_used += int(missing)
        if self.remaining_13 < need_13:
            return False
        self._append_cluster("benzene")
        self.remaining_13 -= need_13
        return True

    def _downgrade_one(self, failed_kind: str) -> bool:
        for next_kind in self.DOWNGRADE_ORDER.get(str(failed_kind), []):
            if next_kind == "benzene":
                return self._make_one_benzene_from_remaining()
            if self._try_make_one(next_kind):
                return True
        return False

    def _consume_remaining_resources(self) -> None:
        progress = True
        while progress:
            progress = False
            for kind in ("tetracene", "chrysene", "triphenylene", "phenanthrene", "anthracene", "naphthalene"):
                if self._try_make_one(kind):
                    progress = True
                    break
        convertible = min(
            int(self.remaining_12),
            max(0, int(self.max_12_to_13 - self.used_12_to_13)),
        )
        if 0 < self.remaining_13 < AROMATIC_CLUSTER_COST["benzene"][1] and convertible > 0:
            self._apply_conversion(min(convertible, int(AROMATIC_CLUSTER_COST["benzene"][1] - self.remaining_13)), 0)
        while self.remaining_13 >= AROMATIC_CLUSTER_COST["benzene"][1]:
            self._append_cluster("benzene")
            self.remaining_13 -= AROMATIC_CLUSTER_COST["benzene"][1]
        if 4 <= self.remaining_13 < AROMATIC_CLUSTER_COST["benzene"][1]:
            self._make_one_benzene_from_remaining()
            self.remaining_13 = max(0.0, float(self.remaining_13))

    def generate(self) -> List[AromaticCluster]:
        self.clusters = []
        self._next_cluster_id = 0
        target_coronene = self.x // 6
        remaining_x = max(0, self.x - target_coronene * 6)
        target_poly5_total = (self.y + 1) // 2
        target_poly4_total = (self.z + 1) // 2
        targets = [
            ("coronene", target_coronene),
            ("pyrene", (remaining_x + 1) // 2),
            ("benzo_pyrene", (target_poly5_total + 1) // 2),
            ("perylene", target_poly5_total // 2),
            ("tetracene", self.m // 2),
            ("chrysene", (2 * target_poly4_total + 2) // 3),
            ("triphenylene", target_poly4_total - (2 * target_poly4_total + 2) // 3),
            ("phenanthrene", self.n // 2),
            ("anthracene", self.p // 2),
            ("naphthalene", self.q // 2),
        ]
        for kind, target_count in targets:
            for _ in range(int(target_count)):
                if not self._try_make_one(kind) and not self._downgrade_one(kind):
                    break
        self._consume_remaining_resources()
        return list(self.clusters)


def generate_aromatic_clusters(
    su_counts: Dict[int, int],
    bridgehead_info: Optional[Tuple[int, ...]] = None,
) -> List[AromaticCluster]:
    return AromaticClusterGenerator(su_counts, bridgehead_info=bridgehead_info).generate()
