from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from model.shared.aromatic_plan import (
    AROMATIC_CLUSTER_RING_COUNTS,
    exact_aromatic_cluster_plan,
    staging_aromatic_counts,
)


@dataclass(frozen=True)
class AromaticCluster:
    cluster_id: int
    kind: str
    rings: int


class AromaticClusterGenerator:
    """Materialize the shared exact aromatic plan from authoritative node counts.

    Direct generator users may inspect ``plan_diagnostics`` after ``generate``.
    ``bridgehead_info`` is diagnostic context only and never changes allocation.
    The convenience function returns clusters, not generator diagnostics.
    """

    def __init__(
        self,
        su_counts: Dict[int, int],
        bridgehead_info: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.original_counts = {
            int(key): int(value) for key, value in dict(su_counts or {}).items()
        }
        self.bridgehead_info = tuple(
            int(value) for value in (bridgehead_info or ())
        )
        self.clusters: List[AromaticCluster] = []
        self._next_cluster_id = 0
        self.original_12, self.original_13, self.five_member_count = (
            staging_aromatic_counts(self.original_counts)
        )
        self.idle_aromatic_count = 0
        self.plan_diagnostics: Dict[str, object] = {}

    def _append_cluster(self, kind: str) -> None:
        self.clusters.append(
            AromaticCluster(
                cluster_id=int(self._next_cluster_id),
                kind=str(kind),
                rings=int(AROMATIC_CLUSTER_RING_COUNTS[str(kind)]),
            )
        )
        self._next_cluster_id += 1

    def generate(self) -> List[AromaticCluster]:
        self.clusters = []
        self._next_cluster_id = 0
        exact_plan = exact_aromatic_cluster_plan(
            int(self.original_12),
            int(self.original_13),
        )
        for kind in list(exact_plan.get("kinds", []) or []):
            self._append_cluster(str(kind))
        self.idle_aromatic_count = int(
            exact_plan.get("idle_aromatic_count", 0) or 0
        )
        self.plan_diagnostics = {
            **dict(exact_plan),
            "five_member_count": int(self.five_member_count),
            "bridgehead_info": list(self.bridgehead_info),
        }
        return list(self.clusters)


def generate_aromatic_clusters(
    su_counts: Dict[int, int],
    bridgehead_info: Optional[Tuple[int, ...]] = None,
) -> List[AromaticCluster]:
    return AromaticClusterGenerator(
        su_counts,
        bridgehead_info=bridgehead_info,
    ).generate()
