"""Shared physical fixed-edge predicates for topology processing."""

from __future__ import annotations

from ...shared.inverse_common import _NodeV3


def nodes_have_fixed_edge(left: _NodeV3, right: _NodeV3) -> bool:
    """Return whether two nodes declare the same real, bidirectional edge."""

    left_id = int(getattr(left, "global_id", -1))
    right_id = int(getattr(right, "global_id", -1))
    return bool(
        right_id
        in {
            int(value)
            for value in set(getattr(left, "fixed_hop1_ids", set()) or set())
        }
        and left_id
        in {
            int(value)
            for value in set(getattr(right, "fixed_hop1_ids", set()) or set())
        }
        and right_id
        in {int(value) for value in list(getattr(left, "hop1_ids", []) or [])}
        and left_id
        in {int(value) for value in list(getattr(right, "hop1_ids", []) or [])}
    )

