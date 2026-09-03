"""Stage-order predicates shared by ring and chain materialization."""

from __future__ import annotations

from ..core.fragment import AssemblyFragmentView


def view_requires_ring(view: AssemblyFragmentView) -> bool:
    return bool(
        bool(getattr(view, "can_ring", False))
        and not bool(getattr(view, "can_side", False))
        and not bool(getattr(view, "can_flex", False))
    )


def ring_candidate_pending_for_ring_stage(
    assembler,
    view: AssemblyFragmentView,
) -> bool:
    return bool(
        bool(getattr(view, "can_ring", False))
        and int(view.fragment_id)
        not in set(getattr(assembler, "ring_exhausted_fragment_ids", set()) or set())
    )


__all__ = ["ring_candidate_pending_for_ring_stage", "view_requires_ring"]
