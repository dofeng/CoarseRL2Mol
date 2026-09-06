"""Stage-order predicates shared by ring and chain materialization."""

from __future__ import annotations

from ..core.fragment import AssemblyFragmentView


def view_requires_ring(view: AssemblyFragmentView) -> bool:
    return bool(
        bool(getattr(view, "can_ring", False))
        and not bool(getattr(view, "can_side", False))
        and not bool(getattr(view, "can_flex", False))
    )


def view_requires_ring_resource_retry(view: AssemblyFragmentView) -> bool:
    """Keep a ring-first view pending while its concrete plan lacks resources.

    ``view_requires_ring`` deliberately describes exclusive capability.  A
    24_D-like view can legally fall back to a side chain, but its routing
    contract still says to try a ring first.  Resource conversion therefore
    gets one chance to satisfy a concrete ring plan before that fallback is
    released.
    """
    return bool(
        bool(getattr(view, "can_ring", False))
        and (
            view_requires_ring(view)
            or str(getattr(view, "preferred_product_type", ""))
            == "ring_candidate"
        )
    )


def ring_candidate_pending_for_ring_stage(
    assembler,
    view: AssemblyFragmentView,
) -> bool:
    if not bool(getattr(view, "can_ring", False)):
        return False
    if int(view.fragment_id) in set(
        getattr(assembler, "ring_exhausted_fragment_ids", set()) or set()
    ):
        return False
    if view_requires_ring(view):
        return True
    # Ring/side dual-use fragments are protected only while the ring phase is
    # still active.  Once the authoritative ring phase has completed without a
    # blocking contract, side-chain fallback must be released even if the ring
    # planner did not write an explicit per-fragment exhausted marker.
    if "real_like24_ring_assembly" in set(
        getattr(assembler, "completed_phases", set()) or set()
    ):
        return False
    return True


__all__ = [
    "ring_candidate_pending_for_ring_stage",
    "view_requires_ring",
    "view_requires_ring_resource_retry",
]
