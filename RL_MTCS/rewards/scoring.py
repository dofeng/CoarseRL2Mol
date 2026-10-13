"""The five reward terms used by staged assembly search.

Expensive forward connectivity is intentionally supplied by the caller at
checkpoints.  This module only combines already computed observations and
computes the cheap shape/SU13 clearance terms.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

from RL_MTCS.rewards.demand import SpatialDemand, spatial_demand


DEFAULT_REWARD_WEIGHTS = (0.40, 0.15, 0.20, 0.20, 0.05)


def normalize_reward_weights(weights: Sequence[float] = DEFAULT_REWARD_WEIGHTS) -> Tuple[float, ...]:
    values = tuple(float(value) for value in weights)
    if len(values) != 5 or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("reward_weights requires five finite nonnegative values")
    total = sum(values)
    if total <= 0:
        raise ValueError("reward_weights must contain a positive value")
    return tuple(value / total for value in values)


def nmr_quality(loss: Optional[float]) -> float:
    if loss is None or not math.isfinite(float(loss)):
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, float(loss)))))


def shape_profile(state: Any) -> Dict[str, Any]:
    """Cheap compactness and aspect observations, without hull/void scans."""
    from RL_MTCS.geometry import coordinate_aspect
    from RL_MTCS.mdp.components import main_units
    if getattr(state, 'stage', None) in {'rigid', 'aromatic'}:
        # Independent rigid clusters have no meaningful relative placement.
        remaining = set(state.placements)
        groups = []
        while remaining:
            group = state.component_unit_ids(min(remaining))
            remaining.difference_update(group)
            groups.append(group)
    else:
        groups = [main_units(state)]
    owner = state.problem.ownership_map()
    points_by_unit = {}
    for node, axial in sorted(state.node_axial.items()):
        points_by_unit.setdefault(owner[node], []).append(
            state.final_node_positions.get(node, axial))
    compact_total, aspect_total, radius_total, weight = 0.0, 0.0, 0.0, 0
    for group in groups:
        points = [p for uid in sorted(group) for p in points_by_unit.get(uid, ())]
        if not points:
            continue
        # Principal-axis ratio is invariant under lattice rotation/reflection.
        shape = coordinate_aspect(points)
        ratio = float(shape['principal_axis_ratio'])
        radius = math.sqrt(sum(shape['covariance_eigenvalues']) / len(points))
        compact_total += len(points) / (1.0 + radius)
        aspect_total += len(points) / (1.0 + max(0.0, ratio - 1.0))
        radius_total += radius * len(points)
        weight += len(points)
    compactness = compact_total / weight if weight else 1.0
    aspect = aspect_total / weight if weight else 1.0
    return dict(quality=0.8 * compactness + 0.2 * aspect,
                compactness_quality=compactness, aspect_quality=aspect,
                normalized_radius_of_gyration=radius_total / weight if weight else 0.0,
                node_count=weight, component_count=len(groups))


def shape_quality(state: Any) -> float:
    """Mostly compactness, with aspect as a secondary rotation-invariant term."""
    return shape_profile(state)['quality']


def su13_clearance_profile(state: Any, radius: int = 3) -> Dict[str, Any]:
    """Count empty honeycomb vertices in the complete axial ball at free SU13s."""
    from RL_MTCS.geometry import honeycomb_ball
    from RL_MTCS.mdp.sites import aromatic_perimeter_ids, hetero_staging_ids
    radius = max(1, int(radius))
    occupied = set(state.occupied_axial)
    protected = set(hetero_staging_ids(state))
    for edge in state.edges:
        if edge.contract_id:
            protected.update((edge.source_node_id, edge.target_node_id))
    rows, union = [], set()
    for unit in state.problem.units:
        if unit.phase != "aromatic" or not state.is_placed(unit.unit_id):
            continue
        for node in unit.nodes:
            gid = int(node.node_id)
            if (state.current_su_types.get(gid) != 13 or gid in protected
                    or gid not in aromatic_perimeter_ids(unit)
                    or f"aromatic_site:{gid}" in state.used_ports):
                continue
            site = state.node_axial[gid]
            if site not in occupied:
                continue
            theoretical = set(honeycomb_ball(site, radius))
            free = theoretical - occupied
            union.update(free)
            rows.append(dict(site_id=gid, unit_id=unit.unit_id,
                             clearance_cells=len(free), theoretical_cells=len(theoretical),
                             quality=len(free) / max(1, len(theoretical))))
    values = [row["clearance_cells"] for row in rows]
    quality = (0.6 * min(row["quality"] for row in rows)
               + 0.4 * sum(row["quality"] for row in rows) / len(rows)) if rows else 0.0
    return dict(radius=radius, sites=rows, site_count=len(rows),
                minimum_clearance=min(values, default=0),
                mean_clearance=sum(values) / len(values) if values else 0.0,
                unique_free_cells=len(union), summed_site_clearance=sum(values),
                quality=quality)


def score_state(
    state: Any,
    *,
    completion: Optional[float] = None,
    nmr: Optional[float] = None,
    forward: Optional[float] = None,
    clearance: Optional[float] = None,
    shape: Optional[float] = None,
    weights: Sequence[float] = DEFAULT_REWARD_WEIGHTS,
    demand: Optional[SpatialDemand] = None,
) -> Dict[str, Any]:
    # Search callers freeze demand at their round/checkpoint origin. Direct
    # callers receive an observation of the supplied state, using the same rule.
    demand = spatial_demand(state) if demand is None else demand
    base = normalize_reward_weights(weights)
    values = list(base)
    released = values[3] * (1.0 - demand.pressure)
    values[3] -= released
    values[4] += released
    # An unavailable NMR scorer must not dilute the remaining objectives.
    if nmr is None:
        values[1] = 0.0
        normalizer = sum(values)
        values = [value / normalizer for value in values] if normalizer else values
    values = tuple(values)
    shape_observation = shape_profile(state) if shape is None else None
    components = {
        "topology_completion": 0.0 if completion is None else max(0.0, min(1.0, float(completion))),
        "nmr_quality": 0.0 if nmr is None else max(0.0, min(1.0, float(nmr))),
        "forward_completion": 0.0 if forward is None else max(0.0, min(1.0, float(forward))),
        "su13_clearance": su13_clearance_profile(state)["quality"] if clearance is None else max(0.0, min(1.0, float(clearance))),
        "shape_quality": shape_observation['quality'] if shape is None else max(0.0, min(1.0, float(shape))),
    }
    total = sum(weight * components[name] for weight, name in zip(values, components))
    return {**components, "total": max(0.0, min(1.0, total)), "weights": list(values),
            "base_weights": list(base), "spatial_demand": demand.to_dict(),
            "shape_metrics": shape_observation,
            "weighted_terms": {name: weight * components[name]
                               for weight, name in zip(values, components)}}


__all__ = [
    "DEFAULT_REWARD_WEIGHTS",
    "normalize_reward_weights",
    "nmr_quality",
    "shape_quality",
    "shape_profile",
    "su13_clearance_profile",
    "score_state",
]
