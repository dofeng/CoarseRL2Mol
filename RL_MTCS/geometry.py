"""Pure hex-grid geometry shared by every assembly stage.

The repository uses axial ``(q, r)`` coordinates projected as
``x = sqrt(3)/2*q`` and ``y = r-q/2``.  This module owns that convention;
assembly modules may transform complete units but must not redefine it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Iterable, Sequence, Tuple


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 2:
        return tuple(points)
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def half(rows):
        result = []
        for point in rows:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 1e-9:
                result.pop()
            result.append(point)
        return result
    return tuple(half(points)[:-1] + half(points[::-1])[:-1])


def outside_hull(hull, axial):
    p = axial_to_cartesian(axial)
    if len(hull) < 3:
        return p not in hull
    return any((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) < -1e-9
               for a, b in zip(hull, hull[1:]+hull[:1]))


def hull_region(hull, points):
    outside = [outside_hull(hull, p) for p in points]
    if outside and all(outside):
        return 'outside'
    if any(outside):
        return 'crossing'
    return 'inside'


Axial = Tuple[int, int]


def connection_axis(direction):
    """World-coordinate axis class under the repository's hex projection."""
    return 'V' if int(direction[0]) == 0 else 'H'

HEX_DIRECTIONS: Tuple[Axial, ...] = (
    (1, 1),
    (1, 0),
    (-1, 0),
    (-1, -1),
    (0, 1),
    (0, -1),
)
HEX_VERTEX_OFFSETS: Tuple[Axial, ...] = (
    (1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1),
)
_HEX_A_DIRECTIONS: Tuple[Axial, ...] = ((1, 0), (0, 1), (-1, -1))
_HEX_B_DIRECTIONS: Tuple[Axial, ...] = ((-1, 0), (0, -1), (1, 1))

MIN_Q_R_ASPECT_RATIO = 0.8
MAX_Q_R_ASPECT_RATIO = 2.0


def valid_hex_directions(axial: Axial) -> Tuple[Axial, ...]:
    """Return the three ordinary honeycomb edges available at ``axial``."""
    phase = (int(axial[0]) + int(axial[1])) % 3
    if phase == 1:
        return _HEX_A_DIRECTIONS
    if phase == 2:
        return _HEX_B_DIRECTIONS
    return ()


def hex_edge_valid(source: Axial, target: Axial, *, honeycomb: bool = False) -> bool:
    """Triangular lattice adjacency, optionally requiring honeycomb vertices.

    Ordinary molecular bonds use ``honeycomb=True``. A declared SU25 fixed
    SU22 branch also uses a unit edge, with its SU22 endpoint at a face centre.
    """
    delta = (target[0] - source[0], target[1] - source[1])
    return delta in HEX_DIRECTIONS and (not honeycomb or (
        delta in valid_hex_directions(source)
        and (-delta[0], -delta[1]) in valid_hex_directions(target)
    ))


def su25_center_branch_errors(unit: Any) -> list[dict[str, Any]]:
    """Validate the narrow geometry declaration independently of coordinates."""
    if not unit.su25_center_branches and not any(n.su_type == 25 for n in unit.nodes):
        return []
    nodes = unit.node_map()
    neighbors = {gid: set() for gid in nodes}
    for edge in unit.internal_edges:
        if edge.source_node_id in nodes and edge.target_node_id in nodes:
            neighbors[edge.source_node_id].add(edge.target_node_id)
            neighbors[edge.target_node_id].add(edge.source_node_id)
    port_counts = {gid: sum(p.node_id == gid for p in unit.ports) for gid in nodes}
    errors, parents, children = [], set(), set()
    for parent, child in unit.su25_center_branches:
        if (parent in parents or child in children or parent not in nodes or child not in nodes
                or nodes[parent].su_type != 25 or nodes[child].su_type != 22
                or neighbors.get(child) != {parent} or port_counts.get(child)
                or len(neighbors.get(parent, ())) + port_counts.get(parent, 0) != 4):
            errors.append(dict(reason='invalid_su25_center_branch', unit_id=unit.unit_id,
                               parent_node_id=parent, branch_node_id=child))
        parents.add(parent)
        children.add(child)
    for gid, node in nodes.items():
        if (node.su_type == 25 and len(neighbors[gid]) + port_counts[gid] == 4
                and gid not in parents):
            errors.append(dict(reason='missing_su25_center_branch', unit_id=unit.unit_id,
                               parent_node_id=gid))
    return errors


def topology_site_valid(node_id: int, point: Axial, center_parents: Mapping[int, int]) -> bool:
    phase = (point[0] + point[1]) % 3
    return phase == 0 if node_id in center_parents else phase != 0


def topology_edge_valid(source_id: int, target_id: int, source: Axial, target: Axial,
                        center_parents: Mapping[int, int]) -> bool:
    """Ordinary honeycomb bonds plus precisely one registered branch per SU25."""
    if source_id in center_parents:
        return (center_parents[source_id] == target_id
                and topology_site_valid(source_id, source, center_parents)
                and topology_site_valid(target_id, target, center_parents)
                and hex_edge_valid(source, target))
    if target_id in center_parents:
        return topology_edge_valid(target_id, source_id, target, source, center_parents)
    return hex_edge_valid(source, target, honeycomb=True)


def axial_add(left: Axial, right: Axial) -> Axial:
    return int(left[0]) + int(right[0]), int(left[1]) + int(right[1])


def axial_neg(value: Axial) -> Axial:
    return -int(value[0]), -int(value[1])


def axial_distance(left: Axial, right: Axial) -> int:
    """Distance for the repository's ``(1, 1)`` diagonal convention."""
    dq = int(right[0]) - int(left[0])
    dr = int(right[1]) - int(left[1])
    return (abs(dq) + abs(dr) + abs(dq - dr)) // 2


def honeycomb_ball(site: Axial, radius: int = 3):
    """Global honeycomb vertices, excluding the site and every face centre."""
    for q in range(site[0] - radius, site[0] + radius + 1):
        for r in range(site[1] - radius, site[1] + radius + 1):
            if (q + r) % 3 and 0 < axial_distance(site, (q, r)) <= radius:
                yield q, r


def segments_intersect(a, b, c, d, tolerance=1e-9):
    """Closed segment intersection; callers exempt edges sharing a node ID."""
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def on(p, q, r):
        return (abs(cross(p, q, r)) <= tolerance
                and min(p[0], q[0])-tolerance <= r[0] <= max(p[0], q[0])+tolerance
                and min(p[1], q[1])-tolerance <= r[1] <= max(p[1], q[1])+tolerance)
    x, y, z, w = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
    return ((x*y < -tolerance and z*w < -tolerance)
            or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b))


def rotate_axial(value: Axial, rotation: int) -> Axial:
    q, r = int(value[0]), int(value[1])
    for _ in range(int(rotation) % 6):
        q, r = q - r, q
    return int(q), int(r)


def reflect_axial(value: Axial) -> Axial:
    q, r = int(value[0]), int(value[1])
    return -q, r - q


def transform_direction(direction: Axial, rotation: int) -> Axial:
    return rotate_axial(direction, rotation)


def transform_axial(
    value: Axial,
    rotation: int,
    reflection: bool = False,
) -> Axial:
    return rotate_axial(reflect_axial(value) if reflection else value, rotation)


def axial_to_cartesian(axial: Axial, scale: float = 1.0) -> Tuple[float, float]:
    """Project axial coordinates with positive ``r`` drawn vertically up."""
    q, r = int(axial[0]), int(axial[1])
    return (
        float(scale) * math.sqrt(3.0) * 0.5 * q,
        float(scale) * (r - 0.5 * q),
    )


def unit_sort_key(unit_id: str) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(unit_id))
    return tuple(int(part) if part.isdigit() else part for part in parts)


def aromatic_slot_direction(unit: Any, node_id: int) -> Axial:
    """Return the outward direction of an aromatic perimeter site."""
    node = unit.node_map()[int(node_id)]
    coord = tuple(node.local_axial)
    occupied = {tuple(value.local_axial) for value in unit.nodes}
    candidates = [
        direction for direction in HEX_DIRECTIONS
        if axial_add(coord, direction) not in occupied
    ]
    if not candidates:
        raise ValueError(
            f"aromatic node {node_id} on {unit.unit_id!r} has no open direction"
        )
    containing = [ring for ring in unit.local_rings if coord in set(ring)]
    if not containing:
        return candidates[0]
    centers = []
    for ring in containing:
        points = [axial_to_cartesian(value) for value in ring]
        centers.append((
            sum(value[0] for value in points) / len(points),
            sum(value[1] for value in points) / len(points),
        ))
    center = (
        sum(value[0] for value in centers) / len(centers),
        sum(value[1] for value in centers) / len(centers),
    )
    node_cart = axial_to_cartesian(coord)
    outward = node_cart[0] - center[0], node_cart[1] - center[1]
    return max(
        candidates,
        key=lambda direction: (
            outward[0] * axial_to_cartesian(direction)[0]
            + outward[1] * axial_to_cartesian(direction)[1],
            direction,
        ),
    )


def occupied_span(axials: Iterable[Axial]) -> Tuple[int, int, int]:
    values = list(axials)
    if not values:
        return 0, 0, 0
    q_span = max(value[0] for value in values) - min(value[0] for value in values)
    r_span = max(value[1] for value in values) - min(value[1] for value in values)
    return int(q_span + r_span), int(q_span), int(r_span)


def coordinate_aspect(axials: Iterable[Axial]) -> dict[str, Any]:
    """Measure the required q/r aspect and projected display aspect."""
    values = list(axials)
    total_span, q_span, r_span = occupied_span(values)
    if q_span == 0 and r_span == 0:
        ratio = 1.0
    elif r_span == 0:
        ratio = math.inf
    else:
        ratio = float(q_span) / float(r_span)
    valid = MIN_Q_R_ASPECT_RATIO <= ratio <= MAX_Q_R_ASPECT_RATIO
    violation = (
        MIN_Q_R_ASPECT_RATIO - ratio
        if ratio < MIN_Q_R_ASPECT_RATIO
        else ratio - MAX_Q_R_ASPECT_RATIO
        if ratio > MAX_Q_R_ASPECT_RATIO
        else 0.0
    )
    cartesian = [axial_to_cartesian(value) for value in values]
    if cartesian:
        x_span = max(value[0] for value in cartesian) - min(value[0] for value in cartesian)
        y_span = max(value[1] for value in cartesian) - min(value[1] for value in cartesian)
    else:
        x_span = y_span = 0.0
    cartesian_ratio = (
        1.0 if x_span == 0.0 and y_span == 0.0
        else math.inf if y_span == 0.0
        else float(x_span) / float(y_span)
    )
    # The q/r bounding-box ratio is useful for preserving the lattice
    # convention, but it cannot detect a diagonal ``q ~= r`` needle.  For
    # example, a long chain along a 60 degree lattice direction can have a
    # perfectly acceptable q/r ratio while still being almost one
    # dimensional in the rendered molecule.  Compute the principal axes in
    # Cartesian space as an additional, rotation-independent shape measure.
    if len(cartesian) < 2:
        principal_axis_ratio = 1.0
        principal_axis_angle = 0.0
        linearity = 0.0
        covariance_eigenvalues = (0.0, 0.0)
    else:
        mean_x = sum(value[0] for value in cartesian) / len(cartesian)
        mean_y = sum(value[1] for value in cartesian) / len(cartesian)
        cov_xx = sum((value[0] - mean_x) ** 2 for value in cartesian) / len(cartesian)
        cov_xy = sum((value[0] - mean_x) * (value[1] - mean_y) for value in cartesian) / len(cartesian)
        cov_yy = sum((value[1] - mean_y) ** 2 for value in cartesian) / len(cartesian)
        discriminant = math.hypot(cov_xx - cov_yy, 2.0 * cov_xy)
        largest = max(0.0, 0.5 * (cov_xx + cov_yy + discriminant))
        smallest = max(0.0, 0.5 * (cov_xx + cov_yy - discriminant))
        if largest <= 1e-12:
            principal_axis_ratio = 1.0
            linearity = 0.0
        elif smallest <= 1e-12:
            principal_axis_ratio = math.inf
            linearity = 1.0
        else:
            principal_axis_ratio = math.sqrt(largest / smallest)
            linearity = max(0.0, min(1.0, 1.0 - smallest / largest))
        principal_axis_angle = 0.5 * math.degrees(
            math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
        )
        covariance_eigenvalues = (float(largest), float(smallest))

    bbox_area = float(x_span * y_span)
    return {
        "q_span": int(q_span),
        "r_span": int(r_span),
        "q_over_r": float(ratio),
        "min_q_over_r": float(MIN_Q_R_ASPECT_RATIO),
        "max_q_over_r": float(MAX_Q_R_ASPECT_RATIO),
        "valid": bool(valid),
        "violation": float(violation),
        "total_span": int(total_span),
        "x_span": float(x_span),
        "y_span": float(y_span),
        "width_over_height": float(cartesian_ratio),
        "rotated_90_width_over_height": (
            math.inf if cartesian_ratio == 0.0
            else 0.0 if math.isinf(cartesian_ratio)
            else float(1.0 / cartesian_ratio)
        ),
        "bbox_area": bbox_area,
        "principal_axis_ratio": float(principal_axis_ratio),
        "principal_axis_angle": float(principal_axis_angle),
        "linearity": float(linearity),
        "covariance_eigenvalues": [
            float(covariance_eigenvalues[0]),
            float(covariance_eigenvalues[1]),
        ],
    }


def _coefficient_of_variation(values: Sequence[float]) -> float:
    """Return a stable coefficient of variation for a finite sample."""
    if not values:
        return 0.0
    mean = sum(float(value) for value in values) / len(values)
    if mean <= 1e-12:
        return 0.0
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(max(0.0, variance)) / mean)


def _binned_density(points: Sequence[Tuple[float, float]], bins: int) -> float:
    """Measure coarse spatial unevenness, including empty bounding-box bins."""
    if len(points) < 2:
        return 0.0
    bins = max(2, int(bins))
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    counts = [0.0] * (bins * bins)
    for x, y in points:
        ix = min(bins - 1, int((x - min_x) / span_x * bins))
        iy = min(bins - 1, int((y - min_y) / span_y * bins))
        counts[iy * bins + ix] += 1.0
    return _coefficient_of_variation(counts)


def _convex_hull(points: Sequence[Tuple[float, float]]) -> list[Tuple[float, float]]:
    """Return a monotonic-chain convex hull in counter-clockwise order."""
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) < 3:
        return unique

    def cross(origin, left, right):
        return (
            (left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0])
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _convex_hull_area(points: Sequence[Tuple[float, float]]) -> float:
    """Return the Cartesian area of a monotonic-chain convex hull."""
    hull = _convex_hull(points)
    if len(hull) < 3:
        return 0.0
    return float(abs(sum(
        hull[index][0] * hull[(index + 1) % len(hull)][1]
        - hull[(index + 1) % len(hull)][0] * hull[index][1]
        for index in range(len(hull))
    )) * 0.5)


def _inside_convex_hull(point: Tuple[float, float], hull: Sequence[Tuple[float, float]]) -> bool:
    if len(hull) < 3:
        return True
    signs = []
    for index, left in enumerate(hull):
        right = hull[(index + 1) % len(hull)]
        cross = (
            (right[0] - left[0]) * (point[1] - left[1])
            - (right[1] - left[1]) * (point[0] - left[0])
        )
        signs.append(cross)
    return not (any(value > 1e-9 for value in signs) and any(value < -1e-9 for value in signs))


def layout_metrics(
    axials: Iterable[Axial],
    unit_groups: Iterable[Iterable[Axial]] | Mapping[Any, Iterable[Axial]] | None = None,
    *,
    void_threshold: float = 2.5,
    full: bool = True,
    include_void: bool = True,
) -> dict[str, Any]:
    """Measure compactness and spatial uniformity of a lattice layout.

    ``coordinate_aspect`` remains the inexpensive lattice-shape contract.  The
    additional metrics detect layouts with an acceptable bounding box but a
    hollow centre or strongly uneven structural-unit distribution.  ``full``
    may be disabled during broad search to skip the convex-hull calculation.
    """
    values = []
    seen = set()
    for value in axials:
        axial = (int(value[0]), int(value[1]))
        if axial not in seen:
            seen.add(axial)
            values.append(axial)
    shape = coordinate_aspect(values)
    cartesian = [axial_to_cartesian(value) for value in values]
    node_density_cv = _binned_density(cartesian, 4)

    groups = []
    if unit_groups is not None:
        source = unit_groups.values() if isinstance(unit_groups, Mapping) else unit_groups
        for group in source:
            group_values = [
                (int(value[0]), int(value[1])) for value in group
                if (int(value[0]), int(value[1])) in seen
            ]
            if group_values:
                groups.append(group_values)
    centroid_points = []
    for group in groups:
        points = [axial_to_cartesian(value) for value in group]
        centroid_points.append((
            sum(value[0] for value in points) / len(points),
            sum(value[1] for value in points) / len(points),
        ))
    unit_density_cv = _binned_density(centroid_points, 3) if centroid_points else 0.0

    largest_void_radius = 0.0
    large_void_fraction = 0.0
    if include_void and len(values) >= 3:
        min_q = min(value[0] for value in values)
        max_q = max(value[0] for value in values)
        min_r = min(value[1] for value in values)
        max_r = max(value[1] for value in values)
        area = max(0, max_q - min_q + 1) * max(0, max_r - min_r + 1)
        # Search is performed on small integer lattice boxes.  Very large
        # boxes are sampled so a pathological candidate cannot dominate run
        # time; the audit path normally remains below this bound.
        if full:
            step = max(1, int(math.ceil(math.sqrt(float(area) / 6000.0))))
            sample_points = [
                (q, r)
                for q in range(min_q, max_q + 1, step)
                for r in range(min_r, max_r + 1, step)
            ]
        else:
            # A small regular grid bounds diagnostic sampling and, unlike
            # the former five-point probe, detects off-centre cavities.  At
            # most 49 points are tested regardless of molecule size.
            q_step = max(1, int(math.ceil(max(1, max_q - min_q) / 6.0)))
            r_step = max(1, int(math.ceil(max(1, max_r - min_r) / 6.0)))
            q_samples = list(range(min_q, max_q + 1, q_step))
            r_samples = list(range(min_r, max_r + 1, r_step))
            if q_samples[-1] != max_q:
                q_samples.append(max_q)
            if r_samples[-1] != max_r:
                r_samples.append(max_r)
            sample_points = [
                (q, r) for q in q_samples[:7] for r in r_samples[:7]
            ]
        hull = _convex_hull([axial_to_cartesian(value) for value in values])
        empty_distances = []
        for point in sample_points:
            if point in seen:
                continue
            if not _inside_convex_hull(axial_to_cartesian(point), hull):
                continue
            nearest = min(axial_distance(point, other) for other in values)
            empty_distances.append(float(nearest))
        if empty_distances:
            largest_void_radius = max(empty_distances)
            large_void_fraction = sum(
                distance > float(void_threshold) for distance in empty_distances
            ) / len(empty_distances)

    hull_area = _convex_hull_area(cartesian) if (full or include_void) else 0.0
    hull_fill = (
        min(1.0, len(values) * (math.sqrt(3.0) * 0.5) / hull_area)
        if hull_area > 1e-12 else 1.0
    )
    eigenvalues = shape.get("covariance_eigenvalues", (0.0, 0.0))
    normalized_radius_of_gyration = (
        math.sqrt(float(eigenvalues[0] + eigenvalues[1]))
        / math.sqrt(max(1, len(values)))
    )
    principal_ratio = float(shape["principal_axis_ratio"])
    energy = (
        2.0 * max(0.0, largest_void_radius - float(void_threshold)) ** 2
        + 1.5 * float(large_void_fraction)
        + 1.0 * float(unit_density_cv)
        + 0.5 * float(node_density_cv)
        + 0.25 * float(normalized_radius_of_gyration)
        + 0.20 * max(0.0, principal_ratio - 1.8) ** 2
        + 0.25 * max(0.0, 0.55 - float(hull_fill)) ** 2
    )
    return {
        **shape,
        "node_density_cv": float(node_density_cv),
        "unit_density_cv": float(unit_density_cv),
        "largest_void_radius": float(largest_void_radius),
        "large_void_fraction": float(large_void_fraction),
        "hull_area": float(hull_area),
        "hull_fill": float(hull_fill),
        "normalized_radius_of_gyration": float(normalized_radius_of_gyration),
        "layout_energy": float(energy),
        "layout_valid": bool(
            largest_void_radius <= max(float(void_threshold) + 3.0, 8.0)
            and unit_density_cv <= 1.5
        ),
    }


def state_layout_metrics(
    state: Any,
    *,
    full: bool = False,
    include_all: bool = False,
    include_void: bool = True,
) -> dict[str, Any]:
    """Compute metrics for the currently active portion of an assembly state."""
    occupied = getattr(state, "occupied_axial", {})
    node_axial = getattr(state, "node_axial", {})
    values = list(node_axial.values()) if include_all else (
        list(occupied.keys()) or list(node_axial.values())
    )
    ownership = {}
    problem = getattr(state, "problem", None)
    if problem is not None and hasattr(problem, "ownership_map"):
        ownership = problem.ownership_map()
    groups = {}
    value_set = {(int(value[0]), int(value[1])) for value in values}
    for node_id, axial in node_axial.items():
        axial = (int(axial[0]), int(axial[1]))
        if axial not in value_set:
            continue
        groups.setdefault(ownership.get(int(node_id), int(node_id)), []).append(axial)
    return layout_metrics(values, groups, full=full, include_void=include_void)


__all__ = [
    "Axial",
    "HEX_DIRECTIONS",
    "HEX_VERTEX_OFFSETS",
    "MAX_Q_R_ASPECT_RATIO",
    "MIN_Q_R_ASPECT_RATIO",
    "aromatic_slot_direction",
    "axial_add",
    "axial_distance",
    "axial_neg",
    "axial_to_cartesian",
    "coordinate_aspect",
    "layout_metrics",
    "occupied_span",
    "reflect_axial",
    "rotate_axial",
    "state_layout_metrics",
    "transform_axial",
    "transform_direction",
    "unit_sort_key",
    "valid_hex_directions",
    "hex_edge_valid",
    "su25_center_branch_errors",
    "topology_site_valid",
    "topology_edge_valid",
    "site_space_profile",
]
