"""Pure hex-grid geometry shared by every assembly stage.

The repository uses axial ``(q, r)`` coordinates projected as
``x = sqrt(3)/2*q`` and ``y = r-q/2``.  This module owns that convention;
assembly modules may transform complete units but must not redefine it.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Tuple


Axial = Tuple[int, int]

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


def axial_add(left: Axial, right: Axial) -> Axial:
    return int(left[0]) + int(right[0]), int(left[1]) + int(right[1])


def axial_neg(value: Axial) -> Axial:
    return -int(value[0]), -int(value[1])


def axial_distance(left: Axial, right: Axial) -> int:
    """Distance for the repository's ``(1, 1)`` diagonal convention."""
    dq = int(right[0]) - int(left[0])
    dr = int(right[1]) - int(left[1])
    return (abs(dq) + abs(dr) + abs(dq - dr)) // 2


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


def state_shape_rank(state: Any) -> Tuple[Any, ...]:
    shape = coordinate_aspect(state.occupied_axial)
    q_span, r_span = int(shape["q_span"]), int(shape["r_span"])
    return (
        float(shape["violation"]),
        float(shape["principal_axis_ratio"]),
        float(shape["linearity"]),
        abs(float(shape["q_over_r"]) - 1.0),
        abs(float(shape["width_over_height"]) - 1.0),
        max(q_span, r_span),
        abs(q_span - r_span),
        int(shape["total_span"]),
        q_span * r_span,
    )


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
    "occupied_span",
    "reflect_axial",
    "rotate_axial",
    "state_shape_rank",
    "transform_axial",
    "transform_direction",
    "unit_sort_key",
    "valid_hex_directions",
]
