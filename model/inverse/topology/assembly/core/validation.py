"""Pure shape validation for materialized topology routes."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .....shared.inverse_common import _NodeV3
from ...classification.classification_tokens import (
    AROMATIC_SU_TYPES,
    normalize_subtype,
)
from .rules import TERMINAL_ENDPOINT_LABELS


def _node_label(node: _NodeV3) -> str:
    subtype = str(normalize_subtype(node))
    try:
        su_type = int(getattr(node, "su_type", -1))
    except (TypeError, ValueError):
        su_type = -1
    return subtype if subtype not in {str(su_type), ""} else str(su_type)


def _path_parenthesis_depth(path: str) -> int:
    depth = 0
    maximum = 0
    for char in str(path or ""):
        if char == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return int(maximum)


def _top_level_route_ids(path: str) -> List[int]:
    ids: List[int] = []
    depth = 0
    for match in re.finditer(r"\(|\)|\d+", str(path or "")):
        token = str(match.group(0))
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            ids.append(int(token))
    return ids


def branch_route_shape_reason(
    branch_topology: str,
    branch_node_ids: str,
    nodes_by_id: Optional[Mapping[int, _NodeV3]] = None,
) -> Optional[str]:
    """Reject a branch payload that is nonlinear or has an internal terminal."""

    if (
        _path_parenthesis_depth(str(branch_topology or "")) > 0
        or _path_parenthesis_depth(str(branch_node_ids or "")) > 0
    ):
        return "nested_branch_topology_forbidden"
    route_ids = _top_level_route_ids(str(branch_node_ids or ""))
    if not route_ids or nodes_by_id is None:
        return None
    terminal_positions = [
        index
        for index, node_id in enumerate(route_ids)
        if int(node_id) in nodes_by_id
        and _node_label(nodes_by_id[int(node_id)]) in set(TERMINAL_ENDPOINT_LABELS)
    ]
    if terminal_positions and terminal_positions != [len(route_ids) - 1]:
        return "branch_terminal_not_at_path_end"
    return None


def validate_chain_product_shape(
    product_kind: str,
    labels: Sequence[str],
    source_views: Sequence[Any],
    node_ids: Optional[Sequence[Optional[int]]] = None,
) -> Optional[str]:
    normalized_kind = str(product_kind)
    token_labels = [str(label) for label in list(labels or []) if str(label)]
    if not token_labels:
        return "empty_topology"
    if normalized_kind not in {"flex_chain", "side_chain"}:
        return None
    materialized_ids = [
        int(value) for value in list(node_ids or []) if value is not None
    ]
    if len(materialized_ids) != len(set(materialized_ids)):
        duplicate_ids = sorted({
            value
            for value in materialized_ids
            if materialized_ids.count(value) > 1
        })
        return f"duplicate_mainline_node_ids:{duplicate_ids}"
    left_label = str(token_labels[0])
    right_label = str(token_labels[-1])
    aromatic_labels = {str(value) for value in AROMATIC_SU_TYPES} | {"11"}
    left_is_aromatic = left_label in aromatic_labels
    right_is_aromatic = right_label in aromatic_labels
    left_is_terminal = left_label in set(TERMINAL_ENDPOINT_LABELS)
    right_is_terminal = right_label in set(TERMINAL_ENDPOINT_LABELS)
    canonical_branch_only_side = bool(
        normalized_kind == "side_chain"
        and any(
            str(getattr(view, "family", ""))
            in {"like24", "like24_secondary", "like24_bridge"}
            and bool(
                dict(getattr(view, "evidence", {}) or {}).get(
                    "edge_semantics_finalized", False
                )
            )
            and not bool(
                getattr(view, "open_left", False)
                or getattr(view, "open_right", False)
            )
            for view in list(source_views or [])
        )
    )
    if normalized_kind == "flex_chain" and not (
        left_is_aromatic and right_is_aromatic
    ):
        return f"invalid_flex_endpoints:{left_label}|{right_label}"
    if normalized_kind == "side_chain" and not (
        (left_is_aromatic and right_is_terminal)
        or (left_is_terminal and right_is_aromatic)
        or canonical_branch_only_side
    ):
        return f"invalid_side_endpoints:{left_label}|{right_label}"
    return None


def validate_branch_capacity(
    labels: Sequence[str],
    branch_map: Dict[int, List[str]],
) -> Optional[str]:
    for index, raw_label in enumerate(list(labels or [])):
        branch_count = sum(
            1
            for value in list(dict(branch_map or {}).get(int(index), []) or [])
            if str(value).strip()
        )
        if str(raw_label) == "14" and branch_count > 1:
            return f"su14_branch_capacity_exceeded:index={index}:count={branch_count}"
    return None


def nested_path_edges(path: str) -> Tuple[Set[Tuple[int, int]], Set[int]]:
    """Parse a rendered node route while preserving parenthesized edges."""

    text = str(path or "")
    index = 0
    edges: Set[Tuple[int, int]] = set()
    rendered: Set[int] = set()

    def _skip_separators() -> None:
        nonlocal index
        while index < len(text) and text[index] in "- ":
            index += 1

    def _level(stop: Optional[str] = None) -> Optional[int]:
        nonlocal index
        previous: Optional[int] = None
        first: Optional[int] = None
        while index < len(text):
            _skip_separators()
            if index >= len(text):
                break
            if stop is not None and text[index] == stop:
                index += 1
                break
            if text[index] == "(":
                index += 1
                child_first = _level(")")
                if previous is not None and child_first is not None:
                    edges.add(tuple(sorted((int(previous), int(child_first)))))
                continue
            if text[index] == ")":
                if stop is not None:
                    index += 1
                break
            start = index
            while index < len(text) and text[index].isdigit():
                index += 1
            if start == index:
                index += 1
                continue
            current = int(text[start:index])
            rendered.add(current)
            if first is None:
                first = current
            if previous is not None:
                edges.add(tuple(sorted((int(previous), current))))
            previous = current
        return first

    _level()
    return edges, rendered
