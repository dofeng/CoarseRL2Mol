"""Core value objects used across topology assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...stage3_models import Stage3Serializable

if TYPE_CHECKING:
    from .fragment import AssemblyFragmentView


@dataclass
class BranchStub(Stage3Serializable):
    topology_notation: str = ""
    node_id_notation: str = ""
    source_fragment_ids: List[int] = field(default_factory=list)
    source_node_ids: List[int] = field(default_factory=list)
    branch_kind: str = ""
    mode: str = ""


@dataclass
class BranchClosureResult(Stage3Serializable):
    branch_topology: str = ""
    branch_node_ids: str = ""
    branch_kind: str = ""
    mode: str = ""
    anchor_index: int = 0
    attach_role: str = "center_branch"
    target_path: str = "center_side_branch"
    suppressed_support_node_ids: List[int] = field(default_factory=list)
    source_fragment_ids: List[int] = field(default_factory=list)
    source_node_ids: List[int] = field(default_factory=list)
    # Canonical physical identity after the branch route has been rendered.
    # Keep source_* above for serialized compatibility with existing outputs.
    fixed_source_node_ids: List[int] = field(default_factory=list)
    rendered_node_ids: List[int] = field(default_factory=list)
    pool_node_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    main_labels: List[str] = field(default_factory=list)
    main_node_ids: List[Optional[int]] = field(default_factory=list)


@dataclass(frozen=True)
class ClosureOperatorSpec(Stage3Serializable):
    name: str = ""
    scope: str = ""
    fixed_side_compatible_only: bool = False


@dataclass(frozen=True)
class AssemblyEndpoint(Stage3Serializable):
    side: str
    node_id: Optional[int]
    label: str
    is_open: bool
    role: str
    bridge_node_id: Optional[int] = None
    bridge_label: str = ""


@dataclass(frozen=True)
class AssemblyCompositionCandidate:
    left: "AssemblyFragmentView"
    right: "AssemblyFragmentView"
    left_reversed: bool
    right_reversed: bool
    product_kind: str
    reason: str
    score: Tuple[Any, ...]
