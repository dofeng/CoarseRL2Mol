"""Internal candidate and materialization records for aliphatic rings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...stage3_models import Stage3Serializable

if TYPE_CHECKING:
    from ..core.fragment import AssemblyFragmentView


@dataclass
class RingCandidatePlan(Stage3Serializable):
    ring_kind: str = ""
    skeleton_kind: str = ""
    missing_slots: List[str] = field(default_factory=list)
    source_views: List["AssemblyFragmentView"] = field(default_factory=list)
    ordered_segments: List[Dict[str, Any]] = field(default_factory=list)
    closure_target_fragment_id: Optional[int] = None
    closure_target_node_id: Optional[int] = None
    like24_count_min: int = 0
    like24_count_max: int = 0
    template_complete: bool = False
    required_counts: Dict[int, int] = field(default_factory=dict)
    available_counts: Dict[int, int] = field(default_factory=dict)
    missing_counts: Dict[int, int] = field(default_factory=dict)
    selected_su24_ids: List[int] = field(default_factory=list)
    protected_node_ids: List[int] = field(default_factory=list)


@dataclass
class ResolvedRingSegment(Stage3Serializable):
    view: Optional["AssemblyFragmentView"] = None
    labels: List[str] = field(default_factory=list)
    node_ids: List[Optional[int]] = field(default_factory=list)
    ring_backbone_labels: List[str] = field(default_factory=list)
    ring_backbone_node_ids: List[Optional[int]] = field(default_factory=list)
    branch_map: Dict[int, List[str]] = field(default_factory=dict)
    branch_id_map: Dict[int, List[str]] = field(default_factory=dict)
    branch_modes: List[str] = field(default_factory=list)
    branch_paths: List[Dict[str, Any]] = field(default_factory=list)
    external_attachments: List[List[int]] = field(default_factory=list)
    like24_node_ids: List[int] = field(default_factory=list)
    node_profiles_by_id: Dict[int, str] = field(default_factory=dict)
    replaced_branch_requirements: List[Dict[str, Any]] = field(default_factory=list)
    pool_node_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    fixed_source_node_ids: List[int] = field(default_factory=list)
    source_views: List["AssemblyFragmentView"] = field(default_factory=list)
    suppressed_support_node_ids: List[int] = field(default_factory=list)
    ring_template_kind: str = ""
    ring_template_roles: List[str] = field(default_factory=list)
    required_counts: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedRingTopology(Stage3Serializable):
    source_views: List["AssemblyFragmentView"] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    node_ids: List[Optional[int]] = field(default_factory=list)
    backbone_labels: List[str] = field(default_factory=list)
    backbone_node_ids: List[Optional[int]] = field(default_factory=list)
    topology_notation: str = ""
    node_id_notation: str = ""
    support_node_ids: List[int] = field(default_factory=list)
    pool_node_ids_by_su: Dict[str, List[int]] = field(default_factory=dict)
    fixed_source_node_ids: List[int] = field(default_factory=list)
    branch_paths: List[Dict[str, Any]] = field(default_factory=list)
    external_attachments: List[List[int]] = field(default_factory=list)
    like24_node_ids: List[int] = field(default_factory=list)
    node_profiles_by_id: Dict[int, str] = field(default_factory=dict)
    replaced_branch_requirements: List[Dict[str, Any]] = field(default_factory=list)
