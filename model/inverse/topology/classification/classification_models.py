from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..stage3_models import Stage3Serializable, TopologyClosureSlot

@dataclass
class TopologyRef(Stage3Serializable):
    kind: str
    role: str
    su_type: Optional[int] = None
    node_id: Optional[int] = None
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyFragment(Stage3Serializable):
    fragment_id: int
    family: str
    class_type: str
    topology_template: str
    main_path: List[TopologyRef] = field(default_factory=list)
    source_node_ids: List[int] = field(default_factory=list)
    claimed_node_ids: List[int] = field(default_factory=list)
    open_slots: List[str] = field(default_factory=list)
    branch_slots: List[str] = field(default_factory=list)
    mainline_node_ids: List[int] = field(default_factory=list)
    mainline_labels: List[str] = field(default_factory=list)
    open_mainline_slots: List[TopologyClosureSlot] = field(default_factory=list)
    branch_contracts: List[TopologyClosureSlot] = field(default_factory=list)
    context_node_ids: List[int] = field(default_factory=list)
    open_boundaries: List[Dict[str, Any]] = field(default_factory=list)
    atomic_group_ids: List[int] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyClassificationResult(Stage3Serializable):
    fragments: List[TopologyFragment] = field(default_factory=list)
    atomic_groups: List[Dict[str, Any]] = field(default_factory=list)
    # The classification pass is the sole authority for fixed physical node
    # ownership.  Assembly may consume open resources, but it must never
    # infer a second owner from a projected view.
    fixed_node_owner_by_id: Dict[int, int] = field(default_factory=dict)
    # Concrete support ownership discovered from raw SU24 hop1 semantics.
    # These are exact graph neighbours, not generic SU11/SU22 reservations.
    su24_support_bindings: List[Dict[str, Any]] = field(default_factory=list)
    resource_pool: Dict[str, Any] = field(default_factory=dict)
    consumed_node_ids: List[int] = field(default_factory=list)
    unclassified_nodes: List[Dict[str, Any]] = field(default_factory=list)
    classification_rejects: List[Dict[str, Any]] = field(default_factory=list)
    claim_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    support_units_only: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
