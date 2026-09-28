"""Immutable inverse-to-MCTS topology contracts.

The legacy RL_MTCS implementation rebuilt aromatic clusters from SU counts and
then allocated new chains.  This module defines the replacement boundary: the
inverse pipeline owns topology and node identity, while MCTS only places the
already-complete units on the hexagonal lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .geometry import Axial

AROMATIC_KINDS = {
    "benzene", "naphthalene", "phenanthrene", "anthracene", "tetracene",
    "chrysene", "triphenylene", "pyrene", "benzo_pyrene", "perylene",
    "coronene", "aromatic", "aromatic_cluster", "aromatic_unit",
}


class InverseInputError(ValueError):
    """Raised when inverse output is not complete enough for placement."""

    def __init__(self, message: str, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class AssemblySearchError(RuntimeError):
    """A bounded placement search failed; this does not prove invalid input."""

    def __init__(self, message: str, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass
class AssemblyConfig:
    """Configuration for the production CSV assembly workflow.

    CSV rigid connections, Flex and Side use persistent PUCT trees.
    rigid_iterations bounds root trials and simulations per rigid contract/root.
    search_budget bounds simulations of one joint Flex tree across all retained
    rigid candidates, and separately the Side tree. candidate_k and beam_width
    control proposal/retention widths; conformer_limit bounds tree geometries.
    c_puct controls exploration and seed controls reproducible tie-breaking.
    """

    rigid_iterations: int = 16
    candidate_k: int = 2
    beam_width: int = 1
    max_steps: int = 128
    c_puct: float = 1.414
    seed: Optional[int] = 7
    search_budget: int = 2000
    conformer_limit: int = 12
    max_tree_nodes: int = 128
    domain_check_limit: int = 64
    domain_pose_limit: int = 16
    max_domain_check_limit: int = 512
    max_domain_pose_limit: int = 64
    strict_geometry: bool = True
    stage_retry_limit: int = 2
    spectrum_csv: Optional[str] = None
    elements: Any = None
    element_composition_csv: Optional[str] = None
    g2s_checkpoint: Optional[str] = None
    nmr_device: str = "auto"
    nmr_batch_size: int = 1
    nmr_graph_cache_limit: int = 256
    nmr_node_cache_limit: int = 2048
    nmr_record_limit: int = 5000
    nmr_model_call_limit: int = 1
    nmr_center_limit: int = 1
    nmr_hwhm: float = 1.0
    nmr_weight: float = 1.0
    geometry_weight: float = 0.05
    contact_weight: float = 0.02
    substitution_beam_width: int = 4
    substitution_candidate_k: int = 32
    substitution_nmr_weight: float = 1.0
    # Runtime reporting is intentionally lightweight: it prints one summary
    # per completed stage and writes one PNG for every stage output directory.
    verbose: bool = True
    save_stage_images: bool = True

    def __post_init__(self) -> None:
        # These are hard process-level bounds.  A large simulation budget is
        # useful only while the retained tree and candidate domains remain
        # bounded; otherwise every state clone can remain reachable from the
        # persistent tree and exhaust a notebook kernel.
        self.search_budget = min(20000, max(1, int(self.search_budget)))
        self.conformer_limit = min(32, max(1, int(self.conformer_limit)))
        self.max_tree_nodes = min(256, max(16, int(self.max_tree_nodes)))
        self.domain_check_limit = min(512, max(1, int(self.domain_check_limit)))
        self.domain_pose_limit = min(64, max(1, int(self.domain_pose_limit)))
        self.max_domain_check_limit = min(
            2048,
            max(self.domain_check_limit, int(self.max_domain_check_limit)),
        )
        self.max_domain_pose_limit = min(
            512,
            max(self.domain_pose_limit, int(self.max_domain_pose_limit)),
        )
        self.rigid_iterations = max(1, int(self.rigid_iterations))
        self.candidate_k = max(1, int(self.candidate_k))
        self.beam_width = max(1, int(self.beam_width))
        self.max_steps = max(1, int(self.max_steps))
        self.stage_retry_limit = min(4, max(0, int(self.stage_retry_limit)))
        # Large PyG batches create a high transient peak during local graph
        # collation.  NMR scoring is already chunked, so enforce a one-graph
        # batch at the configuration boundary.
        self.nmr_batch_size = 1
        self.nmr_graph_cache_limit = min(256, max(1, int(self.nmr_graph_cache_limit)))
        self.nmr_node_cache_limit = min(2048, max(1, int(self.nmr_node_cache_limit)))
        self.nmr_record_limit = min(5000, max(0, int(self.nmr_record_limit)))
        self.nmr_model_call_limit = min(2, max(1, int(self.nmr_model_call_limit)))
        self.nmr_center_limit = 1
        self.nmr_hwhm = max(1e-6, float(self.nmr_hwhm))
        self.nmr_weight = max(0.0, float(self.nmr_weight))
        self.geometry_weight = max(0.0, float(self.geometry_weight))
        self.contact_weight = max(0.0, float(self.contact_weight))
        self.substitution_beam_width = max(1, int(self.substitution_beam_width))
        self.substitution_candidate_k = max(1, int(self.substitution_candidate_k))
        self.substitution_nmr_weight = max(0.0, float(self.substitution_nmr_weight))
        self.verbose = bool(self.verbose)
        self.save_stage_images = bool(self.save_stage_images)

@dataclass(frozen=True)
class UnitNode:
    node_id: int
    su_type: int
    local_axial: Axial
    role: str = ""
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class UnitEdge:
    source_node_id: int
    target_node_id: int
    kind: str = "internal"


@dataclass(frozen=True)
class UnitPort:
    port_id: str
    node_id: int
    role: str = "open"
    phase: str = ""
    direction: Optional[Axial] = None
    contract_id: str = ""
    required_su_types: Tuple[int, ...] = ()
    used: bool = False
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class HeteroAromaticRequest:
    """A final aromatic hetero-atom substitution request.

    ``ring_size`` is the ring represented by the final atom: six-membered
    requests replace one staging SU13, while five-membered requests collapse
    two adjacent staging SU13 sites into one SU26/SU30 node.
    """

    target_su: int
    ring_size: int
    target_node_id: int
    synthetic_node_id: Optional[int] = None
    hop1_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class UnassembledClosureResource:
    """A bounded Stage3 closure resource intentionally absent from the graph."""

    node_id: int
    su_type: int
    state: str = "remaining_closure_resource"
    hop1_ids: Tuple[int, ...] = ()
    reason: str = "stage3_tolerated_unassembled_closure_resource"


@dataclass(frozen=True)
class TopologyUnit:
    unit_id: str
    product_type: str
    nodes: Tuple[UnitNode, ...]
    internal_edges: Tuple[UnitEdge, ...] = ()
    ports: Tuple[UnitPort, ...] = ()
    source_fragment_ids: Tuple[int, ...] = ()
    source_node_ids: Tuple[int, ...] = ()
    topology_notation: str = ""
    node_id_notation: str = ""
    closed: bool = True
    local_rings: Tuple[Tuple[Axial, ...], ...] = ()
    metadata: Tuple[Tuple[str, str], ...] = ()

    @property
    def phase(self) -> str:
        kind = str(self.product_type).lower()
        if kind in AROMATIC_KINDS or kind.startswith("aromatic"):
            return "aromatic"
        if kind in {"flex_chain", "bridge", "aliphatic_ring", "fatty_ring", "fatty_ring_fused", "fatty_ring_side", "fatty_ring_upper_lower"}:
            return "flex"
        if "flex" in kind or "ring" in kind:
            return "flex"
        return "side"

    def node_map(self) -> Dict[int, UnitNode]:
        return {int(node.node_id): node for node in self.nodes}


@dataclass(frozen=True)
class ExternalContract:
    contract_id: str
    source_unit_id: str
    source_node_id: int
    target_unit_id: str
    target_node_id: int
    kind: str
    mandatory: bool = True
    fixed: bool = False
    source_port_id: str = ""
    target_port_id: str = ""
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class InverseAssemblyInput:
    units: Tuple[TopologyUnit, ...]
    contracts: Tuple[ExternalContract, ...]
    node_su_types: Tuple[Tuple[int, int], ...]
    ownership: Tuple[Tuple[int, str], ...]
    diagnostics: Tuple[Tuple[str, str], ...] = ()
    hetero_requests: Tuple[HeteroAromaticRequest, ...] = ()
    unassembled_closure_resources: Tuple[UnassembledClosureResource, ...] = ()
    source_rigid_contracts: Tuple[ExternalContract, ...] = ()
    idle_aromatic_nodes: Tuple[Tuple[int, int], ...] = ()

    def node_map(self) -> Dict[int, int]:
        return {int(node_id): int(su) for node_id, su in self.node_su_types}

    def ownership_map(self) -> Dict[int, str]:
        return {int(node_id): str(unit_id) for node_id, unit_id in self.ownership}

    def unassembled_resource_map(self) -> Dict[int, int]:
        return {
            int(resource.node_id): int(resource.su_type)
            for resource in self.unassembled_closure_resources
        }

    def input_node_map(self) -> Dict[int, int]:
        return {**self.node_map(), **self.unassembled_resource_map(),
                **dict(self.idle_aromatic_nodes)}
