"""Immutable inverse-to-MCTS topology contracts.

The legacy RL_MTCS implementation rebuilt aromatic clusters from SU counts and
then allocated new chains.  This module defines the replacement boundary: the
inverse pipeline owns topology and node identity, while MCTS only places the
already-complete units on the hexagonal lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from model.shared.topology_contract import (
    MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES,
    TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES,
)

from .geometry import (
    Axial,
    HEX_DIRECTIONS,
    HEX_VERTEX_OFFSETS,
    valid_hex_directions,
)

AROMATIC_KINDS = {
    "benzene", "naphthalene", "phenanthrene", "anthracene", "tetracene",
    "chrysene", "triphenylene", "pyrene", "benzo_pyrene", "perylene",
    "coronene", "aromatic", "aromatic_cluster", "aromatic_unit",
}


class InverseInputError(ValueError):
    """Raised when inverse output is not complete enough for placement."""


class InputSourceKind(str, Enum):
    CSV_PAIR = "csv_pair"
    INVERSE_JSON = "inverse_json"


@dataclass
class AssemblyConfig:
    """Configuration for the production CSV assembly workflow.

    Only the rigid stage uses MCTS.  Flex and side placement use the shared
    candidate and beam limits, so their former MCTS-only knobs are omitted.
    """

    rigid_iterations: int = 16
    candidate_k: int = 2
    beam_width: int = 1
    max_steps: int = 128
    c_puct: float = 1.414
    seed: Optional[int] = 7
    strict_geometry: bool = True
    spectrum_csv: Optional[str] = None
    elements: Any = None
    element_composition_csv: Optional[str] = None
    g2s_checkpoint: Optional[str] = None
    nmr_device: str = "auto"
    nmr_batch_size: int = 512
    nmr_hwhm: float = 1.0
    nmr_weight: float = 1.0
    geometry_weight: float = 0.05
    contact_weight: float = 0.02
    substitution_beam_width: int = 4
    substitution_candidate_k: int = 32
    substitution_nmr_weight: float = 1.0

    def __post_init__(self) -> None:
        self.rigid_iterations = max(1, int(self.rigid_iterations))
        self.candidate_k = max(1, int(self.candidate_k))
        self.beam_width = max(1, int(self.beam_width))
        self.max_steps = max(1, int(self.max_steps))
        self.nmr_batch_size = max(1, int(self.nmr_batch_size))
        self.nmr_hwhm = max(1e-6, float(self.nmr_hwhm))
        self.nmr_weight = max(0.0, float(self.nmr_weight))
        self.geometry_weight = max(0.0, float(self.geometry_weight))
        self.contact_weight = max(0.0, float(self.contact_weight))
        self.substitution_beam_width = max(1, int(self.substitution_beam_width))
        self.substitution_candidate_k = max(1, int(self.substitution_candidate_k))
        self.substitution_nmr_weight = max(0.0, float(self.substitution_nmr_weight))

    @property
    def iterations(self) -> int:
        return int(self.rigid_iterations)

    @property
    def flex_iterations(self) -> int:
        return 1

    @property
    def side_iterations(self) -> int:
        return 1

    @property
    def rigid_max_steps(self) -> int:
        return int(self.max_steps)

    @property
    def flex_max_steps(self) -> int:
        return int(self.max_steps)

    @property
    def side_max_steps(self) -> int:
        return int(self.max_steps)


@dataclass
class MCTSConfig:
    """Backward-compatible configuration for older callers and JSON input."""

    c_puct: float = 1.414
    iterations: int = 64
    flex_iterations: int = 64
    side_iterations: int = 64
    candidate_k: int = 16
    beam_width: int = 1
    max_depth: int = 128
    rigid_max_steps: int = 128
    flex_max_steps: int = 128
    side_max_steps: int = 128
    seed: Optional[int] = None
    strict_geometry: bool = False
    spectrum_csv: Optional[str] = None
    elements: Any = None
    element_composition_csv: Optional[str] = None
    g2s_checkpoint: Optional[str] = None
    nmr_device: str = "auto"
    nmr_batch_size: int = 512
    nmr_hwhm: float = 1.0
    nmr_weight: float = 1.0
    geometry_weight: float = 0.05
    contact_weight: float = 0.02
    substitution_beam_width: int = 4
    substitution_candidate_k: int = 32
    substitution_nmr_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "iterations", "flex_iterations", "side_iterations", "candidate_k",
            "beam_width", "max_depth", "rigid_max_steps", "flex_max_steps",
            "side_max_steps",
        ):
            setattr(self, name, max(1, int(getattr(self, name))))
        self.nmr_batch_size = max(1, int(self.nmr_batch_size))
        self.nmr_hwhm = max(1e-6, float(self.nmr_hwhm))
        self.nmr_weight = max(0.0, float(self.nmr_weight))
        self.geometry_weight = max(0.0, float(self.geometry_weight))
        self.contact_weight = max(0.0, float(self.contact_weight))
        self.substitution_beam_width = max(1, int(self.substitution_beam_width))
        self.substitution_candidate_k = max(1, int(self.substitution_candidate_k))
        self.substitution_nmr_weight = max(0.0, float(self.substitution_nmr_weight))



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
    source_kind: InputSourceKind = InputSourceKind.INVERSE_JSON
    hetero_requests: Tuple[HeteroAromaticRequest, ...] = ()
    unassembled_closure_resources: Tuple[UnassembledClosureResource, ...] = ()

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
        return {**self.node_map(), **self.unassembled_resource_map()}


def build_inverse_input(payload: Any) -> InverseAssemblyInput:
    """Compatibility entry point; parsing is owned by :mod:`RL_MTCS.input`."""
    from .input import build_inverse_input as _build

    return _build(payload)
