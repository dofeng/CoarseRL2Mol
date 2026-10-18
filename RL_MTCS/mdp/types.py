"""Immutable input tasks and topology contracts.

The inverse pipeline owns topology, original node identity and allowed closure
resources. MCTS places these units on the hexagonal lattice. Unused SU11
resources retain their original identities and have an assembly target of SU13.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from RL_MTCS.geometry import Axial

AROMATIC_KINDS = {
    "benzene", "naphthalene", "phenanthrene", "anthracene", "tetracene",
    "chrysene", "triphenylene", "pyrene", "benzo_pyrene", "perylene",
    "coronene", "aromatic", "aromatic_cluster", "aromatic_unit",
}
ALIPHATIC_RING_KINDS = frozenset({
    "aliphatic_ring", "fatty_ring", "fatty_ring_fused",
    "fatty_ring_side", "fatty_ring_upper_lower",
})


class InverseInputError(ValueError):
    """Raised when inverse output is not complete enough for placement."""

    def __init__(self, message: str, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class UnitNode:
    node_id: int
    su_type: int
    local_axial: Axial
    role: str = ""
    metadata: Tuple[Tuple[str, str], ...] = ()
    slot_id: str = ""


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
    # Stable logical (SU25 parent, fixed terminal SU22) identities.
    su25_center_branches: Tuple[Tuple[int, int], ...] = ()
    # Set from parsed connectivity. A ring label alone does not imply that
    # the unit shares an edge with an aromatic ring.
    attaches_to_aromatic_edge: bool = False

    @property
    def phase(self) -> str:
        kind = str(self.product_type).lower()
        source = str(dict(self.metadata).get("source_product_type", kind)).lower()
        if kind in AROMATIC_KINDS or kind.startswith("aromatic"):
            return "aromatic"
        if "ring" in kind or "ring" in source:
            return "ring"
        if source in {"bridge", "chain"} or "flex" in kind or "flex" in source:
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
    input_issues: Tuple[str, ...] = ()
    # Unused SU11 closure identities retained on aromatic perimeter slots as
    # actual SU13. These nodes have no external connection task.
    aromatic_residual_node_ids: Tuple[int, ...] = ()

    def node_map(self) -> Dict[int, int]:
        return {int(node_id): int(su) for node_id, su in self.node_su_types}

    def ownership_map(self) -> Dict[int, str]:
        return {int(node_id): str(unit_id) for node_id, unit_id in self.ownership}

    def assembly_target_node_map(self) -> Dict[int, int]:
        """Expected labels after assembly, preserving original SU provenance."""
        return {**self.node_map(), **{gid: 13 for gid in self.aromatic_residual_node_ids}}

    def aromatic_residual_rows(self):
        return [dict(node_id=gid, source_su_type=11, final_su_type=13,
                     state='aromatic_residual', completion_tolerated=True,
                     reason='unused_su11_retained_as_su13')
                for gid in self.aromatic_residual_node_ids]

    def unassembled_resource_map(self) -> Dict[int, int]:
        return {
            int(resource.node_id): int(resource.su_type)
            for resource in self.unassembled_closure_resources
        }

    def input_node_map(self) -> Dict[int, int]:
        return {**self.node_map(), **self.unassembled_resource_map(),
                **dict(self.idle_aromatic_nodes)}


@dataclass(frozen=True)
class UnitPlacement:
    unit_id: str
    rotation: int
    translation: Axial
    reflection: bool = False


@dataclass(frozen=True)
class AssemblyEdge:
    source_node_id: int
    target_node_id: int
    kind: str
    contract_id: str = ""
    direction: Optional[Axial] = None


@dataclass(frozen=True)
class ConnectionAction:
    """A complete validated atomic effect, with no live state snapshots."""

    phase: str
    source_key: bytes
    target_key: bytes
    payload: bytes
    description_json: str
    conformer: Optional[TopologyUnit] = None
    placements: Tuple[Tuple[str, int, Axial, bool], ...] = ()
    endpoint_bindings: Tuple[Tuple[int, int], ...] = ()

    def describe(self):
        import json
        return json.loads(self.description_json)
