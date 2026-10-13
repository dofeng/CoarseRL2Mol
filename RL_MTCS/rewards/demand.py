"""Frozen spatial demand for one selection round and its checkpoint retries.

Reference capacities come from the original input, never from a candidate's
remaining free entrances or a failed feasibility scan. Reports contain counts
only; no search states or candidate coordinates are retained.
"""

from dataclasses import dataclass

from RL_MTCS.mdp.sites import aromatic_perimeter_ids


_FUTURE_PHASES = {
    'aromatic': ('flex', 'ring', 'side'),
    'rigid': ('flex', 'ring', 'side'),
    'flex': ('flex', 'ring', 'side'),
    'ring': ('ring', 'side'),
    'side': ('side',),
    'substitution': (),
    'final': (),
}


@dataclass(frozen=True)
class PhaseDemand:
    phase: str
    units: int
    nodes: int
    ports: int


@dataclass(frozen=True)
class SpatialDemand:
    phase: str
    reference_backbone_nodes: int
    reference_su13_sites: int
    remaining: tuple[PhaseDemand, ...]

    @property
    def remaining_nodes(self):
        return sum(row.nodes for row in self.remaining)

    @property
    def remaining_ports(self):
        return sum(row.ports for row in self.remaining)

    @property
    def remaining_units(self):
        return sum(row.units for row in self.remaining)

    @property
    def pressure(self):
        # A zero capacity with positive demand means high pressure, not zero.
        volume = self.remaining_nodes / max(1, self.reference_backbone_nodes)
        ports = self.remaining_ports / max(1, self.reference_su13_sites)
        return min(1.0, max(volume, ports))

    def to_dict(self):
        return dict(phase=self.phase, pressure=self.pressure,
                    reference_backbone_nodes=self.reference_backbone_nodes,
                    reference_su13_sites=self.reference_su13_sites,
                    remaining_units=self.remaining_units,
                    remaining_nodes=self.remaining_nodes,
                    remaining_ports=self.remaining_ports,
                    stages={r.phase: dict(units=r.units, nodes=r.nodes, ports=r.ports)
                            for r in self.remaining})


def spatial_demand(state, phase=None):
    """Measure tasks still awaiting placement at the START of a search round.

    A failed chain or an excluded unit never shrinks demand within a round.
    Earlier phases stop reserving space only once their stage has ended; their
    unfinished tasks remain untouched in the original completion inventory.
    SU26/SU30 requests do not consume SU13 clearance demand.
    """
    phase = state.stage if phase is None else phase
    future = _FUTURE_PHASES[phase]
    original = state.source_problem or state.problem
    units = {u.unit_id: u for u in original.units}
    backbone = sum(len(u.nodes) for u in original.units
                   if u.phase in ('aromatic', 'flex'))
    reserved = {n for r in original.hetero_requests
                for n in (r.target_node_id, r.synthetic_node_id) if n is not None}
    reserved.update(n for c in original.contracts if c.kind == 'rigid'
                    for n in (c.source_node_id, c.target_node_id))
    # Staging SU13s are physical slots, including slots whose planned labels
    # will change after dynamic binding. Do not restrict them to task GIDs.
    capacity = 0
    for unit in original.units:
        if unit.phase == 'aromatic':
            perimeter = aromatic_perimeter_ids(unit)
            capacity += sum(n.su_type == 13 and n.node_id not in reserved
                            and n.node_id in perimeter for n in unit.nodes)
    pending = {u.unit_id for u in original.units
               if u.phase in future and u.unit_id not in state.placements}
    port_counts = {uid: 0 for uid in pending}
    for contract in original.contracts:
        for uid, other in ((contract.source_unit_id, contract.target_unit_id),
                           (contract.target_unit_id, contract.source_unit_id)):
            if uid in pending and units[other].phase == 'aromatic':
                port_counts[uid] += 1
    rows = tuple(PhaseDemand(p,
        sum(units[uid].phase == p for uid in pending),
        sum(len(units[uid].nodes) for uid in pending if units[uid].phase == p),
        sum(port_counts[uid] for uid in pending if units[uid].phase == p)) for p in future)
    return SpatialDemand(phase, backbone, capacity, rows)


__all__ = ['SpatialDemand', 'PhaseDemand', 'spatial_demand']
