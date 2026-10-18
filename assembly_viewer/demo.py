"""Small synthetic topology inventory for exercising the actual assembly engine."""
from dataclasses import replace

from RL_MTCS.geometry import aromatic_slot_direction
from RL_MTCS.io.input import _make_cluster_template
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import (UnitNode, UnitEdge, UnitPort, TopologyUnit,
                              ExternalContract, InverseAssemblyInput, HeteroAromaticRequest)


def demo_state():
    template = _make_cluster_template("benzene")
    aromatic = []
    planned = {}
    ports = [[] for _ in range(4)]
    for i in range(4):
        nodes = tuple(UnitNode(i*100+j, 13, tuple(point)) for j, point in enumerate(template.coords))
        unit = TopologyUnit(f"aromatic:{i}", "benzene", nodes,
            tuple(UnitEdge(i*100+a, i*100+b, "aromatic") for a, b in template.edges),
            local_rings=tuple(template.rings))
        aromatic.append(unit)
        planned.update({n.node_id:13 for n in nodes})
        gid = i*100
        planned[gid] = 10
        ports[i].append(UnitPort(f"rigid:{gid}", gid, "rigid", "rigid", aromatic_slot_direction(unit,gid)))
    chain = TopologyUnit("chain:0", "flex_chain", (UnitNode(400,23,(0,0)),),
        ports=(UnitPort("a",400,"topology_boundary","flex",(-1,-1)),
               UnitPort("b",400,"topology_boundary","flex",(1,0))))
    ring = TopologyUnit("ring:0", "aliphatic_ring",
        tuple(UnitNode(500+i,23,p) for i,p in enumerate(((0,0),(1,1),(1,2),(0,2)))),
        tuple(UnitEdge(500+i,501+i) for i in range(3)),
        ports=(UnitPort("a",500,"topology_boundary","ring",(-1,0)),
               UnitPort("b",503,"topology_boundary","ring",(-1,-1))),
        metadata=(("ring_geometry_mode","external_contract_ring"),))
    side = TopologyUnit("side:0", "side_chain", (UnitNode(600,22,(0,0)),),
        ports=(UnitPort("a",600,"topology_boundary","side",(1,1)),))
    contracts = (
        ExternalContract("flex:a","chain:0",400,"aromatic:0",2,"flex",source_port_id="a",target_port_id="flex:a"),
        ExternalContract("flex:b","chain:0",400,"aromatic:2",202,"flex",source_port_id="b",target_port_id="flex:b"),
        ExternalContract("ring:a","ring:0",500,"aromatic:0",4,"ring",source_port_id="a",target_port_id="ring:a"),
        ExternalContract("ring:b","ring:0",503,"aromatic:0",5,"ring",source_port_id="b",target_port_id="ring:b"),
        ExternalContract("side:a","side:0",600,"aromatic:1",101,"side",source_port_id="a",target_port_id="side:a"),
    )
    for contract in contracts:
        index = int(contract.target_unit_id.split(":")[1])
        ports[index].append(UnitPort(contract.target_port_id,contract.target_node_id,"aromatic_boundary",
                                     contract.kind,aromatic_slot_direction(aromatic[index],contract.target_node_id)))
        planned[contract.target_node_id] = 11
    aromatic = [replace(u,ports=tuple(ports[i])) for i,u in enumerate(aromatic)]
    planned[302] = 26
    units = (*aromatic,chain,ring,side)
    planned.update({n.node_id:n.su_type for u in (chain,ring,side) for n in u.nodes})
    problem = InverseAssemblyInput(units,contracts,tuple(sorted(planned.items())),
        tuple((n.node_id,u.unit_id) for u in units for n in u.nodes),
        hetero_requests=(HeteroAromaticRequest(26,6,302),))
    state = HexAssemblyState(problem)
    for i,unit in enumerate(aromatic):
        if not state.place_unit(unit.unit_id,0,(i*30,0)):
            raise RuntimeError("synthetic aromatic initialization failed")
    return state
