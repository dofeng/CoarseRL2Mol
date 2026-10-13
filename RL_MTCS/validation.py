"""Common molecular geometry validation, including final substitutions."""
from __future__ import annotations

from collections import Counter
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from RL_MTCS.geometry import (Axial, hex_edge_valid, axial_add, axial_neg,
    transform_axial, state_layout_metrics, unit_sort_key,
    topology_site_valid, topology_edge_valid, su25_center_branch_errors)
from RL_MTCS.mdp.types import InverseInputError, UnitEdge
from RL_MTCS.geometry import segments_intersect
from RL_MTCS.reporting.diagnostics import record_feasibility
from model.shared.topology_contract import (MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES,
    TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES)
from RL_MTCS.mdp.sites import aromatic_template_errors
import re
from dataclasses import replace


def _attachment_pairs(
    attachment_edge: Any = None,
    attachment_edges: Optional[Iterable[Tuple[int, int]]] = None,
) -> Set[frozenset[int]]:
    pairs: Set[frozenset[int]] = set()
    if attachment_edge is not None:
        pairs.add(frozenset(int(value) for value in attachment_edge))
    for pair in attachment_edges or ():
        pairs.add(frozenset(int(value) for value in pair))
    return pairs


class GeometryValidation:
    """Queries and audits; never commit a molecular action."""

    def audit_identity_errors(self):
        errors = list(self.audit_resource_partition_errors())
        owner, planned = self.problem.ownership_map(), self.problem.node_map()
        source = self.source_problem.node_map()
        seen, slots, port_keys = set(), set(), set()
        for unit in self.problem.units:
            if self._units.get(unit.unit_id) != unit:
                errors.append(dict(reason='unit_identity_index_mismatch', unit_id=unit.unit_id))
            for node in unit.nodes:
                gid = node.node_id
                if gid in seen or owner.get(gid) != unit.unit_id:
                    errors.append(dict(reason='node_identity_owner_mismatch', node_id=gid))
                seen.add(gid)
                if planned.get(gid) != source.get(gid):
                    errors.append(dict(reason='planned_identity_changed', node_id=gid))
                if unit.phase != 'aromatic' and node.su_type != planned.get(gid):
                    errors.append(dict(reason='unit_node_su_identity_mismatch', node_id=gid))
                metadata = dict(node.metadata)
                if 'planned_su_type' in metadata and int(metadata['planned_su_type']) != planned.get(gid):
                    errors.append(dict(reason='node_metadata_identity_mismatch', node_id=gid))
                if node.slot_id:
                    if node.slot_id in slots:
                        errors.append(dict(reason='duplicate_physical_slot', slot_id=node.slot_id))
                    slots.add(node.slot_id)
            ids = set(unit.node_map())
            for port in unit.ports:
                key = f'{unit.unit_id}:{port.port_id}'
                if key in port_keys or self._ports.get(key) != port:
                    errors.append(dict(reason='port_identity_index_mismatch', port_id=key))
                port_keys.add(key)
                if port.node_id not in ids:
                    errors.append(dict(reason='port_node_owner_mismatch', unit_id=unit.unit_id, port_id=port.port_id))
        if seen != set(planned):
            errors.append(dict(reason='node_identity_inventory_mismatch'))
        for gid in self.node_axial:
            if gid not in seen or owner.get(gid) not in self.placements:
                errors.append(dict(reason='placed_node_identity_mismatch', node_id=gid))
        requests = {r.target_node_id: r for r in self.source_problem.hetero_requests}
        merged = {a.get('removed_node_id') for a in self.action_history
                  if a.get('kind') == 'substitution' and a.get('ring_size') == 5
                  and a.get('request_target') in requests
                  and a.get('removed_node_id') == requests[a['request_target']].synthetic_node_id}
        if set(source) - seen - merged:
            errors.append(dict(reason='input_identity_lost', node_ids=sorted(set(source) - seen - merged)))
        return errors

    def audit_contract_errors(self):
        errors, contracts = [], {c.contract_id: c for c in self.problem.contracts}
        if len(contracts) != len(self.problem.contracts):
            errors.append(dict(reason='duplicate_contract_id'))
        owner = self.problem.ownership_map()
        source_contracts = {c.contract_id: c for c in self.source_problem.contracts}
        for cid, original in source_contracts.items():
            current = contracts.get(cid)
            if (current is None or (current.kind, current.source_node_id, current.target_node_id,
                    current.source_port_id, current.target_port_id) !=
                    (original.kind, original.source_node_id, original.target_node_id,
                     original.source_port_id, original.target_port_id)):
                errors.append(dict(reason='original_contract_identity_changed', contract_id=cid))
        by_contract = {}
        edge_pairs = set()
        for edge in self.edges:
            pair = frozenset((edge.source_node_id, edge.target_node_id))
            if len(pair) != 2 or pair in edge_pairs or not pair <= self.node_axial.keys():
                errors.append(dict(reason='invalid_actual_edge', node_ids=sorted(pair)))
            edge_pairs.add(pair)
            if not edge.contract_id:
                uid = owner.get(edge.source_node_id)
                unit = self._units.get(uid)
                if (unit is None or owner.get(edge.target_node_id) != uid
                        or not any(pair == frozenset((e.source_node_id, e.target_node_id))
                                   and e.kind == edge.kind for e in unit.internal_edges)):
                    errors.append(dict(reason='undeclared_actual_edge', node_ids=sorted(pair)))
            if edge.contract_id:
                by_contract.setdefault(edge.contract_id, []).append(edge)
                c = contracts.get(edge.contract_id)
                if (c is None or edge.contract_id not in self.satisfied_contracts or edge.kind != c.kind
                    or {edge.source_node_id, edge.target_node_id} != {c.source_node_id, c.target_node_id}):
                    errors.append(dict(reason='edge_contract_identity_mismatch', contract_id=edge.contract_id))
        expected_ports = set()
        for cid in self.satisfied_contracts:
            c = contracts.get(cid)
            if c is None or len(by_contract.get(cid, ())) != 1:
                errors.append(dict(reason='satisfied_contract_edge_mismatch', contract_id=cid))
                continue
            for uid, pid, gid in ((c.source_unit_id, c.source_port_id, c.source_node_id),
                                  (c.target_unit_id, c.target_port_id, c.target_node_id)):
                key = f'{uid}:{pid}'
                if key in expected_ports:
                    errors.append(dict(reason='port_used_by_multiple_contracts', port_id=key))
                expected_ports.add(key)
                port = self._ports.get(key)
                if port is None or port.node_id != gid or owner.get(gid) != uid or key not in self.used_ports:
                    errors.append(dict(reason='contract_port_identity_mismatch', contract_id=cid, node_id=gid))
        unexpected = self.used_ports - expected_ports
        if unexpected:
            errors.append(dict(reason='used_port_without_contract', port_ids=sorted(unexpected)))
        return errors

    def audit_label_errors(self):
        adjacency = {gid: set() for gid in self.node_axial}
        for edge in self.edges:
            if edge.source_node_id in adjacency and edge.target_node_id in adjacency:
                adjacency[edge.source_node_id].add(edge.target_node_id)
                adjacency[edge.target_node_id].add(edge.source_node_id)
        owner, planned = self.problem.ownership_map(), self.problem.node_map()
        errors = []
        metadata = {n.node_id: dict(n.metadata) for u in self.problem.units for n in u.nodes}
        substituted = {a.get('request_target', a.get('representative_node_id', (a.get('site_ids') or [None])[0]))
                       for a in self.action_history if a.get('kind') == 'substitution'}
        for gid, neighbors in adjacency.items():
            unit = self._units.get(owner.get(gid))
            node = unit.node_map().get(gid) if unit is not None else None
            if node is None or gid not in planned:
                errors.append(dict(reason='label_node_identity_missing', node_id=gid))
                continue
            current = self.current_su_types.get(gid)
            valid = current == planned[gid]
            if unit.phase == 'aromatic':
                external = [n for n in neighbors if owner.get(n) != owner[gid]]
                if node.su_type == 12:
                    valid = current == planned[gid] == 12 and len(neighbors) == 3 and not external
                elif current in {26, 30}:
                    valid = (current == planned[gid] and len(neighbors) == 2 and not external
                             and (gid in substituted or node.su_type == current))
                elif not external:
                    valid = current == 13 and len(neighbors) == 2
                else:
                    valid = current == planned[gid] and len(neighbors) == 3 and len(external) == 1
            else:
                target = re.search(r'd(\d+)$', dict(node.metadata).get('su_label', ''))
                if target:
                    valid = valid and len(neighbors) == int(target.group(1))
                declared = metadata[gid].get('target_hop1_degree')
                if declared:
                    valid = valid and len(neighbors) == int(declared)
            if not valid:
                errors.append(dict(reason='actual_node_label_mismatch', node_id=gid,
                    current_su_type=current, planned_su_type=planned[gid],
                    actual_degree=len(neighbors), neighbor_ids=sorted(neighbors)))
        return errors

    def audit_aromatic_errors(self):
        modified = {a['unit_id'] for a in self.action_history
                    if a.get('kind') == 'substitution' and a.get('ring_size') == 5 and 'unit_id' in a}
        errors = aromatic_template_errors(replace(self.problem,
            units=tuple(u for u in self.problem.units if u.unit_id not in modified)))
        for uid in modified:
            unit = self.unit(uid)
            original = next((u for u in self.source_problem.units if u.unit_id == uid), None)
            removed_coordinates = {tuple(a.get('removed_local_axial', ())) for a in self.action_history
                                   if a.get('kind') == 'substitution' and a.get('ring_size') == 5
                                   and a.get('unit_id') == uid}
            if (original is None or unit.local_rings != tuple(tuple(p for p in ring if p not in removed_coordinates)
                                                              for ring in original.local_rings)):
                errors.append(dict(reason='substituted_template_changed', unit_id=uid))
            by_coord = {n.local_axial: n.node_id for n in unit.nodes}
            edges = {frozenset((e.source_node_id, e.target_node_id)) for e in unit.internal_edges}
            ring_edges = set()
            ring_nodes = set()
            for ring in unit.local_rings:
                if len(ring) not in {5, 6} or len(set(ring)) != len(ring) or any(p not in by_coord for p in ring):
                    errors.append(dict(reason='invalid_substituted_ring', unit_id=uid))
                    continue
                ring_nodes.update(by_coord[p] for p in ring)
                for a, b in zip(ring, ring[1:] + ring[:1]):
                    pair = frozenset((by_coord[a], by_coord[b]))
                    ring_edges.add(pair)
                    if pair not in edges:
                        errors.append(dict(reason='substituted_ring_edge_missing', unit_id=uid))
            if ring_edges != edges or ring_nodes != set(unit.node_map()):
                errors.append(dict(reason='substituted_ring_membership_mismatch', unit_id=uid))
        return errors

    @record_feasibility
    def can_transform_component(
        self,
        unit_id: str,
        rotation: int,
        translation: Axial,
        attachment_edge: Optional[Tuple[int, int]] = None,
        attachment_edges: Optional[Iterable[Tuple[int, int]]] = None,
        reflection: bool = False,
    ) -> Tuple[bool, str]:
        component_nodes = self.component_node_ids(unit_id)
        if not component_nodes:
            return False, "component_not_placed"
        transformed = {
            node_id: axial_add(
                transform_axial(self.node_axial[node_id], rotation, reflection),
                translation,
            )
            for node_id in component_nodes
        }
        if len(set(transformed.values())) != len(transformed):
            return False, "component_internal_coordinate_collision"
        geometry_edges = [
            UnitEdge(int(edge.source_node_id), int(edge.target_node_id), str(edge.kind))
            for edge in self.edges
            if int(edge.source_node_id) in component_nodes
            and int(edge.target_node_id) in component_nodes
        ]
        if not self._coordinates_geometry_valid(transformed, geometry_edges):
            return False, "component_non_honeycomb_edge"
        stationary = {
            axial: node_id
            for axial, node_id in self.occupied_axial.items()
            if int(node_id) not in component_nodes
        }
        # CSV initialization places every aromatic unit in a temporary
        # staging pose before the rigid pass.  Those disconnected poses are
        # not physical obstacles: only the component on the other side of
        # the contract may constrain a proposed rigid transform.  Build that
        # stationary map from the authoritative node coordinates rather than
        # from ``occupied_axial``, which also includes temporary parking poses.
        if str(self.stage) == "rigid" and attachment_edge is not None:
            contact_nodes = {int(value) for value in attachment_edge}
            stationary_contact_nodes = contact_nodes - component_nodes
            if stationary_contact_nodes:
                ownership = self.problem.ownership_map()
                stationary_units = {
                    ownership.get(int(node_id), "")
                    for node_id in stationary_contact_nodes
                    if ownership.get(int(node_id), "")
                }
                relevant_nodes: Set[int] = set()
                for unit_id in stationary_units:
                    relevant_nodes.update(self.component_node_ids(unit_id))
                stationary = {
                    self.node_axial[int(node_id)]: int(node_id)
                    for node_id in relevant_nodes
                    if int(node_id) not in component_nodes
                    and int(node_id) in self.node_axial
                }
        for node_id, axial in transformed.items():
            if axial in stationary:
                return False, f"component_axial_collision:{axial}:{stationary[axial]}:{node_id}"
        errors = self._proposed_geometry_errors(transformed, geometry_edges,
            _attachment_pairs(attachment_edge, attachment_edges), stationary)
        if errors:
            return False, errors[0]['reason']
        return True, ""

    def _edge_geometry_valid(
        self,
        source_node_id: int,
        target_node_id: int,
        source: Axial,
        target: Axial,
        kind: str | None = None,
    ) -> bool:
        if frozenset((int(source_node_id), int(target_node_id))) in self.substitution_relaxed_edges:
            return True
        return topology_edge_valid(source_node_id, target_node_id, source, target,
                                   self._su25_center_parents)

    def _node_site_valid(self, node_id, point):
        return topology_site_valid(node_id, point, self._su25_center_parents)

    def _proposed_geometry_errors(self, coordinates, edges, attachment_pairs=(), stationary=None):
        if stationary is None:
            stationary = {p: n for n, p in self.node_axial.items() if n not in coordinates}
        points = {n: self.final_node_positions.get(n, p) for p, n in stationary.items()}
        points.update(coordinates)
        existing = [e for e in self.edges
                    if e.source_node_id not in coordinates and e.target_node_id not in coordinates
                    and e.source_node_id in points and e.target_node_id in points]
        proposed = list(edges)
        for pair in attachment_pairs:
            if len(pair) == 2 and pair <= points.keys():
                left, right = sorted(pair)
                proposed.append(UnitEdge(left, right))
        return _segment_geometry_errors(points, [*existing, *proposed], set(coordinates))

    def _blocking_units(self, coordinates, errors=()):
        owner = self.problem.ownership_map()
        ids = {self.occupied_axial[p] for p in coordinates.values() if p in self.occupied_axial}
        for error in errors:
            ids.update(n for pair in error.get('edges', ()) for n in pair if n in self.node_axial)
            ids.update(n for n in error.get('edge', ()) if n in self.node_axial)
            if error.get('node') in self.node_axial:
                ids.add(error['node'])
        return {owner[n] for n in ids}

    def _coordinates_geometry_valid(
        self,
        coordinates: Dict[int, Axial],
        edges: Iterable[UnitEdge],
    ) -> bool:
        return all(self._node_site_valid(n, p) for n, p in coordinates.items()) and all(
            self._edge_geometry_valid(
                int(edge.source_node_id),
                int(edge.target_node_id),
                coordinates[int(edge.source_node_id)],
                coordinates[int(edge.target_node_id)],
                str(getattr(edge, "kind", "internal")),
            )
            for edge in edges
            if int(edge.source_node_id) in coordinates
            and int(edge.target_node_id) in coordinates
        )

    @record_feasibility
    def can_place(
        self,
        unit_id: str,
        rotation: int,
        translation: Axial,
        attachment_edge: Optional[Tuple[int, int]] = None,
        attachment_edges: Optional[Iterable[Tuple[int, int]]] = None,
        reflection: bool = False,
    ) -> Tuple[bool, str]:
        unit = self.unit(unit_id)
        if self.is_placed(unit_id):
            return False, "unit_already_placed"
        original = next(u for u in self.source_problem.units if u.unit_id == unit_id)
        if (su25_center_branch_errors(unit)
                or unit.su25_center_branches != original.su25_center_branches):
            return False, 'invalid_su25_center_branch'
        coordinates = self._placement_coordinates(
            unit, rotation, translation, reflection
        )
        if len(set(coordinates.values())) != len(coordinates):
            return False, "unit_internal_coordinate_collision"
        if not self._coordinates_geometry_valid(coordinates, unit.internal_edges):
            return False, "unit_non_honeycomb_edge"
        for node_id, axial in coordinates.items():
            occupant = self.occupied_axial.get(axial)
            if occupant is not None:
                return False, f"axial_collision:{axial}:{occupant}:{node_id}"
        attachment_pairs = _attachment_pairs(
            attachment_edge, attachment_edges
        )
        port_nodes = {int(port.node_id) for port in unit.ports}
        for pair in attachment_pairs:
            inside = pair & set(coordinates)
            outside = pair - set(coordinates)
            if len(inside) == len(outside) == 1:
                node_id, other_id = next(iter(inside)), next(iter(outside))
                if node_id in port_nodes and other_id in self.node_axial:
                    if not self._edge_geometry_valid(
                        node_id, other_id, coordinates[node_id], self.node_axial[other_id]
                    ):
                        return False, "external_endpoint_phase_mismatch"
        errors = self._proposed_geometry_errors(coordinates, unit.internal_edges, attachment_pairs)
        if errors:
            return False, errors[0]['reason']
        return True, ""

    def audit_resource_partition_errors(self) -> List[Dict[str, Any]]:
        assembled = self.problem.node_map()
        ownership = self.problem.ownership_map()
        resources = tuple(self.problem.unassembled_closure_resources)
        errors: List[Dict[str, Any]] = []
        if set(assembled) != set(ownership):
            errors.append({
                "reason": "assembled_node_ownership_mismatch",
                "missing_ownership_ids": sorted(set(assembled) - set(ownership)),
                "unexpected_ownership_ids": sorted(set(ownership) - set(assembled)),
            })
        resource_ids = [int(resource.node_id) for resource in resources]
        idle_ids = [node_id for node_id, _su in self.problem.idle_aromatic_nodes]
        if (len(idle_ids) != len(set(idle_ids))
                or set(idle_ids) & (set(assembled) | set(resource_ids))):
            errors.append({"reason": "idle_aromatic_partition_mismatch", "node_ids": idle_ids})
        if len(resource_ids) != len(set(resource_ids)):
            errors.append({
                "reason": "duplicate_unassembled_resource_id",
                "node_ids": sorted(resource_ids),
            })
        overlap = sorted(set(resource_ids) & set(assembled))
        if overlap:
            errors.append({
                "reason": "unassembled_resource_owned_by_graph",
                "node_ids": overlap,
            })
        if len(resources) > MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES:
            errors.append({
                "reason": "unassembled_resource_limit_exceeded",
                "count": len(resources),
                "limit": MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES,
            })
        for resource in resources:
            if str(resource.state) != "remaining_closure_resource":
                errors.append({
                    "reason": "unassembled_resource_state_invalid",
                    "node_id": int(resource.node_id),
                    "state": str(resource.state),
                })
            if int(resource.su_type) not in TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES:
                errors.append({
                    "reason": "unassembled_resource_su_invalid",
                    "node_id": int(resource.node_id),
                    "su_type": int(resource.su_type),
                })
        return errors

    def audit_substitution_errors(self) -> List[Dict[str, Any]]:
        """Validate the bookkeeping of the final SU26/SU30 pass.

        Five-membered replacements deliberately keep an integer staging
        coordinate for lattice audits and expose the physical mean coordinate
        through ``final_node_positions``.  This audit verifies that the two
        source sites were actually collapsed, the target label is present, and
        no final coordinates collide.
        """
        errors: List[Dict[str, Any]] = []
        current = self.current_su_types
        permitted = set()
        requests = {r.target_node_id: r for r in self.source_problem.hetero_requests}
        seen_requests = set()
        for action in self.action_history:
            if action.get("kind") != "substitution":
                continue
            target_su = int(action.get("target_su", -1))
            ring_size = int(action.get("ring_size", 0))
            target = action.get('request_target')
            request = requests.get(target)
            if (request is None or target in seen_requests or request.target_su != target_su
                    or request.ring_size != ring_size):
                errors.append(dict(reason='substitution_request_identity_mismatch', node_id=target))
            seen_requests.add(target)
            if ring_size == 6:
                site_ids = [int(value) for value in action.get("site_ids", [])]
                if len(site_ids) != 1 or site_ids[0] not in self.node_axial:
                    errors.append({"reason": "six_member_site_missing", "site_ids": site_ids})
                elif int(current.get(site_ids[0], -1)) != target_su:
                    errors.append({"reason": "six_member_label_mismatch", "site_id": site_ids[0], "expected": target_su, "actual": current.get(site_ids[0])})
                continue
            if ring_size != 5:
                continue
            permitted.update(frozenset(pair) for pair in action.get('affected_edge_pairs', ()))
            source_ids = [int(value) for value in action.get("site_ids", [])]
            representative = int(action.get("representative_node_id", -1))
            removed = int(action.get("removed_node_id", -1))
            if request is None or request.synthetic_node_id != removed or representative != target:
                errors.append(dict(reason='substitution_removed_real_identity', node_id=removed))
            if len(source_ids) != 2 or representative not in source_ids or removed not in source_ids:
                errors.append({"reason": "five_member_source_metadata_invalid", "site_ids": source_ids})
                continue
            if representative not in self.node_axial:
                errors.append({"reason": "five_member_representative_missing", "node_id": representative})
            if removed in self.node_axial or removed in current:
                errors.append({"reason": "five_member_removed_site_present", "node_id": removed})
            if int(current.get(representative, -1)) != target_su:
                errors.append({"reason": "five_member_label_mismatch", "node_id": representative, "expected": target_su, "actual": current.get(representative)})
            expected = action.get("mean_position")
            actual = self.final_node_positions.get(representative)
            if expected is None or actual is None or len(expected) < 2 or any(abs(float(expected[i]) - float(actual[i])) > 1e-9 for i in (0, 1)):
                errors.append({"reason": "five_member_mean_position_mismatch", "node_id": representative, "expected": expected, "actual": actual})
            source_positions = action.get("source_positions")
            if isinstance(source_positions, (list, tuple)) and len(source_positions) == 2 and all(isinstance(value, (list, tuple)) and len(value) >= 2 for value in source_positions):
                mean = (
                    (float(source_positions[0][0]) + float(source_positions[1][0])) / 2.0,
                    (float(source_positions[0][1]) + float(source_positions[1][1])) / 2.0,
                )
                if actual is None or any(abs(mean[i] - float(actual[i])) > 1e-9 for i in (0, 1)):
                    errors.append({"reason": "five_member_source_mean_mismatch", "node_id": representative, "expected": list(mean), "actual": actual})
            bonded = {
                frozenset((int(edge.source_node_id), int(edge.target_node_id)))
                for edge in self.edges
            }
            if frozenset(source_ids) in bonded:
                errors.append({"reason": "five_member_source_edge_present", "site_ids": source_ids})
        positions: Dict[Tuple[float, float], List[int]] = {}
        for node_id in self.node_axial:
            position = self.final_node_positions.get(int(node_id))
            if position is None:
                continue
            key = (round(float(position[0]), 9), round(float(position[1]), 9))
            positions.setdefault(key, []).append(int(node_id))
        for position, node_ids in positions.items():
            if len(node_ids) > 1:
                errors.append({"reason": "duplicate_final_positions", "position": list(position), "node_ids": sorted(node_ids)})
        for pair in self.substitution_relaxed_edges:
            if any(int(node_id) not in self.node_axial for node_id in pair):
                errors.append({"reason": "relaxed_edge_endpoint_missing", "edge": sorted(int(node_id) for node_id in pair)})
        if permitted != self.substitution_relaxed_edges:
            errors.append({'reason': 'undeclared_substitution_geometry_exception'})
        actual_pairs = {frozenset((e.source_node_id, e.target_node_id)) for e in self.edges}
        representatives = {a.get('representative_node_id') for a in self.action_history
                           if a.get('kind') == 'substitution' and a.get('ring_size') == 5}
        for pair in permitted:
            if pair not in actual_pairs or not pair.intersection(representatives):
                errors.append(dict(reason='invalid_substitution_geometry_exception', node_ids=sorted(pair)))
        if permitted != {p for p in actual_pairs if p.intersection(representatives)}:
            errors.append(dict(reason='substitution_exception_scope_mismatch'))
        return errors

    def audit_flex_endpoint_su_errors(self) -> List[Dict[str, Any]]:
        """Ensure Flex endpoints expose their inverse-topology SU types."""
        ownership = self.problem.ownership_map()
        phases = {
            str(unit.unit_id): str(unit.phase) for unit in self.problem.units
        }
        contracts = {
            str(contract.contract_id): contract
            for contract in self.problem.contracts
        }
        errors: List[Dict[str, Any]] = []
        for edge in self.edges:
            if str(edge.kind) not in {"flex", "ring"} or not str(edge.contract_id):
                continue
            contract = contracts.get(str(edge.contract_id))
            if contract is None:
                continue
            if phases.get(str(contract.source_unit_id)) == "aromatic":
                expected_node_id = int(contract.source_node_id)
            elif phases.get(str(contract.target_unit_id)) == "aromatic":
                expected_node_id = int(contract.target_node_id)
            else:
                continue
            expected_su_type = int(self.problem.node_map().get(expected_node_id, -1))
            for node_id in (int(edge.source_node_id), int(edge.target_node_id)):
                unit_id = ownership.get(node_id, "")
                if phases.get(unit_id) != "aromatic":
                    continue
                actual = int(self.current_su_types.get(node_id, -1))
                if actual != expected_su_type:
                    errors.append({
                        "contract_id": str(edge.contract_id),
                        "node_id": node_id,
                        "expected_su_type": expected_su_type,
                        "actual_su_type": actual,
                    })
        return errors

    def audit_geometry_errors(self) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for edge in self.edges:
            source = self.node_axial.get(int(edge.source_node_id))
            target = self.node_axial.get(int(edge.target_node_id))
            if source is None or target is None:
                continue
            if not self._edge_geometry_valid(
                int(edge.source_node_id), int(edge.target_node_id), source, target,
                str(edge.kind),
            ):
                errors.append({
                    "source_node_id": int(edge.source_node_id),
                    "target_node_id": int(edge.target_node_id),
                    "kind": str(edge.kind),
                    "contract_id": str(edge.contract_id),
                })
        errors.extend({'node_id': n, 'axial': list(p), 'reason': 'non_honeycomb_site'}
                      for n, p in self.node_axial.items() if not self._node_site_valid(n, p))
        return errors

    def audit_su25_center_branch_errors(self):
        errors = []
        original = {u.unit_id: u for u in self.source_problem.units}
        for unit in self.problem.units:
            errors.extend(su25_center_branch_errors(unit))
            source = original.get(unit.unit_id)
            if source is None or unit.su25_center_branches != source.su25_center_branches:
                errors.append(dict(reason='su25_center_branch_identity_changed', unit_id=unit.unit_id))
        expected = {child: parent for u in self.source_problem.units
                    for parent, child in u.su25_center_branches}
        if self._su25_center_parents != expected:
            errors.append(dict(reason='su25_center_branch_index_mismatch'))
        return errors

    def audit_topology_conformation_errors(self) -> List[Dict[str, Any]]:
        """Verify that every placed unit is still its immutable local shape."""
        errors: List[Dict[str, Any]] = []
        from RL_MTCS.actions.conformers import validate_conformer
        for unit_id, original in self.conformer_sources.items():
            try:
                validate_conformer(original, self.unit(unit_id))
            except InverseInputError as exc:
                errors.append({"unit_id": unit_id, "reason": str(exc)})
        for unit_id, placement in sorted(self.placements.items()):
            unit = self.unit(unit_id)
            ids = {n.node_id for n in unit.nodes}
            expected_edges = {frozenset((e.source_node_id, e.target_node_id)) for e in unit.internal_edges}
            actual_edges = {frozenset((e.source_node_id, e.target_node_id)) for e in self.edges
                            if e.source_node_id in ids and e.target_node_id in ids}
            if actual_edges != expected_edges:
                errors.append({
                    'unit_id': unit_id, 'reason': 'internal_connectivity_changed',
                    'missing_edges': sorted(sorted(e) for e in expected_edges - actual_edges),
                    'unexpected_edges': sorted(sorted(e) for e in actual_edges - expected_edges),
                })
            for node in unit.nodes:
                node_id = int(node.node_id)
                expected = axial_add(
                    transform_axial(
                        node.local_axial,
                        placement.rotation,
                        placement.reflection,
                    ),
                    placement.translation,
                )
                actual = self.node_axial.get(node_id)
                if actual != expected:
                    errors.append({
                        "unit_id": str(unit_id),
                        "node_id": node_id,
                        "reason": "local_geometry_changed",
                        "expected_axial": list(expected),
                        "actual_axial": list(actual) if actual is not None else None,
                    })
        return errors

    def audit_port_direction_errors(self) -> List[Dict[str, Any]]:
        """Check every satisfied contract against both transformed ports."""
        errors: List[Dict[str, Any]] = []
        for contract in self.problem.contracts:
            if str(contract.contract_id) not in self.satisfied_contracts:
                continue
            source = self.node_axial.get(int(contract.source_node_id))
            target = self.node_axial.get(int(contract.target_node_id))
            if source is None or target is None:
                errors.append({
                    "contract_id": str(contract.contract_id),
                    "reason": "satisfied_contract_endpoint_missing",
                })
                continue
            direction = self._edge_direction(source, target)
            if direction is None or not self._edge_geometry_valid(
                contract.source_node_id, contract.target_node_id, source, target, contract.kind
            ):
                errors.append({
                    "contract_id": str(contract.contract_id),
                    "reason": "satisfied_contract_geometry_invalid",
                })
                continue
            for unit_id, port_id, expected, partner in (
                (
                    str(contract.source_unit_id),
                    str(contract.source_port_id),
                    direction,
                    contract.target_node_id,
                ),
                (
                    str(contract.target_unit_id),
                    str(contract.target_port_id),
                    axial_neg(direction),
                    contract.source_node_id,
                ),
            ):
                port = self._ports.get(f"{unit_id}:{port_id}")
                if port is None:
                    errors.append({
                        "contract_id": str(contract.contract_id),
                        "unit_id": unit_id,
                        "reason": "contract_port_missing",
                    })
                    continue
                actual = self.global_port_direction(unit_id, port)
                if port.required_su_types and self.current_su_types.get(partner) not in port.required_su_types:
                    errors.append({
                        'contract_id': contract.contract_id, 'unit_id': unit_id, 'port_id': port_id,
                        'reason': 'port_partner_su_type_mismatch', 'partner_node_id': partner,
                        'actual_su_type': self.current_su_types.get(partner),
                        'required_su_types': list(port.required_su_types),
                    })
                if actual is not None and actual != expected:
                    errors.append({
                        "contract_id": str(contract.contract_id),
                        "unit_id": unit_id,
                        "port_id": port_id,
                        "reason": "port_direction_mismatch",
                        "expected_direction": list(expected),
                        "actual_direction": list(actual),
                    })
        return errors

    def aromatic_edge_sites(self, sites) -> bool:
        if len(sites) != 2 or sites[0] == sites[1]:
            return False
        owner = self.problem.ownership_map()
        uid = owner.get(sites[0])
        if uid is None or uid != owner.get(sites[1]) or self.unit(uid).phase != 'aromatic':
            return False
        pair = frozenset(sites)
        return any(frozenset((e.source_node_id, e.target_node_id)) == pair
                   for e in self.unit(uid).internal_edges)

    def audit_ring_attachment_errors(self) -> List[Dict[str, Any]]:
        errors = []
        for unit in self.problem.units:
            if not unit.attaches_to_aromatic_edge or not self.is_placed(unit.unit_id):
                continue
            ids = set(unit.node_map())
            anchors = []
            endpoints = []
            for edge in self.edges:
                a, b = edge.source_node_id, edge.target_node_id
                if (a in ids) != (b in ids):
                    anchors.append(b if a in ids else a)
                    endpoints.append(a if a in ids else b)
            if not self.aromatic_edge_sites(anchors):
                errors.append(dict(unit_id=unit.unit_id, reason='ring_requires_aromatic_edge',
                                   anchor_ids=anchors))
                continue
            adjacency = {gid: set() for gid in ids}
            for edge in unit.internal_edges:
                adjacency[edge.source_node_id].add(edge.target_node_id)
                adjacency[edge.target_node_id].add(edge.source_node_id)
            distance, frontier = {endpoints[0]: 0}, [endpoints[0]]
            for node in frontier:
                for other in adjacency[node]:
                    if other not in distance:
                        distance[other] = distance[node] + 1
                        frontier.append(other)
            if distance.get(endpoints[1]) != 3:
                errors.append(dict(unit_id=unit.unit_id, reason='external_ring_not_six_membered'))
        return errors

    def audit(self, *, diagnostics: bool = True) -> Dict[str, Any]:
        expected_ids = {int(node_id) for node_id, _unit_id in self.problem.ownership}
        placed_ids = set(int(node_id) for node_id in self.node_axial)
        nodes_by_coordinate = {}
        for node_id, axial in self.node_axial.items():
            nodes_by_coordinate.setdefault(axial, []).append(node_id)
        duplicate_axials = {str(p): sorted(ids) for p,ids in nodes_by_coordinate.items() if len(ids)>1}
        non_hex_edges = self.audit_geometry_errors()
        center_branch_errors = self.audit_su25_center_branch_errors()
        conformation_errors = self.audit_topology_conformation_errors()
        aromatic_errors = self.audit_aromatic_errors()
        identity_errors = self.audit_identity_errors()
        contract_errors = self.audit_contract_errors()
        label_errors = self.audit_label_errors()
        port_direction_errors = self.audit_port_direction_errors()
        substitution_errors = self.audit_substitution_errors()
        flex_endpoint_su_errors = self.audit_flex_endpoint_su_errors()
        resource_partition_errors = self.audit_resource_partition_errors()
        ring_attachment_errors = self.audit_ring_attachment_errors()
        from RL_MTCS.validation import actual_geometry_errors
        actual_errors = actual_geometry_errors(self)
        unassembled_resources = tuple(
            self.problem.unassembled_closure_resources
        )
        # Audit the complete assembled molecule, including staging nodes that
        # may be hidden from the active collision index during flex search.
        shape_metrics = state_layout_metrics(self, full=True, include_all=True) if diagnostics else {}
        shape_warnings = []
        if float(shape_metrics.get("principal_axis_ratio", 1.0)) > 2.5:
            shape_warnings.append("excessive_principal_axis_ratio")
        if float(shape_metrics.get("largest_void_radius", 0.0)) > 8.0:
            shape_warnings.append("large_internal_void")
        if float(shape_metrics.get("unit_density_cv", 0.0)) > 0.60:
            shape_warnings.append("uneven_unit_density")
        if float(shape_metrics.get("hull_fill", 1.0)) < 0.30:
            shape_warnings.append("low_convex_hull_fill")

        ownership = self.problem.ownership_map()
        unit_phase = {
            str(unit.unit_id): str(unit.phase) for unit in self.problem.units
        }
        side_unit_loads: Counter[str] = Counter()
        for edge in self.edges:
            if str(edge.kind) != "side":
                continue
            left = ownership.get(int(edge.source_node_id), "")
            right = ownership.get(int(edge.target_node_id), "")
            if left == right:
                continue
            if unit_phase.get(left) == "aromatic" and unit_phase.get(right) == "side":
                side_unit_loads[left] += 1
            elif unit_phase.get(right) == "aromatic" and unit_phase.get(left) == "side":
                side_unit_loads[right] += 1
        side_attachment_count = sum(side_unit_loads.values())
        side_coverage_ratio = (
            len(side_unit_loads) / side_attachment_count
            if side_attachment_count else 1.0
        )
        side_distribution_warnings = []
        if max(side_unit_loads.values(), default=0) > 2:
            side_distribution_warnings.append("side_unit_overloaded")
        if side_attachment_count >= 4 and side_coverage_ratio < 0.65:
            side_distribution_warnings.append("side_attachments_clustered")
        return {
            "complete": bool(self.is_complete()),
            "input_valid": not self.problem.input_issues,
            "input_errors": [json.loads(issue) for issue in self.problem.input_issues],
            "rigid_stage_complete": bool(self.phase_done("rigid")),
            "rigid_connections_done": sum(
                contract.kind == "rigid" and contract.contract_id in self.satisfied_contracts
                for contract in self.problem.contracts
            ),
            "rigid_connections_total": sum(
                su == 10 for _node_id, su in self.problem.node_su_types
            ) // 2,
            "stage": str(self.stage),
            "placed_unit_count": len(self.placements),
            "unit_count": len(self.problem.units),
            "placed_node_count": len(placed_ids),
            "expected_node_count": len(expected_ids),
            "source_node_count": int(dict(self.problem.diagnostics).get(
                "source_node_count",
                len(expected_ids) + len(unassembled_resources),
            )),
            "unassembled_resource_count": len(unassembled_resources) + len(expected_ids - placed_ids),
            "unassembled_resource_ids": sorted(
                {int(resource.node_id) for resource in unassembled_resources} | (expected_ids-placed_ids)
            ),
            "idle_aromatic_nodes": [
                {"node_id": node_id, "su_type": su_type}
                for node_id, su_type in self.problem.idle_aromatic_nodes
            ],
            "idle_aromatic_count": len(self.problem.idle_aromatic_nodes),
            "strict_all_nodes_assembled": not bool(
                unassembled_resources or self.problem.idle_aromatic_nodes or (expected_ids-placed_ids)),
            "completion_mode": "best_legal_partial_allowed",
            "resource_partition_errors": resource_partition_errors,
            "resource_partition_valid": not resource_partition_errors,
            "missing_node_ids": sorted(expected_ids - placed_ids),
            "unexpected_node_ids": sorted(placed_ids - expected_ids),
            "pending_contract_ids": sorted(contract.contract_id for contract in self.mandatory_contracts()),
            "duplicate_axials": duplicate_axials,
            "non_hex_edges": non_hex_edges,
            "su25_center_branch_errors": center_branch_errors,
            "su25_center_branches_valid": not center_branch_errors,
            "topology_conformation_errors": conformation_errors,
            "topology_conformation_valid": not conformation_errors,
            "port_direction_errors": port_direction_errors,
            "port_directions_valid": not port_direction_errors,
            "substitution_errors": substitution_errors,
            "substitution_valid": not substitution_errors,
            "flex_endpoint_su_errors": flex_endpoint_su_errors,
            "flex_endpoint_su_valid": not flex_endpoint_su_errors,
            "ring_attachment_errors": ring_attachment_errors,
            "ring_attachments_valid": not ring_attachment_errors,
            "shape_metrics": shape_metrics,
            "shape_valid": not shape_warnings,
            "shape_warnings": shape_warnings,
            "side_distribution": {
                "attachment_count": int(side_attachment_count),
                "covered_aromatic_unit_count": len(side_unit_loads),
                "coverage_ratio": float(side_coverage_ratio),
                "max_unit_load": max(side_unit_loads.values(), default=0),
                "unit_loads": dict(sorted(
                    side_unit_loads.items(), key=lambda row: unit_sort_key(row[0])
                )),
            },
            "side_distribution_valid": not side_distribution_warnings,
            "side_distribution_warnings": side_distribution_warnings,
            "actual_geometry_errors": actual_errors,
            "main_anchor": self.main_anchor,
            "excluded_unit_ids": sorted(self.excluded_unit_ids),
            "geometry_valid": not (
                actual_errors or
                aromatic_errors or duplicate_axials
                or non_hex_edges
                or center_branch_errors
                or conformation_errors
                or port_direction_errors
                or substitution_errors
                or flex_endpoint_su_errors
                or ring_attachment_errors
            ),
            "identity_valid": not identity_errors,
            "identity_errors": identity_errors,
            "topology_valid": not (contract_errors or aromatic_errors or any(
                e['reason'] == 'internal_connectivity_changed' for e in conformation_errors)),
            "topology_errors": [*contract_errors, *aromatic_errors, *(
                e for e in conformation_errors if e['reason'] == 'internal_connectivity_changed')],
            "contract_errors": contract_errors,
            "label_valid": not label_errors,
            "label_errors": label_errors,
            "legal": not (identity_errors or contract_errors or label_errors or actual_errors
                          or aromatic_errors or duplicate_axials or non_hex_edges or center_branch_errors or conformation_errors
                          or port_direction_errors or substitution_errors
                          or flex_endpoint_su_errors or ring_attachment_errors),
            "component_count": self.component_count(),
            "aromatic_template_errors": aromatic_errors,
            "edge_count": len(self.edges),
            "substitution_relaxed_edge_count": len(self.substitution_relaxed_edges),
        }


def actual_geometry_errors(state):
    """Audit actual positions, including the SU26/SU30 midpoint geometry."""
    points = {n:state.final_node_positions.get(n,p) for n,p in state.node_axial.items()}
    errors, occupied = [], {}
    midpoint_ids = {a.get('representative_node_id') for a in state.action_history
                    if a.get('kind') == 'substitution' and a.get('ring_size') == 5}
    for n,p in points.items():
        if n not in midpoint_ids and tuple(p) != tuple(state.node_axial[n]):
            errors.append(dict(reason='ordinary_final_position_changed', node=n))
        key = tuple(round(float(v),9) for v in p)
        if key in occupied:
            errors.append(dict(reason='coordinate_collision', nodes=[occupied[key],n]))
        occupied[key] = n
    return errors + _segment_geometry_errors(points, state.edges)


def _segment_geometry_errors(points, edges, changed=None):
    """Explicit bonds may intersect only at their shared graph endpoint."""
    errors = []
    edges = [e for e in edges if e.source_node_id in points and e.target_node_id in points]
    segments=[]
    for e in edges:
        a,b=points[e.source_node_id],points[e.target_node_id]
        segments.append((min(a[0],b[0]),max(a[0],b[0]),min(a[1],b[1]),max(a[1],b[1]),e,a,b))
    segments.sort(key=lambda row:row[0])
    for i,(xmin,xmax,ymin,ymax,e,a,b) in enumerate(segments):
        ids={e.source_node_id,e.target_node_id}
        touched = changed is None or bool(ids & changed)
        # Incremental placement checks compare only changed bonds/points.
        comparisons = segments[i+1:] if changed is None else segments if touched else ()
        for fxmin,fxmax,fymin,fymax,f,c,d in comparisons:
            if fxmin>xmax+1e-9:
                break
            other_ids = {f.source_node_id,f.target_node_id}
            if fxmax<xmin-1e-9 or fymin>ymax+1e-9 or fymax<ymin-1e-9 or ids & other_ids:
                continue
            if changed is not None and other_ids & changed and sorted(other_ids)<sorted(ids):
                continue
            if segments_intersect(a,b,c,d):
                errors.append(dict(reason='edge_intersection',edges=[sorted(ids),[f.source_node_id,f.target_node_id]]))
        scan_points = points.items() if touched else ((n,points[n]) for n in changed)
        for n,p in scan_points:
            if n in ids or not (xmin-1e-9<=p[0]<=xmax+1e-9 and ymin-1e-9<=p[1]<=ymax+1e-9):
                continue
            if segments_intersect(a,b,p,p):
                errors.append(dict(reason='node_on_edge',node=n,edge=sorted(ids)))
    return errors
