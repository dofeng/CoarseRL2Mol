"""Extract molecular fragments and coarse-graining attributes from ReaxFF bonds."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import struct
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from pseudo_species import FragmentProperties, ThreeDimensionalClassifier


DEFAULT_ELEMENTS = {1: "C", 2: "H", 3: "O"}
DEFAULT_MASSES = {"H": 1.008, "C": 12.011, "O": 15.999}
DEFAULT_VALENCES = {"H": 1.0, "C": 4.0, "O": 2.0}
PARSER_VERSION = "3-mol-column"
DESCRIPTOR_VERSION = "2-cycle-rank-topology"


@dataclass(frozen=True, slots=True)
class AtomRecord:
    atom_id: int
    atom_type: int
    element: str
    neighbors: tuple[int, ...]
    bond_orders: tuple[float, ...]
    abo: float
    nlp: float
    charge: float
    molecule_id: int = 0


@dataclass(slots=True)
class FragmentRecord:
    fragment_id: int
    atom_ids: list[int]
    atom_types: dict[str, int]
    element_counts: dict[str, int]
    molecular_mass: float
    ring_count: int
    ring_sizes: list[int]
    radical_sites: int
    radical_atom_ids: list[int]
    valence_deficit: float
    edge_count: int
    node_id: str
    carbon_bin: str
    ring_bin: str
    oxygen_bin: str
    atom_details: list[dict]
    topology_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _cycle_basis(nodes: set[int], adjacency: Mapping[int, set[int]]) -> list[list[int]]:
    """Return an independent graph cycle basis without external dependencies."""
    cycles: list[list[int]] = []
    visited: set[int] = set()
    parent: dict[int, int | None] = {}
    for root in sorted(nodes):
        if root in visited:
            continue
        visited.add(root)
        parent[root] = None
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in sorted(adjacency[current]):
                if neighbor == parent.get(current):
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    stack.append(neighbor)
                    continue
                if neighbor > current:
                    continue
                ancestors: set[int] = set()
                path_current: list[int] = []
                cursor: int | None = current
                while cursor is not None:
                    ancestors.add(cursor)
                    path_current.append(cursor)
                    cursor = parent.get(cursor)
                path_neighbor: list[int] = []
                cursor = neighbor
                while cursor not in ancestors:
                    path_neighbor.append(cursor)
                    cursor = parent.get(cursor)
                    if cursor is None:
                        break
                if cursor is None:
                    continue
                path_neighbor.append(cursor)
                cycle = path_current[: path_current.index(cursor) + 1]
                cycle.extend(reversed(path_neighbor[:-1]))
                if len(cycle) >= 3:
                    cycles.append(cycle)
    return cycles


class ReaxFFBondFile:
    """Parser and fragment extractor for one ReaxFF bonds file."""

    def __init__(
        self,
        path: str | Path,
        element_map: Mapping[int, str] | None = None,
        mass_map: Mapping[str, float] | None = None,
        valence_map: Mapping[str, float] | None = None,
        bond_order_cutoff: float = 0.3,
    ) -> None:
        self.path = Path(path)
        self.element_map = dict(element_map or DEFAULT_ELEMENTS)
        self.mass_map = dict(mass_map or DEFAULT_MASSES)
        if any(not math.isfinite(value) or value <= 0 for value in self.mass_map.values()):
            raise ValueError("Atomic masses must be finite and positive")
        self.valence_map = dict(valence_map or DEFAULT_VALENCES)
        self.bond_order_cutoff = bond_order_cutoff
        if not math.isfinite(bond_order_cutoff) or bond_order_cutoff < 0:
            raise ValueError("Bond order cutoff must be finite and non-negative")
        self.timestep: int | None = None
        self.declared_particle_count: int | None = None
        self.atoms: dict[int, AtomRecord] = {}
        self.edges: dict[tuple[int, int], float] = {}

    def parse(self) -> "ReaxFFBondFile":
        self.atoms.clear()
        self.edges.clear()
        self.timestep = None
        self.declared_particle_count = None
        with self.path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    self._parse_header(line)
                    continue
                fields = line.split()
                if len(fields) < 7:
                    raise ValueError(f"Malformed data line: {line}")
                atom_id, atom_type, neighbor_count = map(int, fields[:3])
                expected = 7 + 2 * neighbor_count
                if neighbor_count < 0 or len(fields) != expected:
                    raise ValueError(
                        f"Atom {atom_id} declares {neighbor_count} neighbors, "
                        f"but line has {len(fields)} fields"
                    )
                start_neighbors = 3
                end_neighbors = start_neighbors + neighbor_count
                neighbors = tuple(map(int, fields[start_neighbors:end_neighbors]))
                if atom_id <= 0 or atom_id in self.atoms:
                    raise ValueError(f"Invalid or duplicate atom ID: {atom_id}")
                if atom_type not in self.element_map:
                    raise ValueError(f"Unknown atom type: {atom_type}")
                if len(set(neighbors)) != neighbor_count or atom_id in neighbors:
                    raise ValueError(f"Duplicate neighbor or self bond for atom {atom_id}")
                molecule_id = int(fields[end_neighbors])
                start_orders = end_neighbors + 1
                bond_orders = tuple(
                    map(float, fields[start_orders : start_orders + neighbor_count])
                )
                abo, nlp, charge = map(
                    float, fields[start_orders + neighbor_count : expected]
                )
                if not all(math.isfinite(value) for value in (*bond_orders, abo, nlp, charge)):
                    raise ValueError(f"Non-finite atom data: {atom_id}")
                if any(order < 0 for order in bond_orders):
                    raise ValueError(f"Negative bond order: {atom_id}")
                element = self.element_map[atom_type]
                if element not in self.mass_map:
                    raise ValueError(f"Missing molecular mass for element: {element}")
                self.atoms[atom_id] = AtomRecord(
                    atom_id, atom_type, element, neighbors, bond_orders, abo, nlp, charge,
                    molecule_id,
                )
                for neighbor, order in zip(neighbors, bond_orders):
                    if order >= self.bond_order_cutoff and neighbor != atom_id:
                        key = tuple(sorted((atom_id, neighbor)))
                        self.edges[key] = max(order, self.edges.get(key, 0.0))
        if self.declared_particle_count is not None and len(self.atoms) != self.declared_particle_count:
            raise ValueError(
                f"Expected {self.declared_particle_count} atoms, parsed {len(self.atoms)}"
            )
        if self.timestep is None or self.declared_particle_count is None:
            raise ValueError("Missing Timestep or Number of particles header")
        for atom in self.atoms.values():
            if any(neighbor not in self.atoms for neighbor in atom.neighbors):
                raise ValueError(f"Unknown neighbor for atom {atom.atom_id}")
        return self

    def _parse_header(self, line: str) -> None:
        stored_cutoff = re.search(r"coarse bond order cutoff\s+([\d.eE+-]+)", line)
        if stored_cutoff and self.bond_order_cutoff < float(stored_cutoff.group(1)):
            raise ValueError("Requested cutoff is below the bond file's stored cutoff; missing bonds cannot be reconstructed")
        fields = line.lstrip("# ").split()
        if len(fields) >= 2 and fields[0] == "Timestep":
            if self.timestep is not None:
                raise ValueError("Multiple timesteps in one file are not supported; split the file first")
            self.timestep = int(fields[1])
        elif len(fields) >= 4 and fields[:3] == ["Number", "of", "particles"]:
            self.declared_particle_count = int(fields[3])

    def _adjacency(self) -> dict[int, set[int]]:
        adjacency = {atom_id: set() for atom_id in self.atoms}
        for left, right in self.edges:
            if left in adjacency and right in adjacency:
                adjacency[left].add(right)
                adjacency[right].add(left)
        return adjacency

    def _components(self, adjacency: Mapping[int, set[int]]) -> list[set[int]]:
        components = []
        unseen = set(adjacency)
        for root in sorted(adjacency):
            if root not in unseen:
                continue
            component = set([root])
            queue = deque([root])
            unseen.remove(root)
            while queue:
                atom_id = queue.popleft()
                for neighbor in adjacency[atom_id]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return sorted(components, key=lambda component: min(component))

    def extract_fragments(
        self,
        classifier: ThreeDimensionalClassifier | None = None,
        include_ring_sizes: bool = True,
        include_atom_details: bool = True,
    ) -> list[FragmentRecord]:
        classifier = classifier or ThreeDimensionalClassifier()
        adjacency = self._adjacency()
        components = self._components(adjacency)
        atom_components = {
            atom_id: index for index, component in enumerate(components) for atom_id in component
        }
        edge_counts = [0] * len(components)
        topology_hashes = [hashlib.sha256() for component in components]
        for left, right in sorted(self.edges):
            index = atom_components[left]
            edge_counts[index] += 1
            topology_hashes[index].update(struct.pack("<qq", left, right))
        records = []
        for fragment_id, atom_ids_set in enumerate(components, start=1):
            atom_ids = sorted(atom_ids_set)
            atoms = [self.atoms[atom_id] for atom_id in atom_ids]
            element_counts = Counter(atom.element for atom in atoms)
            atom_types = Counter(str(atom.atom_type) for atom in atoms) if include_atom_details else {}
            edge_count = edge_counts[fragment_id - 1]
            cycles = (
                _cycle_basis(atom_ids_set, {a: adjacency[a] & atom_ids_set for a in atom_ids_set})
                if include_ring_sizes else []
            )
            ring_count = max(0, edge_count - len(atom_ids_set) + 1)
            valence_deficit = 0.0
            radical_atoms = []
            radical_count = 0
            for atom in atoms:
                target = self.valence_map.get(atom.element)
                if target is None:
                    continue
                deficit = max(0.0, target - atom.abo)
                valence_deficit += deficit
                if deficit >= 0.75:
                    radical_count += 1
                    if include_atom_details:
                        radical_atoms.append(atom.atom_id)
            carbon_count = element_counts.get("C", 0)
            oxygen_count = element_counts.get("O", 0)
            props = FragmentProperties(
                carbon_count=carbon_count,
                ring_count=ring_count,
                oxygen_count=oxygen_count,
                radical_sites=radical_count,
                molecular_mass=sum(self.mass_map[atom.element] for atom in atoms),
            )
            species = classifier.classify(props)
            records.append(FragmentRecord(
                fragment_id=fragment_id,
                atom_ids=atom_ids,
                atom_types=dict(atom_types),
                element_counts=dict(element_counts),
                molecular_mass=props.molecular_mass,
                ring_count=ring_count,
                ring_sizes=sorted(len(cycle) for cycle in cycles),
                radical_sites=radical_count,
                radical_atom_ids=radical_atoms,
                valence_deficit=round(valence_deficit, 6),
                edge_count=edge_count,
                node_id=species.node_id,
                carbon_bin=species.carbon_bin,
                ring_bin=species.ring_bin,
                oxygen_bin=species.oxygen_bin,
                topology_hash=topology_hashes[fragment_id - 1].hexdigest(),
                atom_details=[{
                    "atom_id": atom.atom_id,
                    "atom_type": atom.atom_type,
                    "element": atom.element,
                    "neighbors": list(atom.neighbors),
                    "bond_orders": list(atom.bond_orders),
                    "abo": atom.abo,
                    "nlp": atom.nlp,
                    "charge": atom.charge,
                    "molecule_id": atom.molecule_id,
                } for atom in atoms] if include_atom_details else [],
            ))
        return records


def write_outputs(
    source: ReaxFFBondFile,
    fragments: Iterable[FragmentRecord],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fragments = list(fragments)
    json_path = output_dir / f"{source.path.stem}_fragments.json"
    csv_path = output_dir / f"{source.path.stem}_fragments.csv"
    nodes: dict[str, dict] = {}
    for fragment in fragments:
        node = nodes.setdefault(fragment.node_id, {
            "node_id": fragment.node_id,
            "carbon_bin": fragment.carbon_bin,
            "ring_bin": fragment.ring_bin,
            "oxygen_bin": fragment.oxygen_bin,
            "population": 0,
            "total_mass": 0.0,
            "fragment_ids": [],
            "total_radical_sites": 0,
        })
        node["population"] += 1
        node["total_mass"] += fragment.molecular_mass
        node["fragment_ids"].append(fragment.fragment_id)
        node["total_radical_sites"] += fragment.radical_sites
    for node in nodes.values():
        node["total_mass"] = round(node["total_mass"], 6)
        node["mean_mass"] = round(node["total_mass"] / node["population"], 6)

    payload = {
        "source_file": str(source.path),
        "timestep": source.timestep,
        "declared_particle_count": source.declared_particle_count,
        "parsed_particle_count": len(source.atoms),
        "bond_order_cutoff": source.bond_order_cutoff,
        "element_map": source.element_map,
        "mass_map": source.mass_map,
        "radical_definition": "estimated from max(0, typical_valence - ReaxFF abo); deficit >= 0.75 counts as a radical site",
        "fragment_count": len(fragments),
        "pseudo_species_nodes": sorted(nodes.values(), key=lambda node: node["node_id"]),
        "fragments": [fragment.to_dict() for fragment in fragments],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = list(fragments[0].to_dict()) if fragments else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fragment in fragments:
            row = fragment.to_dict()
            row["atom_ids"] = " ".join(map(str, row["atom_ids"]))
            row["radical_atom_ids"] = " ".join(map(str, row["radical_atom_ids"]))
            row["atom_types"] = json.dumps(row["atom_types"], ensure_ascii=False)
            row["element_counts"] = json.dumps(row["element_counts"], ensure_ascii=False)
            row["ring_sizes"] = " ".join(map(str, row["ring_sizes"]))
            row["atom_details"] = json.dumps(row["atom_details"], ensure_ascii=False)
            writer.writerow(row)
    return json_path, csv_path
