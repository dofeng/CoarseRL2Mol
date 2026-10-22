"""Match ReaxFF fragments between two frames and aggregate node transitions."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pseudo_species import ThreeDimensionalClassifier
from reaxff_fragments import FragmentRecord, ReaxFFBondFile


@dataclass(frozen=True)
class LineageLink:
    parent_fragment_id: int
    child_fragment_id: int
    shared_atom_count: int
    parent_overlap: float
    child_overlap: float


@dataclass
class TransitionEvent:
    event_id: int
    parent_fragment_ids: list[int]
    child_fragment_ids: list[int]
    reactant_nodes: dict[str, int]
    product_nodes: dict[str, int]
    parent_atom_count: int
    child_atom_count: int
    shared_atom_count: int
    reaction_signature: str
    event_type: str
    resolution: str
    confidence: float
    atom_conservation_ratio: float
    mass_error: float
    element_balance: dict[str, int]


def _descriptor(fragment: FragmentRecord) -> str:
    return fragment.node_id


def _node_label(descriptor: str) -> str:
    return descriptor


def _multiset_signature(nodes: dict[str, int]) -> str:
    return " + ".join(
        f"{node}" + (f"*{count}" if count != 1 else "")
        for node, count in sorted(nodes.items())
    ) or "∅"


def _event_type(parent_count: int, child_count: int) -> str:
    if parent_count == 0 and child_count > 0:
        return "birth"
    if parent_count > 0 and child_count == 0:
        return "death"
    if parent_count == 1 and child_count == 1:
        return "transformation"
    if parent_count == 1 and child_count > 1:
        return "fragmentation"
    if parent_count > 1 and child_count == 1:
        return "fusion"
    return "many_to_many"


def _connected_event_groups(
    links: list[LineageLink],
) -> list[tuple[list[int], list[int]]]:
    parents: dict[int, set[int]] = defaultdict(set)
    children: dict[int, set[int]] = defaultdict(set)
    for link in links:
        parents[link.parent_fragment_id].add(link.child_fragment_id)
        children[link.child_fragment_id].add(link.parent_fragment_id)
    unseen_parents = set(parents)
    unseen_children = set(children)
    groups = []
    for identifier in sorted(parents):
        if identifier not in unseen_parents:
            continue
        start = ("p", identifier)
        queue = deque([start])
        group_parents: set[int] = set()
        group_children: set[int] = set()
        while queue:
            kind, identifier = queue.popleft()
            if kind == "p":
                if identifier not in unseen_parents:
                    continue
                unseen_parents.remove(identifier)
                group_parents.add(identifier)
                queue.extend(("c", child) for child in parents[identifier])
            else:
                if identifier not in unseen_children:
                    continue
                unseen_children.remove(identifier)
                group_children.add(identifier)
                queue.extend(("p", parent) for parent in children[identifier])
        groups.append((sorted(group_parents), sorted(group_children)))
    return groups


def match_fragments(
    parents: Iterable[FragmentRecord],
    children: Iterable[FragmentRecord],
    minimum_shared_atoms: int = 1,
    minimum_parent_overlap: float = 0.05,
    minimum_child_overlap: float = 0.05,
    preserve_all_overlaps: bool = True,
) -> list[LineageLink]:
    """Build parent-child links using exact atom-ID intersections."""
    parents = list(parents)
    children = list(children)
    child_map = {child.fragment_id: child for child in children}
    atom_to_children: dict[int, FragmentRecord] = {}
    for child in children:
        for atom_id in child.atom_ids:
            if atom_id in atom_to_children:
                raise ValueError(f"Atom {atom_id} belongs to multiple child fragments")
            atom_to_children[atom_id] = child
    links = []
    seen_parent_atoms = set()
    for parent in parents:
        shared_by_child: Counter[int] = Counter()
        parent_atoms = set(parent.atom_ids)
        if len(parent_atoms) != len(parent.atom_ids) or seen_parent_atoms & parent_atoms:
            raise ValueError("Atoms must belong to exactly one parent fragment")
        seen_parent_atoms.update(parent_atoms)
        for atom_id in parent_atoms:
            child = atom_to_children.get(atom_id)
            if child is not None:
                shared_by_child[child.fragment_id] += 1
        for child_id, shared in shared_by_child.items():
            child = child_map[child_id]
            parent_overlap = shared / len(parent_atoms) if parent_atoms else 0.0
            child_overlap = shared / len(child.atom_ids) if child.atom_ids else 0.0
            if preserve_all_overlaps or (shared >= minimum_shared_atoms and (
                parent_overlap >= minimum_parent_overlap
                or child_overlap >= minimum_child_overlap
            )):
                links.append(LineageLink(
                    parent.fragment_id, child.fragment_id, shared,
                    parent_overlap, child_overlap,
                ))
    return links


def build_transition_events(
    parents: Iterable[FragmentRecord],
    children: Iterable[FragmentRecord],
    links: list[LineageLink],
    minimum_event_conservation: float = 0.90,
) -> list[TransitionEvent]:
    parent_map = {fragment.fragment_id: fragment for fragment in parents}
    child_map = {fragment.fragment_id: fragment for fragment in children}
    groups = _connected_event_groups(links)
    linked_parents = {item for parent_ids, _ in groups for item in parent_ids}
    linked_children = {item for _, child_ids in groups for item in child_ids}
    groups.extend(([parent_id], []) for parent_id in sorted(set(parent_map) - linked_parents))
    groups.extend(([], [child_id]) for child_id in sorted(set(child_map) - linked_children))

    links_by_parent = defaultdict(list)
    for link in links:
        links_by_parent[link.parent_fragment_id].append(link)
    events: list[TransitionEvent] = []
    for event_id, (parent_ids, child_ids) in enumerate(groups, start=1):
        parent_fragments = [parent_map[item] for item in parent_ids]
        child_fragments = [child_map[item] for item in child_ids]
        parent_atoms = set().union(*(set(item.atom_ids) for item in parent_fragments)) if parent_fragments else set()
        child_atoms = set().union(*(set(item.atom_ids) for item in child_fragments)) if child_fragments else set()
        shared_atoms = parent_atoms & child_atoms
        parent_mass = sum(item.molecular_mass for item in parent_fragments)
        child_mass = sum(item.molecular_mass for item in child_fragments)
        denominator = max(len(parent_atoms), len(child_atoms), 1)
        conservation = len(shared_atoms) / denominator
        mass_error = abs(parent_mass - child_mass)

        parent_elements: Counter[str] = Counter()
        child_elements: Counter[str] = Counter()
        for fragment in parent_fragments:
            parent_elements.update(fragment.element_counts)
        for fragment in child_fragments:
            child_elements.update(fragment.element_counts)
        element_balance = {
            element: child_elements[element] - parent_elements[element]
            for element in sorted(set(parent_elements) | set(child_elements))
            if child_elements[element] != parent_elements[element]
        }

        reactants = Counter(_node_label(_descriptor(item)) for item in parent_fragments)
        products = Counter(_node_label(_descriptor(item)) for item in child_fragments)
        event_type = _event_type(len(parent_ids), len(child_ids))
        if (len(parent_ids) == len(child_ids) == 1
                and parent_atoms == child_atoms
                and reactants == products):
            parent_hash = parent_fragments[0].topology_hash
            child_hash = child_fragments[0].topology_hash
            event_type = (
                "internal_rearrangement" if parent_hash and child_hash and parent_hash != child_hash
                else "persistence" if parent_hash and child_hash else "coarse_persistence"
            )

        relevant_links = [link for parent_id in parent_ids for link in links_by_parent[parent_id]]
        link_quality = (
            sum(max(item.parent_overlap, item.child_overlap) for item in relevant_links)
            / len(relevant_links)
            if relevant_links else 0.0
        )
        confidence = conservation if event_type == "persistence" else (conservation + link_quality) / 2
        if event_type in {"birth", "death"}:
            resolution = "unresolved"
        elif event_type == "many_to_many":
            resolution = "composite"
        elif conservation >= minimum_event_conservation:
            resolution = "resolved"
        else:
            resolution = "composite"

        events.append(TransitionEvent(
            event_id=event_id,
            parent_fragment_ids=parent_ids,
            child_fragment_ids=child_ids,
            reactant_nodes=dict(reactants),
            product_nodes=dict(products),
            parent_atom_count=len(parent_atoms),
            child_atom_count=len(child_atoms),
            shared_atom_count=len(shared_atoms),
            reaction_signature=f"{_multiset_signature(reactants)} -> {_multiset_signature(products)}",
            event_type=event_type,
            resolution=resolution,
            confidence=round(confidence, 6),
            atom_conservation_ratio=round(conservation, 6),
            mass_error=round(mass_error, 6),
            element_balance=element_balance,
        ))
    return events


def aggregate_transitions(events: Iterable[TransitionEvent]) -> list[dict]:
    groups: dict[str, dict] = {}
    for event in events:
        if event.event_type in {"persistence", "coarse_persistence"}:
            continue
        group = groups.setdefault(event.reaction_signature, {
            "reaction_signature": event.reaction_signature,
            "event_type": event.event_type,
            "event_type_counts": {},
            "event_count": 0,
            "event_ids": [],
            "mean_confidence": 0.0,
            "mean_parent_atom_count": 0.0,
            "mean_child_atom_count": 0.0,
            "mean_atom_conservation_ratio": 0.0,
            "mean_mass_error": 0.0,
            "resolution_counts": {},
            "reactant_nodes": event.reactant_nodes,
            "product_nodes": event.product_nodes,
        })
        group["event_count"] += 1
        group["event_type_counts"][event.event_type] = group["event_type_counts"].get(event.event_type, 0) + 1
        group["event_ids"].append(event.event_id)
        group["mean_confidence"] += event.confidence
        group["mean_parent_atom_count"] += event.parent_atom_count
        group["mean_child_atom_count"] += event.child_atom_count
        group["mean_atom_conservation_ratio"] += event.atom_conservation_ratio
        group["mean_mass_error"] += event.mass_error
        group["resolution_counts"][event.resolution] = group["resolution_counts"].get(event.resolution, 0) + 1
    for group in groups.values():
        if len(group["event_type_counts"]) > 1:
            group["event_type"] = "mixed"
        count = group["event_count"]
        group["mean_confidence"] = round(group["mean_confidence"] / count, 6)
        group["mean_parent_atom_count"] = round(group["mean_parent_atom_count"] / count, 3)
        group["mean_child_atom_count"] = round(group["mean_child_atom_count"] / count, 3)
        group["mean_atom_conservation_ratio"] = round(group["mean_atom_conservation_ratio"] / count, 6)
        group["mean_mass_error"] = round(group["mean_mass_error"] / count, 6)
    return sorted(groups.values(), key=lambda item: (-item["event_count"], item["reaction_signature"]))


def aggregate_node_edges(events: Iterable[TransitionEvent]) -> list[dict]:
    """Project reaction hyperedges onto directed node-to-node relationships."""
    edges: dict[tuple[str, str], dict] = {}
    for event in events:
        if event.event_type in {"persistence", "coarse_persistence"}:
            continue
        for reactant in event.reactant_nodes:
            for product in event.product_nodes:
                key = (reactant, product)
                edge = edges.setdefault(key, {
                    "source": reactant,
                    "target": product,
                    "event_count": 0,
                    "event_ids": [],
                })
                edge["event_count"] += 1
                edge["event_ids"].append(event.event_id)
    return sorted(edges.values(), key=lambda item: (-item["event_count"], item["source"], item["target"]))


def compare_bonds_files(
    parent_path: str | Path,
    child_path: str | Path,
    cutoff: float = 0.3,
    element_map: dict[int, str] | None = None,
    minimum_shared_atoms: int = 1,
    minimum_parent_overlap: float = 0.05,
    minimum_child_overlap: float = 0.05,
    minimum_event_conservation: float = 0.90,
) -> dict:
    classifier = ThreeDimensionalClassifier()
    parent_source = ReaxFFBondFile(parent_path, element_map=element_map, bond_order_cutoff=cutoff).parse()
    child_source = ReaxFFBondFile(child_path, element_map=element_map, bond_order_cutoff=cutoff).parse()
    parent_fragments = parent_source.extract_fragments(classifier)
    child_fragments = child_source.extract_fragments(classifier)
    links = match_fragments(parent_fragments, child_fragments, minimum_shared_atoms, minimum_parent_overlap, minimum_child_overlap)
    events = build_transition_events(
        parent_fragments, child_fragments, links, minimum_event_conservation,
    )
    return {
        "parent_file": str(parent_path),
        "child_file": str(child_path),
        "parent_timestep": parent_source.timestep,
        "child_timestep": child_source.timestep,
        "bond_order_cutoff": cutoff,
        "matching": {
            "minimum_shared_atoms": minimum_shared_atoms,
            "minimum_parent_overlap": minimum_parent_overlap,
            "minimum_child_overlap": minimum_child_overlap,
            "minimum_event_conservation": minimum_event_conservation,
        },
        "parent_fragment_count": len(parent_fragments),
        "child_fragment_count": len(child_fragments),
        "lineage_link_count": len(links),
        "event_count": len(events),
        "links": [asdict(link) for link in links],
        "events": [asdict(event) for event in events],
        "aggregated_reactions": aggregate_transitions(events),
        "node_edges": aggregate_node_edges(events),
    }


def write_transition_outputs(result: dict, output_dir: str | Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bond{result['parent_timestep']}_to_bond{result['child_timestep']}"
    json_path = output_dir / f"{stem}_transitions.json"
    csv_path = output_dir / f"{stem}_reactions.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "reaction_signature", "event_type", "event_type_counts", "event_count", "mean_confidence",
        "mean_parent_atom_count", "mean_child_atom_count",
        "mean_atom_conservation_ratio", "mean_mass_error",
        "resolution_counts", "event_ids",
        "reactant_nodes", "product_nodes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for reaction in result["aggregated_reactions"]:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in reaction.items()})
    edge_path = output_dir / f"{stem}_node_edges.csv"
    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target", "event_count", "event_ids"])
        writer.writeheader()
        writer.writerows(result["node_edges"])
    return json_path, csv_path, edge_path
