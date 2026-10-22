"""Scalable multi-frame pipeline for coarse ReaxFF reaction networks."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import shutil
import signal
import sqlite3
import struct
import sys
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from fragment_lineage import build_transition_events, match_fragments
from network_postprocess import (
    annotate_events,
    build_fragment_tracks,
    build_network_views,
    ensure_schema as ensure_postprocess_schema,
    export_network,
    invalidate_tracks,
    write_csv,
)
from reaxff_fragments import FragmentRecord, ReaxFFBondFile
from reaxff_fragments import DEFAULT_MASSES, DESCRIPTOR_VERSION, PARSER_VERSION
from pseudo_species import ThreeDimensionalClassifier
from pipeline_runtime import Progress, atomic_json, bounded_results, database_lock
from coarse_network.build import build_analytics, ensure_schema as ensure_coarse_schema, invalidate_pair
from coarse_network.export import export_analytics


SCHEMA_VERSION = "3"


def _parse_element_map(value: str) -> dict[int, str]:
    return {
        int(item.split(":", 1)[0]): item.split(":", 1)[1]
        for item in value.split(",")
    }


def _read_timestep(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("# Timestep"):
                return int(line.lstrip("# ").split()[1])
            if line and not line.startswith("#"):
                break
    raise ValueError(f"No '# Timestep' header found in {path}")


def _filename_frame_number(path: Path) -> int | None:
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def discover_frames(
    inputs: Iterable[str],
    pattern: str,
    frame_order: str = "filename",
) -> list[dict]:
    paths: set[Path] = set()
    for value in inputs:
        if any(character in value for character in "*?["):
            paths.update(Path(item).resolve() for item in glob.glob(value, recursive=True))
            continue
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            paths.update(item.resolve() for item in path.rglob(pattern))
        elif path.is_file():
            paths.add(path)
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise ValueError("No ReaxFF bonds files were discovered")
    indexed = [
        {
            "source_path": str(path),
            "timestep": _read_timestep(path),
            "filename_frame_number": _filename_frame_number(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]
    if frame_order == "timestep":
        indexed.sort(key=lambda item: (item["timestep"], item["source_path"]))
    elif frame_order == "filename":
        numbers = [item["filename_frame_number"] for item in indexed if item["filename_frame_number"] is not None]
        if len(set(numbers)) != len(numbers):
            raise ValueError("Duplicate filename frame numbers; do not combine independent trajectories")
        indexed.sort(key=lambda item: (
            item["filename_frame_number"] is None,
            item["filename_frame_number"] if item["filename_frame_number"] is not None else 0,
            item["source_path"],
        ))
    else:
        raise ValueError(f"Unsupported frame order: {frame_order}")
    for frame_id, item in enumerate(indexed):
        item["frame_id"] = frame_id
    return indexed


def _configuration_hash(
    cutoff: float,
    element_map: dict[int, str],
    matching: dict,
    classification: dict | None = None,
    mass_map: dict | None = None,
) -> str:
    classifier = ThreeDimensionalClassifier(**(classification or {}))
    payload = json.dumps({
        "cutoff": cutoff,
        "element_map": element_map,
        "matching": matching,
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "descriptor_version": DESCRIPTOR_VERSION,
        "classification": {"carbon_bins": classifier.carbon_bins, "ring_bins": classifier.ring_bins,
                           "oxygen_bins": classifier.oxygen_bins},
        "masses": mass_map or DEFAULT_MASSES,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_hash(frames: list[dict]) -> str:
    payload = json.dumps(
        [(item["frame_id"], item["timestep"], item["source_path"], item.get("size_bytes"),
          item.get("mtime_ns"), item.get("sha256")) for item in frames],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode_atom_ids(atom_ids: list[int]) -> bytes:
    packed = struct.pack(f"<{len(atom_ids)}q", *atom_ids) if atom_ids else b""
    return zlib.compress(packed, level=6)


def _decode_atom_ids(payload: bytes, atom_count: int) -> list[int]:
    if atom_count == 0:
        return []
    unpacked = zlib.decompress(payload)
    return list(struct.unpack(f"<{atom_count}q", unpacked))


def _connect_database(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=60)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='metadata'").fetchone():
        version = _get_metadata(connection, "schema_version")
        if version and version != SCHEMA_VERSION:
            raise ValueError("Legacy database uses the incorrect bond parser. Keep it as an archive "
                             "and choose a NEW database path; automatic reuse is unsafe.")
    connection.executescript("""
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS frames (
        frame_id INTEGER PRIMARY KEY,
        timestep INTEGER NOT NULL,
        source_path TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL DEFAULT 'pending',
        atom_count INTEGER,
        edge_count INTEGER,
        fragment_count INTEGER,
        duration_seconds REAL,
        error TEXT,
        atom_signature TEXT
    );
    CREATE TABLE IF NOT EXISTS species (
        species_id TEXT PRIMARY KEY,
        carbon_bin TEXT NOT NULL,
        ring_bin TEXT NOT NULL,
        oxygen_bin TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS fragments (
        frame_id INTEGER NOT NULL,
        fragment_id INTEGER NOT NULL,
        species_id TEXT NOT NULL,
        carbon_bin TEXT NOT NULL,
        ring_bin TEXT NOT NULL,
        oxygen_bin TEXT NOT NULL,
        atom_count INTEGER NOT NULL,
        atom_ids BLOB NOT NULL,
        carbon_count INTEGER NOT NULL,
        hydrogen_count INTEGER NOT NULL,
        oxygen_count INTEGER NOT NULL,
        other_elements_json TEXT NOT NULL,
        molecular_mass REAL NOT NULL,
        ring_count INTEGER NOT NULL,
        radical_sites INTEGER NOT NULL,
        valence_deficit REAL NOT NULL,
        edge_count INTEGER NOT NULL,
        topology_hash TEXT NOT NULL,
        PRIMARY KEY (frame_id, fragment_id),
        FOREIGN KEY (frame_id) REFERENCES frames(frame_id),
        FOREIGN KEY (species_id) REFERENCES species(species_id)
    );
    CREATE INDEX IF NOT EXISTS idx_fragments_species ON fragments(species_id);
    CREATE TABLE IF NOT EXISTS species_states (
        frame_id INTEGER NOT NULL,
        species_id TEXT NOT NULL,
        population INTEGER NOT NULL,
        total_mass REAL NOT NULL,
        mean_mass REAL NOT NULL,
        total_radical_sites INTEGER NOT NULL,
        total_carbon INTEGER NOT NULL,
        total_hydrogen INTEGER NOT NULL,
        total_oxygen INTEGER NOT NULL,
        PRIMARY KEY (frame_id, species_id)
    );
    CREATE TABLE IF NOT EXISTS frame_pairs (
        parent_frame_id INTEGER NOT NULL,
        child_frame_id INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        link_count INTEGER,
        event_count INTEGER,
        duration_seconds REAL,
        error TEXT,
        PRIMARY KEY (parent_frame_id, child_frame_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pairs_child ON frame_pairs(child_frame_id, status);
    CREATE INDEX IF NOT EXISTS idx_states_species_frame ON species_states(species_id, frame_id);
    CREATE TABLE IF NOT EXISTS lineage_links (
        parent_frame_id INTEGER NOT NULL,
        child_frame_id INTEGER NOT NULL,
        parent_fragment_id INTEGER NOT NULL,
        child_fragment_id INTEGER NOT NULL,
        shared_atom_count INTEGER NOT NULL,
        parent_overlap REAL NOT NULL,
        child_overlap REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_lineage_pair
        ON lineage_links(parent_frame_id, child_frame_id);
    CREATE TABLE IF NOT EXISTS events (
        parent_frame_id INTEGER NOT NULL,
        child_frame_id INTEGER NOT NULL,
        event_id INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        resolution TEXT NOT NULL,
        reaction_signature TEXT NOT NULL,
        reactant_nodes_json TEXT NOT NULL,
        product_nodes_json TEXT NOT NULL,
        parent_fragment_ids_json TEXT NOT NULL,
        child_fragment_ids_json TEXT NOT NULL,
        parent_atom_count INTEGER NOT NULL,
        child_atom_count INTEGER NOT NULL,
        shared_atom_count INTEGER NOT NULL,
        confidence REAL NOT NULL,
        atom_conservation_ratio REAL NOT NULL,
        mass_error REAL NOT NULL,
        element_balance_json TEXT NOT NULL,
        PRIMARY KEY (parent_frame_id, child_frame_id, event_id)
    );
    DROP INDEX IF EXISTS idx_events_signature;
    CREATE INDEX IF NOT EXISTS idx_events_reaction_signature
        ON events(reaction_signature,event_type,resolution,parent_frame_id)
        WHERE event_type NOT IN ('persistence','coarse_persistence');
    CREATE INDEX IF NOT EXISTS idx_events_reaction_frame
        ON events(parent_frame_id,child_frame_id,event_id)
        WHERE event_type NOT IN ('persistence','coarse_persistence');
    CREATE TABLE IF NOT EXISTS event_members (
        parent_frame_id INTEGER NOT NULL,
        child_frame_id INTEGER NOT NULL,
        event_id INTEGER NOT NULL,
        side TEXT NOT NULL,
        species_id TEXT NOT NULL,
        coefficient INTEGER NOT NULL,
        fragment_ids_json TEXT NOT NULL,
        PRIMARY KEY (parent_frame_id, child_frame_id, event_id, side, species_id)
    );
    CREATE TABLE IF NOT EXISTS reaction_channels (
        reaction_signature TEXT PRIMARY KEY,
        event_type TEXT NOT NULL,
        event_count INTEGER NOT NULL,
        first_timestep INTEGER NOT NULL,
        last_timestep INTEGER NOT NULL,
        mean_confidence REAL NOT NULL,
        mean_atom_conservation_ratio REAL NOT NULL,
        mean_mass_error REAL NOT NULL,
        resolution_counts_json TEXT NOT NULL,
        reactant_nodes_json TEXT NOT NULL,
        product_nodes_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS reaction_stoichiometry (
        reaction_signature TEXT NOT NULL,
        side TEXT NOT NULL,
        species_id TEXT NOT NULL,
        coefficient INTEGER NOT NULL,
        PRIMARY KEY (reaction_signature, side, species_id)
    );
    """)
    ensure_coarse_schema(connection)
    connection.commit()


def _set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _get_metadata(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def _initialize_manifest(
    connection: sqlite3.Connection,
    frames: list[dict],
    config_hash: str,
) -> None:
    existing_config = _get_metadata(connection, "configuration_hash")
    existing_manifest = _get_metadata(connection, "manifest_hash")
    current_manifest = _manifest_hash(frames)
    if existing_config and existing_config != config_hash:
        raise ValueError("Existing database was created with a different analysis configuration")
    if existing_manifest and existing_manifest != current_manifest:
        raise ValueError("Existing database contains a different input frame manifest")
    _set_metadata(connection, "schema_version", SCHEMA_VERSION)
    _set_metadata(connection, "configuration_hash", config_hash)
    _set_metadata(connection, "manifest_hash", current_manifest)
    _set_metadata(connection, "frame_count", str(len(frames)))
    for item in frames:
        connection.execute(
            "INSERT OR IGNORE INTO frames(frame_id, timestep, source_path) VALUES (?, ?, ?)",
            (item["frame_id"], item["timestep"], item["source_path"]),
        )
    connection.execute("UPDATE frames SET status='pending' WHERE status='processing'")
    connection.execute("UPDATE frame_pairs SET status='pending' WHERE status='processing'")
    connection.commit()


def _process_frame_worker(task: dict) -> dict:
    started = time.monotonic()
    source = ReaxFFBondFile(
        task["source_path"],
        element_map=task["element_map"],
        mass_map=task.get("mass_map"),
        bond_order_cutoff=task["cutoff"],
    ).parse()
    fragments = source.extract_fragments(
        classifier=ThreeDimensionalClassifier(**task.get("classification", {})),
        include_ring_sizes=False,
        include_atom_details=False,
    )
    compact = []
    for fragment in fragments:
        other_elements = {
            key: value for key, value in fragment.element_counts.items()
            if key not in {"C", "H", "O"}
        }
        compact.append({
            "fragment_id": fragment.fragment_id,
            "species_id": fragment.node_id,
            "carbon_bin": fragment.carbon_bin,
            "ring_bin": fragment.ring_bin,
            "oxygen_bin": fragment.oxygen_bin,
            "atom_count": len(fragment.atom_ids),
            "atom_ids": _encode_atom_ids(fragment.atom_ids),
            "carbon_count": fragment.element_counts.get("C", 0),
            "hydrogen_count": fragment.element_counts.get("H", 0),
            "oxygen_count": fragment.element_counts.get("O", 0),
            "other_elements": other_elements,
            "molecular_mass": fragment.molecular_mass,
            "ring_count": fragment.ring_count,
            "radical_sites": fragment.radical_sites,
            "valence_deficit": fragment.valence_deficit,
            "edge_count": fragment.edge_count,
            "topology_hash": fragment.topology_hash,
        })
    return {
        "frame_id": task["frame_id"],
        "timestep": source.timestep,
        "atom_count": len(source.atoms),
        "edge_count": len(source.edges),
        "fragment_count": len(fragments),
        "duration_seconds": time.monotonic() - started,
        "fragments": compact,
        "atom_signature": hashlib.sha256(json.dumps(
            [(atom_id, source.atoms[atom_id].element) for atom_id in sorted(source.atoms)]
        ).encode()).hexdigest(),
    }


def _store_frame_result(connection: sqlite3.Connection, result: dict) -> None:
    frame_id = result["frame_id"]
    invalidate_tracks(connection, frame_id)
    for pair in connection.execute("SELECT parent_frame_id,child_frame_id FROM frame_pairs WHERE parent_frame_id=? OR child_frame_id=?", (frame_id, frame_id)).fetchall():
        invalidate_pair(connection, *pair)
    connection.execute("UPDATE frame_pairs SET status='pending' WHERE parent_frame_id=? OR child_frame_id=?", (frame_id, frame_id))
    _set_metadata(connection, "views_ready", "false")
    _set_metadata(connection, "coarse_ready", "false")
    connection.execute("DELETE FROM species_states WHERE frame_id=?", (frame_id,))
    connection.execute("DELETE FROM fragments WHERE frame_id=?", (frame_id,))
    states: dict[str, dict] = {}
    for fragment in result["fragments"]:
        connection.execute(
            "INSERT OR IGNORE INTO species VALUES (?, ?, ?, ?)",
            (fragment["species_id"], fragment["carbon_bin"], fragment["ring_bin"], fragment["oxygen_bin"]),
        )
        atom_ids = fragment["atom_ids"]
        connection.execute("""
            INSERT INTO fragments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            frame_id, fragment["fragment_id"], fragment["species_id"],
            fragment["carbon_bin"], fragment["ring_bin"], fragment["oxygen_bin"],
            fragment["atom_count"], atom_ids, fragment["carbon_count"],
            fragment["hydrogen_count"], fragment["oxygen_count"],
            json.dumps(fragment["other_elements"], sort_keys=True),
            fragment["molecular_mass"], fragment["ring_count"], fragment["radical_sites"],
            fragment["valence_deficit"], fragment["edge_count"], fragment["topology_hash"],
        ))
        state = states.setdefault(fragment["species_id"], {
            "population": 0, "total_mass": 0.0, "total_radical_sites": 0,
            "total_carbon": 0, "total_hydrogen": 0, "total_oxygen": 0,
        })
        state["population"] += 1
        state["total_mass"] += fragment["molecular_mass"]
        state["total_radical_sites"] += fragment["radical_sites"]
        state["total_carbon"] += fragment["carbon_count"]
        state["total_hydrogen"] += fragment["hydrogen_count"]
        state["total_oxygen"] += fragment["oxygen_count"]
    for species_id, state in states.items():
        connection.execute(
            "INSERT INTO species_states VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (frame_id, species_id, state["population"], state["total_mass"],
             state["total_mass"] / state["population"], state["total_radical_sites"],
             state["total_carbon"], state["total_hydrogen"], state["total_oxygen"]),
        )
    connection.execute("""
        UPDATE frames SET status='completed', atom_count=?, edge_count=?, fragment_count=?,
            duration_seconds=?, atom_signature=?, error=NULL WHERE frame_id=?
    """, (result["atom_count"], result["edge_count"], result["fragment_count"],
          result["duration_seconds"], result["atom_signature"], frame_id))


def _load_fragments(connection: sqlite3.Connection, frame_id: int) -> list[FragmentRecord]:
    rows = connection.execute(
        "SELECT * FROM fragments WHERE frame_id=? ORDER BY fragment_id", (frame_id,),
    )
    fragments = []
    for row in rows:
        element_counts = {
            "C": row["carbon_count"],
            "H": row["hydrogen_count"],
            "O": row["oxygen_count"],
        }
        element_counts.update(json.loads(row["other_elements_json"]))
        element_counts = {key: value for key, value in element_counts.items() if value}
        fragments.append(FragmentRecord(
            fragment_id=row["fragment_id"],
            atom_ids=_decode_atom_ids(row["atom_ids"], row["atom_count"]),
            atom_types={},
            element_counts=element_counts,
            molecular_mass=row["molecular_mass"],
            ring_count=row["ring_count"],
            ring_sizes=[],
            radical_sites=row["radical_sites"],
            radical_atom_ids=[],
            valence_deficit=row["valence_deficit"],
            edge_count=row["edge_count"],
            node_id=row["species_id"],
            carbon_bin=row["carbon_bin"],
            ring_bin=row["ring_bin"],
            oxygen_bin=row["oxygen_bin"],
            atom_details=[],
            topology_hash=row["topology_hash"],
        ))
    return fragments


def _process_pair_worker(task: dict) -> dict:
    started = time.monotonic()
    connection = _connect_database(Path(task["database"]), readonly=True)
    try:
        parents = _load_fragments(connection, task["parent_frame_id"])
        children = _load_fragments(connection, task["child_frame_id"])
    finally:
        connection.close()
    links = match_fragments(
        parents, children,
        task["minimum_shared_atoms"],
        task["minimum_parent_overlap"],
        task["minimum_child_overlap"],
    )
    events = build_transition_events(
        parents, children, links, task["minimum_event_conservation"],
    )
    parent_map = {item.fragment_id: item for item in parents}
    child_map = {item.fragment_id: item for item in children}
    members = []
    for event in events:
        for side, fragment_ids, fragment_map in (
            ("reactant", event.parent_fragment_ids, parent_map),
            ("product", event.child_fragment_ids, child_map),
        ):
            grouped: dict[str, list[int]] = {}
            for fragment_id in fragment_ids:
                grouped.setdefault(fragment_map[fragment_id].node_id, []).append(fragment_id)
            for species_id, identifiers in grouped.items():
                members.append({
                    "event_id": event.event_id,
                    "side": side,
                    "species_id": species_id,
                    "coefficient": len(identifiers),
                    "fragment_ids": identifiers,
                })
    return {
        "parent_frame_id": task["parent_frame_id"],
        "child_frame_id": task["child_frame_id"],
        "links": [asdict(item) for item in links],
        "events": [asdict(item) for item in events],
        "members": members,
        "duration_seconds": time.monotonic() - started,
    }


def _store_pair_result(connection: sqlite3.Connection, result: dict) -> None:
    parent_frame = result["parent_frame_id"]
    child_frame = result["child_frame_id"]
    pair = (parent_frame, child_frame)
    invalidate_tracks(connection, child_frame)
    invalidate_pair(connection, *pair)
    connection.execute(
        "DELETE FROM lineage_links WHERE parent_frame_id=? AND child_frame_id=?", pair,
    )
    connection.execute(
        "DELETE FROM event_members WHERE parent_frame_id=? AND child_frame_id=?", pair,
    )
    connection.execute(
        "DELETE FROM events WHERE parent_frame_id=? AND child_frame_id=?", pair,
    )
    for link in result["links"]:
        connection.execute(
            "INSERT INTO lineage_links VALUES (?, ?, ?, ?, ?, ?, ?)",
            pair + (link["parent_fragment_id"], link["child_fragment_id"],
                    link["shared_atom_count"], link["parent_overlap"], link["child_overlap"]),
        )
    for event in result["events"]:
        connection.execute("""
            INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, pair + (
            event["event_id"], event["event_type"], event["resolution"],
            event["reaction_signature"], json.dumps(event["reactant_nodes"], sort_keys=True),
            json.dumps(event["product_nodes"], sort_keys=True),
            json.dumps(event["parent_fragment_ids"]), json.dumps(event["child_fragment_ids"]),
            event["parent_atom_count"], event["child_atom_count"], event["shared_atom_count"],
            event["confidence"], event["atom_conservation_ratio"], event["mass_error"],
            json.dumps(event["element_balance"], sort_keys=True),
        ))
    for member in result["members"]:
        connection.execute(
            "INSERT INTO event_members VALUES (?, ?, ?, ?, ?, ?, ?)",
            pair + (member["event_id"], member["side"], member["species_id"],
                    member["coefficient"], json.dumps(member["fragment_ids"])),
        )
    connection.execute("""
        UPDATE frame_pairs SET status='completed', link_count=?, event_count=?,
            duration_seconds=?, error=NULL WHERE parent_frame_id=? AND child_frame_id=?
    """, (len(result["links"]), len(result["events"]), result["duration_seconds"],
          parent_frame, child_frame))


def _run_frame_stage(
    connection: sqlite3.Connection,
    workers: int,
    cutoff: float,
    element_map: dict[int, str],
    retry_failed: bool,
    max_inflight: int | None = None,
    memory_limit_mib: int = 3072,
    progress: Progress | None = None,
    classification: dict | None = None,
    mass_map: dict | None = None,
) -> None:
    statuses = ["pending"] + (["failed"] if retry_failed else [])
    placeholders = ",".join("?" for _ in statuses)
    rows = connection.execute(
        f"SELECT frame_id, source_path FROM frames WHERE status IN ({placeholders}) ORDER BY frame_id",
        statuses,
    ).fetchall()
    if not rows:
        print("Frame stage: nothing to process")
        return
    tasks = [{
        "frame_id": row["frame_id"], "source_path": row["source_path"],
        "cutoff": cutoff, "element_map": element_map,
    } for row in rows]
    completed = 0
    for task in tasks:
        task.update(classification=classification or {}, mass_map=mass_map)
    def pending_tasks():
        for task in tasks:
            with connection:
                connection.execute("UPDATE frames SET status='processing', error=NULL WHERE frame_id=?",
                                   (task["frame_id"],))
            yield task
    for task, future in bounded_results(_process_frame_worker, pending_tasks(), workers,
                                         max_inflight or workers * 2, memory_limit_mib):
        try:
            result = future.result()
            with connection:
                _store_frame_result(connection, result)
            del result
            completed += 1
        except Exception as exc:
            with connection:
                connection.execute("UPDATE frames SET status='failed', error=? WHERE frame_id=?",
                                   (str(exc), task["frame_id"]))
            print(f"Frame stage: frame {task['frame_id']} failed: {exc}", flush=True)
        del future
        if progress:
            progress.update(stage="frames", completed=completed, total=len(tasks))
        print(f"Frame stage: {completed}/{len(tasks)}", flush=True)


def _prepare_pairs(connection: sqlite3.Connection) -> None:
    frames = connection.execute(
        "SELECT frame_id, status FROM frames ORDER BY frame_id",
    ).fetchall()
    for parent, child in zip(frames, frames[1:]):
        if child["frame_id"] != parent["frame_id"] + 1:
            continue
        connection.execute(
            "INSERT OR IGNORE INTO frame_pairs(parent_frame_id, child_frame_id, status) VALUES (?, ?, ?)",
            (parent["frame_id"], child["frame_id"],
             "pending" if parent["status"] == child["status"] == "completed" else "blocked"),
        )
    connection.execute("""UPDATE frame_pairs SET status='pending' WHERE status='blocked'
        AND parent_frame_id IN (SELECT frame_id FROM frames WHERE status='completed')
        AND child_frame_id IN (SELECT frame_id FROM frames WHERE status='completed')""")
    connection.commit()


def _run_pair_stage(
    connection: sqlite3.Connection,
    database: Path,
    workers: int,
    matching: dict,
    retry_failed: bool,
    max_inflight: int | None = None,
    memory_limit_mib: int = 3072,
    progress: Progress | None = None,
) -> None:
    _prepare_pairs(connection)
    statuses = ["pending"] + (["failed"] if retry_failed else [])
    placeholders = ",".join("?" for _ in statuses)
    rows = connection.execute(
        f"SELECT parent_frame_id, child_frame_id FROM frame_pairs "
        f"WHERE status IN ({placeholders}) ORDER BY parent_frame_id",
        statuses,
    ).fetchall()
    if not rows:
        print("Transition stage: nothing to process")
        return
    tasks = [{
        "database": str(database),
        "parent_frame_id": row["parent_frame_id"],
        "child_frame_id": row["child_frame_id"],
        **matching,
    } for row in rows]
    completed = 0
    def pending_tasks():
        for task in tasks:
            with connection:
                connection.execute("UPDATE frame_pairs SET status='processing', error=NULL "
                                   "WHERE parent_frame_id=? AND child_frame_id=?",
                                   (task["parent_frame_id"], task["child_frame_id"]))
            yield task
    for task, future in bounded_results(_process_pair_worker, pending_tasks(), workers,
                                         max_inflight or workers * 2, memory_limit_mib):
        pair = (task["parent_frame_id"], task["child_frame_id"])
        try:
            result = future.result()
            with connection:
                _store_pair_result(connection, result)
            del result
            completed += 1
        except Exception as exc:
            with connection:
                connection.execute("UPDATE frame_pairs SET status='failed', error=? "
                                   "WHERE parent_frame_id=? AND child_frame_id=?", (str(exc), *pair))
            print(f"Transition stage: pair {pair[0]}->{pair[1]} failed: {exc}", flush=True)
        del future
        if progress:
            progress.update(stage="transitions", completed=completed, total=len(tasks))
        print(f"Transition stage: {completed}/{len(tasks)}", flush=True)


def _rebuild_reaction_channels(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute("DELETE FROM reaction_stoichiometry")
        connection.execute("DELETE FROM reaction_channels")
        connection.execute("""INSERT INTO reaction_channels SELECT reaction_signature,event_type,event_count,
            first_timestep,last_timestep,mean_confidence,mean_atom_conservation_ratio,mean_mass_error,
            resolution_counts_json,reactant_nodes_json,product_nodes_json
            FROM network_view_channels WHERE view_name='raw'""")
        for row in connection.execute("SELECT reaction_signature,reactant_nodes_json,product_nodes_json FROM reaction_channels"):
            for side, field in (("reactant", "reactant_nodes_json"), ("product", "product_nodes_json")):
                connection.executemany("INSERT INTO reaction_stoichiometry VALUES(?,?,?,?)",
                                       ((row["reaction_signature"], side, species, amount)
                                        for species, amount in json.loads(row[field]).items()))


def _write_exports(connection: sqlite3.Connection, output_dir: Path, export_states: bool = False,
                   export_views: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = {}
    queries = {
        "species": "SELECT * FROM species ORDER BY species_id",
        "reaction_channels": "SELECT * FROM reaction_channels ORDER BY event_count DESC, reaction_signature",
    }
    if export_states:
        queries["species_states"] = "SELECT species_states.*, frames.timestep FROM species_states JOIN frames USING(frame_id) ORDER BY frame_id, species_id"
    for name, query in queries.items():
        path = output_dir / f"{name}.csv"
        write_csv(connection, query, path)
        exports[name] = str(path)
    if export_views:
        exports["network_views"] = export_network(connection, output_dir / "network_views")
    summary = {
        "frames": connection.execute("SELECT COUNT(*) FROM frames WHERE status='completed'").fetchone()[0],
        "failed_frames": connection.execute("SELECT COUNT(*) FROM frames WHERE status='failed'").fetchone()[0],
        "frame_pairs": connection.execute("SELECT COUNT(*) FROM frame_pairs WHERE status='completed'").fetchone()[0],
        "failed_pairs": connection.execute("SELECT COUNT(*) FROM frame_pairs WHERE status='failed'").fetchone()[0],
        "species": connection.execute("SELECT COUNT(*) FROM species").fetchone()[0],
        "fragments": connection.execute("SELECT COALESCE(SUM(fragment_count),0) FROM frames WHERE status='completed'").fetchone()[0],
        "events": connection.execute("SELECT COALESCE(SUM(event_count),0) FROM frame_pairs WHERE status='completed'").fetchone()[0],
        "reaction_channels": connection.execute("SELECT COUNT(*) FROM reaction_channels").fetchone()[0],
        "exports": exports,
    }
    summary_path = output_dir / "network_summary.json"
    atomic_json(summary_path, summary)
    summary["summary"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/network.json")
    parser.add_argument("inputs", nargs="+", help="Input files, directories, or quoted glob patterns")
    parser.add_argument("--pattern", default="*.reaxff")
    parser.add_argument(
        "--frame-order", choices=("filename", "timestep"), default="filename",
        help="filename orders bond0, bond1, ... by frame number; timestep uses the file header",
    )
    parser.add_argument("--database", type=Path, default=Path("results/reaction_network_v3/network.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pair-workers", type=int, default=None)
    parser.add_argument("--max-inflight", type=int, default=None)
    parser.add_argument("--memory-limit-mib", type=int, default=3072)
    parser.add_argument("--minimum-free-disk-mib", type=int, default=512)
    parser.add_argument("--expected-frame-count", type=int)
    parser.add_argument("--expected-frame-step", type=int)
    parser.add_argument("--strict-input-hash", action="store_true")
    parser.add_argument("--allow-atom-changes", action="store_true")
    parser.add_argument("--build-tracks", action="store_true")
    parser.add_argument("--export-states", action="store_true")
    parser.add_argument("--no-network-export", action="store_true")
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--elements", default="1:C,2:H,3:O")
    parser.add_argument("--minimum-shared-atoms", type=int, default=1)
    parser.add_argument("--minimum-parent-overlap", type=float, default=0.05)
    parser.add_argument("--minimum-child-overlap", type=float, default=0.05)
    parser.add_argument("--minimum-event-conservation", type=float, default=0.90)
    parser.add_argument("--stage", choices=("all", "frames", "transitions", "postprocess"), default="all")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--temporal-mode", choices=("off", "annotate", "exclude"), default="off",
        help="off keeps all events; annotate labels candidates; exclude builds a filtered view only",
    )
    parser.add_argument(
        "--minimum-lifetime-frames", type=int, default=1,
        help="tracks shorter than this are candidates; default 1 disables short-lifetime filtering",
    )
    parser.add_argument(
        "--reversal-window-frames", type=int, default=0,
        help="mark reversed reaction signatures within this frame window; 0 disables it",
    )
    config_path = parser.parse_known_args()[0].config.expanduser()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    temporal_config = config.get("temporal_filter", {})
    parser.set_defaults(cutoff=config.get("bond_order_cutoff", 0.3),
                        elements=",".join(f"{key}:{value}" for key, value in config.get("elements", {"1":"C", "2":"H", "3":"O"}).items()),
                        temporal_mode=temporal_config.get("mode", "off"),
                        minimum_lifetime_frames=temporal_config.get("minimum_lifetime_frames", 1),
                        reversal_window_frames=temporal_config.get("reversal_window_frames", 0),
                        **{key.replace("-", "_"): value for key, value in config.get("matching", {}).items()},
                        **config.get("execution", {}))
    args = parser.parse_args()

    database = args.database.expanduser().resolve()
    output_dir = (args.output_dir or database.parent).expanduser().resolve()
    element_map = _parse_element_map(args.elements)
    classification = config.get("classification", {})
    mass_map = config.get("masses")
    ThreeDimensionalClassifier(**classification)
    pair_workers = args.workers if args.pair_workers is None else args.pair_workers
    max_inflight = 2 * max(args.workers, pair_workers) if args.max_inflight is None else args.max_inflight
    if min(args.workers, pair_workers, args.memory_limit_mib) < 1 or max_inflight < max(args.workers, pair_workers):
        parser.error("Positive workers/memory and max-inflight >= worker counts are required")
    if args.expected_frame_step is not None and args.expected_frame_step < 1:
        parser.error("expected-frame-step must be positive")
    if not math.isfinite(args.cutoff) or args.cutoff < 0:
        parser.error("cutoff must be finite and non-negative")
    if not 0 <= args.minimum_event_conservation <= 1:
        parser.error("minimum-event-conservation must be between 0 and 1")
    if args.minimum_lifetime_frames < 1 or args.reversal_window_frames < 0:
        parser.error("Temporal frame parameters require lifetime >= 1 and reversal >= 0")
    matching = {
        "minimum_shared_atoms": args.minimum_shared_atoms,
        "minimum_parent_overlap": args.minimum_parent_overlap,
        "minimum_child_overlap": args.minimum_child_overlap,
        "minimum_event_conservation": args.minimum_event_conservation,
    }
    temporal = {
        "mode": args.temporal_mode,
        "minimum_lifetime_frames": args.minimum_lifetime_frames,
        "reversal_window_frames": args.reversal_window_frames,
    }
    config_hash = _configuration_hash(args.cutoff, element_map, {}, classification, mass_map)
    def cancel_analysis(signum, frame):
        raise KeyboardInterrupt("Analysis cancelled; committed frames and pairs can be resumed")
    signal.signal(signal.SIGTERM, cancel_analysis)
    with database_lock(database):
        progress = Progress(args.progress_file or database.with_suffix(".progress.json"), args.memory_limit_mib)
        connection = None
        owns_manifest = False
        try:
            progress.update(force=True, database=str(database))
            if shutil.disk_usage(database.parent).free < args.minimum_free_disk_mib * 2**20:
                raise OSError("Insufficient free disk space for analysis")
            frames = discover_frames(args.inputs, args.pattern, args.frame_order)
            if args.expected_frame_count is not None and len(frames) != args.expected_frame_count:
                raise ValueError(f"Expected {args.expected_frame_count} frames, found {len(frames)}")
            if args.expected_frame_step is not None:
                numbers = [frame["filename_frame_number"] for frame in frames]
                if any(number is None for number in numbers) or any(
                    right - left != args.expected_frame_step for left, right in zip(numbers, numbers[1:])
                ):
                    raise ValueError("Missing, duplicate or out-of-order filename frame numbers")
            if args.strict_input_hash:
                for frame in frames:
                    digest = hashlib.sha256()
                    with Path(frame["source_path"]).open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    frame["sha256"] = digest.hexdigest()
            connection = _connect_database(database)
            _initialize_schema(connection)
            _initialize_manifest(connection, frames, config_hash)
            owns_manifest = True
            ensure_postprocess_schema(connection)
            with connection:
                previous_threshold = _get_metadata(connection, "minimum_event_conservation")
                if previous_threshold is not None and float(previous_threshold) != args.minimum_event_conservation:
                    connection.execute("""UPDATE events SET resolution=CASE
                        WHEN event_type IN ('birth','death') THEN 'unresolved'
                        WHEN event_type='many_to_many' THEN 'composite'
                        WHEN atom_conservation_ratio>=? THEN 'resolved' ELSE 'composite' END""",
                                       (args.minimum_event_conservation,))
                for key, value in {"temporal_profile": json.dumps(temporal, sort_keys=True),
                                   "frame_order": args.frame_order, "parser_version": PARSER_VERSION,
                                   "descriptor_version": DESCRIPTOR_VERSION,
                                   "classification": json.dumps(classification, sort_keys=True),
                                   "minimum_event_conservation": str(args.minimum_event_conservation),
                                   "matching_version": "2-exact-raw-overlaps",
                                   "coverage_scope": "manifest; filename gaps checked" if args.expected_frame_step else "manifest only; filename gaps not validated",
                                   "run_status": "running", "views_ready": "false", "coarse_ready": "false"}.items():
                    _set_metadata(connection, key, value)
            if args.stage in {"all", "frames"}:
                progress.update(force=True, stage="frames", completed=0, total=len(frames))
                _run_frame_stage(connection, args.workers, args.cutoff, element_map, args.retry_failed,
                                 max_inflight, args.memory_limit_mib, progress, classification, mass_map)
            _prepare_pairs(connection)
            if not args.allow_atom_changes and connection.execute(
                "SELECT COUNT(DISTINCT atom_signature) FROM frames WHERE status='completed'"
            ).fetchone()[0] > 1:
                raise ValueError("Atom membership/types change across frames; verify inputs or explicitly use --allow-atom-changes")
            if args.stage in {"all", "transitions"}:
                progress.update(force=True, stage="transitions", completed=0, total=max(0, len(frames)-1))
                _run_pair_stage(connection, database, pair_workers, matching, args.retry_failed,
                                max_inflight, args.memory_limit_mib, progress)
            unfinished_frames = connection.execute("SELECT COUNT(*) FROM frames WHERE status!='completed'").fetchone()[0]
            unfinished_pairs = connection.execute("SELECT COUNT(*) FROM frame_pairs WHERE status!='completed'").fetchone()[0]
            complete = unfinished_frames == 0 and unfinished_pairs == 0
            track_summary = {"tracks": 0, "fragments": 0}
            annotation_summary = {"mode": args.temporal_mode, "statuses": {}}
            coarse_summary = {}
            if args.stage != "frames" and complete:
                if args.build_tracks or args.temporal_mode != "off":
                    progress.update(force=True, stage="tracks", completed=0, total=len(frames))
                    track_summary = build_fragment_tracks(connection, progress=progress)
                progress.update(force=True, stage="annotations", completed=0, total=0)
                annotation_summary = annotate_events(connection, mode=args.temporal_mode,
                    minimum_lifetime_frames=args.minimum_lifetime_frames,
                    reversal_window_frames=args.reversal_window_frames,
                    preserve_fast_intermediates=temporal_config.get("preserve_fast_intermediates", True),
                    progress=progress)
                progress.update(force=True, stage="views", completed=0, total=0)
                build_network_views(connection, mode=args.temporal_mode, progress=progress)
                _rebuild_reaction_channels(connection)
                progress.update(force=True, stage="coarse_flows", completed=0, total=max(0, len(frames)-1))
                coarse_summary = build_analytics(connection, element_map, mass_map, args.cutoff, progress)
                with connection:
                    _set_metadata(connection, "views_ready", "true")
            status = ("frames_completed" if not unfinished_frames else "partial") if args.stage == "frames" else ("completed" if complete else "partial")
            progress.update(force=True, stage="export", completed=0, total=0)
            summary = _write_exports(connection, output_dir, args.export_states,
                                     not args.no_network_export and args.stage != "frames" and complete)
            if not args.no_network_export and args.stage != "frames" and complete:
                summary["exports"].update(export_analytics(connection, output_dir))
            summary.update(status=status, expected_frames=len(frames), expected_pairs=max(0, len(frames)-1),
                           unfinished_frames=unfinished_frames, unfinished_pairs=unfinished_pairs,
                           tracks=track_summary["tracks"], temporal_annotations=annotation_summary,
                           coarse_analytics=coarse_summary,
                           coverage_scope=_get_metadata(connection, "coverage_scope"))
            atomic_json(output_dir / "network_summary.json", summary)
            with connection:
                _set_metadata(connection, "run_status", status)
            progress.update(force=True, status=status, stage="done", completed=1, total=1, summary=summary)
            print(json.dumps({"database": str(database), **summary}, ensure_ascii=False, indent=2), flush=True)
            return 0 if status != "partial" else 2
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            if connection:
                connection.rollback()
                if owns_manifest:
                    with connection:
                        _set_metadata(connection, "run_status", status)
            progress.memory_limit_mib = 0
            progress.update(force=True, status=status, error=str(exc))
            raise
        finally:
            if connection:
                connection.close()


if __name__ == "__main__":
    sys.exit(main())
