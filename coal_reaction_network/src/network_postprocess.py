"""Disk-backed tracks, residence lifetimes and reaction-network views."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


POSTPROCESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS fragment_tracks (
    track_id INTEGER NOT NULL, frame_id INTEGER NOT NULL, fragment_id INTEGER NOT NULL,
    first_parent_track_id INTEGER, segment_id INTEGER NOT NULL,
    PRIMARY KEY (frame_id, fragment_id)
);
CREATE INDEX IF NOT EXISTS idx_fragment_tracks_track ON fragment_tracks(track_id);
CREATE TABLE IF NOT EXISTS track_summaries (
    track_id INTEGER PRIMARY KEY, first_frame_id INTEGER NOT NULL, last_frame_id INTEGER NOT NULL,
    frame_count INTEGER NOT NULL, first_timestep INTEGER NOT NULL, last_timestep INTEGER NOT NULL,
    dominant_species_id TEXT, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS residence_segments (
    segment_id INTEGER PRIMARY KEY, track_id INTEGER NOT NULL, species_id TEXT NOT NULL,
    first_frame_id INTEGER NOT NULL, last_frame_id INTEGER NOT NULL,
    frame_count INTEGER NOT NULL, left_censored INTEGER NOT NULL, right_censored INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS track_species_counts (
    track_id INTEGER NOT NULL, species_id TEXT NOT NULL, frame_count INTEGER NOT NULL,
    PRIMARY KEY(track_id, species_id)
);
CREATE TABLE IF NOT EXISTS track_links (
    parent_frame_id INTEGER NOT NULL, child_frame_id INTEGER NOT NULL,
    parent_fragment_id INTEGER NOT NULL, child_fragment_id INTEGER NOT NULL,
    parent_track_id INTEGER NOT NULL, child_track_id INTEGER NOT NULL, shared_atom_count INTEGER NOT NULL,
    PRIMARY KEY(parent_frame_id, child_frame_id, parent_fragment_id, child_fragment_id)
);
CREATE TABLE IF NOT EXISTS event_annotations (
    parent_frame_id INTEGER NOT NULL, child_frame_id INTEGER NOT NULL, event_id INTEGER NOT NULL,
    temporal_status TEXT NOT NULL, filter_decision TEXT NOT NULL, filter_reason TEXT NOT NULL,
    involved_track_ids_json TEXT NOT NULL, minimum_track_lifetime INTEGER,
    reversal_candidate INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(parent_frame_id, child_frame_id, event_id)
);
CREATE TABLE IF NOT EXISTS network_view_channels (
    view_name TEXT NOT NULL, reaction_signature TEXT NOT NULL, event_type TEXT NOT NULL,
    event_count INTEGER NOT NULL, first_timestep INTEGER NOT NULL, last_timestep INTEGER NOT NULL,
    mean_confidence REAL NOT NULL, mean_atom_conservation_ratio REAL NOT NULL,
    mean_mass_error REAL NOT NULL, resolution_counts_json TEXT NOT NULL,
    reactant_nodes_json TEXT NOT NULL, product_nodes_json TEXT NOT NULL,
    PRIMARY KEY(view_name, reaction_signature)
);
CREATE TABLE IF NOT EXISTS postprocess_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
DROP INDEX IF EXISTS idx_events_pair_type;
CREATE INDEX IF NOT EXISTS idx_events_reaction_frame ON events(parent_frame_id,child_frame_id,event_id)
    WHERE event_type NOT IN ('persistence','coarse_persistence');
"""


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(POSTPROCESS_SCHEMA)
    connection.execute("""CREATE VIEW IF NOT EXISTS network_view_events AS
        SELECT 'raw' AS view_name, parent_frame_id, child_frame_id, event_id, 'keep' AS decision
        FROM events WHERE event_type NOT IN ('persistence', 'coarse_persistence')
        UNION ALL
        SELECT 'annotated', parent_frame_id, child_frame_id, event_id, 'keep'
        FROM events WHERE event_type NOT IN ('persistence', 'coarse_persistence')
        UNION ALL
        SELECT 'filtered', e.parent_frame_id, e.child_frame_id, e.event_id,
               COALESCE(a.filter_decision, 'keep')
        FROM events e LEFT JOIN event_annotations a USING(parent_frame_id, child_frame_id, event_id)
        WHERE e.event_type NOT IN ('persistence', 'coarse_persistence')""")
    connection.commit()


def _tick(progress, stage: str, completed: int, total: int = 0) -> None:
    if progress:
        progress.update(stage=stage, completed=completed, total=total)


def invalidate_tracks(connection: sqlite3.Connection, frame_id: int) -> None:
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='postprocess_state'").fetchone():
        return
    marker = connection.execute("SELECT value FROM postprocess_state WHERE key='tracks_frame'").fetchone()
    if marker and frame_id <= int(marker[0]):
        # A checkpoint cannot be resumed after its already processed inputs change.
        connection.execute("DELETE FROM postprocess_state WHERE key='tracks_frame'")
        for table in ("fragment_tracks", "track_summaries", "track_links", "residence_segments",
                      "track_species_counts", "event_annotations"):
            connection.execute(f"DELETE FROM {table}")


def build_fragment_tracks(connection: sqlite3.Connection, progress=None, resume: bool = True) -> dict:
    """Continue an instance only for exact one-to-one atom membership; split residence on class change."""
    ensure_schema(connection)
    marker = connection.execute("SELECT value FROM postprocess_state WHERE key='tracks_frame'").fetchone()
    last_frame = int(marker[0]) if marker and resume else -1
    if last_frame < 0:
        with connection:
            for table in ("fragment_tracks", "track_summaries", "track_links", "residence_segments",
                          "track_species_counts"):
                connection.execute(f"DELETE FROM {table}")
            connection.execute("DELETE FROM postprocess_state WHERE key='tracks_frame'")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(fragments)")}
    atom_count = "atom_count" if "atom_count" in columns else "NULL AS atom_count"
    previous = {}
    if last_frame >= 0:
        for row in connection.execute(f"""SELECT ft.*, f.species_id, {atom_count}
                FROM fragment_tracks ft JOIN fragments f USING(frame_id, fragment_id)
                WHERE ft.frame_id=?""", (last_frame,)):
            previous[row["fragment_id"]] = dict(row)
    next_track = connection.execute("SELECT COALESCE(MAX(track_id),0)+1 FROM track_summaries").fetchone()[0]
    next_segment = connection.execute("SELECT COALESCE(MAX(segment_id),0)+1 FROM residence_segments").fetchone()[0]
    total = connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    frame_ids = connection.execute("SELECT frame_id,timestep FROM frames WHERE status='completed' AND frame_id>? ORDER BY frame_id",
                                   (last_frame,))
    for frame in frame_ids:
        frame_id = frame["frame_id"]
        pair = connection.execute("SELECT parent_frame_id FROM frame_pairs WHERE child_frame_id=? AND status='completed'",
                                  (frame_id,)).fetchone()
        linked = bool(pair and pair[0] == last_frame and frame_id == last_frame + 1)
        if not linked:
            previous = {}
        children = defaultdict(list)
        parent_degrees = Counter()
        if linked:
            for row in connection.execute("SELECT * FROM lineage_links WHERE parent_frame_id=? AND child_frame_id=?",
                                          (last_frame, frame_id)):
                children[row["child_fragment_id"]].append(row)
                parent_degrees[row["parent_fragment_id"]] += 1
        current = {}
        with connection:
            for fragment in connection.execute(f"SELECT fragment_id,species_id,{atom_count} FROM fragments WHERE frame_id=? ORDER BY fragment_id",
                                               (frame_id,)):
                fragment_id = fragment["fragment_id"]
                candidates = children.get(fragment_id, [])
                parent = previous.get(candidates[0]["parent_fragment_id"]) if len(candidates) == 1 else None
                continuation = bool(parent and parent_degrees[candidates[0]["parent_fragment_id"]] == 1
                                    and (fragment["atom_count"] is None or (
                                        candidates[0]["shared_atom_count"] == fragment["atom_count"] == parent["atom_count"])))
                if continuation:
                    track_id = parent["track_id"]
                else:
                    track_id = next_track
                    next_track += 1
                same_species = continuation and parent["species_id"] == fragment["species_id"]
                if same_species:
                    segment_id = parent["segment_id"]
                else:
                    segment_id = next_segment
                    next_segment += 1
                current[fragment_id] = dict(fragment, track_id=track_id, segment_id=segment_id)
                connection.execute("INSERT INTO fragment_tracks VALUES(?,?,?,?,?)",
                                   (track_id, frame_id, fragment_id, parent["track_id"] if parent else None, segment_id))
                connection.execute("""INSERT INTO track_summaries VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(track_id) DO UPDATE SET last_frame_id=excluded.last_frame_id,
                    last_timestep=excluded.last_timestep, frame_count=track_summaries.frame_count+1""",
                                   (track_id, frame_id, frame_id, 1, frame["timestep"], frame["timestep"], fragment["species_id"], "observed"))
                connection.execute("""INSERT INTO track_species_counts VALUES(?,?,1)
                    ON CONFLICT(track_id,species_id) DO UPDATE SET frame_count=frame_count+1""",
                                   (track_id, fragment["species_id"]))
                connection.execute("""INSERT INTO residence_segments VALUES(?,?,?,?,?,1,?,1)
                    ON CONFLICT(segment_id) DO UPDATE SET last_frame_id=excluded.last_frame_id,
                    frame_count=residence_segments.frame_count+1""",
                                   (segment_id, track_id, fragment["species_id"], frame_id, frame_id, int(not linked)))
                for candidate in candidates:
                    ancestor = previous.get(candidate["parent_fragment_id"])
                    if ancestor:
                        connection.execute("INSERT INTO track_links VALUES(?,?,?,?,?,?,?)",
                                           (last_frame, frame_id, candidate["parent_fragment_id"], fragment_id,
                                            ancestor["track_id"], track_id, candidate["shared_atom_count"]))
            active_segments = {item["segment_id"] for item in current.values()}
            if linked:
                connection.executemany("UPDATE residence_segments SET right_censored=0 WHERE segment_id=?",
                                       ((item["segment_id"],) for item in previous.values() if item["segment_id"] not in active_segments))
            connection.execute("INSERT OR REPLACE INTO postprocess_state VALUES('tracks_frame',?)", (str(frame_id),))
        previous = current
        last_frame = frame_id
        _tick(progress, "tracks", frame_id + 1, total)
    with connection:
        connection.execute("""UPDATE track_summaries SET dominant_species_id=(
            SELECT species_id FROM track_species_counts c WHERE c.track_id=track_summaries.track_id
            ORDER BY frame_count DESC,species_id LIMIT 1)""")
    return dict(zip(("tracks", "fragments"), connection.execute(
        "SELECT COUNT(*),COALESCE(SUM(frame_count),0) FROM track_summaries").fetchone()))


def _event_residences(connection: sqlite3.Connection, row) -> list[sqlite3.Row]:
    segments = {}
    for field, frame_field in (("parent_fragment_ids_json", "parent_frame_id"),
                               ("child_fragment_ids_json", "child_frame_id")):
        for fragment_id in json.loads(row[field]):
            result = connection.execute("""SELECT s.* FROM fragment_tracks ft
                JOIN residence_segments s USING(segment_id) WHERE ft.frame_id=? AND ft.fragment_id=?""",
                                        (row[frame_field], fragment_id)).fetchone()
            if result:
                segments[result["segment_id"]] = result
    return list(segments.values())


def _instance_key(connection, frame_id: int, fragment_ids: str) -> str:
    segments = []
    for fragment_id in json.loads(fragment_ids):
        row = connection.execute("SELECT track_id FROM fragment_tracks WHERE frame_id=? AND fragment_id=?",
                                 (frame_id, fragment_id)).fetchone()
        if not row:
            return ""
        segments.append(row[0])
    return json.dumps(sorted(segments)) if segments else ""


def annotate_events(connection: sqlite3.Connection, mode: str = "off", minimum_lifetime_frames: int = 1,
                    reversal_window_frames: int = 0, preserve_fast_intermediates: bool = True,
                    progress=None) -> dict:
    """Use bounded, disk-backed reversal candidates and uncensored species residence segments."""
    ensure_schema(connection)
    if mode not in {"off", "annotate", "exclude"}:
        raise ValueError("Invalid temporal mode")
    if minimum_lifetime_frames < 1 or reversal_window_frames < 0:
        raise ValueError("Invalid temporal window")
    connection.execute("DROP TABLE IF EXISTS temp.annotation_build")
    connection.execute("CREATE TEMP TABLE annotation_build AS SELECT * FROM event_annotations WHERE 0")
    connection.execute("CREATE UNIQUE INDEX annotation_build_key ON annotation_build(parent_frame_id,child_frame_id,event_id)")
    connection.execute("DROP TABLE IF EXISTS temp.reversal_window")
    connection.execute("""CREATE TEMP TABLE reversal_window(
        parent_frame INTEGER, child_frame INTEGER, event_id INTEGER,
        reactants TEXT, products TEXT, instance_key TEXT)""")
    connection.execute("CREATE INDEX reversal_lookup ON reversal_window(products,reactants,instance_key,child_frame)")
    total = connection.execute("SELECT COUNT(*) FROM events WHERE event_type NOT IN ('persistence','coarse_persistence')").fetchone()[0]
    counts = Counter()
    processed = 0
    if mode != "off":
        for row in connection.execute("SELECT * FROM events WHERE event_type NOT IN ('persistence','coarse_persistence') ORDER BY parent_frame_id,child_frame_id,event_id"):
            key = (row["parent_frame_id"], row["child_frame_id"], row["event_id"])
            residences = _event_residences(connection, row)
            complete = [segment["frame_count"] for segment in residences if not segment["left_censored"] and not segment["right_censored"]]
            lifetime = min(complete) if complete else None
            status, reason = ("stable", "") if complete else ("boundary_censored", "residence_not_fully_observed")
            if lifetime is not None and lifetime < minimum_lifetime_frames:
                status, reason = "short_lived_candidate", "species_residence_below_threshold"
                if preserve_fast_intermediates and row["event_type"] not in {"birth", "death"} and row["atom_conservation_ratio"] >= 0.9:
                    status = "fast_intermediate_candidate"
            decision = "exclude" if mode == "exclude" and status == "short_lived_candidate" else "keep"
            tracks = sorted({segment["track_id"] for segment in residences})
            connection.execute("INSERT INTO annotation_build VALUES(?,?,?,?,?,?,?,?,0)",
                               (*key, status, decision, reason, json.dumps(tracks), lifetime))
            if reversal_window_frames and row["reactant_nodes_json"] != row["product_nodes_json"]:
                connection.execute("DELETE FROM reversal_window WHERE child_frame < ?",
                                   (row["parent_frame_id"] - reversal_window_frames,))
                instance = _instance_key(connection, row["parent_frame_id"], row["parent_fragment_ids_json"])
                if instance:
                    for reverse in connection.execute("""SELECT parent_frame,child_frame,event_id FROM reversal_window
                        WHERE products=? AND reactants=? AND instance_key=? AND child_frame<=? AND parent_frame<?""",
                        (row["reactant_nodes_json"], row["product_nodes_json"], instance, row["parent_frame_id"], row["parent_frame_id"])):
                        for match in (tuple(reverse), key):
                            connection.execute("""UPDATE annotation_build SET reversal_candidate=1,
                                temporal_status=CASE WHEN temporal_status='fast_intermediate_candidate' THEN temporal_status ELSE 'reversal_candidate' END,
                                filter_reason='instance_linked_reverse_within_window',
                                filter_decision=CASE WHEN temporal_status='fast_intermediate_candidate' THEN 'keep' ELSE ? END
                                WHERE parent_frame_id=? AND child_frame_id=? AND event_id=?""",
                                               ("exclude" if mode == "exclude" else "keep", *match))
                child_instance = _instance_key(connection, row["child_frame_id"], row["child_fragment_ids_json"])
                if child_instance:
                    connection.execute("INSERT INTO reversal_window VALUES(?,?,?,?,?,?)",
                                       (*key, row["reactant_nodes_json"], row["product_nodes_json"], child_instance))
            processed += 1
            if processed % 500 == 0:
                connection.commit()
                _tick(progress, "annotations", processed, total)
    connection.commit()
    with connection:
        connection.execute("DELETE FROM event_annotations")
        connection.execute("INSERT INTO event_annotations SELECT * FROM annotation_build")
    for row in connection.execute("SELECT temporal_status,COUNT(*) FROM event_annotations GROUP BY temporal_status"):
        counts[row[0]] = row[1]
    connection.execute("DROP TABLE annotation_build")
    connection.execute("DROP TABLE reversal_window")
    _tick(progress, "annotations", total, total)
    return {"events": total, "annotated_events": processed, "statuses": dict(counts), "mode": mode}


def iter_channel_rows(connection: sqlite3.Connection, filtered: bool = False):
    where = "AND COALESCE(a.filter_decision,'keep')='keep'" if filtered else ""
    rows = connection.execute(f"""SELECT e.reaction_signature,e.event_type,e.resolution,
        COUNT(*) AS amount,MIN(f.timestep) AS first_time,MAX(f.timestep) AS last_time,
        SUM(e.confidence) AS confidence,SUM(e.atom_conservation_ratio) AS conservation,
        SUM(e.mass_error) AS mass_error,e.reactant_nodes_json,e.product_nodes_json
        FROM events e JOIN frames f ON f.frame_id=e.parent_frame_id
        LEFT JOIN event_annotations a USING(parent_frame_id,child_frame_id,event_id)
        WHERE e.event_type NOT IN ('persistence','coarse_persistence') {where}
        GROUP BY e.reaction_signature,e.event_type,e.resolution ORDER BY e.reaction_signature""")
    signature = None
    group = None
    def packed():
        amount = group["amount"]
        return (signature, next(iter(group["types"])) if len(group["types"]) == 1 else "mixed", amount,
                group["first"], group["last"], group["confidence"] / amount, group["conservation"] / amount,
                group["mass_error"] / amount, json.dumps(group["resolutions"], sort_keys=True),
                group["reactants"], group["products"])
    for row in rows:
        if row["reaction_signature"] != signature:
            if group:
                yield packed()
            signature = row["reaction_signature"]
            group = dict(amount=0, first=row["first_time"], last=row["last_time"], confidence=0.0,
                         conservation=0.0, mass_error=0.0, types=set(), resolutions=Counter(),
                         reactants=row["reactant_nodes_json"], products=row["product_nodes_json"])
        group["amount"] += row["amount"]
        group["first"] = min(group["first"], row["first_time"])
        group["last"] = max(group["last"], row["last_time"])
        for field in ("confidence", "conservation", "mass_error"):
            group[field] += row[field]
        group["types"].add(row["event_type"])
        group["resolutions"][row["resolution"]] += row["amount"]
    if group:
        yield packed()


def build_network_views(connection: sqlite3.Connection, mode: str = "off", progress=None) -> None:
    ensure_schema(connection)
    connection.execute("DROP TABLE IF EXISTS temp.channels_build")
    connection.execute("CREATE TEMP TABLE channels_build AS SELECT * FROM network_view_channels WHERE 0")
    for index, channel in enumerate(iter_channel_rows(connection), start=1):
        for view in ("raw", "annotated") if mode == "exclude" else ("raw", "annotated", "filtered"):
            connection.execute("INSERT INTO channels_build VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (view, *channel))
        if index % 500 == 0:
            _tick(progress, "views", index)
    if mode == "exclude":
        for index, channel in enumerate(iter_channel_rows(connection, filtered=True), start=1):
            connection.execute("INSERT INTO channels_build VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", ("filtered", *channel))
            if index % 500 == 0:
                _tick(progress, "views", index)
    connection.commit()
    with connection:
        connection.execute("DELETE FROM network_view_channels")
        connection.execute("INSERT INTO network_view_channels SELECT * FROM channels_build")
    connection.execute("DROP TABLE channels_build")


def write_csv(connection: sqlite3.Connection, query: str, path: Path, params: tuple = ()) -> int:
    cursor = connection.execute(query, params)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([column[0] for column in cursor.description])
        for row in cursor:
            writer.writerow(tuple(row))
            count += 1
    temporary.replace(path)
    return count


def export_network(connection: sqlite3.Connection, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for view in ("raw", "annotated", "filtered"):
        query = "SELECT * FROM network_view_channels WHERE view_name=? ORDER BY event_count DESC,reaction_signature"
        csv_path = output_dir / f"{view}_reaction_channels.csv"
        count = write_csv(connection, query, csv_path, (view,))
        json_path = output_dir / f"{view}_reaction_network.json"
        temporary = json_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write('{"view":' + json.dumps(view) + ',"species":[')
            for index, row in enumerate(connection.execute("SELECT * FROM species ORDER BY species_id")):
                handle.write(("," if index else "") + json.dumps(dict(row), ensure_ascii=False))
            handle.write('],"reaction_channels":[')
            for index, row in enumerate(connection.execute(query, (view,))):
                handle.write(("," if index else "") + json.dumps(dict(row), ensure_ascii=False))
            handle.write('],"stoichiometry_note":"Full reactant and product coefficients are retained"}')
        temporary.replace(json_path)
        result[view] = {"channels": count, "csv": str(csv_path), "json": str(json_path)}
    return result
