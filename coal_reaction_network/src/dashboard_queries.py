"""Read-only, bounded dashboard queries. All filters are parameterized."""

from __future__ import annotations

import base64
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def db_connect(path: Path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA cache_size=-8192")
    connection.execute("PRAGMA temp_store=FILE")
    deadline = time.monotonic() + 30
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    try:
        yield connection
    finally:
        connection.close()


def table_exists(connection, name):
    return connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None


def metadata(connection):
    return dict(connection.execute("SELECT key,value FROM metadata")) if table_exists(connection, "metadata") else {}


def db_summary(path: Path) -> dict:
    with db_connect(path) as connection:
        frame_status = dict(connection.execute("SELECT status,COUNT(*) FROM frames GROUP BY status"))
        pair_status = dict(connection.execute("SELECT status,COUNT(*) FROM frame_pairs GROUP BY status"))
        frame = connection.execute("SELECT COUNT(*) AS total,COALESCE(SUM(fragment_count),0) AS fragments,MIN(frame_id) AS first,MAX(frame_id) AS last FROM frames").fetchone()
        pair = connection.execute("SELECT COALESCE(SUM(event_count),0) FROM frame_pairs WHERE status='completed'").fetchone()[0]
        meta = metadata(connection)
        reaction_events = connection.execute("SELECT COALESCE(SUM(event_count),0) FROM reaction_channels").fetchone()[0]
        return dict(database=str(path), frames=frame["total"], fragments=frame["fragments"],
                    first_frame=frame["first"], last_frame=frame["last"],
                    completed_frames=frame_status.get("completed", 0), failed_frames=frame_status.get("failed", 0),
                    completed_pairs=pair_status.get("completed", 0), failed_pairs=pair_status.get("failed", 0),
                    frame_pairs=sum(pair_status.values()), expected_pairs=max(0, frame["total"]-1),
                    events=pair, reaction_events=reaction_events if meta.get("views_ready") == "true" else None,
                    persistence_events=pair-reaction_events if meta.get("views_ready") == "true" else None,
                    species=connection.execute("SELECT COUNT(*) FROM species").fetchone()[0],
                    reaction_channels=connection.execute("SELECT COUNT(*) FROM reaction_channels").fetchone()[0],
                    metadata=meta, frame_status=frame_status, pair_status=pair_status,
                    legacy=meta.get("schema_version") != "3")


def integer(query: dict, name: str, default: int, minimum: int = 0, maximum: int = 2**63-1) -> int:
    value = int(query.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def encode_cursor(values) -> str:
    return base64.urlsafe_b64encode(json.dumps(values).encode()).decode()


def decode_cursor(value: str, length: int):
    try:
        result = json.loads(base64.urlsafe_b64decode(value.encode()))
        if not isinstance(result, list) or len(result) != length:
            raise ValueError()
        return result
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("Invalid pagination cursor") from exc


def decoded_row(row):
    result = dict(row)
    for field in list(result):
        if field.endswith("_json"):
            result[field[:-5]] = json.loads(result.pop(field))
    return result


def event_filters(query: dict, alias: str = "e") -> tuple[str, list]:
    """Use the same event population for channels, flow projections and evidence."""
    start, end = integer(query, "start", 0), integer(query, "end", 2**63-1)
    if end < start:
        raise ValueError("Frame range is reversed")
    view = query.get("view", "raw")
    activity = query.get("activity", "all")
    if view not in {"raw", "annotated", "filtered"}:
        raise ValueError("Unknown network view")
    if activity not in {"all", "changes", "internal"}:
        raise ValueError("activity must be all, changes or internal")
    clauses = [f"{alias}.parent_frame_id>=?", f"{alias}.child_frame_id<=?",
               f"{alias}.event_type NOT IN ('persistence','coarse_persistence')"]
    params = [start, end]
    if view == "filtered":
        clauses.append(f"NOT EXISTS(SELECT 1 FROM event_annotations a WHERE a.parent_frame_id={alias}.parent_frame_id AND a.child_frame_id={alias}.child_frame_id AND a.event_id={alias}.event_id AND a.filter_decision='exclude')")
    if activity != "all":
        operator = "!=" if activity == "changes" else "="
        clauses.append(f"{alias}.reactant_nodes_json{operator}{alias}.product_nodes_json")
    for key, column in (("event_type", "event_type"), ("resolution", "resolution"), ("signature", "reaction_signature")):
        if query.get(key):
            clauses.append(f"{alias}.{column}=?")
            params.append(query[key])
    species_clauses = []
    for key, column in (("species", "species_id"), ("carbon_bin", "carbon_bin"),
                        ("ring_bin", "ring_bin"), ("oxygen_bin", "oxygen_bin")):
        if query.get(key):
            species_clauses.append(f"s.{column}=?")
            params.append(query[key])
    if species_clauses:
        clauses.append(f"EXISTS(SELECT 1 FROM species s WHERE (EXISTS(SELECT 1 FROM json_each({alias}.reactant_nodes_json) WHERE key=s.species_id) OR EXISTS(SELECT 1 FROM json_each({alias}.product_nodes_json) WHERE key=s.species_id)) AND " + " AND ".join(species_clauses) + ")")
    return " AND ".join(clauses), params


def channel_details(channel: dict) -> dict:
    reactants, products = channel["reactant_nodes"], channel["product_nodes"]
    channel["net_coefficients"] = {species: products.get(species, 0)-reactants.get(species, 0)
                                   for species in sorted(reactants.keys() | products.keys())
                                   if products.get(species, 0) != reactants.get(species, 0)}
    channel["activity"] = "changes" if channel["net_coefficients"] else "internal"
    return channel


def network(path: Path, query: dict | str) -> dict:
    if isinstance(query, str):
        query = {"view": query}
    view = query.get("view", "raw")
    if view not in {"raw", "annotated", "filtered"}:
        raise ValueError("Unknown network view")
    limit = integer(query, "limit", 40, 1, 200)
    start = integer(query, "start", 0)
    end = integer(query, "end", 2**63-1)
    if end < start:
        raise ValueError("Frame range is reversed")
    minimum_count = integer(query, "minimum_count", 1, 1)
    with db_connect(path) as connection:
        meta = metadata(connection)
        if meta.get("schema_version") != "3":
            raise ValueError("Legacy parser results: rebuild with a NEW v3 database before visualization")
        if meta.get("views_ready") != "true":
            return {"view": view, "channels": [], "total_channels": 0, "total_events": 0,
                    "displayed_events": 0, "next_cursor": None, "ready": False,
                    "message": "反应网络尚未完整构建；请查看任务阶段和失败帧。"}
        where, filter_params = event_filters(query)
        temporal = ("a.temporal_status" if view != "raw" else "NULL")
        base = f"""SELECT e.reaction_signature,CASE WHEN COUNT(DISTINCT e.event_type)=1 THEN MIN(e.event_type) ELSE 'mixed' END event_type,
            COUNT(*) event_count,MIN(e.parent_frame_id) first_frame,MAX(e.child_frame_id) last_frame,
            AVG(e.confidence) mean_confidence,AVG(e.atom_conservation_ratio) mean_atom_conservation_ratio,
            AVG(e.mass_error) mean_mass_error,e.reactant_nodes_json,e.product_nodes_json,
            SUM(e.resolution!='resolved') uncertain_events,
            SUM(CASE WHEN {temporal} IN ('short_lived_candidate','fast_intermediate_candidate','reversal_candidate') THEN 1 ELSE 0 END) flagged_events
            FROM events e LEFT JOIN event_annotations a USING(parent_frame_id,child_frame_id,event_id)
            WHERE {where} GROUP BY e.reaction_signature HAVING COUNT(*)>=?"""
        params = [*filter_params, minimum_count]
        full_range = start == 0 and end >= int(meta.get("frame_count", 0)) - 1
        cached = full_range and not any(query.get(key) for key in ("species", "event_type", "resolution", "carbon_bin", "ring_bin", "oxygen_bin", "signature")) and query.get("activity", "all") == "all" and view != "annotated"
        if cached:
            base = """SELECT reaction_signature,event_type,event_count,NULL AS first_frame,NULL AS last_frame,
                mean_confidence,mean_atom_conservation_ratio,mean_mass_error,reactant_nodes_json,product_nodes_json,
                (SELECT COALESCE(SUM(value),0) FROM json_each(resolution_counts_json) WHERE key!='resolved') uncertain_events,
                NULL AS flagged_events FROM network_view_channels WHERE view_name=? AND event_count>=?"""
            params = [view, minimum_count]
        totals = connection.execute(f"SELECT COUNT(*),COALESCE(SUM(event_count),0) FROM ({base})", params).fetchone()
        outer, page_params = "", list(params)
        if query.get("cursor"):
            frequency, signature = decode_cursor(query["cursor"], 2)
            outer = "WHERE event_count<? OR (event_count=? AND reaction_signature>?)"
            page_params.extend([int(frequency), int(frequency), str(signature)])
        rows = connection.execute(f"SELECT * FROM ({base}) {outer} ORDER BY event_count DESC,reaction_signature LIMIT ?",
                                  (*page_params, limit+1)).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        cursor = encode_cursor([rows[-1]["event_count"], rows[-1]["reaction_signature"]]) if more else None
        channels = []
        for row in rows:
            channel = channel_details(decoded_row(row))
            breakdown = connection.execute(f"""SELECT e.event_type,COUNT(*) amount,
                MIN(e.parent_frame_id) first_frame,MAX(e.child_frame_id) last_frame
                FROM events e WHERE {where} AND e.reaction_signature=? GROUP BY e.event_type""",
                (*filter_params, channel["reaction_signature"])).fetchall()
            channel["event_type_counts"] = {item["event_type"]: item["amount"] for item in breakdown}
            if breakdown:
                channel["first_frame"] = min(item["first_frame"] for item in breakdown)
                channel["last_frame"] = max(item["last_frame"] for item in breakdown)
            channels.append(channel)
        return dict(view=view, channels=channels, total_channels=totals[0],
                    total_events=totals[1], displayed_events=sum(row["event_count"] for row in rows),
                    next_cursor=cursor, ready=True, start=start, end=end, scope="current filters and page")


def records(path: Path, kind: str, query: dict) -> dict:
    limit = integer(query, "limit", 100, 1, 500)
    params = []
    where = []
    start = integer(query, "start", 0)
    end = integer(query, "end", 2**63-1)
    if end < start:
        raise ValueError("Frame range is reversed")
    definitions = {
        "frames": ("frames", ["frame_id"], "*"),
        "events": ("events", ["parent_frame_id", "child_frame_id", "event_id"], "*"),
        "tracks": ("track_summaries", ["track_id"], "*"),
        "species-states": ("species_states", ["frame_id", "species_id"], "*"),
        "fragments": ("fragments", ["frame_id", "fragment_id"],
                      "frame_id,fragment_id,species_id,atom_count,molecular_mass,ring_count,radical_sites,topology_hash"),
        "lineage": ("lineage_links", ["parent_frame_id", "child_frame_id", "parent_fragment_id", "child_fragment_id"], "*"),
    }
    table, keys, columns = definitions[kind]
    if kind in {"events", "lineage"}:
        where.extend(["parent_frame_id>=?", "child_frame_id<=?"])
        params.extend([start, end])
        if query.get("parent_frame") is not None:
            where.append("parent_frame_id=?")
            params.append(integer(query, "parent_frame", 0))
        if kind == "events":
            event_where, event_params = event_filters(query, "events")
            where.append(event_where)
            params.extend(event_params)
            columns = "events.*,(SELECT temporal_status FROM event_annotations a WHERE a.parent_frame_id=events.parent_frame_id AND a.child_frame_id=events.child_frame_id AND a.event_id=events.event_id) AS temporal_status"
            if query.get("event_id"):
                where.append("event_id=?")
                params.append(integer(query, "event_id", 1, 1))
            if query.get("flow_source") and query.get("flow_target"):
                where.append("EXISTS(SELECT 1 FROM coarse_event_flows f WHERE f.parent_frame_id=events.parent_frame_id AND f.child_frame_id=events.child_frame_id AND f.event_id=events.event_id AND f.source_species=? AND f.target_species=?)")
                params.extend([query["flow_source"], query["flow_target"]])
    elif kind == "tracks":
        where.extend(["last_frame_id>=?", "first_frame_id<=?"])
        params.extend([start, end])
    else:
        where.extend(["frame_id>=?", "frame_id<=?"])
        params.extend([start, end])
        if kind == "frames" and query.get("status"):
            where.append("status=?")
            params.append(query["status"])
        if kind != "frames" and query.get("species_id"):
            where.append("species_id=?")
            params.append(query["species_id"])
    if query.get("cursor"):
        values = decode_cursor(query["cursor"], len(keys))
        where.append(f"({','.join(keys)}) > ({','.join('?' for key in keys)})" if len(keys)>1 else f"{keys[0]}>?")
        params.extend(values)
    with db_connect(path) as connection:
        rows = connection.execute(f"SELECT {columns} FROM {table} WHERE {' AND '.join(where)} ORDER BY {','.join(keys)} LIMIT ?",
                                  (*params, limit+1)).fetchall()
    more = len(rows)>limit
    rows = rows[:limit]
    return dict(rows=[decoded_row(row) for row in rows], has_more=more,
                next_cursor=encode_cursor([rows[-1][key] for key in keys]) if more else None,
                limit=limit, scope="current filters and page")


def fragment_detail(path: Path, query: dict) -> dict:
    from trajectory_pipeline import _decode_atom_ids
    frame = integer(query, "frame", 0)
    fragment = integer(query, "fragment", 1, 1)
    offset = integer(query, "offset", 0)
    limit = integer(query, "limit", 500, 1, 2000)
    with db_connect(path) as connection:
        row = connection.execute("SELECT * FROM fragments WHERE frame_id=? AND fragment_id=?", (frame, fragment)).fetchone()
        if row is None:
            raise ValueError("Fragment not found")
        result = decoded_row(row)
        atom_ids = _decode_atom_ids(result.pop("atom_ids"), row["atom_count"])
        result["atom_ids"] = atom_ids[offset:offset+limit]
        result["next_offset"] = offset+limit if offset+limit < len(atom_ids) else None
        track = connection.execute("""SELECT ft.track_id,s.* FROM fragment_tracks ft
            JOIN residence_segments s USING(segment_id) WHERE ft.frame_id=? AND ft.fragment_id=?""",
                                   (frame, fragment)).fetchone()
        result["residence"] = dict(track) if track else None
        result["source"] = dict(connection.execute("SELECT frame_id,timestep,source_path,status FROM frames WHERE frame_id=?", (frame,)).fetchone())
        return result


def timeseries(path: Path, query: dict) -> dict:
    points = integer(query, "points", 400, 10, 800)
    start = integer(query, "start", 0)
    with db_connect(path) as connection:
        maximum = connection.execute("SELECT COALESCE(MAX(frame_id),0) FROM frames").fetchone()[0]
        end = min(integer(query, "end", maximum), maximum)
        if start > end:
            return {"rows": [], "bucket_frames": 1}
        species = query.get("species_id")
        if not species:
            selected = connection.execute("SELECT species_id FROM species_states WHERE frame_id BETWEEN ? AND ? GROUP BY species_id ORDER BY SUM(total_mass) DESC LIMIT 1",
                                          (start, end)).fetchone()
            species = selected[0] if selected else ""
        width = max(1, (end-start+points)//points)
        rows = connection.execute("""SELECT (f.frame_id-?)/? bucket,MIN(f.frame_id) first_frame,
            MAX(f.frame_id) last_frame,COUNT(*) sample_frames,
            AVG(COALESCE(s.population,0)) population,MIN(COALESCE(s.population,0)) population_min,
            MAX(COALESCE(s.population,0)) population_max,
            AVG(COALESCE(s.total_mass,0)) total_mass,AVG(COALESCE(s.total_radical_sites,0)) total_radical_sites
            ,AVG(COALESCE(s.total_mass/NULLIF(t.mass,0),0)) mass_fraction
            ,AVG(COALESCE(s.total_carbon*1.0/NULLIF(t.carbon,0),0)) carbon_fraction
            ,AVG(COALESCE(s.total_oxygen*1.0/NULLIF(t.oxygen,0),0)) oxygen_fraction
            FROM frames f LEFT JOIN species_states s ON s.frame_id=f.frame_id AND s.species_id=?
            LEFT JOIN (SELECT frame_id,SUM(total_mass) mass,SUM(total_carbon) carbon,SUM(total_oxygen) oxygen
                FROM species_states WHERE frame_id BETWEEN ? AND ? GROUP BY frame_id) t ON t.frame_id=f.frame_id
            WHERE f.status='completed' AND f.frame_id BETWEEN ? AND ? GROUP BY bucket ORDER BY bucket""",
                                  (start, width, species, start, end, start, end)).fetchall()
        return {"species_id": species, "rows": [dict(row) for row in rows], "bucket_frames": width,
                "aggregation": "mean including absent species as zero; population min/max retained"}
