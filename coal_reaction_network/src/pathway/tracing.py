"""Read-only, frame-bounded tracing with explicit display pagination."""

import json

from dashboard_queries import db_connect, integer, metadata
from trajectory_pipeline import _decode_atom_ids
from pathway.summary import ProcessSummary
from pathway.compression import compress_page


def trace(path, query, cancelled=lambda: False, progress=lambda value: None):
    start = integer(query, "start", 0)
    end = integer(query, "end", start)
    if end <= start:
        raise ValueError("结束帧必须大于起始帧")
    direction = query.get("direction", "forward")
    if direction not in {"forward", "backward"}:
        raise ValueError("Unknown tracing direction")
    limit = integer(query, "event_limit", 100, 1, 300)
    offset = integer(query, "offset", 0)
    with db_connect(path) as connection:
        connection.set_progress_handler(lambda: int(cancelled()), 10000)
        connection.execute("BEGIN")
        if metadata(connection).get("schema_version") != "3":
            raise ValueError("路径需要正确解析的 v3 数据库")
        frames = connection.execute("SELECT COUNT(*) FROM frames WHERE status='completed' AND frame_id BETWEEN ? AND ?", (start, end)).fetchone()[0]
        pairs = connection.execute("SELECT COUNT(*) FROM frame_pairs WHERE status='completed' AND parent_frame_id>=? AND child_frame_id<=?", (start, end)).fetchone()[0]
        if frames != end-start+1 or pairs != end-start:
            raise ValueError("区间内存在缺帧或未完成帧对，不能构建连续路径")

        def fragments(frame):
            return {row["fragment_id"]: dict(species=row["species_id"], atoms=set(_decode_atom_ids(row["atom_ids"], row["atom_count"])))
                    for row in connection.execute("SELECT fragment_id,species_id,atom_ids,atom_count FROM fragments WHERE frame_id=?", (frame,))}

        anchor = start if direction == "forward" else end
        anchor_fragments = fragments(anchor)
        identifier = integer(query, "fragment", 0)
        species = query.get("species", "")
        if not identifier and not species:
            raise ValueError("请选择起点 / 终点类别，或输入该帧的片段 ID")
        selected = [item for key, item in anchor_fragments.items()
                    if (key == identifier if identifier else item["species"] == species)]
        seed = set().union(*(item["atoms"] for item in selected))
        if not seed:
            raise ValueError("锚点帧中不存在所选片段 / 类别")
        target_species = query.get("target_species", "")
        if target_species:
            opposite = fragments(end if direction == "forward" else start)
            target_atoms = set().union(*(item["atoms"] for item in opposite.values() if item["species"] == target_species))
            seed.intersection_update(target_atoms)
            if not seed:
                raise ValueError("起终点之间没有共享原子证据")
        states, events, links = [], [], []
        active = {}
        total_events = 0
        missing_occurrences = 0
        overview = ProcessSummary(start, end, seed, integer(query, 'summary_stages', 8, 2, 16))

        def state(frame, fragment_id, item, context=False):
            if len(states) >= 4000:
                raise ValueError("当前页超过 4000 个状态，请减少每页事件或选择具体片段")
            node = dict(id=f"s{len(states)}", kind="state", species=item["species"],
                        first_frame=frame, last_frame=frame, fragment_id=fragment_id,
                        tracked=len(item["atoms"] & seed), context=context)
            states.append(node)
            return node

        parents = fragments(start)
        overview.observe(start, parents)
        for frame in range(start, end):
            if cancelled():
                raise InterruptedError("路径任务已取消")
            children = fragments(frame+1)
            overview.observe(frame+1, children)
            present = set().union(*(item["atoms"] & seed for item in children.values()))
            missing_occurrences += len(seed-present)
            if not active and total_events <= offset:
                active = {key: state(frame, key, item) for key, item in parents.items() if item["atoms"] & seed}
            next_active = {}
            for row in connection.execute("SELECT * FROM events WHERE parent_frame_id=? AND child_frame_id=? ORDER BY event_id", (frame, frame+1)):
                parent_ids = json.loads(row["parent_fragment_ids_json"])
                child_ids = json.loads(row["child_fragment_ids_json"])
                relevant = any(parents[key]["atoms"] & seed for key in parent_ids) or any(children[key]["atoms"] & seed for key in child_ids)
                if not relevant:
                    continue
                if row["event_type"] in {"persistence", "coarse_persistence"}:
                    previous = active.pop(parent_ids[0], None)
                    if previous:
                        previous["last_frame"] = frame+1
                        next_active[child_ids[0]] = previous
                    continue
                total_events += 1
                overview.event(frame, row['event_type'], total_events-1)
                visible = offset < total_events <= offset+limit
                if offset and total_events == offset+1:
                    states.clear()
                    active = {key: state(frame, key, item) for key, item in parents.items() if item["atoms"] & seed}
                    next_active = {key: state(frame+1, key, children[key]) for key in next_active}
                if not visible:
                    active.clear()
                    if total_events <= offset:
                        states.clear()
                        next_active.clear()
                    continue
                event = dict(id=f"e{frame}-{row['event_id']}", kind="event", frame=frame,
                             child_frame=frame+1, event_id=row["event_id"], event_type=row["event_type"],
                             resolution=row["resolution"], signature=row["reaction_signature"],
                             reactants=json.loads(row["reactant_nodes_json"]), products=json.loads(row["product_nodes_json"]))
                events.append(event)
                largest_input = max((len(parents[key]["atoms"] & seed) for key in parent_ids), default=0)
                for key in parent_ids:
                    node = active.pop(key, None) or state(frame, key, parents[key], not bool(parents[key]["atoms"] & seed))
                    node["last_frame"] = frame
                    links.append(dict(source=node["id"], target=event["id"], tracked=node["tracked"],
                                      role="dominant_input" if node["tracked"] and node["tracked"] == largest_input else "input"))
                weights = {key: len(children[key]["atoms"] & seed) for key in child_ids}
                maximum = max(weights.values(), default=0)
                for key in child_ids:
                    node = state(frame+1, key, children[key], not bool(weights[key]))
                    if weights[key]:
                        next_active[key] = node
                    links.append(dict(source=event["id"], target=node["id"], tracked=weights[key],
                                      role="dominant" if weights[key] and weights[key] == maximum else "branch"))
            parents = children
            active = next_active if total_events <= offset+limit else {}
            progress(dict(frame=frame+1, start=start, end=end, relevant_events=total_events))
        used = {link[side] for link in links for side in ("source", "target")}
        if events:
            states = [node for node in states if node["id"] in used]
        return dict(states=states, events=events, links=links, start=start, end=end, anchor=anchor,
                    compact=compress_page(states, events, links),
                    overview=overview.result(),
                    direction=direction, seed_atoms=len(seed), total_events=total_events, offset=offset,
                    next_offset=offset+limit if total_events > offset+limit else None,
                    missing_atom_frame_occurrences=missing_occurrences,
                    source_metadata={key:value for key,value in metadata(connection).items()
                                     if key in {"configuration_hash","manifest_hash","parser_version","schema_version"}},
                    scope="chronological event page; no artificial edge across omitted pages",
                    weighting="fixed anchor atom count; dominant means largest local outgoing branch, not unique global mechanism")
