"""Bounded C/R/O network projections and classification diagnostics."""

import json
import math

from dashboard_queries import channel_details, db_connect, decoded_row, event_filters, integer, metadata, table_exists
from pseudo_species import ThreeDimensionalClassifier


def _filters(query):
    return event_filters(query)


def coarse_network(path, query):
    weight = query.get("weight", "carbon")
    if weight not in {"events", "carbon", "oxygen", "mass"}:
        raise ValueError("weight must be events, carbon, oxygen or mass")
    coverage = float(query.get("coverage", .9))
    if not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0,1]")
    limit = integer(query, "limit", 40, 1, 200)
    minimum = integer(query, "minimum_count", 1, 1)
    include_internal = query.get("include_internal", "true") == "true"
    where, params = _filters(query)
    temporal = "NULL" if query.get("view", "raw") == "raw" else "(SELECT temporal_status FROM event_annotations a WHERE a.parent_frame_id=e.parent_frame_id AND a.child_frame_id=e.child_frame_id AND a.event_id=e.event_id)"
    prefix = f"""WITH eligible AS (SELECT e.parent_frame_id,e.child_frame_id,e.event_id,
        e.reaction_signature,e.event_type,e.resolution,e.confidence,e.atom_conservation_ratio,
        e.mass_error,e.reactant_nodes_json,e.product_nodes_json,{temporal} temporal_status FROM events e WHERE {where}),
        amounts AS (SELECT f.parent_frame_id,f.child_frame_id,f.event_id,
            SUM(CASE WHEN source_species!=target_species THEN carbon ELSE 0 END) cross_carbon,
            SUM(CASE WHEN source_species!=target_species THEN oxygen ELSE 0 END) cross_oxygen,
            SUM(CASE WHEN source_species!=target_species THEN mass ELSE 0 END) cross_mass,
            SUM(CASE WHEN source_species=target_species THEN carbon ELSE 0 END) internal_carbon,
            SUM(CASE WHEN source_species=target_species THEN oxygen ELSE 0 END) internal_oxygen,
            SUM(CASE WHEN source_species=target_species THEN mass ELSE 0 END) internal_mass
            FROM coarse_event_flows f JOIN eligible USING(parent_frame_id,child_frame_id,event_id)
            GROUP BY f.parent_frame_id,f.child_frame_id,f.event_id),
        types AS (SELECT reaction_signature,json_group_object(event_type,amount) event_type_counts_json
            FROM (SELECT reaction_signature,event_type,COUNT(*) amount FROM eligible GROUP BY reaction_signature,event_type)
            GROUP BY reaction_signature),
        channels AS (SELECT e.reaction_signature,e.reactant_nodes_json,e.product_nodes_json,
            CASE WHEN COUNT(DISTINCT e.event_type)=1 THEN MIN(e.event_type) ELSE 'mixed' END event_type,
            COUNT(*) event_count,SUM(e.resolution!='resolved') uncertain_events,
            SUM(CASE WHEN e.temporal_status IN ('short_lived_candidate','fast_intermediate_candidate','reversal_candidate') THEN 1 ELSE 0 END) flagged_events,
            MIN(e.parent_frame_id) first_frame,MAX(e.child_frame_id) last_frame,
            AVG(e.confidence) mean_confidence,AVG(e.atom_conservation_ratio) mean_atom_conservation_ratio,
            AVG(e.mass_error) mean_mass_error,
            SUM(COALESCE(a.cross_carbon,0)+{int(include_internal)}*COALESCE(a.internal_carbon,0)) carbon,
            SUM(COALESCE(a.cross_oxygen,0)+{int(include_internal)}*COALESCE(a.internal_oxygen,0)) oxygen,
            SUM(COALESCE(a.cross_mass,0)+{int(include_internal)}*COALESCE(a.internal_mass,0)) mass,
            SUM(COALESCE(a.cross_carbon,0)) cross_carbon,SUM(COALESCE(a.cross_oxygen,0)) cross_oxygen,
            SUM(COALESCE(a.cross_mass,0)) cross_mass,SUM(COALESCE(a.internal_carbon,0)) internal_carbon,
            SUM(COALESCE(a.internal_oxygen,0)) internal_oxygen,SUM(COALESCE(a.internal_mass,0)) internal_mass,
            t.event_type_counts_json FROM eligible e
            LEFT JOIN amounts a USING(parent_frame_id,child_frame_id,event_id)
            JOIN types t USING(reaction_signature)
            GROUP BY e.reaction_signature) """
    metric = "event_count" if weight == "events" else weight
    with db_connect(path) as connection:
        meta = metadata(connection)
        if meta.get("schema_version") != "3":
            raise ValueError("Legacy database: rebuild with corrected parser in a new database")
        if meta.get("coarse_ready") != "true" or meta.get("views_ready") != "true":
            return {"ready": False, "channels": [], "edges": [], "nodes": [],
                    "message": "C/R/O 物质流尚未构建。对原输入和 v3 数据库运行 --stage postprocess；未完成的轨迹请先续算。"}
        connection.execute("CREATE TEMP TABLE selected_channel_totals AS " + prefix + "SELECT * FROM channels", params)
        totals = connection.execute(f"""SELECT COUNT(*) channels,COALESCE(SUM(event_count),0) events,
            COALESCE(SUM({metric}),0) weight,COALESCE(SUM(cross_carbon),0) cross_carbon,
            COALESCE(SUM(internal_carbon),0) internal_carbon,COALESCE(SUM(cross_oxygen),0) cross_oxygen,
            COALESCE(SUM(internal_oxygen),0) internal_oxygen,COALESCE(SUM(cross_mass),0) cross_mass,
            COALESCE(SUM(internal_mass),0) internal_mass,
            COALESCE(SUM(CASE WHEN {metric}=0 THEN event_count ELSE 0 END),0) zero_weight_events
            FROM selected_channel_totals""").fetchone()
        selected = []
        displayed = 0
        rows = connection.execute(f"SELECT * FROM selected_channel_totals WHERE event_count>=? AND {metric}>0 ORDER BY {metric} DESC,reaction_signature LIMIT ?",
                                  (minimum, limit))
        for row in rows:
            selected.append(channel_details(decoded_row(row)))
            displayed += row[metric]
            if displayed >= coverage * totals["weight"]:
                break
        edges = []
        if selected:
            placeholders = ",".join("?" for channel in selected)
            projection = "COUNT(*)" if weight == "events" else f"SUM(f.{weight})"
            internal_clause = "" if include_internal else "AND f.source_species!=f.target_species"
            edges = [dict(row) for row in connection.execute(f"""SELECT f.source_species source,f.target_species target,
                COUNT(*) event_count,SUM(f.carbon) carbon,SUM(f.oxygen) oxygen,SUM(f.mass) mass,
                {projection} weight FROM coarse_event_flows f
                JOIN events e USING(parent_frame_id,child_frame_id,event_id)
                WHERE {where} AND e.reaction_signature IN ({placeholders}) {internal_clause}
                GROUP BY f.source_species,f.target_species HAVING {projection}>0
                ORDER BY weight DESC,source,target LIMIT 4001""", (*params, *(channel["reaction_signature"] for channel in selected)))]
        truncated = len(edges) > 4000
        edges = edges[:4000]
        node_ids = {species for channel in selected for side in ("reactant_nodes", "product_nodes") for species in channel[side]}
        nodes = [dict(row) for row in connection.execute("SELECT * FROM species ORDER BY species_id") if row["species_id"] in node_ids]
        return dict(ready=True, channels=selected, edges=edges, nodes=nodes, weight=weight,
                    total_channels=totals["channels"], total_events=totals["events"], total_weight=totals["weight"],
                    displayed_events=sum(channel["event_count"] for channel in selected), displayed_weight=displayed,
                    achieved_coverage=displayed/totals["weight"] if totals["weight"] else 0,
                    requested_coverage=coverage, target_reached=bool(totals["weight"] and displayed >= coverage*totals["weight"]),
                    edges_truncated=truncated, include_internal=include_internal,
                    flow_totals={key:totals[key] for key in ("cross_carbon", "internal_carbon", "cross_oxygen", "internal_oxygen", "cross_mass", "internal_mass")},
                    zero_weight_events=totals["zero_weight_events"],
                    scope="current time/view/participant/type filters before minimum_count and display limit",
                    warning="累计反应转移可重复计入同一原子，不是产率或速率；流边不是独立反应式。")


def diagnostics(path, query):
    start = integer(query, "start", 0)
    end = integer(query, "end", 2**63-1)
    if end < start:
        raise ValueError("Frame range is reversed")
    with db_connect(path) as connection:
        meta = metadata(connection)
        if meta.get("schema_version") != "3":
            raise ValueError("Legacy database: rebuild with corrected parser")
        classifier = ThreeDimensionalClassifier(**json.loads(meta.get("classification", "{}")))
        frame_count = connection.execute("SELECT COUNT(*) FROM frames WHERE status='completed' AND frame_id BETWEEN ? AND ?", (start, end)).fetchone()[0]
        rows = connection.execute("""WITH totals AS (
            SELECT frame_id,SUM(total_mass) mass,SUM(total_carbon) carbon,SUM(total_oxygen) oxygen
            FROM species_states JOIN frames USING(frame_id)
            WHERE frames.status='completed' AND frame_id BETWEEN ? AND ? GROUP BY frame_id)
            SELECT s.species_id,SUM(s.population) instances,MAX(s.population) peak_population,
            SUM(s.population)*1.0/? mean_population,SUM(s.total_mass)/? mean_mass,
            SUM(COALESCE(s.total_mass/NULLIF(t.mass,0),0))/? mean_mass_fraction,
            SUM(COALESCE(s.total_carbon*1.0/NULLIF(t.carbon,0),0))/? mean_carbon_fraction,
            SUM(COALESCE(s.total_oxygen*1.0/NULLIF(t.oxygen,0),0))/? mean_oxygen_fraction
            FROM species_states s JOIN totals t USING(frame_id)
            GROUP BY s.species_id ORDER BY mean_mass_fraction DESC,s.species_id LIMIT 1001""",
            (start, end, *([max(1, frame_count)]*5))).fetchall()
        species = [dict(row) for row in rows[:1000]]
        audit = None
        if table_exists(connection, "coarse_pair_audits"):
            audit = dict(connection.execute("""SELECT COUNT(*) audited_pairs,
                COALESCE(SUM(unbalanced_events),0) unbalanced_events,
                COALESCE(SUM(missing_atoms),0) unmatched_atom_occurrences
                FROM coarse_pair_audits WHERE parent_frame_id>=? AND child_frame_id<=?""", (start, end)).fetchone())
        activity = [dict(row) for row in connection.execute("""SELECT event_type,COUNT(*) event_count,
            SUM(reactant_nodes_json=product_nodes_json) same_category_stoichiometry
            FROM events JOIN frame_pairs USING(parent_frame_id,child_frame_id)
            WHERE frame_pairs.status='completed' AND parent_frame_id>=? AND child_frame_id<=?
            AND event_type NOT IN ('persistence','coarse_persistence') GROUP BY event_type""", (start, end))]
        all_events = connection.execute("""SELECT COALESCE(SUM(event_count),0) FROM frame_pairs
            WHERE status='completed' AND parent_frame_id>=? AND child_frame_id<=?""", (start, end)).fetchone()[0]
        activity.append(dict(event_type="persistence / coarse_persistence",
                             event_count=all_events-sum(row["event_count"] for row in activity),
                             same_category_stoichiometry=0))
        reaction_events = sum(row["event_count"] for row in activity[:-1])
        internal_events = sum(row["same_category_stoichiometry"] for row in activity[:-1])
        largest = "_".join(bins[-1][0] for bins in (classifier.carbon_bins, classifier.ring_bins, classifier.oxygen_bins))
        largest_state = next((row for row in species if row["species_id"] == largest), None)
        return dict(frame_count=frame_count, species=species, species_truncated=len(rows)>1000,
                    bins=dict(carbon=classifier.carbon_bins, ring=classifier.ring_bins, oxygen=classifier.oxygen_bins),
                    largest_category=largest, largest_category_state=largest_state, activity=activity, audit=audit,
                    coarse_ready=meta.get("coarse_ready")=="true", views_ready=meta.get("views_ready")=="true",
                    event_totals=dict(observed=all_events, reactions=reaction_events,
                        population_changes=reaction_events-internal_events, internal_activity=internal_events,
                        persistence=all_events-reaction_events),
                    definition="C=carbon count; R=E-V+1 for a connected fragment; O=oxygen count",
                    warning="类别数量少并不代表反应少；检查最大区间占用率与类别不变事件。")
