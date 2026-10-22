"""Resumable, pair-bounded atom-flow accounting and population audits."""

from collections import Counter, defaultdict
import hashlib
import json

from reaxff_fragments import ReaxFFBondFile


VERSION = "1"


def ensure_schema(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS coarse_event_flows (
            parent_frame_id INTEGER NOT NULL, child_frame_id INTEGER NOT NULL,
            event_id INTEGER NOT NULL, source_species TEXT NOT NULL,
            target_species TEXT NOT NULL, atoms INTEGER NOT NULL,
            carbon INTEGER NOT NULL, oxygen INTEGER NOT NULL, mass REAL NOT NULL,
            PRIMARY KEY(parent_frame_id,child_frame_id,event_id,source_species,target_species)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS coarse_pair_audits (
            parent_frame_id INTEGER NOT NULL, child_frame_id INTEGER NOT NULL,
            version TEXT NOT NULL, population_errors_json TEXT NOT NULL,
            unbalanced_events INTEGER NOT NULL, missing_atoms INTEGER NOT NULL,
            flow_rows INTEGER NOT NULL,
            PRIMARY KEY(parent_frame_id,child_frame_id)
        ) WITHOUT ROWID;
    """)


def invalidate_pair(connection, parent, child):
    for table in ("coarse_event_flows", "coarse_pair_audits"):
        connection.execute(f"DELETE FROM {table} WHERE parent_frame_id=? AND child_frame_id=?", (parent, child))
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('coarse_ready','false')")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('views_ready','false')")


def event_flows(event, parents, children, inventory):
    """Count each shared atom once; untouched persistence is not reaction flow."""
    if event["event_type"] in {"persistence", "coarse_persistence"}:
        return []
    destinations = {}
    for identifier in json.loads(event["child_fragment_ids_json"]):
        fragment = children[identifier]
        for atom in fragment.atom_ids:
            if atom in destinations:
                raise ValueError("Atom belongs to multiple product fragments")
            destinations[atom] = fragment.node_id
    grouped = defaultdict(lambda: [0, 0, 0, 0.0])
    for identifier in json.loads(event["parent_fragment_ids_json"]):
        fragment = parents[identifier]
        for atom in fragment.atom_ids:
            target = destinations.get(atom)
            if target is None:
                continue
            element, mass = inventory[atom]
            values = grouped[(fragment.node_id, target)]
            values[0] += 1
            values[1] += element == "C"
            values[2] += element == "O"
            values[3] += mass
    return [(source, target, *values) for (source, target), values in sorted(grouped.items())]


def build_analytics(connection, element_map, mass_map=None, cutoff=.3, progress=None):
    from trajectory_pipeline import _load_fragments

    ensure_schema(connection)
    with connection:
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('coarse_ready','false')")
    if connection.execute("SELECT COUNT(*) FROM frames WHERE status!='completed'").fetchone()[0] or connection.execute(
        "SELECT COUNT(*) FROM frame_pairs WHERE status!='completed'"
    ).fetchone()[0]:
        raise ValueError("Coarse analytics requires all frames and pairs to be completed")
    total = connection.execute("SELECT COUNT(*) FROM frame_pairs").fetchone()[0]
    expected_pairs = max(0, connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]-1)
    if total != expected_pairs:
        raise ValueError("Coarse analytics requires a complete adjacent-pair manifest")
    inventories = {}
    cached_frame = None
    cached_fragments = None

    def inventory_for(frame_id):
        frame = connection.execute("SELECT * FROM frames WHERE frame_id=?", (frame_id,)).fetchone()
        signature = frame["atom_signature"]
        if signature and signature in inventories:
            return inventories[signature]
        source = ReaxFFBondFile(frame["source_path"], element_map=element_map,
                                mass_map=mass_map, bond_order_cutoff=cutoff).parse()
        actual = hashlib.sha256(json.dumps([(atom, source.atoms[atom].element)
                                           for atom in sorted(source.atoms)]).encode()).hexdigest()
        if signature != actual:
            raise ValueError("Source atom inventory differs from parsed frame; rebuild in a new database")
        inventory = {atom: (record.element, source.mass_map[record.element]) for atom, record in source.atoms.items()}
        if len(inventories) >= 2:
            inventories.pop(next(iter(inventories)))
        inventories[signature] = inventory
        return inventory

    pairs = connection.execute("""SELECT p.parent_frame_id,p.child_frame_id FROM frame_pairs p
        LEFT JOIN coarse_pair_audits a USING(parent_frame_id,child_frame_id)
        WHERE a.version IS NULL OR a.version!=? ORDER BY p.parent_frame_id""", (VERSION,))
    completed = connection.execute("SELECT COUNT(*) FROM coarse_pair_audits WHERE version=?", (VERSION,)).fetchone()[0]
    for parent, child in pairs:
        parents = cached_fragments if cached_frame == parent else {item.fragment_id: item for item in _load_fragments(connection, parent)}
        children = {item.fragment_id: item for item in _load_fragments(connection, child)}
        inventory = inventory_for(parent)
        child_inventory = inventory_for(child)
        if any(inventory[atom] != child_inventory[atom] for atom in inventory.keys() & child_inventory.keys()):
            raise ValueError("An atom ID changes element or mass across a frame pair")
        expected = Counter(item.node_id for item in children.values())
        expected.subtract(item.node_id for item in parents.values())
        observed = Counter()
        seen_parents, seen_children = set(), set()
        unbalanced = missing = flow_count = 0
        with connection:
            invalidate_pair(connection, parent, child)
            events = connection.execute("SELECT * FROM events WHERE parent_frame_id=? AND child_frame_id=? ORDER BY event_id", (parent, child))
            for event in events:
                for field, fragments, seen, side in (
                    ("parent_fragment_ids_json", parents, seen_parents, "reactant_nodes_json"),
                    ("child_fragment_ids_json", children, seen_children, "product_nodes_json"),
                ):
                    identifiers = json.loads(event[field])
                    if any(identifier in seen for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
                        raise ValueError("Fragment counted in multiple events")
                    seen.update(identifiers)
                    if Counter(fragments[identifier].node_id for identifier in identifiers) != json.loads(event[side]):
                        raise ValueError("Population balance failed: event coefficients differ from fragment instances")
                observed.update(json.loads(event["product_nodes_json"]))
                observed.subtract(json.loads(event["reactant_nodes_json"]))
                unbalanced += any(value != 0 for value in json.loads(event["element_balance_json"]).values())
                missing += event["parent_atom_count"] + event["child_atom_count"] - 2 * event["shared_atom_count"]
                flows = event_flows(event, parents, children, inventory)
                if flows and sum(row[2] for row in flows) != event["shared_atom_count"]:
                    raise ValueError("Shared atom flow does not match event accounting")
                connection.executemany("INSERT INTO coarse_event_flows VALUES (?,?,?,?,?,?,?,?,?)",
                                       ((parent, child, event["event_id"], *row) for row in flows))
                flow_count += len(flows)
            if seen_parents != parents.keys() or seen_children != children.keys():
                raise ValueError("Not every fragment is represented exactly once in frame-pair events")
            errors = {species: {"expected": expected[species], "observed": observed[species]}
                      for species in expected.keys() | observed.keys() if expected[species] != observed[species]}
            if errors:
                raise ValueError(f"Population balance failed for pair {parent}->{child}: {errors}")
            connection.execute("INSERT INTO coarse_pair_audits VALUES (?,?,?,?,?,?,?)",
                               (parent, child, VERSION, json.dumps(errors), unbalanced, missing, flow_count))
        cached_frame, cached_fragments = child, children
        completed += 1
        if progress:
            progress.update(stage="coarse_flows", completed=completed, total=total)
    with connection:
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('coarse_ready','true')")
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('coarse_version',?)", (VERSION,))
    return dict(connection.execute("""SELECT 'audited_pairs',COUNT(*) FROM coarse_pair_audits
        UNION ALL SELECT 'unbalanced_events',COALESCE(SUM(unbalanced_events),0) FROM coarse_pair_audits
        UNION ALL SELECT 'unmatched_atom_occurrences',COALESCE(SUM(missing_atoms),0) FROM coarse_pair_audits
        UNION ALL SELECT 'flow_rows',COALESCE(SUM(flow_rows),0) FROM coarse_pair_audits"""))
