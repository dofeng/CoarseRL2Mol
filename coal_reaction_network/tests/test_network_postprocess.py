import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from network_postprocess import (
    annotate_events,
    build_fragment_tracks,
    build_network_views,
    ensure_schema,
)


def test_postprocess_keeps_raw_events_when_excluding_short_tracks():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
    CREATE TABLE frames(frame_id INTEGER PRIMARY KEY, timestep INTEGER, status TEXT);
    CREATE TABLE fragments(frame_id INTEGER, fragment_id INTEGER, species_id TEXT);
    CREATE TABLE frame_pairs(parent_frame_id INTEGER, child_frame_id INTEGER, status TEXT);
    CREATE TABLE lineage_links(parent_frame_id INTEGER, child_frame_id INTEGER,
        parent_fragment_id INTEGER, child_fragment_id INTEGER, shared_atom_count INTEGER);
    CREATE TABLE events(parent_frame_id INTEGER, child_frame_id INTEGER, event_id INTEGER,
        event_type TEXT, reaction_signature TEXT, reactant_nodes_json TEXT,
        product_nodes_json TEXT, parent_fragment_ids_json TEXT, child_fragment_ids_json TEXT,
        atom_conservation_ratio REAL, confidence REAL, mass_error REAL, resolution TEXT);
    """)
    connection.executemany("INSERT INTO frames VALUES (?, ?, 'completed')", [(0, 0), (1, 1), (2, 2)])
    connection.executemany("INSERT INTO fragments VALUES (?, ?, ?)", [(0, 1, "A"), (1, 1, "B"), (2, 1, "C")])
    connection.execute("INSERT INTO frame_pairs VALUES (0, 1, 'completed')")
    connection.execute("INSERT INTO frame_pairs VALUES (1, 2, 'completed')")
    connection.execute("INSERT INTO lineage_links VALUES (0, 1, 1, 1, 1)")
    connection.execute("INSERT INTO lineage_links VALUES (1, 2, 1, 1, 1)")
    connection.execute("""INSERT INTO events VALUES
        (0, 1, 1, 'transformation', 'A -> B', '{"A":1}', '{"B":1}', '[1]', '[1]', 0.5, 1.0, 0.0, 'composite')
    """)
    connection.execute("""INSERT INTO events VALUES
        (1, 2, 1, 'transformation', 'B -> C', '{"B":1}', '{"C":1}', '[1]', '[1]', 0.5, 1.0, 0.0, 'composite')
    """)
    ensure_schema(connection)
    build_fragment_tracks(connection)
    annotate_events(connection, mode="exclude", minimum_lifetime_frames=4)
    build_network_views(connection, mode="exclude")
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM network_view_channels WHERE view_name='raw'").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM network_view_channels WHERE view_name='filtered'").fetchone()[0] == 0
    connection.close()
