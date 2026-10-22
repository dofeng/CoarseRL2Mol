import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dashboard_queries import db_summary, fragment_detail, network, records, timeseries
from fragment_lineage import build_transition_events, match_fragments
from network_postprocess import annotate_events, build_fragment_tracks, build_network_views
from pipeline_runtime import bounded_results, database_lock
from reaxff_fragments import ReaxFFBondFile
from trajectory_pipeline import (
    _configuration_hash, _connect_database, _initialize_manifest, _initialize_schema,
    _load_fragments, _manifest_hash, _prepare_pairs, _process_frame_worker, _process_pair_worker,
    _rebuild_reaction_channels, _set_metadata, _store_frame_result, _store_pair_result, discover_frames,
)
from visualize_frame import write_fragment_scatter
from visualize_transitions import write_transition_svg


ROOT = Path(__file__).parents[1]


def write_bonds(path, edges=(), elements=(1, 1, 1, 1), timestep=0):
    neighbors = {identifier: [] for identifier in range(1, len(elements)+1)}
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    lines = [f"# Timestep {timestep}", f"# Number of particles {len(elements)}",
             "# id type nb id_1...id_nb mol bo_1...bo_nb abo nlp q"]
    for identifier, atom_type in enumerate(elements, start=1):
        bonded = neighbors[identifier]
        fields = [identifier, atom_type, len(bonded), *bonded, 42, *([1.0]*len(bonded)), float(len(bonded)), 0.0, -0.25]
        lines.append(" ".join(map(str, fields)))
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def identity_worker(task):
    return task


class Workspace(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def file(self, name="bond0.reaxff", edges=(), elements=(1, 1, 1, 1)):
        path = self.root / name
        write_bonds(path, edges, elements)
        return path

    def database(self, edge_frames, elements=(1, 1, 1, 1)):
        paths = [self.file(f"bond{index}.reaxff", edges, elements) for index, edges in enumerate(edge_frames)]
        database = self.root / "network.sqlite"
        connection = _connect_database(database)
        _initialize_schema(connection)
        from network_postprocess import ensure_schema
        ensure_schema(connection)
        frames = discover_frames([str(self.root)], "*.reaxff")
        _initialize_manifest(connection, frames, _configuration_hash(.3, {1:"C",2:"H",3:"O"}, {}))
        for index, path in enumerate(paths):
            result = _process_frame_worker(dict(frame_id=index, source_path=str(path), cutoff=.3, element_map={1:"C",2:"H",3:"O"}))
            with connection:
                _store_frame_result(connection, result)
        _prepare_pairs(connection)
        for index in range(len(paths)-1):
            result = _process_pair_worker(dict(database=str(database), parent_frame_id=index, child_frame_id=index+1,
                minimum_shared_atoms=1, minimum_parent_overlap=.05, minimum_child_overlap=.05, minimum_event_conservation=.9))
            with connection:
                _store_pair_result(connection, result)
        return database, connection


class ParserTests(Workspace):
    def test_lower_than_stored_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"stored cutoff"):
            ReaxFFBondFile(ROOT / "data/bond0.reaxff",bond_order_cutoff=.2).parse()

    def test_mol_column_and_zero_neighbor(self):
        source = ReaxFFBondFile(self.file(edges=((1, 2),))).parse()
        self.assertEqual(source.atoms[1].bond_orders, (1.0,))
        self.assertEqual(source.atoms[1].molecule_id, 42)
        self.assertEqual(source.atoms[1].abo, 1.0)
        self.assertEqual(source.atoms[1].charge, -.25)
        self.assertEqual(source.atoms[3].neighbors, ())
        self.assertEqual(source.atoms[3].abo, 0)
        self.assertEqual(source.atoms[3].charge, -.25)

    def test_real_sample_initial_radicals(self):
        source = ReaxFFBondFile(ROOT / "data/bond0.reaxff").parse()
        fragments = source.extract_fragments(include_atom_details=False, include_ring_sizes=False)
        self.assertEqual(sum(fragment.radical_sites for fragment in fragments), 0)
        self.assertEqual(len(fragments), 12)

    def test_ring_count_consistent_between_modes(self):
        source = ReaxFFBondFile(self.file(edges=((1,2),(2,3),(3,4),(4,1),(1,3)))).parse()
        self.assertEqual(source.extract_fragments()[0].ring_count, 2)
        self.assertEqual(source.extract_fragments(include_ring_sizes=False)[0].ring_count, 2)

    def test_multi_frame_file_rejected(self):
        path = self.file()
        path.write_text(path.read_text() + "# Timestep 10\n")
        with self.assertRaisesRegex(ValueError, "Multiple timesteps"):
            ReaxFFBondFile(path).parse()

    def test_duplicate_atom_rejected(self):
        path = self.file()
        path.write_text(path.read_text() + "1 1 0 42 0 0 0\n")
        with self.assertRaisesRegex(ValueError, "duplicate atom"):
            ReaxFFBondFile(path).parse()

    def test_unknown_neighbor_rejected(self):
        path = self.file()
        path.write_text("# Timestep 0\n# Number of particles 1\n1 1 1 999 42 1 1 0 0\n")
        with self.assertRaisesRegex(ValueError, "Unknown neighbor"):
            ReaxFFBondFile(path).parse()

    def test_truncated_and_nonfinite_rows_rejected(self):
        path = self.file()
        for line in ("1 1 1 2 42 1 0 0", "1 1 0 42 nan 0 0"):
            path.write_text("# Timestep 0\n# Number of particles 1\n"+line+"\n")
            with self.assertRaises(ValueError):
                ReaxFFBondFile(path).parse()

    def test_unknown_element_mass_rejected(self):
        path = self.file(elements=(4,))
        with self.assertRaisesRegex(ValueError, "Missing molecular mass"):
            ReaxFFBondFile(path, element_map={4:"N"}).parse()
        source = ReaxFFBondFile(path, element_map={4:"N"}, mass_map={"N":14.007}).parse()
        self.assertAlmostEqual(source.extract_fragments()[0].molecular_mass, 14.007)

    def test_topology_fingerprint_tracks_connectivity_not_bond_noise(self):
        first = ReaxFFBondFile(self.file("bond0.reaxff", ((1,2),(2,3),(3,4)))).parse().extract_fragments()[0]
        second = ReaxFFBondFile(self.file("bond1.reaxff", ((1,2),(1,3),(3,4)))).parse().extract_fragments()[0]
        self.assertEqual(first.node_id, second.node_id)
        self.assertNotEqual(first.topology_hash, second.topology_hash)
        event = build_transition_events([first], [second], match_fragments([first],[second]))[0]
        self.assertEqual(event.event_type, "internal_rearrangement")
        unchanged = build_transition_events([first],[first],match_fragments([first],[first]))[0]
        self.assertEqual(unchanged.event_type, "persistence")
        unknown = replace(first, topology_hash="")
        self.assertEqual(build_transition_events([unknown],[unknown],match_fragments([unknown],[unknown]))[0].event_type, "coarse_persistence")


class LineageAndTrackTests(Workspace):
    def test_reprocessed_pair_invalidates_track_checkpoint(self):
        database,connection=self.database([((1,2),)]*3)
        try:
            expected=build_fragment_tracks(connection)
            annotate_events(connection,"annotate")
            pair=_process_pair_worker(dict(database=str(database),parent_frame_id=0,child_frame_id=1,
                minimum_shared_atoms=1,minimum_parent_overlap=.05,minimum_child_overlap=.05,minimum_event_conservation=.9))
            with connection:
                _store_pair_result(connection,pair)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM postprocess_state WHERE key='tracks_frame'").fetchone()[0],0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event_annotations").fetchone()[0],0)
            self.assertEqual(build_fragment_tracks(connection),expected)
            self.assertEqual(connection.execute("SELECT MAX(frame_count) FROM track_summaries").fetchone()[0],3)
        finally:
            connection.close()

    def test_reprocessed_frame_invalidates_residence_statistics(self):
        database,connection=self.database([((1,2),)]*3)
        try:
            build_fragment_tracks(connection)
            frame=_process_frame_worker(dict(frame_id=1,source_path=str(self.root/'bond1.reaxff'),cutoff=.3,element_map={1:'C',2:'H',3:'O'}))
            with connection:
                _store_frame_result(connection,frame)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM residence_segments").fetchone()[0],0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM fragment_tracks").fetchone()[0],0)
        finally:
            connection.close()

    def test_same_pair_reverse_signatures_are_not_reversals(self):
        first=((1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(8,5))
        second=((1,2),(2,3),(3,4),(4,1),(5,6),(6,7),(7,8))
        database, connection = self.database([first,second], elements=(1,)*8)
        try:
            build_fragment_tracks(connection)
            annotate_events(connection,"annotate",reversal_window_frames=1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event_annotations").fetchone()[0],2)
            self.assertEqual(connection.execute("SELECT SUM(reversal_candidate) FROM event_annotations").fetchone()[0],0)
        finally:
            connection.close()

    def test_unrelated_later_reverse_signature_is_not_reversal(self):
        first=((1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(8,5))
        second=first+((4,1),)
        third=((1,2),(2,3),(3,4),(4,1),(5,6),(6,7),(7,8))
        database, connection = self.database([first,second,third], elements=(1,)*8)
        try:
            build_fragment_tracks(connection)
            annotate_events(connection,"annotate",reversal_window_frames=1)
            self.assertEqual(connection.execute("SELECT SUM(reversal_candidate) FROM event_annotations").fetchone()[0],0)
        finally:
            connection.close()

    def test_track_stage_resumes_after_interruption(self):
        database, connection = self.database([((1,2),)]*4)
        class Interrupt:
            def update(self, **values):
                if values["completed"] == 2:
                    raise RuntimeError("simulated interruption")
        try:
            with self.assertRaisesRegex(RuntimeError,"simulated"):
                build_fragment_tracks(connection,progress=Interrupt())
            resumed=build_fragment_tracks(connection)
            rebuilt=build_fragment_tracks(connection,resume=False)
            self.assertEqual(resumed,rebuilt)
            self.assertEqual(connection.execute("SELECT MAX(frame_count) FROM track_summaries").fetchone()[0],4)
        finally:
            connection.close()

    def test_small_atom_exchange_is_not_dropped(self):
        source = ReaxFFBondFile(self.file(edges=((1,2),(2,3),(3,4)))).parse().extract_fragments()[0]
        parents = [replace(source,fragment_id=1,atom_ids=list(range(100))), replace(source,fragment_id=2,atom_ids=list(range(100,200)))]
        children = [replace(source,fragment_id=1,atom_ids=list(range(1,100))+[100]), replace(source,fragment_id=2,atom_ids=[0]+list(range(101,200)))]
        links = match_fragments(parents,children)
        self.assertEqual(len(links),4)
        self.assertEqual(len(build_transition_events(parents,children,links)),1)

    def test_split_has_unique_instance_tracks(self):
        database, connection = self.database([((1,2),(2,3),(3,4)),((1,2),(3,4))])
        try:
            build_fragment_tracks(connection)
            tracks = connection.execute("SELECT track_id FROM fragment_tracks WHERE frame_id=1").fetchall()
            self.assertEqual(len({row[0] for row in tracks}),2)
            self.assertEqual(connection.execute("SELECT MAX(frame_count) FROM track_summaries").fetchone()[0],1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM track_links").fetchone()[0],2)
        finally:
            connection.close()

    def test_fusion_starts_new_track(self):
        database, connection = self.database([((1,2),(3,4)),((1,2),(2,3),(3,4))])
        try:
            build_fragment_tracks(connection)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM track_summaries").fetchone()[0],3)
            self.assertEqual(connection.execute("SELECT MAX(frame_count) FROM track_summaries").fetchone()[0],1)
        finally:
            connection.close()

    def test_species_residence_and_raw_preservation(self):
        edges = [((1,2),(2,3),(3,4)),((1,2),(2,3),(3,4),(4,1)),((1,2),(2,3),(3,4))]
        database, connection = self.database(edges)
        try:
            build_fragment_tracks(connection)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM track_summaries").fetchone()[0],1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM residence_segments").fetchone()[0],3)
            annotate_events(connection, "exclude", 2, preserve_fast_intermediates=False)
            build_network_views(connection, "exclude")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM network_view_channels WHERE view_name='raw'").fetchone()[0],2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM network_view_channels WHERE view_name='filtered'").fetchone()[0],0)
            annotate_events(connection, "exclude", 2, preserve_fast_intermediates=True)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event_annotations WHERE temporal_status='fast_intermediate_candidate'").fetchone()[0],2)
            annotate_events(connection, "exclude", 2, reversal_window_frames=1, preserve_fast_intermediates=True)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event_annotations WHERE filter_decision='exclude'").fetchone()[0],0)
        finally:
            connection.close()

    def test_censored_boundary_segments_are_not_short_lived(self):
        database, connection = self.database([((1,2),),((1,2),(2,3))])
        try:
            build_fragment_tracks(connection)
            annotate_events(connection,"exclude",10,preserve_fast_intermediates=False)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event_annotations WHERE filter_decision='exclude'").fetchone()[0],0)
        finally:
            connection.close()

    def test_instance_linked_reverse_detected(self):
        database, connection = self.database([((1,2),(2,3),(3,4)),((1,2),(2,3),(3,4),(4,1)),((1,2),(2,3),(3,4))])
        try:
            build_fragment_tracks(connection)
            annotate_events(connection,"annotate",reversal_window_frames=1)
            self.assertEqual(connection.execute("SELECT SUM(reversal_candidate) FROM event_annotations").fetchone()[0],2)
            annotate_events(connection,"off",reversal_window_frames=1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event_annotations").fetchone()[0],0)
        finally:
            connection.close()

    def test_track_restart_does_not_duplicate_rows(self):
        database, connection = self.database([((1,2),),((1,2),),((1,2),)])
        try:
            first = build_fragment_tracks(connection)
            second = build_fragment_tracks(connection)
            self.assertEqual(first,second)
            self.assertEqual(connection.execute("SELECT MAX(frame_count) FROM track_summaries").fetchone()[0],3)
        finally:
            connection.close()


class RuntimeTests(Workspace):
    def test_cli_detects_missing_frames(self):
        self.file("bond0.reaxff")
        self.file("bond2.reaxff")
        process=subprocess.run([sys.executable,"-B",str(ROOT / "src/trajectory_pipeline.py"),str(self.root),
                                "--database",str(self.root / "network.sqlite"),"--expected-frame-step","1"],
                               capture_output=True,text=True,timeout=20)
        self.assertNotEqual(process.returncode,0)
        self.assertIn("Missing, duplicate",process.stderr)

    def test_config_bins_and_threshold_changes_reuse_frames(self):
        self.file("bond0.reaxff",((1,2),(2,3),(3,4)))
        self.file("bond1.reaxff",((1,2),(3,4)))
        config=json.loads((ROOT / "configs/network.json").read_text())
        config["classification"]={"carbon_bins":[["C0",0,0],["C1+",1,None]]}
        config_path=self.root / "config.json"
        config_path.write_text(json.dumps(config))
        database=self.root / "network.sqlite"
        command=[sys.executable,"-B",str(ROOT / "src/trajectory_pipeline.py"),str(self.root),"--config",str(config_path),
                 "--database",str(database),"--workers","1"]
        first=subprocess.run(command,capture_output=True,text=True,timeout=20)
        self.assertEqual(first.returncode,0,first.stderr)
        config["matching"]["minimum_event_conservation"]=.99
        config_path.write_text(json.dumps(config))
        second=subprocess.run(command,capture_output=True,text=True,timeout=20)
        self.assertEqual(second.returncode,0,second.stderr)
        self.assertIn("Frame stage: nothing to process",second.stdout)
        connection=sqlite3.connect(database)
        self.assertTrue(all(row[0].startswith("C1+") for row in connection.execute("SELECT species_id FROM species")))
        connection.close()

    def test_scheduler_applies_backpressure(self):
        submitted = []
        def tasks():
            for value in range(15):
                submitted.append(value)
                yield value
        results = bounded_results(identity_worker,tasks(),workers=1,max_inflight=2)
        task, future = next(results)
        self.assertEqual(len(submitted),2)
        self.assertEqual(future.result(),task)
        time.sleep(.05)
        self.assertEqual(len(submitted),2)
        values = [task]+[future.result() for task,future in results]
        self.assertEqual(sorted(values),list(range(15)))

    def test_scheduler_memory_guard(self):
        with patch("pipeline_runtime.process_tree_rss_mib", return_value=5000):
            with self.assertRaises(MemoryError):
                list(bounded_results(identity_worker,[1,2],1,2,3072))

    def test_database_exclusive_lock(self):
        database = self.root / "network.sqlite"
        with database_lock(database):
            with self.assertRaises(RuntimeError):
                with database_lock(database):
                    pass
        with database_lock(database):
            pass

    def test_legacy_database_rejected_without_migration(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);INSERT INTO metadata VALUES('schema_version','2');")
        with self.assertRaisesRegex(ValueError,"NEW database"):
            _initialize_schema(connection)
        self.assertEqual(connection.execute("SELECT value FROM metadata").fetchone()[0],"2")
        connection.close()

    def test_manifest_changes_when_input_changes(self):
        path = self.file()
        first = _manifest_hash(discover_frames([str(path)],"*.reaxff"))
        path.write_text(path.read_text()+"\n")
        second = _manifest_hash(discover_frames([str(path)],"*.reaxff"))
        self.assertNotEqual(first,second)

    def test_cli_resume_and_worker_determinism(self):
        self.file("bond0.reaxff",((1,2),(2,3),(3,4)))
        self.file("bond1.reaxff",((1,2),(3,4)))
        self.file("bond2.reaxff",((1,2),(2,3),(3,4)))
        snapshots = []
        for workers in (1,2):
            database = self.root / f"workers{workers}/network.sqlite"
            command = [sys.executable,"-B",str(ROOT / "src/trajectory_pipeline.py"),str(self.root),
                       "--database",str(database),"--workers",str(workers),"--expected-frame-step","1", "--build-tracks"]
            for attempt in range(2):
                process = subprocess.run(command,capture_output=True,text=True,timeout=30)
                self.assertEqual(process.returncode,0,process.stderr)
                connection = sqlite3.connect(database)
                rows = connection.execute("SELECT * FROM events ORDER BY parent_frame_id,event_id").fetchall()
                if attempt == 0:
                    snapshots.append(rows)
                else:
                    self.assertEqual(rows,snapshots[-1])
                connection.close()
        self.assertEqual(snapshots[0],snapshots[1])

    def test_cli_partial_failure_has_nonzero_exit(self):
        self.file("bond0.reaxff")
        bad = self.root / "bond1.reaxff"
        bad.write_text("# Timestep 1\n# Number of particles 1\nmalformed\n")
        database = self.root / "network.sqlite"
        command = [sys.executable,"-B",str(ROOT / "src/trajectory_pipeline.py"),str(self.root),"--database",str(database),"--workers","1"]
        process = subprocess.run(command,capture_output=True,text=True,timeout=30)
        self.assertEqual(process.returncode,2,process.stderr)
        summary = json.loads((self.root / "network_summary.json").read_text())
        self.assertEqual(summary["status"],"partial")
        self.assertEqual(summary["unfinished_pairs"],1)


class QueryAndVisualTests(Workspace):
    def test_single_pair_outputs_include_stoichiometry(self):
        from fragment_lineage import compare_bonds_files, write_transition_outputs
        first=self.file("bond0.reaxff",((1,2),(2,3),(3,4)))
        second=self.file("bond1.reaxff",((1,2),(3,4)))
        result=compare_bonds_files(first,second)
        paths=write_transition_outputs(result,self.root)
        self.assertTrue(all(path.is_file() for path in paths))
        write_transition_svg(result,self.root / "network.svg")

    def prepared(self):
        database,connection = self.database([((1,2),(2,3),(3,4)),((1,2),(3,4)),((1,2),(2,3),(3,4))])
        build_network_views(connection)
        _rebuild_reaction_channels(connection)
        with connection:
            _set_metadata(connection,"views_ready","true")
        connection.close()
        return database

    def test_network_pagination_and_time_scope(self):
        database = self.prepared()
        first = network(database,{"limit":"1"})
        second = network(database,{"limit":"1","cursor":first["next_cursor"]})
        self.assertEqual(first["total_channels"],2)
        self.assertNotEqual(first["channels"][0]["reaction_signature"],second["channels"][0]["reaction_signature"])
        self.assertIsNone(second["next_cursor"])
        subset = network(database,{"start":"0","end":"1"})
        self.assertEqual(subset["total_channels"],1)

    def test_events_and_fragments_have_real_pagination(self):
        database=self.prepared()
        first=records(database,"events",{"limit":"1"})
        self.assertTrue(first["has_more"])
        second=records(database,"events",{"limit":"1","cursor":first["next_cursor"]})
        self.assertFalse(second["has_more"])
        fragments=records(database,"fragments",{"start":"1","end":"1","limit":"1"})
        self.assertTrue(fragments["has_more"])
        self.assertNotIn("atom_ids",fragments["rows"][0])
        detail=fragment_detail(database,{"frame":"1","fragment":"1","limit":"1"})
        self.assertEqual(len(detail["atom_ids"]),1)
        self.assertEqual(detail["next_offset"],1)

    def test_empty_species_states_are_zero_in_series(self):
        database=self.prepared()
        data=timeseries(database,{"species_id":"absent","points":"10"})
        self.assertEqual(len(data["rows"]),3)
        self.assertTrue(all(row["population"] == 0 for row in data["rows"]))

    def test_svg_preserves_self_participation_and_coefficients(self):
        path=self.root / "network.svg"
        write_transition_svg({"aggregated_reactions":[{"reaction_signature":"A -> A + 2B","event_count":3,
                              "reactant_nodes":{"A":1},"product_nodes":{"A":1,"B":2}}]},path)
        text=path.read_text()
        self.assertIn("PRODUCTS",text)
        self.assertIn("REACTANTS",text)
        self.assertIn(">2</text>",text)
        self.assertEqual(text.count("marker-end"),3)
        self.assertNotIn("<line ",text)

    def test_scatter_handles_zero_radicals(self):
        write_fragment_scatter({"fragments":[{"fragment_id":1,"node_id":"C1-5_R0_O0","molecular_mass":12,"radical_sites":0}]},self.root / "scatter.svg")
        self.assertTrue((self.root / "scatter.svg").is_file())


if __name__ == "__main__":
    unittest.main()
