import csv

from coarse_network.build import build_analytics
from coarse_network.export import export_analytics
from coarse_network.queries import coarse_network, diagnostics
from dashboard_queries import channel_details, db_summary, network, records, timeseries
from network_postprocess import annotate_events, build_network_views
from test_regressions import Workspace
from trajectory_pipeline import _process_pair_worker, _rebuild_reaction_channels, _store_pair_result


class CoarseNetworkTests(Workspace):
    def build(self, edge_frames, elements=(1, 1, 1, 1)):
        path, connection = self.database(edge_frames, elements)
        annotate_events(connection, mode="off")
        build_network_views(connection, mode="off")
        _rebuild_reaction_channels(connection)
        report = build_analytics(connection, {1: "C", 2: "H", 3: "O"})
        with connection:
            connection.execute("INSERT OR REPLACE INTO metadata VALUES ('views_ready','true')")
        return path, connection, report

    def test_split_conserves_elements_and_same_category_coefficients(self):
        path, connection, report = self.build([[(1, 2), (2, 3), (3, 4)], [(1, 2), (3, 4)]], (1, 1, 3, 2))
        self.assertEqual(report["audited_pairs"], 1)
        totals = connection.execute("SELECT SUM(atoms),SUM(carbon),SUM(oxygen),SUM(mass) FROM coarse_event_flows").fetchone()
        self.assertEqual(tuple(totals[:3]), (4, 2, 1))
        self.assertAlmostEqual(totals[3], 2*12.011+15.999+1.008)
        data = coarse_network(path, {"coverage": "1"})
        self.assertEqual(data["total_events"], 1)
        self.assertEqual(data["total_weight"], 2)
        self.assertEqual(sum(data["channels"][0]["product_nodes"].values()), 2)
        self.assertEqual(report["unbalanced_events"], 0)
        connection.close()

    def test_persistence_is_not_flow(self):
        path, connection, report = self.build([[(1, 2)], [(1, 2)]])
        self.assertEqual(report["flow_rows"], 0)
        self.assertEqual(coarse_network(path, {})["total_events"], 0)
        connection.close()

    def activity_fixture(self):
        chain=[(atom,atom+1) for atom in range(1,7)]
        split=[edge for edge in chain if edge!=(3,4)]
        rewired=[(1,2),(2,3),(4,5),(4,6),(4,7)]
        return self.build([chain,split,rewired,rewired], (1,)*7+(3,))

    def test_activity_filters_agree_across_channels_flows_and_evidence(self):
        path, connection, report = self.activity_fixture()
        for activity, count in (("all",2),("changes",1),("internal",1)):
            query={"activity":activity,"weight":"events","coverage":"1"}
            self.assertEqual(network(path,query)["total_events"],count)
            self.assertEqual(coarse_network(path,query)["total_events"],count)
            self.assertEqual(len(records(path,"events",query)["rows"]),count)
        connection.close()

    def test_cross_and_internal_flow_totals_are_separate(self):
        path, connection, report = self.activity_fixture()
        cross=coarse_network(path,{"include_internal":"false","coverage":"1"})
        full=coarse_network(path,{"include_internal":"true","coverage":"1"})
        self.assertEqual(cross["flow_totals"],full["flow_totals"])
        self.assertEqual(cross["flow_totals"]["cross_carbon"],7)
        self.assertEqual(cross["flow_totals"]["internal_carbon"],4)
        self.assertEqual(cross["total_weight"],7)
        self.assertEqual(full["total_weight"],11)
        self.assertEqual(cross["total_events"],2)
        self.assertEqual(cross["zero_weight_events"],1)
        for include in ("false","true"):
            self.assertEqual(coarse_network(path,{"weight":"events","include_internal":include})["total_weight"],2)
        connection.close()

    def test_summary_and_diagnostics_do_not_count_persistence_as_reactions(self):
        path, connection, report = self.activity_fixture()
        summary=db_summary(path)
        totals=diagnostics(path,{})["event_totals"]
        self.assertEqual(summary["reaction_events"],2)
        self.assertGreater(summary["persistence_events"],0)
        self.assertEqual(summary["events"],summary["reaction_events"]+summary["persistence_events"])
        self.assertEqual(totals["reactions"],2)
        self.assertEqual(totals["population_changes"],1)
        self.assertEqual(totals["internal_activity"],1)
        self.assertEqual(totals["observed"],totals["reactions"]+totals["persistence"])
        connection.close()

    def test_combined_category_filters_match_one_participant(self):
        path, connection, report = self.activity_fixture()
        for query, count in (({"carbon_bin":"C6-20","oxygen_bin":"O0"},1),
                             ({"carbon_bin":"C6-20","oxygen_bin":"O1"},0),
                             ({"species":"C1-5_R0_O0","carbon_bin":"C6-20"},0)):
            self.assertEqual(network(path,query)["total_events"],count)
            self.assertEqual(coarse_network(path,{**query,"weight":"events"})["total_events"],count)
            self.assertEqual(len(records(path,"events",query)["rows"]),count)
        connection.close()

    def test_cached_network_reports_actual_frame_ranges(self):
        path, connection, report = self.activity_fixture()
        cached=network(path,{})
        scoped=network(path,{"end":"2"})
        self.assertEqual({row["reaction_signature"]:(row["first_frame"],row["last_frame"]) for row in cached["channels"]},
                         {row["reaction_signature"]:(row["first_frame"],row["last_frame"]) for row in scoped["channels"]})
        self.assertTrue(all(row["event_type_counts"] for row in cached["channels"]))
        connection.close()

    def test_net_coefficients_preserve_both_sides(self):
        row=channel_details({"reactant_nodes":{"A":1},"product_nodes":{"A":1,"B":2}})
        self.assertEqual(row["net_coefficients"],{"B":2})
        self.assertEqual(row["reactant_nodes"],{"A":1})
        self.assertEqual(row["product_nodes"],{"A":1,"B":2})
        self.assertEqual(row["activity"],"changes")
        self.assertEqual(channel_details({"reactant_nodes":{"A":2},"product_nodes":{"A":2}})["activity"],"internal")

    def test_diagnostic_occupancy_uses_only_requested_frames(self):
        path, connection, report = self.activity_fixture()
        result=diagnostics(path,{"start":"0","end":"0"})
        self.assertEqual(result["frame_count"],1)
        self.assertEqual({row["species_id"] for row in result["species"]},{"C6-20_R0_O0","C0_R0_O1"})
        self.assertAlmostEqual(sum(row["mean_mass_fraction"] for row in result["species"]),1)
        connection.close()

    def test_same_category_exchange_is_retained(self):
        path, connection, report = self.build([[(1, 2), (3, 4)], [(1, 3), (2, 4)]])
        data = coarse_network(path, {})
        channel = data["channels"][0]
        self.assertEqual(channel["event_type"], "many_to_many")
        self.assertEqual(channel["reactant_nodes"], channel["product_nodes"])
        self.assertEqual(channel["carbon"], 4)
        self.assertEqual(data["edges"][0]["event_count"], 1)
        self.assertEqual(diagnostics(path, {})["activity"][0]["same_category_stoichiometry"], 1)
        connection.close()

    def test_internal_rewiring_is_not_persistence(self):
        path, connection, report = self.build([[(1, 2), (2, 3), (3, 4)], [(1, 2), (1, 3), (1, 4)]])
        data = coarse_network(path, {})
        self.assertEqual(data["channels"][0]["event_type"], "internal_rearrangement")
        self.assertEqual(data["total_weight"], 4)
        connection.close()

    def test_bin_filter_and_source_target_evidence(self):
        from dashboard_queries import records
        path, connection, report = self.build([[(1, 2)], []], (1, 3, 2, 2))
        self.assertEqual(coarse_network(path, {"carbon_bin":"not-a-bin"})["total_events"], 0)
        data = coarse_network(path, {"oxygen_bin":"O1", "coverage":"1"})
        self.assertEqual(data["total_events"], 1)
        edge = data["edges"][0]
        evidence = records(path, "events", {"flow_source":edge["source"], "flow_target":edge["target"]})
        self.assertEqual(len(evidence["rows"]), 1)
        connection.close()

    def test_resuming_does_not_double_count(self):
        path, connection, report = self.build([[(1, 2)], [], [(1, 2)]])
        second = build_analytics(connection, {1: "C", 2: "H", 3: "O"})
        self.assertEqual(second, report)
        self.assertEqual(coarse_network(path, {"weight": "events", "coverage": "1"})["total_events"], 2)
        connection.close()

    def test_pair_replacement_invalidates_derived_rows(self):
        path, connection, report = self.build([[(1, 2)], []])
        result = _process_pair_worker(dict(database=str(path), parent_frame_id=0, child_frame_id=1,
            minimum_shared_atoms=1, minimum_parent_overlap=.05, minimum_child_overlap=.05,
            minimum_event_conservation=.9))
        with connection:
            _store_pair_result(connection, result)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM coarse_pair_audits").fetchone()[0], 0)
        self.assertFalse(coarse_network(path, {})["ready"])
        self.assertEqual(build_analytics(connection, {1:"C", 2:"H", 3:"O"}), report)
        connection.close()

    def test_population_corruption_prevents_publication(self):
        path, connection = self.database([[(1, 2)], []])
        with connection:
            connection.execute("UPDATE events SET product_nodes_json='{}' WHERE event_type='fragmentation'")
        with self.assertRaisesRegex(ValueError, "Population balance"):
            build_analytics(connection, {1:"C", 2:"H", 3:"O"})
        self.assertEqual(connection.execute("SELECT value FROM metadata WHERE key='coarse_ready'").fetchone()[0], "false")
        connection.close()

    def test_window_coverage_and_display_cap(self):
        path, connection, report = self.build([[(1, 2)], [], [(1, 2)], []])
        data = coarse_network(path, {"weight":"events", "coverage":"1", "limit":"1"})
        self.assertEqual(data["total_events"], 3)
        self.assertEqual(data["displayed_events"], 2)
        self.assertFalse(data["target_reached"])
        window = coarse_network(path, {"start":"1", "end":"2", "weight":"events"})
        self.assertEqual(window["total_events"], 1)
        self.assertEqual(window["channels"][0]["event_type"], "fusion")
        connection.close()

    def test_internal_transfer_toggle_and_oxygen_zero(self):
        path, connection, report = self.build([[(1, 2)], []])
        self.assertGreater(coarse_network(path, {})["total_weight"], 0)
        self.assertEqual(coarse_network(path, {"include_internal":"false"})["total_weight"], 0)
        self.assertEqual(coarse_network(path, {"weight":"oxygen"})["edges"], [])
        connection.close()

    def test_filtered_flow_agrees_with_event_filter(self):
        path, connection, report = self.build([[(1, 2)], []])
        annotate_events(connection, mode="annotate")
        with connection:
            connection.execute("UPDATE event_annotations SET filter_decision='exclude'")
        self.assertEqual(coarse_network(path, {"view":"filtered"})["total_events"], 0)
        self.assertEqual(coarse_network(path, {"view":"raw"})["total_events"], 1)
        connection.close()

    def test_diagnostics_and_fraction_series(self):
        path, connection, report = self.build([[(1, 2)], []], (1, 3, 2, 2))
        result = diagnostics(path, {})
        self.assertEqual(result["frame_count"], 2)
        self.assertAlmostEqual(sum(row["mean_mass_fraction"] for row in result["species"]), 1)
        self.assertAlmostEqual(sum(row["mean_carbon_fraction"] for row in result["species"]), 1)
        self.assertEqual(result["audit"]["audited_pairs"], 1)
        for row in result["species"]:
            series = timeseries(path, {"species_id":row["species_id"]})
            self.assertAlmostEqual(sum(point["mass_fraction"] for point in series["rows"])/2, row["mean_mass_fraction"])
        connection.close()

    def test_export_is_complete_and_preserves_signatures(self):
        path, connection, report = self.build([[(1, 2)], [], [(1, 2)]])
        export_analytics(connection, self.root)
        with (self.root / "coarse_channel_flows.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sum(int(row["event_count"]) for row in rows), 2)
        self.assertTrue(all("->" in row["reaction_signature"] for row in rows))
        connection.close()

    def test_invalid_parameters(self):
        path, connection, report = self.build([[], []])
        for query in ({"coverage":"nan"}, {"weight":"evil"}, {"start":"3", "end":"1"}, {"limit":"10000"}):
            with self.assertRaises(ValueError):
                coarse_network(path, query)
        connection.close()

    def test_old_v3_requires_postprocess_instead_of_empty_success(self):
        path, connection = self.database([[], []])
        self.assertFalse(coarse_network(path, {})["ready"])
        connection.close()

    def test_open_boundary_is_reported_not_invented_as_flow(self):
        from test_regressions import write_bonds
        from trajectory_pipeline import _process_frame_worker, _store_frame_result
        path, connection = self.database([[], []])
        source = self.root / "bond1.reaxff"
        write_bonds(source, elements=(1, 1, 1))
        frame = _process_frame_worker(dict(frame_id=1,source_path=str(source),cutoff=.3,element_map={1:"C",2:"H",3:"O"}))
        with connection:
            _store_frame_result(connection, frame)
        pair = _process_pair_worker(dict(database=str(path),parent_frame_id=0,child_frame_id=1,
            minimum_shared_atoms=1,minimum_parent_overlap=.05,minimum_child_overlap=.05,minimum_event_conservation=.9))
        with connection:
            _store_pair_result(connection, pair)
        report = build_analytics(connection, {1:"C",2:"H",3:"O"})
        self.assertEqual(report["unmatched_atom_occurrences"], 1)
        self.assertEqual(report["unbalanced_events"], 1)
        self.assertEqual(report["flow_rows"], 0)
        connection.close()
