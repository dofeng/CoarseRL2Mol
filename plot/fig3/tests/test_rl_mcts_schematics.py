"""Keep method artwork faithful to molecular glyphs, stage semantics and exports."""

import ast
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

from model.shared.coarse_graph import _node_color
from outputs.figure_plans.rl_mcts_schematics import plot_rl_mcts as drawing
from plot.coarse_style import DEFAULT_STYLE
from RL_MTCS.geometry import hex_edge_valid, reflect_axial, rotate_axial


def component_count(graph):
    adjacency = {node_id: set() for node_id in graph["positions"]}
    for source, target in graph["edges"]:
        adjacency[source].add(target)
        adjacency[target].add(source)
    remaining, count = set(adjacency), 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            node_id = pending.pop()
            for neighbor in adjacency[node_id] & remaining:
                remaining.remove(neighbor)
                pending.append(neighbor)
    return count


class RLMCTSSchematicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = drawing.load_inputs()

    def tearDown(self):
        plt.close("all")

    def assert_su_style(self, artists, su_types):
        self.assertEqual(set(artists["nodes"]), set(su_types))
        self.assertEqual(set(artists["texts"]), set(su_types))
        self.assertEqual(artists["radius_pt"], 7)
        for node_id, su_type in su_types.items():
            node, label = artists["nodes"][node_id], artists["texts"][node_id]
            np.testing.assert_allclose(node.get_facecolors()[0, :3], _node_color(su_type))
            np.testing.assert_allclose(node.get_edgecolors()[0, :3], to_rgb("white"))
            self.assertEqual(node.get_alpha(), 0.9)
            np.testing.assert_allclose(node.get_sizes(), [196])
            np.testing.assert_allclose(node.get_linewidths(), [0.8])
            self.assertEqual(label.get_text(), str(su_type))
            self.assertEqual(label.get_fontsize(), 7)
            self.assertEqual(label.get_fontweight(), "bold")
            np.testing.assert_allclose(to_rgb(label.get_color()), to_rgb("white"))
        for edge in artists["edges"]:
            self.assertEqual(edge.get_linewidth(), 1.5)

    def test_glyph_defaults_match_the_original_molecule_export_not_a_new_style(self):
        self.assertEqual(self.data["style"], DEFAULT_STYLE)
        self.assertTrue(self.data["style_baseline"]["all_match"])
        self.assertTrue(all(self.data["style_baseline"]["checks"].values()))
        reference = json.loads((drawing.PROJECT / self.data["style_baseline"]["reference_file"]).read_text())["coarse_drawing"]
        self.assertEqual(DEFAULT_STYLE.edge_width_pt, reference["edge_linewidth_pt"])
        self.assertEqual(DEFAULT_STYLE.node_alpha, reference["node_alpha"])
        self.assertEqual(DEFAULT_STYLE.font_color, reference["node_font_color"])
        self.assertAlmostEqual(DEFAULT_STYLE.molecular_edge_length_mm,
                               self.data["style_baseline"]["reference_center_spacing_mm"], delta=0.02)

    def test_old_molecule_renderer_really_uses_the_claimed_white_outline(self):
        source = ast.parse((drawing.PROJECT / "plot/fig3/plot_molecule_panels.py").read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "create_coarse_figure")
        circles = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Circle"]
        keywords = {argument.arg: ast.literal_eval(argument.value) for argument in circles[0].keywords
                    if isinstance(argument.value, ast.Constant)}
        self.assertEqual(keywords["edgecolor"], DEFAULT_STYLE.node_outline_color)
        self.assertEqual(keywords["linewidth"], DEFAULT_STYLE.outline_width_pt)

    def test_style_drift_is_rejected_instead_of_silently_redefining_the_baseline(self):
        for overrides in ({"font_color": "black"}, {"node_alpha": 1}, {"outline_width_pt": 0.65}, {"edge_width_pt": 0.9}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                config = deepcopy(self.data["config"])
                config["style"] = overrides
                config["spectrum_config"] = str(self.data["spectrum_path"])
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "original molecule-panel"):
                    drawing.load_inputs(path)

    def test_phase_order_and_branch_membership_match_the_real_scheduler(self):
        source = ast.parse((drawing.PROJECT / "RL_MTCS/runner.py").read_text())
        function = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "run_all_stages")
        tuples = [tuple(value.value for value in node.elts) for node in ast.walk(function)
                  if isinstance(node, ast.Tuple) and all(isinstance(value, ast.Constant) for value in node.elts)]
        self.assertIn(drawing.PHASES, tuples)
        self.assertEqual(self.data["connections"]["branch"]["phase"], "flex")
        self.assertNotIn("branch", drawing.PHASES)

    def test_forward_completion_is_called_at_checkpoints_not_each_tree_evaluation(self):
        source = ast.parse((drawing.PROJECT / "RL_MTCS/policy/checkpoints.py").read_text())
        methods = {node.name: node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)}
        calls = lambda method: {node.func.id for node in ast.walk(method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("fast_rollout_remaining", calls(methods["search"]))
        self.assertIn("fast_rollout_remaining", calls(methods["checkpoint"]))
        self.assertIn("MCTS", calls(methods["search"]))

    def test_cumulative_example_is_connected_with_unique_honeycomb_edges(self):
        example = self.data["example"]
        self.assertEqual(len(example["positions"]), 32)
        self.assertEqual(len(example["edges"]), 36)
        self.assertEqual(component_count(example), 1)
        self.assertEqual(len(set(map(tuple, example["edges"]))), 36)
        for source, target in example["edges"]:
            self.assertTrue(hex_edge_valid(tuple(example["axial"][source]), tuple(example["axial"][target]), honeycomb=True))

    def test_cumulative_stages_preserve_inventory_until_five_membered_contraction(self):
        example = self.data["example"]
        expected = ((16, 18), (24, 27), (30, 34), (32, 36), (31, 35))
        previous_nodes, previous_edges = set(), set()
        for phase, (node_count, edge_count) in zip(drawing.PHASES, expected):
            graph = example["stages"][phase]
            nodes, edges = set(graph["positions"]), set(map(tuple, graph["edges"]))
            self.assertEqual((len(nodes), len(edges)), (node_count, edge_count))
            if phase != "substitution":
                self.assertTrue(previous_nodes <= nodes)
                self.assertTrue(previous_edges <= edges)
            self.assertEqual(nodes, set(graph["su_types"]))
            previous_nodes, previous_edges = nodes, edges
        before, after = example["stages"]["side"], example["stages"]["substitution"]
        changes = {node_id for node_id, su_type in after["su_types"].items() if su_type != before["su_types"][node_id]}
        self.assertEqual(changes, {example["hetero_node"]})
        representative, removed = example["hetero_site_ids"]
        self.assertEqual(set(before["positions"]) - set(after["positions"]), {removed})
        self.assertEqual(after["su_types"][representative], 30)
        np.testing.assert_allclose(after["positions"][representative],
                                   np.mean([before["positions"][site] for site in (representative, removed)], axis=0))
        self.assertEqual(component_count(after), 1)
        ring = set(example["groups"]["aromatic_c"]) - {removed}
        self.assertEqual(len(ring), 5)
        self.assertEqual(sum(set(edge) <= ring for edge in after["edges"]), 5)
        for node_id in set(after["positions"]) - {representative}:
            self.assertEqual(after["positions"][node_id], before["positions"][node_id])

    def test_each_mechanism_has_an_explicit_before_and_after(self):
        expected = {"rigid": (12, 12, 12, 13), "flex": (8, 8, 5, 7), "branch": (7, 7, 3, 6),
                    "ring": (12, 12, 11, 13), "side": (8, 8, 7, 8), "substitution": (6, 5, 6, 5), "six_membered": (6, 6, 6, 6)}
        for key, counts in expected.items():
            with self.subTest(key=key):
                row = self.data["connections"][key]
                before, after = row["before"], row["after"]
                self.assertEqual((len(before["positions"]), len(after["positions"]), len(before["edges"]), len(after["edges"])), counts)
                self.assertEqual(len(after["edges"]), len(set(map(tuple, after["edges"]))))
                self.assertEqual(component_count(after), 1)

    def test_independent_connection_panels_keep_complete_aromatic_contexts(self):
        expected = {"flex": (14, 13, 15, 3), "branch": (22, 21, 24, 4), "ring": (12, 11, 13, 2)}
        for key, (node_count, before_edges, after_edges, components) in expected.items():
            with self.subTest(key=key):
                row = self.data["connections"][key]
                before, after = row.get("detail_before", row["before"]), row.get("detail_after", row["after"])
                self.assertEqual(len(before["positions"]), node_count)
                self.assertEqual(before["positions"], after["positions"])
                self.assertEqual(len({tuple(point) for point in after["positions"].values()}), node_count)
                self.assertEqual((len(before["edges"]), len(after["edges"])), (before_edges, after_edges))
                self.assertEqual((component_count(before), component_count(after)), (components, 1))
                retained = {tuple(edge) for edge in before["edges"]}
                self.assertEqual({tuple(edge) for edge in after["edges"]} - retained,
                                 {tuple(edge) for edge in row.get("detail_added_edges", row["added_edges"])})
                for source, target in after["edges"]:
                    length = np.linalg.norm(np.asarray(after["positions"][source]) - after["positions"][target])
                    self.assertAlmostEqual(length, 1)
                self.assertIn("not an observed", row["provenance"])

    def test_rigid_is_a_direct_su10_bond_not_fusing_or_inserting_nodes(self):
        row = self.data["connections"]["rigid"]
        self.assertEqual(component_count(row["before"]), 2)
        self.assertEqual(len(row["added_edges"]), 1)
        self.assertEqual({row["after"]["su_types"][node_id] for node_id in row["added_edges"][0]}, {10})
        self.assertEqual(row["before"]["su_types"], row["after"]["su_types"])

    def test_flexible_link_and_branched_link_bind_all_boundaries(self):
        for key, ports in (("flex", 2), ("branch", 3)):
            row = self.data["connections"][key]
            self.assertEqual(component_count(row["before"]), ports + 1)
            self.assertEqual(len(row["added_edges"]), ports)
            for source, target in row["added_edges"]:
                self.assertEqual({row["after"]["su_types"][source], row["after"]["su_types"][target]}, {11, 23})
        center = next(node_id for node_id, su_type in self.data["connections"]["branch"]["after"]["su_types"].items() if su_type == 24)
        self.assertEqual(sum(center in edge for edge in self.data["connections"]["branch"]["after"]["edges"]), 3)

    def test_side_ring_retains_its_open_path_and_side_chain_has_one_attachment(self):
        ring = self.data["connections"]["ring"]
        cycle = set(ring["groups"]["aliphatic_ring"])
        before_edges = {tuple(edge) for edge in ring["before"]["edges"] if set(edge) <= cycle}
        after_edges = {tuple(edge) for edge in ring["after"]["edges"] if set(edge) <= cycle}
        self.assertEqual(before_edges, after_edges)
        self.assertEqual(len(after_edges), len(cycle) - 1)
        self.assertEqual(len(ring["added_edges"]), 2)
        side = self.data["connections"]["side"]
        self.assertEqual(len(side["added_edges"]), 1)
        terminal = next(node_id for node_id, su_type in side["after"]["su_types"].items() if su_type == 22)
        self.assertEqual(sum(terminal in edge for edge in side["after"]["edges"]), 1)

    def test_side_ring_shares_two_aromatic_anchors_and_their_existing_edge(self):
        row = self.data["connections"]["ring"]
        aromatic = set(row["groups"]["aromatic"])
        backbone = row["ring_backbone"]
        ring_edges = {tuple(sorted((backbone[index], backbone[(index + 1) % 6]))) for index in range(6)}
        before_edges = {tuple(edge) for edge in row["before"]["edges"]}
        after_edges = {tuple(edge) for edge in row["after"]["edges"]}
        aromatic_edges = {edge for edge in after_edges if set(edge) <= aromatic}
        shared_edge = tuple(sorted(row["aromatic_edge"]))
        self.assertEqual(row["ring_type"], "fatty_ring_side")
        self.assertEqual(set(backbone) & aromatic, set(shared_edge))
        self.assertEqual(ring_edges & aromatic_edges, {shared_edge})
        self.assertIn(shared_edge, before_edges)
        self.assertTrue(ring_edges <= after_edges)
        self.assertEqual(ring_edges - before_edges, set(map(tuple, row["added_edges"])))
        self.assertEqual([row["after"]["su_types"][node_id] for node_id in backbone], [11, 23, 24, 24, 23, 11])
        for node_id in backbone[2:4]:
            self.assertEqual(sum(node_id in edge for edge in after_edges), 3)
        cumulative = self.data["example"]
        cumulative_backbone = cumulative["ring_backbone"]
        self.assertEqual(cumulative["ring_type"], "fatty_ring_side")
        self.assertEqual([cumulative["su_types"][node_id] for node_id in cumulative_backbone], [11, 23, 24, 24, 23, 11])
        self.assertEqual(set(cumulative_backbone) & set(cumulative["groups"]["aromatic_a"]), set(cumulative["ring_aromatic_edge"]))
        self.assertEqual(len(cumulative["external_edges"]["ring"]), 2)
        existing_edges = {tuple(sorted(edge)) for edge in cumulative["stages"]["flex"]["edges"]}
        self.assertIn(tuple(sorted(cumulative["ring_aromatic_edge"])), existing_edges)

    def test_side_ring_su_motif_is_present_in_the_recorded_topology_input(self):
        row = self.data["connections"]["ring"]
        source = row["source_motif"]
        with (drawing.PROJECT / source["path"]).open(encoding="utf-8-sig") as stream:
            self.assertTrue(any(record["actual_su"] == source["actual_su"]
                                and record["product_type"] == "aliphatic_ring" for record in csv.DictReader(stream)))
        graph, backbone = row["after"], row["ring_backbone"]
        labels = []
        for node_id in backbone:
            branches = [target if source == node_id else source for source, target in graph["edges"]
                        if node_id in (source, target) and not {source, target} <= set(backbone)]
            branch_types = [graph["su_types"][branch] for branch in branches if branch not in row["groups"]["aromatic"]]
            labels.append(str(graph["su_types"][node_id]) + "".join(f"(-{su_type})" for su_type in branch_types))
        self.assertEqual("-".join(labels), source["actual_su"])

    def test_main_gallery_uses_the_five_membered_substitution_with_a_shared_reference(self):
        figure, axis = drawing.canvas(*self.data["config"]["sizes_mm"]["stage_connections"])
        result = drawing.draw_stage_connections(axis, self.data)["substitution"]
        row = self.data["connections"]["substitution"]
        self.assertEqual((len(result["before"]["nodes"]), len(result["after"]["nodes"])), (6, 5))
        self.assertEqual(len(result["after"]["edges"]), 5)
        representative, removed = row["staging_site_ids"]
        panel_offset = np.asarray((31, 0))
        expected = np.mean([result["before"]["positions"][site] for site in (representative, removed)], axis=0) + panel_offset
        np.testing.assert_allclose(result["after"]["positions"][representative], expected)
        self.assertEqual(result["after"]["texts"][representative].get_text(), "30")
        for node_id in set(result["after"]["positions"]) - {representative}:
            np.testing.assert_allclose(result["after"]["positions"][node_id],
                                       result["before"]["positions"][node_id] + panel_offset)
        self.assertIn("Five-membered: 2 × SU13 → SU30", {label.get_text() for label in axis.texts})

    def test_five_membered_substitution_rejects_invalid_or_attached_staging_sites(self):
        row = self.data["connections"]["substitution"]
        pair = row["staging_site_ids"]
        graph = row["before"]
        unchanged = deepcopy(graph)
        for sites, target in (([pair[0]], 30), ([pair[0], pair[0]], 30), ([pair[0], 999], 30),
                              ([pair[0], next(node_id for node_id in graph["positions"] if graph["su_types"][node_id] == 7)], 30),
                              (pair, 22), (pair, True), (pair, 30.0)):
            with self.subTest(sites=sites, target=target), self.assertRaises(ValueError):
                drawing.contract_staging_pair(graph, sites, target)
        attached = deepcopy(graph)
        attached["positions"][99], attached["su_types"][99] = [2, 2], 22
        attached["edges"].append((pair[0], 99))
        with self.assertRaisesRegex(ValueError, "external attachments"):
            drawing.contract_staging_pair(attached, pair, 30)
        nonadjacent = next(node_id for node_id, su_type in graph["su_types"].items()
                           if su_type == 13 and node_id not in pair
                           and not any(set(edge) == {node_id, pair[0]} for edge in graph["edges"]))
        with self.assertRaisesRegex(ValueError, "adjacent staging sites"):
            drawing.contract_staging_pair(graph, (pair[0], nonadjacent), 30)
        self.assertEqual(graph, unchanged)

    def test_six_and_five_membered_substitution_are_not_conflated(self):
        six = self.data["connections"]["six_membered"]
        self.assertEqual(six["before"]["positions"], six["after"]["positions"])
        self.assertEqual(six["before"]["edges"], six["after"]["edges"])
        target = six["changed_nodes"][0]
        self.assertEqual((six["before"]["su_types"][target], six["after"]["su_types"][target]), (13, 26))
        five = self.data["connections"]["substitution"]
        removed = (set(five["before"]["positions"]) - set(five["after"]["positions"])).pop()
        target = five["changed_nodes"][0]
        expected = np.mean([five["before"]["positions"][target], five["before"]["positions"][removed]], axis=0)
        np.testing.assert_allclose(five["after"]["positions"][target], expected)
        self.assertEqual(five["after"]["su_types"][target], 30)

    def test_all_before_after_panels_use_the_original_molecule_glyph(self):
        for key, row in self.data["connections"].items():
            with self.subTest(key=key):
                figure, axis = drawing.canvas(*self.data["config"]["sizes_mm"]["connection_pair"])
                result = drawing.draw_connection_pair(axis, self.data, key=key)
                self.assert_su_style(result["before"], row.get("detail_before", row["before"])["su_types"])
                self.assert_su_style(result["after"], row.get("detail_after", row["after"])["su_types"])

    def test_stage_frames_keep_common_coordinates_sizes_and_node_spacing(self):
        reference = self.data["example"]["stages"]["side"]
        baseline = None
        representative, removed = self.data["example"]["hetero_site_ids"]
        for phase in ("side", "rigid", "flex", "ring", "substitution"):
            figure, axis = drawing.canvas(160, 90)
            graph = self.data["example"]["stages"][phase]
            result = drawing.molecular_graph(axis, graph, (80, 42), self.data, reference=reference)
            self.assert_su_style(result, graph["su_types"])
            if baseline is None:
                baseline = result
            for node_id in result["positions"]:
                expected = np.mean([baseline["positions"][site] for site in (representative, removed)], axis=0) \
                    if phase == "substitution" and node_id == representative else baseline["positions"][node_id]
                np.testing.assert_allclose(result["positions"][node_id], expected)
            for source, target in graph["edges"]:
                if phase != "substitution" or representative not in (source, target):
                    self.assertAlmostEqual(np.linalg.norm(result["positions"][source] - result["positions"][target]), 10.5)

    def test_tree_boxes_are_states_and_no_fictitious_metrics_are_drawn(self):
        figure, axis = drawing.canvas(88, 56)
        result = drawing.draw_search(axis, self.data)
        tree = result["tree"]
        self.assertTrue(set(tree["selected_edges"]) <= set(tree["edges"]))
        self.assertFalse({"Q", "N", "probabilities", "reward"} & tree.keys())
        self.assertTrue(all(isinstance(node, FancyBboxPatch) for node in result["artists"]["nodes"].values()))
        self.assertEqual(len(axis.collections), 0)

    def test_checkpoint_labels_separate_clone_forward_audit_commit_and_retry(self):
        figure, axis = drawing.canvas(90, 60)
        result = drawing.draw_checkpoint(axis, self.data)
        self.assertEqual(set(result), {"clone", "forward", "audit", "commit", "retry"})
        labels = {label.get_text() for label in axis.texts}
        self.assertIn("Clone endpoint", labels)
        self.assertIn("Commit endpoint", labels)
        self.assertIn("Audit + forward completion", labels)
        self.assertIn("Exclude + retry", labels)

    def test_reward_terms_and_optional_nmr_do_not_invent_a_policy_network(self):
        figure, axis = drawing.canvas(88, 60)
        result = drawing.draw_reward(axis, self.data)
        self.assert_su_style(result["state"], {0: 22, 1: 24, 2: 22})
        labels = {label.get_text() for label in axis.texts}
        self.assertTrue({"Task completion", "NMR agreement", "Free SU13 sites", "Shape"} <= labels)
        self.assertFalse(any("trained policy" in label for label in labels))

    def test_axial_symmetries_preserve_projected_lengths(self):
        for point in ((1, 0), (2, 1), (-3, 2)):
            initial_length = np.linalg.norm(drawing.axial_to_cartesian(point))
            for rotation in range(6):
                transformed = rotate_axial(point, rotation)
                self.assertTrue(all(isinstance(coordinate, int) for coordinate in transformed))
                self.assertAlmostEqual(np.linalg.norm(drawing.axial_to_cartesian(transformed)), initial_length)
                self.assertAlmostEqual(np.linalg.norm(drawing.axial_to_cartesian(reflect_axial(transformed))), initial_length)

    def test_hex_asset_draws_real_su_glyphs_not_a_second_role_palette(self):
        figure, axis = drawing.canvas(132, 66)
        result = drawing.draw_hex_coordinates(axis, self.data)
        self.assert_su_style(result["graph"], {0: 24, 1: 22, 2: 22, 3: 22})
        labels = {label.get_text() for label in axis.texts}
        self.assertTrue(any("≠ a 3D conformation" in label for label in labels))

    def assert_inside_canvas(self, figure):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            node_centers = []
            for collection in axis.collections:
                if isinstance(collection, PathCollection):
                    self.assertTrue(np.allclose(collection.get_sizes(), [196]))
                    for position in collection.get_offsets():
                        point = axis.transData.transform(position)
                        node_centers.append(point)
                        radius = 7 * figure.dpi / 72
                        self.assertGreaterEqual(point[0] - radius, -1)
                        self.assertGreaterEqual(point[1] - radius, -1)
                        self.assertLessEqual(point[0] + radius, figure.bbox.width + 1)
                        self.assertLessEqual(point[1] + radius, figure.bbox.height + 1)
            for index, center in enumerate(node_centers):
                for other in node_centers[index + 1:]:
                    self.assertGreaterEqual(np.linalg.norm(center - other), 14 * figure.dpi / 72 - 1)
            for label in axis.texts:
                self.assertEqual(label.get_fontsize(), 7)
                extent = label.get_window_extent(renderer)
                self.assertGreaterEqual(extent.x0, -1)
                self.assertGreaterEqual(extent.y0, -1)
                self.assertLessEqual(extent.x1, figure.bbox.width + 1)
                self.assertLessEqual(extent.y1, figure.bbox.height + 1)

    def test_component_and_full_figure_labels_stay_inside_declared_canvases(self):
        for name, draw in drawing.DRAWERS.items():
            with self.subTest(component=name):
                figure, axis = drawing.canvas(*self.data["config"]["sizes_mm"][name])
                draw(axis, self.data)
                self.assert_inside_canvas(figure)
        self.assert_inside_canvas(drawing.draw_full_figure(self.data))

    def test_mechanism_panels_and_stage_frames_fit_without_shrinking_glyphs(self):
        for key in self.data["connections"]:
            with self.subTest(key=key):
                figure, axis = drawing.canvas(*self.data["config"]["sizes_mm"]["connection_pair"])
                drawing.draw_connection_pair(axis, self.data, key=key)
                self.assert_inside_canvas(figure)
        for phase in drawing.PHASES:
            figure, axis = drawing.canvas(160, 90)
            drawing.molecular_graph(axis, self.data["example"]["stages"][phase], (80, 42), self.data,
                                    reference=self.data["example"]["stages"]["side"])
            self.assert_inside_canvas(figure)

    def test_malformed_graph_and_su_assignments_are_rejected(self):
        for key, value in (("aromatic_a_faces", [[0.5, 0]]), ("aromatic_a_faces", [[1, 0]]),
                           ("flex_nodes", [[7, 1], [7, 1]]), ("side_nodes", [[0, 1], [-3, -2]]),
                           ("rigid_bond", [[3, 1], [99, 99]]), ("su_types", self.data["config"]["graph"]["su_types"][:-1]),
                           ("aliphatic_ring_path", [[-2, 0], [-3, -1], [-3, -2]]),
                           ("aliphatic_ring_branches", [[[-2, 0], [-3, -1]]]),
                           ("ring_bonds", [[[-1, 0], [-2, 0]]]),
                           ("ring_bonds", [[[-1, 0], [-2, 0]], [[-2, -2], [-1, 0]]]),
                           ("hetero_sites", [[11, 2]]), ("hetero_sites", [[11, 2], [11, 2]]),
                           ("hetero_sites", [[11, 2], [-1, 0]]), ("hetero_sites", [[11, 2], [10, 3]])):
            with self.subTest(key=key):
                specification = deepcopy(self.data["config"]["graph"])
                specification[key] = value
                with self.assertRaises(ValueError):
                    drawing.build_example(specification)

    def test_invalid_exports_and_undersized_canvases_fail_early(self):
        for key, value in (("dpi", 0), ("dpi", True), ("formats", ["jpg"]),
                           ("colors", {"committed": "not-a-color"}), ("sizes_mm", {"figure5_rl_mcts_schematic": [100, 100]})):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                config = deepcopy(self.data["config"])
                config[key] = value
                config["spectrum_config"] = str(self.data["spectrum_path"])
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    drawing.load_inputs(path)

    def test_all_exports_are_editable_and_original_data_are_not_modified(self):
        inputs = [drawing.PROJECT / name for name in drawing.SOURCE_FILES]
        inputs.extend([self.data["spectrum_data"], drawing.PROJECT / self.data["style_baseline"]["reference_file"]])
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        with tempfile.TemporaryDirectory() as directory:
            destination = drawing.generate(output_dir=Path(directory), dpi=80)
            paths = sorted(destination.glob("*.svg"))
            self.assertEqual(len(paths), 20)
            self.assertTrue(all((destination / f"{filename}.svg").exists() for filename in drawing.CONNECTION_FILES.values()))
            for path in paths:
                document = ElementTree.parse(path).getroot()
                self.assertFalse(document.findall(".//{http://www.w3.org/2000/svg}image"))
                self.assertIn("<text", path.read_text())
                self.assertTrue(path.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
                with Image.open(path.with_suffix(".png")) as image:
                    self.assertAlmostEqual(image.info["dpi"][0], 80, delta=0.1)
            metadata = json.loads((destination / "schematic_data.json").read_text())
            self.assertEqual(metadata["phase_order"], list(drawing.PHASES))
            self.assertTrue(metadata["style_baseline"]["all_match"])
            self.assertIn("not a recorded", metadata["purpose"])
            curves = np.loadtxt(destination / "illustrative_spectra.csv", delimiter=",", skiprows=1)
            np.testing.assert_array_equal(curves, self.data["curves"])
        self.assertEqual(hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})


if __name__ == "__main__":
    unittest.main()
