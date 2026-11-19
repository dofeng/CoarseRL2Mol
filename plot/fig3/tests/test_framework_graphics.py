"""Keep framework artwork consistent without rewriting observed graph topology."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch
import numpy as np

from outputs.figure_plans.topology_data_demo import plot_topology_demo as demo
from outputs.figure_plans.topology_schematics import plot_topology_motifs as motifs
from plot.coarse_style import CoarseStyle, DEFAULT_STYLE, contrasting_label_color, draw_graph, draw_neighborhood
from plot.coarse_style import model_su_colors, normalize_positions


ROOT = Path(__file__).resolve().parents[3]


class FrameworkGraphicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = demo.extract_data(ROOT)

    def tearDown(self):
        plt.close("all")

    def test_node_radius_and_font_do_not_scale_with_canvas(self):
        for width, height in ((44, 44), (88, 44), (88, 88)):
            figure, axis = demo.canvas(width, height)
            artists = draw_neighborhood(axis, self.data["neighborhood"], (22, 22))
            figure.canvas.draw()
            for node in artists["nodes"].values():
                np.testing.assert_allclose(node.get_sizes(), [(2 * DEFAULT_STYLE.labeled_radius_pt) ** 2])
            for text in artists["texts"].values():
                self.assertEqual(text.get_fontsize(), DEFAULT_STYLE.font_size_pt)

    def test_labeled_and_overview_modes_have_explicit_physical_sizes(self):
        self.assertEqual(DEFAULT_STYLE.labeled_radius_pt, 7)
        self.assertEqual(DEFAULT_STYLE.font_size_pt, 7)
        self.assertEqual(DEFAULT_STYLE.overview_radius_pt, 2.5)

    def test_neighborhood_expansion_preserves_existing_positions(self):
        neighborhood = self.data["neighborhood"]
        _, axis = demo.canvas(44, 44)
        first = draw_neighborhood(axis, neighborhood, (22, 22), max_hop=1)
        second = draw_neighborhood(axis, neighborhood, (22, 22), max_hop=2)
        self.assertEqual(set(first["nodes"]), {neighborhood["root_id"], *neighborhood["hop1_ids"]})
        self.assertEqual(set(second["nodes"]), set(map(int, neighborhood["graph"]["positions"])))
        for node_id, position in first["positions"].items():
            np.testing.assert_allclose(position, second["positions"][node_id])
        self.assertEqual(len(first["edges"]), 2)
        self.assertEqual(len(second["edges"]), 6)
        with self.assertRaises(ValueError):
            draw_neighborhood(axis, neighborhood, (22, 22), max_hop=3)

    def test_true_su_colors_are_not_overwritten_by_hop_distance(self):
        neighborhood = self.data["neighborhood"]
        _, axis = demo.canvas(44, 44)
        artists = draw_neighborhood(axis, neighborhood, (22, 22))
        colors = model_su_colors(neighborhood["graph"]["su_types"])
        for node_id, node in artists["nodes"].items():
            np.testing.assert_allclose(node.get_facecolors()[0, :3], colors[node_id])

    def test_label_color_handles_light_and_dark_fills(self):
        self.assertEqual(contrasting_label_color("#FFFF99"), "#111111")
        self.assertEqual(contrasting_label_color("#000055"), "white")

    def test_audit_pair_changes_styles_not_node_positions(self):
        graph = self.data["audit_graph"]
        original = deepcopy(graph)
        changed = [row["node_id"] for row in self.data["changes"]]
        _, axis = demo.canvas(92, 36)
        before = demo.draw_graph(axis, graph, (16, 17), 7.3, changed=changed, required=True, labels=True)
        after = demo.draw_graph(axis, graph, (66, 17), 7.3, changed=changed, labels=True)
        self.assertEqual(graph, original)
        self.assertEqual(len(before["nodes"]), 9)
        self.assertEqual(len(before["edges"]), 9)
        for node_id in before["nodes"]:
            np.testing.assert_allclose(after["positions"][node_id] - before["positions"][node_id], [50, 0])
            if node_id in changed:
                np.testing.assert_allclose(before["nodes"][node_id].get_facecolors()[0, :3], to_rgb("white"))
            else:
                np.testing.assert_allclose(before["nodes"][node_id].get_facecolors(), after["nodes"][node_id].get_facecolors())

    def test_edges_end_at_physical_node_boundaries(self):
        graph = {"positions": {0: [10, 10], 1: [30, 10]}, "edges": [(0, 1)], "su_types": {0: 2, 1: 13}}
        for size in (44, 88):
            figure, axis = demo.canvas(size, size)
            artists = draw_graph(axis, graph, labels=True)
            figure.canvas.draw()
            edge = artists["edges"][0]
            vertices = edge.get_transform().transform(edge.get_path().vertices)
            source = axis.transData.transform(graph["positions"][0])
            target = axis.transData.transform(graph["positions"][1])
            expected = DEFAULT_STYLE.labeled_radius_pt * figure.dpi / 72
            self.assertAlmostEqual(np.linalg.norm(vertices[0] - source), expected, delta=0.05)
            self.assertAlmostEqual(np.linalg.norm(vertices[-1] - target), expected, delta=0.05)
            self.assertEqual(edge.get_linewidth(), DEFAULT_STYLE.edge_width_pt)

    def test_schematic_motifs_share_overview_size_and_median_edge_length(self):
        for build in (motifs.fused_aromatic, motifs.flexible_linker, motifs.aliphatic_ring, motifs.branch_junction, motifs.side_chain):
            motif = build()
            original = deepcopy(motif)
            _, axis = motifs.canvas(54, 40)
            artists = motifs.draw_motif(axis, motif, (27, 21))
            self.assertEqual(motif, original)
            self.assertTrue(all(isinstance(patch, FancyArrowPatch) for patch in axis.patches))
            self.assertEqual(len(artists["nodes"]), len(motif["positions"]))
            self.assertEqual(len(artists["edges"]), len(motif["edges"]))
            lengths = [np.linalg.norm(artists["positions"][source] - artists["positions"][target]) for source, target in motif["edges"]]
            self.assertAlmostEqual(float(np.median(lengths)), DEFAULT_STYLE.motif_edge_length_mm)
            for node in artists["nodes"].values():
                np.testing.assert_allclose(node.get_sizes(), [(2 * DEFAULT_STYLE.overview_radius_pt) ** 2])

    def test_uniform_normalization_does_not_distort_fused_rings(self):
        motif = motifs.fused_aromatic()
        positions = normalize_positions(motif["positions"], motif["edges"])
        lengths = [np.linalg.norm(positions[source] - positions[target]) for source, target in motif["edges"]]
        np.testing.assert_allclose(lengths, DEFAULT_STYLE.motif_edge_length_mm, atol=1e-6)
        self.assertEqual(len(motif["positions"]), 18)
        self.assertEqual(len(motif["edges"]), 21)
        self.assertEqual(len(motif["shared_edges"]), 3)

    def test_real_output_topology_and_counts_are_preserved(self):
        recorded = json.loads((ROOT / "outputs/figure_plans/topology_data_demo/source_data.json").read_text())
        for key in ("distribution", "product_counts", "aromatic_cluster_counts", "changes", "source_files"):
            self.assertEqual(self.data[key], recorded[key])
        for current, previous in zip(self.data["motifs"], recorded["motifs"]):
            self.assertEqual(current["record"], previous["record"])
            self.assertEqual(current["count"], previous["count"])
            self.assertEqual([tuple(edge) for edge in current["graph"]["edges"]], [tuple(edge) for edge in previous["graph"]["edges"]])
            self.assertEqual(set(map(int, current["graph"]["positions"])), set(map(int, previous["graph"]["positions"])))

    def test_style_and_bad_graph_inputs_are_validated(self):
        for invalid in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                CoarseStyle(labeled_radius_pt=invalid)
        with self.assertRaises(ValueError):
            normalize_positions({0: [0, 0], 1: [0, 0]}, [(0, 1)])
        _, axis = demo.canvas(44, 44)
        with self.assertRaises(ValueError):
            draw_graph(axis, {"positions": {0: [0, 0]}, "edges": []}, labels=True)
        with self.assertRaises(ValueError):
            draw_graph(axis, {"positions": {0: [0, 0]}, "edges": [(0, 1)]})

    def test_framework_assets_export_editable_svg_at_declared_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with plt.rc_context(demo.STYLE):
                demo.export_framework_assets(self.data, directory, dpi=72)
            for name, size in (("layer1_neighborhood", [44, 44]), ("layer2_neighborhood", [44, 44]), ("topology_audit_pair", [92, 36])):
                svg = (directory / f"{name}.svg").read_text()
                self.assertIn("<text", svg)
                self.assertNotIn("<image", svg)
                element = ElementTree.fromstring(svg)
                self.assertAlmostEqual(float(element.attrib["width"].removesuffix("pt")), size[0] / 25.4 * 72, places=5)
                self.assertAlmostEqual(float(element.attrib["height"].removesuffix("pt")), size[1] / 25.4 * 72, places=5)
                self.assertTrue((directory / f"{name}.pdf").read_bytes().startswith(b"%PDF"))
                self.assertTrue((directory / f"{name}.png").exists())

    def test_overview_size_can_be_adjusted_without_changing_labeled_size(self):
        style = replace(DEFAULT_STYLE, overview_radius_pt=3.0)
        self.assertEqual(style.labeled_radius_pt, DEFAULT_STYLE.labeled_radius_pt)
        self.assertEqual(asdict(style)["overview_radius_pt"], 3.0)

    def test_compact_neighborhood_uses_overview_markers_without_type_labels(self):
        _, axis = demo.canvas(32, 32)
        artists = demo.draw_neighborhood(axis, self.data["neighborhood"], (16, 16), labels=False, scale=0.7)
        self.assertEqual(artists["radius_pt"], DEFAULT_STYLE.overview_radius_pt)
        self.assertEqual(artists["texts"], {})
        self.assertEqual({text.get_text() for text in axis.texts}, {"1-hop", "2-hop"})
        self.assertEqual(len(artists["nodes"]), 7)


if __name__ == "__main__":
    unittest.main()
