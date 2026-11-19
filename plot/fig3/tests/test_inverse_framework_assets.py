"""Verify schematic components preserve spectra, SU inventories, and paired samples."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from outputs.figure_plans.inverse_framework_assets import plot_framework_assets as assets
from outputs.figure_plans.nmr_schematics import plot_nmr_schematics as spectra
from plot.coarse_style import model_su_colors


class InverseFrameworkAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = assets.load_inputs(assets.ROOT / "config.json")

    def tearDown(self):
        plt.close("all")

    def test_input_and_feedback_reuse_identical_target_and_coordinates(self):
        figure, _ = assets.canvas(120, 45)
        target = assets.draw_component(figure, (0, 0, 40, 30), "target_spectrum", self.data)
        feedback = assets.draw_component(figure, (50, 0, 64, 35), "layer4_spectral_feedback", self.data)
        target_line = next(line for line in target.lines if line.get_label() == "Target")
        feedback_line = next(line for line in feedback["comparison"].lines if line.get_label() == "Target")
        np.testing.assert_array_equal(target_line.get_xdata(), feedback_line.get_xdata())
        np.testing.assert_array_equal(target_line.get_ydata(), feedback_line.get_ydata())
        self.assertEqual(target_line.get_color(), feedback_line.get_color())
        self.assertEqual(target_line.get_linewidth(), feedback_line.get_linewidth())
        self.assertEqual(target.get_xlim(), feedback["comparison"].get_xlim())
        self.assertEqual(target.get_ylim(), feedback["comparison"].get_ylim())
        self.assertFalse(target.axison)
        self.assertFalse(feedback["difference"].axison)

    def test_complete_filled_regions_are_retained(self):
        figure, axis = plt.subplots()
        spectra.draw_panel(axis, self.data["curves"], self.data["spectrum_config"], "target")
        self.assertEqual(len(axis.collections), 1)
        figure, axis = plt.subplots()
        spectra.draw_panel(axis, self.data["curves"], self.data["spectrum_config"], "comparison")
        self.assertEqual(len(axis.collections), 2)
        self.assertEqual({line.get_label() for line in axis.lines if not line.get_label().startswith("_")}, {"Target", "Reconstructed"})

    def test_source_spectrum_config_is_supported_without_dropping_rows(self):
        config = deepcopy(self.data["config"])
        source_config = assets.ROOT.parent / "nmr_schematics/config.json"
        config["spectrum_config"] = str(source_config)
        config["neighborhood_data"] = str(self.data["source_paths"]["neighborhood_data"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            data = assets.load_inputs(path)
        self.assertEqual(data["spectrum_config"]["mode"], "source")
        self.assertEqual(data["input_diagnostics"]["rows_excluded"], 0)
        self.assertEqual(data["input_diagnostics"]["input_rows"], len(data["curves"]))
        np.testing.assert_array_equal(data["curves"][:, 3], data["curves"][:, 1] - data["curves"][:, 2])

    def test_residual_is_exactly_target_minus_reconstructed(self):
        curves = self.data["curves"]
        np.testing.assert_array_equal(curves[:, 3], curves[:, 1] - curves[:, 2])
        figure, axis = plt.subplots()
        spectra.draw_panel(axis, curves, self.data["spectrum_config"], "difference")
        np.testing.assert_array_equal(axis.lines[-1].get_ydata(), curves[:, 3])
        self.assertAlmostEqual(sum(axis.get_ylim()), 0)
        self.assertGreater(curves[:, 3].max(), 0)
        self.assertLess(curves[:, 3].min(), 0)

    def test_invalid_spectrum_kind_is_not_silently_a_residual(self):
        figure, axis = plt.subplots()
        with self.assertRaisesRegex(ValueError, "kind"):
            spectra.draw_panel(axis, self.data["curves"], self.data["spectrum_config"], "typo")
        self.assertEqual(len(axis.lines), 0)

    def test_layer0_pool_contains_every_integer_unit_and_no_edges(self):
        figure, axis = assets.canvas(36, 34)
        result = assets.draw_inventory(axis, self.data)
        self.assertEqual(len(result["pool"]["nodes"]), self.data["inventory"].sum())
        self.assertEqual(len(result["pool"]["edges"]), 0)
        self.assertEqual(len(result["bars"]), np.count_nonzero(self.data["inventory"]))
        np.testing.assert_allclose(
            [bar.get_height() for bar in result["bars"]],
            self.data["inventory"][result["active_ids"]] / self.data["inventory"].max() * 34 * 0.32,
        )

    def test_layer1_layer2_preserve_common_positions_and_model_colors(self):
        neighborhood = deepcopy(self.data["neighborhood"])
        figure, first_axis = assets.canvas(36, 34)
        figure, second_axis = assets.canvas(36, 34)
        first = assets.draw_hops(first_axis, self.data, 1)
        second = assets.draw_hops(second_axis, self.data, 2)
        self.assertEqual(len(first["nodes"]), 3)
        self.assertEqual(len(first["edges"]), 2)
        self.assertEqual(len(second["nodes"]), 7)
        self.assertEqual(len(second["edges"]), 6)
        for node_id in first["nodes"]:
            np.testing.assert_array_equal(first["positions"][node_id], second["positions"][node_id])
            np.testing.assert_array_equal(first["nodes"][node_id].get_facecolors(), second["nodes"][node_id].get_facecolors())
            np.testing.assert_array_equal(first["nodes"][node_id].get_sizes(), second["nodes"][node_id].get_sizes())
        root_position = first["positions"][neighborhood["root_id"]]
        required_distance = (2 * self.data["style"].labeled_radius_pt + 2) * 25.4 / 72
        for node_id in neighborhood["hop1_ids"]:
            self.assertGreater(np.linalg.norm(first["positions"][node_id] - root_position), required_distance)
        colors = model_su_colors(neighborhood["graph"]["su_types"])
        for node_id, node in second["nodes"].items():
            np.testing.assert_allclose(node.get_facecolors()[0, :3], matplotlib.colors.to_rgb(colors[node_id]))
        self.assertEqual(neighborhood, self.data["neighborhood"])

    def test_too_small_labeled_neighborhood_reports_overlap(self):
        figure, axis = assets.canvas(20, 20)
        with self.assertRaisesRegex(ValueError, "overlap"):
            assets.draw_hops(axis, self.data, 2)

    def test_sample_cards_keep_z_mu_paired_with_one_highlight(self):
        for selected in (1, 2, 3):
            data = {**self.data, "config": {**self.data["config"], "selected_sample": selected}}
            figure, axis = assets.canvas(36, 34)
            result = assets.draw_sample_selection(axis, data)
            self.assertEqual(result["selected_sample"], selected)
            self.assertEqual(len(result["cards"]), 3)
            self.assertEqual(sum(card.get_linewidth() == 1 for card in result["cards"]), 1)
            texts = {text.get_text() for text in axis.texts}
            self.assertIn("Fixed key", texts)
            for index in (1, 2, 3):
                self.assertIn(f"$(z_{index},\\,\\mu_{index})$", texts)

    def test_invalid_inventory_sample_size_and_style_fail_early(self):
        config = deepcopy(self.data["config"])
        config["spectrum_config"] = str(self.data["source_paths"]["spectrum_config"])
        config["neighborhood_data"] = str(self.data["source_paths"]["neighborhood_data"])
        for field, value in (("selected_sample", 0), ("selected_sample", True), ("dpi", -1)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                invalid = deepcopy(config)
                invalid[field] = value
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    assets.load_inputs(path)
        for counts in ({"2": 1.2}, {"2": -1}, {"2": True}, {"33": 1}, {"-1": 1}, {"2": 0}):
            with self.subTest(counts=counts), tempfile.TemporaryDirectory() as directory:
                invalid = deepcopy(config)
                invalid["inventory"]["counts"] = counts
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    assets.load_inputs(path)

    def test_all_exports_are_vector_or_declared_raster_with_source_data(self):
        input_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.data["source_paths"].values()}
        with tempfile.TemporaryDirectory() as directory:
            destination = assets.generate(output_dir=Path(directory), dpi=100)
            paths = sorted(destination.glob("*.svg"))
            self.assertEqual(len(paths), 9)
            for path in paths:
                root = ElementTree.fromstring(path.read_text())
                self.assertFalse(root.findall(".//{http://www.w3.org/2000/svg}image"))
                self.assertTrue(path.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
                with Image.open(path.with_suffix(".png")) as image:
                    self.assertAlmostEqual(image.info["dpi"][0], 100, delta=0.1)
            layer3 = (destination / "layer3_sample_selection.svg").read_text()
            self.assertIn("<text", layer3)
            self.assertIn("Fixed key", layer3)
            curves = np.loadtxt(destination / "curves.csv", delimiter=",", skiprows=1)
            np.testing.assert_array_equal(curves, self.data["curves"])
            inventory = np.loadtxt(destination / "illustrative_inventory.csv", delimiter=",", skiprows=1)
            self.assertEqual(len(inventory), 33)
            np.testing.assert_array_equal(inventory[:, 1], self.data["inventory"])
            metadata = json.loads((destination / "metadata.json").read_text())
            self.assertIn("SCHEMATIC ONLY", metadata["purpose"])
            self.assertEqual(metadata["spectra"]["rows_excluded"], 0)
            self.assertEqual(metadata["inventory"]["nodes_drawn"], self.data["inventory"].sum())
        self.assertEqual(input_hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_hashes})


if __name__ == "__main__":
    unittest.main()
