"""Readability, exact geometry, vector content and output integration."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from assembly_viewer.demo import demo_state
from assembly_viewer.export import export_molecule, render_state
from assembly_viewer.adapter import record_state
from assembly_viewer.tests.test_trace import settings, baseline


def test_labels_fit_nodes_and_scale_with_viewport(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    original = Figure.savefig
    measurements = []

    def measure(figure, *args, **kwargs):
        figure.canvas.draw()
        axis = figure.axes[0]
        renderer = figure.canvas.get_renderer()
        for circle, label in zip(axis.patches, axis.texts):
            diameter = circle.get_window_extent(renderer).width
            box = label.get_window_extent(renderer)
            assert box.width <= diameter * .9
            assert box.height <= diameter * .9
            assert box.x0 >= circle.get_window_extent(renderer).x0
            assert box.x1 <= circle.get_window_extent(renderer).x1
        diameter = axis.patches[0].get_window_extent(renderer).width
        measurements.append((diameter, axis.texts[0].get_fontsize() / diameter))
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", measure)
    state = demo_state().to_dict()
    render_state(state, tmp_path / "small.svg", bounds=(-10, 10, -10, 10))
    render_state(state, tmp_path / "large.svg", bounds=(-60, 60, -60, 60))
    assert measurements[0][0] > measurements[1][0]
    assert measurements[0][1] == pytest.approx(measurements[1][1], rel=1e-10)


def test_legacy_result_exports_editable_vector_without_assembly(tmp_path, monkeypatch):
    import RL_MTCS.runner as engine
    def forbidden(*args, **kwargs):
        raise AssertionError("saved molecule export must not start assembly")
    monkeypatch.setattr(engine.MCTSRunner, "__init__", forbidden)
    molecule = demo_state().to_dict()
    before = deepcopy(molecule)
    source = tmp_path / "sample.final.json"
    source.write_text(json.dumps(dict(molecule=molecule)))
    original_bytes = source.read_bytes()
    paths = export_molecule(source)
    assert {Path(p).suffix for p in paths} == {".svg", ".pdf", ".png"}
    assert all(Path(p).stem == "sample.final" for p in paths)
    svg = ET.parse(next(p for p in paths if p.endswith(".svg")))
    ns = {"s": "http://www.w3.org/2000/svg"}
    identifiers = {e.attrib.get("id") for e in svg.iter()}
    assert {f"node-{n['node_id']}" for n in molecule['nodes']} <= identifiers
    assert {f"label-{n['node_id']}" for n in molecule['nodes']} <= identifiers
    assert len(svg.findall(".//s:text", ns)) == len(molecule["nodes"])
    assert not svg.findall(".//s:image", ns)
    pdf = next(p for p in paths if p.endswith(".pdf"))
    assert Path(pdf).read_bytes().startswith(b"%PDF-")
    assert b"/FontFile2" in Path(pdf).read_bytes()
    if shutil.which("pdftotext"):
        content = subprocess.check_output(["pdftotext", pdf, "-"], text=True)
        assert "13" in content
    assert molecule == before and source.read_bytes() == original_bytes


def test_distant_components_do_not_allocate_unbounded_canvas(tmp_path):
    state = demo_state().to_dict()
    state["nodes"][-1]["final_position"] = [100000, 100000]
    path = render_state(state, tmp_path / "distant.svg", main_only=False)
    root = ET.parse(path).getroot()
    assert float(root.attrib["width"].removesuffix("pt")) <= 20 * 72
    assert float(root.attrib["height"].removesuffix("pt")) <= 20 * 72
    assert len(list(root.iter("{http://www.w3.org/2000/svg}text"))) == len(state["nodes"])


def test_stage_and_final_vectors_saved_with_original_structure(tmp_path):
    config = replace(settings(), save_stage_images=True)
    initial = demo_state()
    result = record_state(initial, output_dir=tmp_path, config=config)
    expected = baseline(initial, config)
    assert result["state"].state_signature() == expected["state"].state_signature()
    assert result["state"].action_history == expected["state"].action_history
    for phase, payload in result["stages"].items():
        summary = json.loads((tmp_path / phase / f"{phase}_summary.json").read_text())
        for ext in ("svg", "pdf", "png"):
            path = tmp_path / phase / f"{phase}_molecule.{ext}"
            assert path.is_file()
            assert Path(summary["stage_outputs"][ext]) == path
            assert payload["stage_outputs"][ext] == str(path)
    for ext in ("svg", "pdf", "png"):
        assert Path(result["outputs"][ext]).is_file()
    saved = json.loads((tmp_path / "final.json").read_text())
    assert saved["outputs"] == result["outputs"]


@pytest.mark.parametrize("option", [{"formats": ()}, {"formats": ("jpg",)}, {"dpi": 0}])
def test_invalid_export_request_reports_error(tmp_path, option):
    source = tmp_path / "final.json"
    source.write_text(json.dumps(demo_state().to_dict()))
    with pytest.raises(ValueError):
        export_molecule(source, **option)
