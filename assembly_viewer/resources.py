"""Immutable input catalog and provenance; preview never runs search or NMR."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from .recorder import json_text


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def provenance(nodes_csv, topology_csv, config):
    project = Path(__file__).resolve().parents[1]
    paths = {"nodes": Path(nodes_csv).resolve(), "topology": Path(topology_csv).resolve()}
    for name, filename in (("aromatic_clusters", "final_aromatic_clusters.json"),
                           ("topology_resource_pool", "final_topology_resource_pool.csv")):
        sidecar = paths["nodes"].parent / filename
        if sidecar.is_file():
            paths[name] = sidecar
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project,
                              capture_output=True, text=True, check=False).stdout.strip()
    return dict(config=asdict(config), inputs={name: dict(path=str(path), sha256=checksum(path))
                for name, path in paths.items()}, revision=revision, recording="external_observer",
                source="RL_MTCS", engine_sha256={str(path.relative_to(project)): checksum(path)
                for path in sorted((project / "RL_MTCS").rglob("*.py"))})


def resource_catalog(state):
    from RL_MTCS.mdp.components import phase_of
    from RL_MTCS.actions.side import is_complex_side
    problem = state.source_problem
    original = problem.input_node_map()
    planned = problem.node_map()
    targets = problem.assembly_target_node_map()
    units = []
    for unit in problem.units:
        row = asdict(unit)
        row["phase"] = phase_of(unit)
        row["node_count"] = len(unit.nodes)
        if row["phase"] == "side":
            row["side_category"] = "complex" if is_complex_side(unit) else "simple"
        center_parents = {child: parent for parent, child in unit.su25_center_branches}
        for node in row["nodes"]:
            gid = node["node_id"]
            node.update(original_su=original.get(gid, node["su_type"]),
                        planned_su=planned.get(gid, node["su_type"]),
                        assembly_target_su=targets.get(gid, node["su_type"]),
                        residual_conversion=gid in problem.aromatic_residual_node_ids,
                        su25_center_parent_id=center_parents.get(gid))
        units.append(row)
    return dict(units=units, contracts=[asdict(c) for c in problem.contracts],
                substitutions=[asdict(r) for r in problem.hetero_requests],
                aromatic_residual_nodes=problem.aromatic_residual_rows(),
                counts=dict(Counter(row["phase"] for row in units)),
                node_count=len(original), original_su_counts=dict(Counter(str(su) for su in original.values())),
                input_issues=[json.loads(issue) for issue in problem.input_issues])


def register_input(recorder, state):
    catalog = resource_catalog(state)
    (recorder.directory / "resources.json").write_text(json_text(catalog), encoding="utf-8")
    frame = recorder.capture(state, snapshot=True)
    recorder.update_manifest(initial_state=frame, latest_formal_state=frame,
                             original_su_counts=catalog["original_su_counts"], resource_counts=catalog["counts"])
    seq = recorder.emit("input.ready", scope="input", phase="input", state_id=frame,
                        data=dict(counts=catalog["counts"], node_count=catalog["node_count"]))
    recorder.phase_start("input", seq, frame)
    recorder.phase_end("input", seq, frame)
    return frame
