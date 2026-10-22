import json
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from cg2mol import (
    AllAtomToRDKit,
    RDKitConversionError,
    convert_allatom_to_rdkit,
    convert_cg_to_allatom,
    convert_rl_mtcs_file,
    export_molecule,
    simplify_cg_molecule,
)
from cg2mol.cg_to_allatom import CG2MolConversionError


def assembled_graph(su_types, edges, *, complete=True, action_history=()):
    molecule = {
        "nodes": [
            {"node_id": index + 100, "su_type": su_type, "axial": [index, 0]}
            for index, su_type in enumerate(su_types)
        ],
        "edges": [
            {"source_node_id": left + 100, "target_node_id": right + 100, "kind": kind}
            for left, right, kind in edges
        ],
        "action_history": list(action_history),
        "audit": {"complete": complete, "strict_all_nodes_assembled": complete},
    }
    return {
        "status": "complete" if complete else "partial",
        "all_complete": complete,
        "legal": True,
        "molecule": molecule,
    }


def reconstruct(graph):
    allatom = convert_cg_to_allatom(simplify_cg_molecule(graph))
    molecule = convert_allatom_to_rdkit(allatom, verbose=False)
    assert Chem.SanitizeMol(Chem.Mol(molecule), catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
    assert not Chem.DetectChemistryProblems(Chem.Mol(molecule))
    return allatom, molecule


def test_current_graph_identity_and_element_conservation():
    graph = assembled_graph([22, 22], [(0, 1, "side")])
    graph["molecule"]["nodes"][0].update(original_su_type=13, planned_su_type=13)
    allatom, molecule = reconstruct(graph)
    assert allatom["metadata"]["element_counts"] == {"C": 2, "H": 6}
    assert molecule.GetNumAtoms() == 8
    assert set(allatom["node_id_to_atom_map"]) == {"100", "101"}
    assert allatom["atoms"][0]["su_type"] == 22


@pytest.mark.parametrize(
    "su_types,edges,expected_h",
    [
        ([19, 29, 19], [(0, 1, "side"), (1, 2, "side")], 6),
        ([19, 28, 22], [(0, 1, "side"), (0, 2, "side")], 6),
        ([19, 28, 22, 22], [(0, 1, "side"), (0, 2, "side"), (0, 3, "side")], 8),
        ([21, 32, 22], [(0, 1, "side"), (0, 2, "side")], 5),
    ],
)
def test_special_carbon_hydrogens_follow_actual_degree(su_types, edges, expected_h):
    allatom, _ = reconstruct(assembled_graph(su_types, edges))
    assert allatom["metadata"]["element_counts"]["H"] == expected_h


@pytest.mark.parametrize("su_type,hetero_type", [(0, 20), (2, 19)])
def test_carbonyl_ports_assigned_jointly(su_type, hetero_type):
    graph = assembled_graph(
        [su_type, hetero_type, 23, 22],
        [(0, 1, "flex"), (0, 2, "flex"), (2, 3, "side")],
    )
    allatom, _ = reconstruct(graph)
    primary_atoms = allatom["cg_to_atom_map"]["0"]
    hetero_edge = next(detail for detail in allatom["cg_bond_details"] if detail["target_seq"] == 1)
    body_edge = next(detail for detail in allatom["cg_bond_details"] if detail["target_seq"] == 2)
    assert hetero_edge["atom_indices"][0] == primary_atoms[2]
    assert body_edge["atom_indices"][0] == primary_atoms[0]


def test_adjacent_unsaturated_pairs_do_not_all_become_double():
    allatom, _ = reconstruct(assembled_graph(
        [16, 15, 15, 16],
        [(0, 1, "flex"), (1, 2, "flex"), (2, 3, "flex")],
    ))
    assert [detail["bond_type"] for detail in allatom["cg_bond_details"]] == ["DOUBLE", "SINGLE", "DOUBLE"]
    assert allatom["metadata"]["element_counts"] == {"C": 4, "H": 6}


def test_unpairable_unsaturation_is_rejected():
    graph = assembled_graph([16, 15, 16], [(0, 1, "flex"), (1, 2, "flex")])
    with pytest.raises(CG2MolConversionError, match="perfect matching"):
        convert_cg_to_allatom(simplify_cg_molecule(graph))


def test_interaryl_rigid_edge_remains_single():
    edges = [(index, (index + 1) % 6, "aromatic") for index in range(6)]
    edges += [(index + 6, (index + 1) % 6 + 6, "aromatic") for index in range(6)]
    edges.append((0, 6, "rigid"))
    allatom, _ = reconstruct(assembled_graph([10, *([13] * 5), 10, *([13] * 5)], edges))
    assert allatom["cg_bond_details"][-1]["bond_type"] == "SINGLE"
    assert allatom["metadata"]["element_counts"] == {"C": 12, "H": 10}


@pytest.mark.parametrize("with_provenance", [True, False])
def test_pyrrolic_nitrogen_is_not_pyridine_nitrogen(with_provenance):
    action = [{"kind": "substitution", "target_su": 26, "ring_size": 5, "request_target": 100}]
    graph = assembled_graph(
        [26, 13, 13, 13, 13],
        [(index, (index + 1) % 5, "aromatic") for index in range(5)],
        action_history=action if with_provenance else (),
    )
    allatom, _ = reconstruct(graph)
    assert allatom["metadata"]["element_counts"] == {"N": 1, "H": 5, "C": 4}


def test_pyridine_and_thiophene():
    for hetero_type, ring_size, expected in (
        (26, 6, {"N": 1, "C": 5, "H": 5}),
        (30, 5, {"S": 1, "C": 4, "H": 4}),
    ):
        graph = assembled_graph(
            [hetero_type, *([13] * (ring_size - 1))],
            [(index, (index + 1) % ring_size, "aromatic") for index in range(ring_size)],
        )
        allatom, _ = reconstruct(graph)
        assert allatom["metadata"]["element_counts"] == expected


@pytest.mark.parametrize("wrapped", [True, False])
def test_partial_assembly_rejected_including_raw_molecule(wrapped):
    payload = assembled_graph([22, 22], [(0, 1, "side")], complete=False)
    if not wrapped:
        payload = payload["molecule"]
    with pytest.raises(CG2MolConversionError, match="not complete"):
        convert_cg_to_allatom(payload)


def test_wrong_su_degree_is_not_silently_relabelled():
    with pytest.raises(CG2MolConversionError, match="SU 22 degree=2"):
        convert_cg_to_allatom(assembled_graph([22, 22, 22], [(0, 1, "side"), (1, 2, "side")]))


def test_duplicate_and_dangling_edges_rejected():
    for edges in (
        [(0, 1, "side"), (0, 1, "side")],
        [(0, 2, "side")],
    ):
        with pytest.raises(ValueError):
            simplify_cg_molecule(assembled_graph([22, 22], edges))


def test_edge_details_and_declared_degree_cannot_hide_connectivity_errors():
    simplified = simplify_cg_molecule(assembled_graph([22, 22], [(0, 1, "side")]))
    simplified["bonds"] = []
    with pytest.raises(CG2MolConversionError, match="different connectivity"):
        convert_cg_to_allatom(simplified)
    graph = assembled_graph([22, 22], [(0, 1, "side")])
    graph["molecule"]["nodes"][0]["actual_degree"] = 2
    with pytest.raises(CG2MolConversionError, match="actual_degree"):
        convert_cg_to_allatom(graph)


def test_disconnected_and_unknown_su_graphs_are_rejected():
    with pytest.raises(CG2MolConversionError, match="unsupported SU"):
        convert_cg_to_allatom(assembled_graph([33, 22], [(0, 1, "side")]))
    graph = assembled_graph([22, 22, 22, 22], [(0, 1, "side"), (2, 3, "side")])
    allatom = convert_cg_to_allatom(graph)
    with pytest.raises(RDKitConversionError, match="disconnected"):
        convert_allatom_to_rdkit(allatom, verbose=False)


def test_raw_final_molecule_input_works(tmp_path):
    source = tmp_path / "final_molecule.json"
    source.write_text(json.dumps(assembled_graph([22, 22], [(0, 1, "side")])["molecule"]))
    paths = convert_rl_mtcs_file(str(source), str(tmp_path / "output"), coord_mode="2d", verbose=False)
    assert json.loads(Path(paths["report"]).read_text())["formula"] == "C2H6"


def test_strict_converter_rejects_valence_failure():
    allatom, _ = reconstruct(assembled_graph([22, 22], [(0, 1, "side")]))
    carbon_indices = [atom["idx"] for atom in allatom["atoms"] if atom["element"] == "C"]
    for bond in allatom["bonds"]:
        if set(bond[:2]) == set(carbon_indices):
            bond[2] = "DOUBLE"
    with rdBase.BlockLogs(), pytest.raises(RDKitConversionError, match="sanitization"):
        convert_allatom_to_rdkit(allatom, verbose=False)


def test_embedding_and_optimization_failure_do_not_export(tmp_path, monkeypatch):
    _, molecule = reconstruct(assembled_graph([22, 22], [(0, 1, "side")]))
    target = tmp_path / "failed.sdf"
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *arguments: -1)
    with pytest.raises(RDKitConversionError, match="embedding"):
        export_molecule(molecule, str(target), verbose=False)
    assert not target.exists()
    monkeypatch.setattr(AllChem, "MMFFHasAllMoleculeParams", lambda *arguments: True)
    monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", lambda *arguments, **kwargs: 1)
    with pytest.raises(RDKitConversionError, match="did not converge"):
        AllAtomToRDKit(verbose=False).optimize_3d(molecule)


def test_large_rl_graph_uses_cg_seed_without_global_embedding(monkeypatch):
    node_count = 102
    graph = assembled_graph(
        [22, *([23] * (node_count - 2)), 22],
        [(index, index + 1, "flex") for index in range(node_count - 1)],
    )
    _, molecule = reconstruct(graph)
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *arguments: pytest.fail("unexpected global embedding"))
    monkeypatch.setattr(AllAtomToRDKit, "optimize_3d", lambda self, candidate, **kwargs: Chem.Mol(candidate))
    prepared = AllAtomToRDKit(verbose=False).generate_3d_coords(molecule)
    assert prepared.GetNumAtoms() == molecule.GetNumAtoms()
    assert prepared.GetProp("cg2mol_coordinate_method") == "cg_layout_seeded"
    assert prepared.GetConformer().Is3D()
    assert any(abs(position[2]) > 0.01 for position in prepared.GetConformer().GetPositions())


def test_one_conformer_shared_by_all_export_formats(tmp_path, monkeypatch):
    payload = assembled_graph([22, 22], [(0, 1, "side")])
    source = tmp_path / "final.json"
    source.write_text(json.dumps(payload))
    calls = []
    original_embed = AllChem.EmbedMolecule

    def counted_embed(*arguments):
        calls.append(1)
        return original_embed(*arguments)

    monkeypatch.setattr(AllChem, "EmbedMolecule", counted_embed)
    paths = convert_rl_mtcs_file(str(source), str(tmp_path / "output"), formats=("sdf", "mol", "pdb", "xyz"), verbose=False)
    assert len(calls) == 1
    allatom = json.loads(Path(paths["allatom_json"]).read_text())
    report = json.loads(Path(paths["report"]).read_text())
    molecule = Chem.SDMolSupplier(paths["sdf"], removeHs=False)[0]
    assert molecule.GetNumAtoms() == len(allatom["atoms"]) == 8
    assert report["optimization_converged"] is True
    assert report["sanitized"] is True
    assert report["formula"] == "C2H6"
    assert "$$$$" not in Path(paths["mol"]).read_text()
    for atom_data in allatom["atoms"]:
        position = molecule.GetConformer().GetAtomPosition(atom_data["idx"])
        assert [position.x, position.y, position.z] == pytest.approx(atom_data["coord"], abs=1e-4)


def test_saved_ym1_graph_regression_if_available():
    root = Path(__file__).resolve().parents[1]
    source = root / "test_results/YM-1/rl_mtcs_nmr/20261019_142850_682077/final.json"
    if not source.exists():
        pytest.skip("local YM-1 inference output is not distributed with source code")
    allatom, molecule = reconstruct(json.loads(source.read_text()))
    assert allatom["metadata"]["element_counts"] == {"C": 499, "H": 516, "N": 6, "O": 166}
    assert molecule.GetNumAtoms() == 1187
    assert Counter(detail["bond_type"] for detail in allatom["cg_bond_details"] if detail["kind"] == "rigid") == {"SINGLE": 13}
