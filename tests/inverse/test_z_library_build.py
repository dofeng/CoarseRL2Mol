from __future__ import annotations

import pickle

import torch
import pytest

import model.inverse.library.z_library as z_library
from model.forward.g2s.g2s_model import G2SConfig, NMR_VAE
from model.forward.g2s.g2s_model import LocalNMRDataset
from model.inverse.library.z_library import (
    MIN_SOURCE_GRAPH_NODES,
    _filter_source_graphs,
    _load_library_g2s_checkpoint,
    filter_subgraph_library,
    merge_3hop_libraries_v2,
)


class _AttrGraph:
    def __init__(self, node_count: int) -> None:
        self.x = torch.zeros((int(node_count), 39), dtype=torch.float)


def test_source_graph_filter_excludes_only_graphs_below_five_nodes() -> None:
    graphs = [
        {"x": torch.zeros((4, 39), dtype=torch.float)},
        {"x": torch.zeros((5, 39), dtype=torch.float)},
        _AttrGraph(7),
    ]

    kept, excluded = _filter_source_graphs(graphs)

    assert MIN_SOURCE_GRAPH_NODES == 5
    assert excluded == 1
    assert [_graph.x.size(0) if hasattr(_graph, "x") else _graph["x"].size(0) for _graph in kept] == [5, 7]


def test_build_filters_small_graphs_before_subgraph_extraction(monkeypatch) -> None:
    seen_node_counts = []

    class _StopAfterDatasetInput(RuntimeError):
        pass

    def fake_dataset(graphs, k, **_kwargs):
        del k
        seen_node_counts.extend(int(graph["x"].size(0)) for graph in graphs)
        raise _StopAfterDatasetInput

    config = G2SConfig(k_hop=3)
    monkeypatch.setattr(
        z_library,
        "load_raw_data",
        lambda _path: [
            {"x": torch.zeros((4, 39), dtype=torch.float)},
            {"x": torch.zeros((5, 39), dtype=torch.float)},
        ],
    )
    monkeypatch.setattr(
        z_library,
        "_load_library_g2s_checkpoint",
        lambda **_kwargs: (object(), config, {}),
    )
    monkeypatch.setattr(z_library, "LocalNMRDataset", fake_dataset)

    with pytest.raises(_StopAfterDatasetInput):
        z_library.build_3hop_library_v2(
            "unused",
            "unused",
            device="cpu",
            num_workers=0,
        )

    assert seen_node_counts == [5]


def test_local_dataset_structure_mode_keeps_centers_without_real_ppm() -> None:
    node_count = 5
    x = torch.zeros((node_count, 39), dtype=torch.float)
    x[:, 0] = 1.0
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    graph = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": torch.ones((4, 2), dtype=torch.float),
        "is_carbon": torch.ones(node_count, dtype=torch.bool),
        "real_ppm": torch.zeros(node_count),
        "real_intensity": torch.zeros(node_count),
        "su_hist": x[:, :33].sum(0),
        "total_atom_counts": torch.tensor([5.0, 0, 0, 0, 0, 0]),
    }

    dataset = LocalNMRDataset([graph], k=3, require_real_ppm=False)

    assert len(dataset) == node_count
    assert torch.equal(dataset[0].real_ppm, torch.zeros(1))
    with pytest.raises(ValueError, match="No valid carbon-centered"):
        LocalNMRDataset([graph], k=3, require_real_ppm=True)


def test_wrapped_g2s_checkpoint_uses_its_saved_config(tmp_path) -> None:
    config = G2SConfig(k_hop=3, hid=16, latent_dim=4, n_layers=2, predict_intensity=False)
    original = NMR_VAE.from_config(config)
    checkpoint_path = tmp_path / "wrapped_g2s.pt"
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "config": config.to_dict(),
            "epoch": 3,
        },
        checkpoint_path,
    )

    loaded, loaded_config, metadata = _load_library_g2s_checkpoint(
        str(checkpoint_path),
        device="cpu",
        k_hop=3,
        hid_dim=384,
        latent_dim=16,
    )

    assert loaded_config.hid == 16
    assert loaded_config.latent_dim == 4
    assert loaded_config.predict_intensity is False
    assert metadata["epoch"] == 3
    for key, value in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_g2s_checkpoint_rejects_k_hop_mismatch(tmp_path) -> None:
    config = G2SConfig(k_hop=2, hid=16, latent_dim=4, n_layers=2)
    model = NMR_VAE.from_config(config)
    checkpoint_path = tmp_path / "k2_g2s.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict()},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="k_hop=2.*k_hop=3"):
        _load_library_g2s_checkpoint(
            str(checkpoint_path),
            device="cpu",
            k_hop=3,
            hid_dim=384,
            latent_dim=16,
        )


def test_filter_library_keeps_only_matching_hop1_port_templates(tmp_path) -> None:
    valid_key = (4, (23,), ())
    invalid_key = (4, (13,), ())
    low_count_key = (4, (23,), (1,))
    template_info = {
        "sample_count": 3,
        "mu_min": 110.0,
        "mu_max": 112.0,
        "center_mu": 111.0,
    }
    input_path = tmp_path / "raw" / "subgraph_library.pt"
    output_path = tmp_path / "filtered" / "subgraph_library_filtered.pt"
    input_path.parent.mkdir(parents=True)
    torch.save(
        {
            "templates": {
                valid_key: dict(template_info),
                invalid_key: dict(template_info),
                low_count_key: {**template_info, "sample_count": 2},
            },
            "center_index": {4: [valid_key, invalid_key, low_count_key]},
            "role_index": {},
            "library_type": "3hop_enhanced_v2",
            "build_stats": {"source_graphs_excluded_small": 2},
            "g2s_config": {"k_hop": 3},
        },
        input_path,
    )

    filtered = filter_subgraph_library(
        str(input_path),
        str(output_path),
        verbose=False,
    )

    assert list(filtered["templates"]) == [valid_key]
    assert filtered["center_index"] == {4: [valid_key]}
    assert filtered["filter_stats"]["removed_count"] == 2
    assert filtered["filter_stats"]["min_samples"] == 3
    assert filtered["build_stats"]["source_graphs_excluded_small"] == 2
    assert output_path.exists()
    with (output_path.parent / "indexes" / "center_index.pkl").open("rb") as file_obj:
        assert pickle.load(file_obj) == {4: [valid_key]}


def test_library_merge_recomputes_range_and_accumulates_build_stats() -> None:
    key = (4, (23,), ())

    def entry(mu: float, sample_count: int) -> dict:
        z = torch.tensor([[mu, 0.0]], dtype=torch.float)
        return {
            "samples": {
                "z": z,
                "mu": torch.tensor([mu]),
                "pi": torch.ones(1),
            },
            "center_sample_idx": 0,
            "center_z": z[0],
            "center_mu": mu,
            "center_pi": 1.0,
            "mu_min": mu,
            "mu_max": mu,
            "nmr_range": (mu, mu),
            "center_su": 4,
            "hop1_counts": {23: 1},
            "hop2_counts": {},
            "sample_count": sample_count,
        }

    base = {
        "templates": {key: entry(100.0, 1)},
        "center_index": {4: [key]},
        "role_index": {},
        "build_stats": {"source_graphs_loaded": 2},
    }
    addition = {
        "templates": {key: entry(120.0, 1)},
        "center_index": {4: [key]},
        "role_index": {},
        "build_stats": {"source_graphs_loaded": 3},
        "g2s_config": {"k_hop": 3},
    }

    merged = merge_3hop_libraries_v2(base, addition)
    merged_entry = merged["templates"][key]

    assert merged_entry["sample_count"] == 2
    assert merged_entry["nmr_range"] == (100.0, 120.0)
    assert merged["build_stats"]["source_graphs_loaded"] == 5
    assert merged["g2s_config"] == {"k_hop": 3}


def test_library_merge_rejects_incompatible_g2s_config() -> None:
    with pytest.raises(ValueError, match="incompatible G2S configs"):
        merge_3hop_libraries_v2(
            {"templates": {}, "g2s_config": {"k_hop": 3, "latent_dim": 16}},
            {"templates": {}, "g2s_config": {"k_hop": 3, "latent_dim": 8}},
        )
