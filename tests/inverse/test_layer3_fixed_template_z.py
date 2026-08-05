from __future__ import annotations

import torch
import pytest

from model.inverse.layer3.inverse_layer3 import Layer3Estimator
from model.shared.coarse_graph import PPM_AXIS
from model.shared.inverse_common import _NodeV3, lorentzian_spectrum


class _FakeG2S:
    def global_mlp(self, elements: torch.Tensor) -> torch.Tensor:
        return torch.zeros((elements.shape[0], 2), dtype=elements.dtype, device=elements.device)

    def decoder(self, su_feat: torch.Tensor, z: torch.Tensor, global_embed: torch.Tensor) -> torch.Tensor:
        mu = 80.0 + 15.0 * z[:, :1]
        return torch.cat([mu, torch.zeros_like(mu)], dim=1)


def test_layer3_can_adjust_su19_above_90_without_changing_template() -> None:
    template_key = (19, (), ())
    estimator = Layer3Estimator(device=torch.device("cpu"), vae_model=_FakeG2S())
    estimator._template_cache = {
        "templates": {
            template_key: {
                "samples": {
                    "z": torch.tensor([[0.0], [1.0]]),
                    "mu": torch.tensor([80.0, 95.0]),
                }
            }
        }
    }
    node = _NodeV3(global_id=0, su_type=19)
    node.template_key = template_key
    node.z_vec = torch.tensor([0.0])
    node.mu = 80.0
    node.pi = 1.0
    target = lorentzian_spectrum(
        torch.tensor([95.0]),
        torch.ones(1),
        PPM_AXIS,
        hwhm=1.0,
    )

    result, r2 = estimator.layer3_adjust_templates(
        [node],
        S_target=target,
        E_target=torch.zeros(6),
        max_iters=2,
        pos_search_window=20.0,
        top_k_samples=2,
        verbose=False,
    )

    assert result[0].template_key == template_key
    assert result[0].mu == 95.0
    assert result[0].mu > 90.0
    assert result[0].pi == 1.0
    assert result[0].score_components["layer3_spectral_source"] == "template_sample_mu_layer3"
    assert result[0].score_components["spectral_source"] == "template_sample_mu_layer3"
    assert r2 > 0.99


def test_layer3_zero_move_assigns_every_carbon_node_from_paired_template_mu() -> None:
    template_key = (22, (), ())
    estimator = Layer3Estimator(device=torch.device("cpu"), vae_model=_FakeG2S())
    estimator._template_cache = {
        "templates": {
            template_key: {
                "samples": {
                    "z": torch.tensor([[0.25], [0.5]]),
                    "mu": torch.tensor([83.75, 87.5]),
                },
            }
        }
    }
    nodes = [_NodeV3(global_id=index, su_type=22) for index in range(2)]
    for node, z_value in zip(nodes, (0.25, 0.5)):
        node.template_key = template_key
        node.z_vec = torch.tensor([z_value])
        node.mu = -999.0

    result, _ = estimator.layer3_adjust_templates(
        nodes,
        S_target=torch.zeros_like(PPM_AXIS),
        E_target=torch.zeros(6),
        max_iters=0,
        verbose=False,
    )

    assert [node.mu for node in result] == [83.75, 87.5]
    assert all(node.pi == 1.0 for node in result)
    assert all(
        node.score_components["spectral_source"] == "template_sample_mu_layer3"
        for node in result
    )
    assert all(
        node.score_components["layer3_final_template_mu_error"] == 0.0
        for node in result
    )


def test_layer3_rolls_back_full_decode_when_it_is_worse_than_input() -> None:
    template_key = (22, (), ())
    estimator = Layer3Estimator(device=torch.device("cpu"), vae_model=_FakeG2S())
    estimator._template_cache = {
        "templates": {
            template_key: {
                "samples": {
                    "z": torch.tensor([[0.0], [0.5]]),
                    "mu": torch.tensor([80.0, 87.5]),
                },
            }
        }
    }
    node = _NodeV3(global_id=0, su_type=22)
    node.template_key = template_key
    node.z_vec = torch.tensor([0.0])
    node.mu = 110.0
    node.pi = 1.0
    node.score_components["spectral_source"] = "layer2_template_center_mu"
    target = lorentzian_spectrum(
        torch.tensor([110.0]),
        torch.ones(1),
        PPM_AXIS,
        hwhm=1.0,
    )

    result, r2 = estimator.layer3_adjust_templates(
        [node],
        S_target=target,
        E_target=torch.zeros(6),
        max_iters=0,
        verbose=False,
    )

    assert result[0].mu == 110.0
    assert torch.equal(result[0].z_vec, torch.tensor([0.0]))
    assert result[0].score_components["spectral_source"] == "layer2_template_center_mu"
    assert r2 > 0.99
    assert estimator.last_diagnostics["accepted"] is False
    assert estimator.last_diagnostics["candidate_r2"] < estimator.last_diagnostics["baseline_r2"]


class _FailingG2S(_FakeG2S):
    def decoder(self, su_feat: torch.Tensor, z: torch.Tensor, global_embed: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("decoder failure")


def test_layer3_transition_does_not_call_g2s_decoder() -> None:
    template_key = (22, (), ())
    estimator = Layer3Estimator(device=torch.device("cpu"), vae_model=_FailingG2S())
    estimator._template_cache = {
        "templates": {
            template_key: {
                "samples": {
                    "z": torch.tensor([[0.0]]),
                    "mu": torch.tensor([80.0]),
                }
            }
        }
    }
    node = _NodeV3(global_id=0, su_type=22)
    node.template_key = template_key
    node.z_vec = torch.tensor([0.0])

    result, _ = estimator.layer3_adjust_templates(
        [node],
        S_target=torch.zeros_like(PPM_AXIS),
        E_target=torch.zeros(6),
        max_iters=0,
        verbose=False,
    )

    assert result[0].mu == 80.0
    assert result[0].score_components["spectral_source"] == "template_sample_mu_layer3"


def test_layer3_transition_rejects_template_without_paired_mu_bank() -> None:
    template_key = (22, (), ())
    estimator = Layer3Estimator(device=torch.device("cpu"), vae_model=None)
    estimator._template_cache = {
        "templates": {template_key: {"samples": {"z": torch.tensor([[0.0]])}}}
    }
    node = _NodeV3(global_id=0, su_type=22)
    node.template_key = template_key
    node.z_vec = torch.tensor([0.0])

    with pytest.raises(RuntimeError, match="has no mu sample bank"):
        estimator.layer3_adjust_templates(
            [node],
            S_target=torch.zeros_like(PPM_AXIS),
            E_target=torch.zeros(6),
            max_iters=0,
            verbose=False,
        )
