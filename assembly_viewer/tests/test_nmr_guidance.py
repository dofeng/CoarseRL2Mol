from types import SimpleNamespace

import pytest
import torch

from RL_MTCS import AssemblyConfig
from RL_MTCS.runner import MCTSRunner
from assembly_viewer.tests.test_trace import minimal_csv_inputs
from model.forward.g2s import rl_mtcs_utils


@pytest.mark.parametrize(
    "composition",
    [
        "C=560,H=388,O=20,N=10,S=4,X=0",
        "c=560,h=388,o=20,n=10,s=4,x=0",
        "C560H388O20N10S4",
        " C = 560, H = 388, O = 20, N = 10, S = 4, X = 0 ",
    ],
)
def test_element_composition_expression(composition):
    parsed = rl_mtcs_utils.parse_element_composition(composition)

    torch.testing.assert_close(
        parsed, torch.tensor([560, 388, 20, 10, 4, 0], dtype=torch.float)
    )


def test_runner_initializes_nmr_guidance_with_composition_expression(tmp_path, monkeypatch):
    nodes_csv, topology_csv = minimal_csv_inputs(tmp_path / "inputs")
    spectrum_csv = tmp_path / "spectrum.csv"
    spectrum_csv.write_text("ppm,intensity\n0,1\n100,2\n200,1\n", encoding="utf-8")
    checkpoint = tmp_path / "g2s.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        rl_mtcs_utils,
        "load_g2s_checkpoint",
        lambda *args, **kwargs: (torch.nn.Identity(), SimpleNamespace(k_hop=3), {}),
    )
    config = AssemblyConfig(
        spectrum_csv=str(spectrum_csv),
        element_composition_csv="C=560,H=388,O=20,N=10,S=4,X=0",
        g2s_checkpoint=str(checkpoint),
        nmr_device="cpu",
        verbose=False,
        save_stage_images=False,
    )

    runner = MCTSRunner(nodes_csv, config=config, topology_specs_csv=topology_csv)

    assert runner.nmr_scorer is not None
    assert runner.nmr_scorer.k_hop == 3
    torch.testing.assert_close(
        runner.nmr_scorer.elements,
        torch.tensor([560, 388, 20, 10, 4, 0], dtype=torch.float),
    )
    assert torch.isfinite(runner.nmr_scorer.target_spectrum).all()
    assert runner.nmr_scorer.target_spectrum.sum().item() > 0
