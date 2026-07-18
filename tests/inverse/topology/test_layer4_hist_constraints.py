from __future__ import annotations

import torch

from model.inverse.layer4.inverse_layer4 import Layer4Adjuster


def test_required_hist_constraints_reject_hydrogen_error_above_tolerance() -> None:
    adjuster = Layer4Adjuster(device=torch.device("cpu"))
    H = torch.zeros(33, dtype=torch.long)
    H[22] = 1
    E_target = torch.zeros(6, dtype=torch.float)
    E_target[0] = 1.0

    result = adjuster._evaluate_required_hist_constraints(
        H,
        E_target,
        enforce_h_rel=True,
    )

    assert result["h_ok"] is False
    assert result["current_h"] == 3.0
    assert result["target_h"] == 0.0
    assert result["h_rel"] == 3.0
    assert "元素H误差过大" in result["reason"]


def test_aromatic_balance_does_not_enforce_su13_minimum() -> None:
    adjuster = Layer4Adjuster(device=torch.device("cpu"))
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 2

    result = adjuster._evaluate_aromatic_balance_constraints(H)

    assert result["su13"] == 0
    assert result["su13_min"] == 0
    assert result["su13_ok"] is True
    assert result["ok"] is True
