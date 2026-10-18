"""Canonical checkpoint readers for forward prediction and inverse setup."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .g2s.g2s_model import G2SConfig, NMR_VAE
    from .s2n.s2n_model import S2NConfig, S2NModel


def _checkpoint_parts(checkpoint) -> tuple[dict, dict]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("A checkpoint must contain a state dictionary")
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            if not isinstance(state, Mapping):
                raise TypeError(f"Checkpoint {key} must be a state dictionary")
            return dict(state), dict(checkpoint)
    return dict(checkpoint), {"legacy_state_dict_only": True}


def load_g2s_checkpoint(
    model_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    fallback_config: G2SConfig | None = None,
) -> tuple[NMR_VAE, G2SConfig, dict]:
    from .g2s.g2s_model import G2SConfig, NMR_VAE

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state, metadata = _checkpoint_parts(checkpoint)
    normalized = {
        "decoder." + key[len("decoder.decoder."):] if key.startswith("decoder.decoder.") else key: value
        for key, value in state.items()
    }
    config_data = metadata.get("config")
    if not config_data and fallback_config is not None:
        config_data = fallback_config.to_dict()
    config = G2SConfig.from_dict(config_data)
    decoder_weight = normalized.get("decoder.net.5.weight")
    if decoder_weight is not None:
        config.predict_intensity = int(decoder_weight.shape[0]) > 1
    model = NMR_VAE.from_config(config).to(device)
    model.load_state_dict(normalized)
    model.eval()
    return model, config, metadata


def load_s2n_checkpoint(
    model_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[S2NModel, S2NConfig, dict]:
    from .s2n.s2n_model import S2NConfig, S2NModel

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state, metadata = _checkpoint_parts(checkpoint)
    config_data = metadata.get("config")
    config = S2NConfig.from_dict(config_data) if config_data else S2NConfig(target_mode="absolute")
    model = S2NModel.from_config(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, config, metadata
