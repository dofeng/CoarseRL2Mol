"""Configuration for staged assembly; candidate caps are not scan budgets."""

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple
from RL_MTCS.rewards.scoring import DEFAULT_REWARD_WEIGHTS, normalize_reward_weights


@dataclass
class AssemblyConfig:
    """Cap legal proposals per action and distinct results per selection round."""

    connection_candidate_limit: int = 10
    forward_min_completion: float = 0.80
    forward_max_rollbacks: int = 4
    c_puct: float = 1.414
    seed: Optional[int] = 7
    spectrum_csv: Optional[str] = None
    elements: Any = None
    element_composition_csv: Optional[str] = None
    g2s_checkpoint: Optional[str] = None
    nmr_device: str = "auto"
    nmr_batch_size: int = 1
    nmr_graph_cache_limit: int = 256
    nmr_node_cache_limit: int = 2048
    nmr_record_limit: int = 5000
    nmr_center_limit: int = 8
    nmr_aromatic_response_scale: float = 1.0
    nmr_hwhm: float = 1.0
    # Unified staged reward: completion, NMR, forward rollout, SU13 clearance,
    # and shape. These are base weights: spatial demand transfers unused
    # clearance weight to compactness/aspect within each frozen search round.
    # Forward completion is measured only at checkpoints.
    reward_weights: Tuple[float, ...] = DEFAULT_REWARD_WEIGHTS
    forward_checkpoint_interval: int = 4
    clearance_radius: int = 3
    # Report stage/checkpoint events and periodic search memory diagnostics.
    verbose: bool = True
    save_stage_images: bool = True

    def __post_init__(self) -> None:
        for name, minimum in (('connection_candidate_limit', 1), ('forward_max_rollbacks', 0),
                              ('forward_checkpoint_interval', 1), ('clearance_radius', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not math.isfinite(self.forward_min_completion) or not 0 <= self.forward_min_completion <= 1:
            raise ValueError("forward_min_completion must be in [0, 1]")
        if not math.isfinite(self.c_puct) or self.c_puct <= 0:
            raise ValueError("c_puct must be finite and positive")
        self.nmr_batch_size = max(1, int(self.nmr_batch_size))
        self.nmr_graph_cache_limit = min(256, max(1, int(self.nmr_graph_cache_limit)))
        self.nmr_node_cache_limit = min(2048, max(1, int(self.nmr_node_cache_limit)))
        self.nmr_record_limit = min(5000, max(0, int(self.nmr_record_limit)))
        self.nmr_center_limit = min(32, max(1, int(self.nmr_center_limit)))
        self.nmr_hwhm = max(1e-6, float(self.nmr_hwhm))
        self.reward_weights = normalize_reward_weights(self.reward_weights)
        self.verbose = bool(self.verbose)
        self.save_stage_images = bool(self.save_stage_images)
