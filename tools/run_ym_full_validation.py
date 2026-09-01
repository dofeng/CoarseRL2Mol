from pathlib import Path
import os
import sys

ROOT = Path("/home/dofengqi/Graphnmr")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.inverse.inverse_pipeline import (
    load_pipeline,
    parse_elements,
    read_spectrum_csv,
)
from model.shared.inverse_common import normalize_spectrum_to_carbon_count


RUN_ID = os.environ.get("YM_RUN_ID", "full-2round")
OUTPUT_DIR = Path(os.environ.get(
    "YM_OUTPUT_DIR",
    str(ROOT / "test_results" / f"YM-full-validation-{RUN_ID}"),
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = load_pipeline(
    s2n_ckpt=str(ROOT / "checkpoints_s2n_v1" / "best_s2n_model.pt"),
    vae_ckpt=str(ROOT / "checkpoints_g2s_k3" / "best_g2s_model.pt"),
    templates_pkl=str(ROOT / "z_library" / "subgraph_library_filtered.pt"),
    device="cpu",
)

S_target_raw = read_spectrum_csv(str(ROOT / "data" / "YM.csv"))
E_target = parse_elements("C=500,H=520,O=166,N=6,S=0,X=0")
S_target = normalize_spectrum_to_carbon_count(
    S_target_raw,
    float(E_target[0].item()),
)

stage_configs = {
    "block_a": {
        "max_cycles": 2,
        "max_moves": 6,
        "carbonyl_max_moves": 4,
        "score_rel_threshold": 0.02,
        "peak_rel_threshold": 0.01,
    },
    "block_b": {
        "max_cycles": 2,
        "max_moves_each": 4,
        "peak_rel_threshold": 0.01,
    },
    "block_c_resource": {
        "enabled": True,
        "flex_max_primary_adjustments": 14,
        "external_max_primary_adjustments": 12,
        "internal_max_primary_adjustments": 30,
        "max_h_compensation_ops_per_adjustment": 8,
        "h_tolerance": 0.08,
        "enable_post_external_spectrum": True,
        "spectrum_active_through_cycle": 10,
        "spectrum_candidate_limit": 2,
        "spectrum_max_primary_adjustments": 10,
        "spectrum_min_r2_gain": 0.0003,
        "residual_flex_preferred_23": 3,
        "residual_flex_min_23": 2,
        "residual_side_preferred_23": 2,
        "residual_side_min_23": 1,
        "verbose": False,
    },
}

layer1_config = {
    "neg_peak_threshold": -0.8,
    "pos_peak_threshold": 0.8,
    "max_candidate_trials_per_region": 60,
    "max_moves_per_region": 4,
    "region_adjustment_cycles": 2,
    "layer1_multistart_count": 2,
    "layer1_multistart_refine_count": 2,
    "layer1_multistart_branch_limit": 4,
}

layer3_config = {
    "max_iters": 500,
    "top_k_samples": 50,
    "layer3_verbose": True,
}

nodes, H_final = pipeline.infer(
    S_target=S_target,
    E_target=E_target,
    save_intermediates=True,
    output_dir=str(OUTPUT_DIR),
    layer1_config=layer1_config,
    layer3_config=layer3_config,
    stage_configs=stage_configs,
    outer_max_cycles=int(os.environ.get("YM_OUTER_MAX_CYCLES", "2")),
)

print("Done", flush=True)
print("nodes:", len(nodes), flush=True)
print("H_final:", H_final.detach().cpu().long().tolist(), flush=True)
print("output_dir:", OUTPUT_DIR, flush=True)
