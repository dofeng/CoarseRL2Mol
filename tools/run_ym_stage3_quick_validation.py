from pathlib import Path
import os
import json
import sys
import time

ROOT = Path("/home/dofengqi/Graphnmr")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.inverse.inverse_pipeline import (
    load_pipeline,
    parse_elements,
    read_spectrum_csv,
)
from model.shared.inverse_common import normalize_spectrum_to_carbon_count


RUN_ID = os.environ.get("YM_RUN_ID") or (
    time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
)
OUTPUT_DIR = Path(
    os.environ.get(
        "YM_OUTPUT_DIR",
        str(ROOT / "test_results" / f"YM-stage3-fix-{RUN_ID}"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRAPHNMR_RUN_ID", RUN_ID)

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
        "max_cycles": 1,
        "max_moves": 2,
        "carbonyl_max_moves": 1,
        "score_rel_threshold": 0.02,
        "peak_rel_threshold": 0.01,
    },
    "block_b": {
        "max_cycles": 1,
        "max_moves_each": 2,
        "peak_rel_threshold": 0.01,
    },
    "block_c_resource": {
        "enabled": True,
        "flex_max_primary_adjustments": 10,
        "external_max_primary_adjustments": 10,
        "internal_max_primary_adjustments": 30,
        "max_h_compensation_ops_per_adjustment": 8,
        "h_tolerance": 0.08,
        "enable_post_external_spectrum": True,
        "spectrum_active_through_cycle": 1,
        "spectrum_candidate_limit": 1,
        "spectrum_max_primary_adjustments": 1,
        "spectrum_min_r2_gain": 0.0005,
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
    "max_candidate_trials_per_region": 8,
    "max_moves_per_region": 1,
    "layer1_multistart_count": 1,
    "layer1_multistart_refine_count": 1,
    "layer1_multistart_branch_limit": 4,
}

layer3_config = {
    "max_iters": 5,
    "top_k_samples": 3,
    "layer3_verbose": False,
}

try:
    nodes, H_final = pipeline.infer(
        S_target=S_target,
        E_target=E_target,
        save_intermediates=True,
        output_dir=str(OUTPUT_DIR),
        layer1_config=layer1_config,
        layer3_config=layer3_config,
        stage_configs=stage_configs,
        outer_max_cycles=int(os.environ.get("YM_OUTER_MAX_CYCLES", "1")),
    )
except Exception as exc:
    failure_dir = OUTPUT_DIR.parent / "failure_outputs" / RUN_ID
    failure_dir.mkdir(parents=True, exist_ok=True)
    status_path = failure_dir / "run_status.json"
    status = {
        "ok": False,
        "run_id": RUN_ID,
        "output_dir": str(OUTPUT_DIR),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "terminal_diagnostics": dict(
            getattr(pipeline, "_final_stage3_terminal_diagnostics", {}) or {}
        ),
    }
    if status_path.exists():
        persisted = json.loads(status_path.read_text(encoding="utf-8"))
        persisted.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        status = persisted
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("VALIDATION_FAILED")
    print(json.dumps(status, ensure_ascii=False))
    raise

(OUTPUT_DIR / "run_status.json").write_text(
    json.dumps(
        {"ok": True, "run_id": RUN_ID, "output_dir": str(OUTPUT_DIR)},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("VALIDATION_DONE")
print("nodes:", len(nodes))
print("H_final:", H_final.detach().cpu().long().tolist())
print("output_dir:", OUTPUT_DIR)
