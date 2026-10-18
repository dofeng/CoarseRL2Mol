"""YM-3 example: python -m RL_MTCS.run_ym3 [--preflight-only]."""

from datetime import datetime
from pathlib import Path

from . import AssemblyConfig, MCTSRunner


def run(*, preflight_only=False):
    project = Path(__file__).resolve().parents[1]
    root = project / "test_results" / "YM-3"
    inputs = root / "final_outputs"
    nodes_csv = inputs / "final_nodes_by_global_id.csv"
    topology_csv = inputs / "final_topology_specs.csv"
    for path in (nodes_csv, topology_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = AssemblyConfig(
        # 每次连接的合法候选及每轮选优的不同结果候选，均最多 10 个。
        connection_candidate_limit=10,
        forward_checkpoint_interval=4,
        forward_min_completion=0.80,
        forward_max_rollbacks=4,
        clearance_radius=3,
        c_puct=1.414,
        seed=7,
        spectrum_csv=None,
        element_composition_csv=None,
        g2s_checkpoint=None,
        reward_weights=(0.40, 0.15, 0.20, 0.20, 0.05),
        verbose=True,
        save_stage_images=False,
    )
    runner = MCTSRunner(nodes_csv, config=config, topology_specs_csv=topology_csv)
    audit = runner.initial_state.audit()
    if not audit["legal"]:
        raise RuntimeError(f"初始化身份、标签、拓扑或几何审计未通过: {audit}")
    if preflight_only:
        print("初始状态审计通过；尚未执行阶段搜索。输入问题数量:", len(audit['input_errors']))
        return {"status": "preflight_passed", "audit": audit}

    output = root / "rl_mtcs_nmr" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    print("输出目录:", output, flush=True)
    result = runner.run_all_stages(str(output))
    print("运行状态:", result["status"])
    print("结构合法:", result["legal"])
    print("是否全部完成:", result["all_complete"])
    print("所有原始节点均已使用:", result["strict_all_nodes_assembled"])
    print("允许剩余资源:", result["allowed_remaining_resources"])

    report = result["completion_report"]
    print("完成任务:", f"{report['completed']}/{report['total']}",
          f"({report['completion']:.2%})")
    print("主组件锚点:", report["main_anchor"])
    print("未接入单元:", report["excluded_units"])
    labels = {
        "rigid": "芳香单元（最终主组件内）",
        "flex": "柔性链",
        "ring": "脂肪环",
        "side": "侧链",
        "substitution": "取代请求",
    }
    for phase, row in report["stages"].items():
        print(labels[phase], f"{row['completed']}/{row['total']}",
              "未完成:", row["unconnected"])
    for checkpoint in result["checkpoints"]:
        selected = checkpoint["selected_attempt"]
        attempt = next(row for row in checkpoint["attempts"]
                       if row["attempt"] == selected)
        print("检查点:", checkpoint["phase"],
              f"{checkpoint['start_count']} → {checkpoint['end_count']}",
              "评估次数:", len(checkpoint["attempts"]),
              "选中尝试:", selected,
              "前瞻完成率:", f"{attempt['forward']['forward_completion']:.2%}",
              "是否保底选择:", checkpoint["fallback"])
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(preflight_only=args.preflight_only)
