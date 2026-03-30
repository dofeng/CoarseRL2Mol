import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# 复用 coarse_graph 中的 SU 信息与配色
try:
    from .coarse_graph import SU_DEFS, _node_color
except ImportError:
    try:
        from model.coarse_graph import SU_DEFS, _node_color
    except ImportError:
        from coarse_graph import SU_DEFS, _node_color


def _set_plot_defaults():
    """与 dataset_utils._set_plot_defaults 保持一致的绘图风格"""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0


def _find_latest_matching_file(base_dir: Path, file_names: list[str]) -> Path | None:
    """优先查找当前目录下的文件；若不存在，再递归查找最近更新的匹配文件。"""
    for name in file_names:
        direct_path = base_dir / name
        if direct_path.exists():
            return direct_path

    matches: list[Path] = []
    for name in file_names:
        matches.extend(p for p in base_dir.rglob(name) if p.is_file())

    if not matches:
        return None

    return max(matches, key=lambda p: p.stat().st_mtime)


def _parse_template_key(template_key: str) -> tuple[int | None, list[int], list[int]]:
    """解析形如 '(3, (9, 9), (11, 13, 13, 13))' 的模板键。"""
    if template_key is None or (isinstance(template_key, float) and np.isnan(template_key)):
        return None, [], []

    try:
        value = ast.literal_eval(str(template_key))
    except Exception:
        return None, [], []

    if not isinstance(value, tuple) or len(value) != 3:
        return None, [], []

    try:
        center_su = int(value[0])
        hop1 = [int(x) for x in value[1]]
        hop2 = [int(x) for x in value[2]]
        return center_su, hop1, hop2
    except Exception:
        return None, [], []


def _format_multiset(values: list[int]) -> str:
    if not values:
        return "[]"
    return "[" + " ".join(str(int(x)) for x in values) + "]"


def _standardize_nodes_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    将旧版/新版 layer2/layer3 节点 CSV 统一为:
    node_id, center_su_idx, center_su, hop1_ms, hop2_ms, mu_pred, pi_pred
    """
    if df.empty:
        return pd.DataFrame(
            columns=["node_id", "center_su_idx", "center_su", "hop1_ms", "hop2_ms", "mu_pred", "pi_pred"]
        )

    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        if "node_id" in df.columns:
            node_id = int(row["node_id"])
        elif "global_id" in df.columns:
            node_id = int(row["global_id"])
        else:
            continue

        if "center_su_idx" in df.columns:
            center_su_idx = int(row["center_su_idx"])
        elif "su_type" in df.columns:
            center_su_idx = int(row["su_type"])
        else:
            center_su_idx = None

        if "center_su" in df.columns:
            center_su = str(row["center_su"])
        elif "su_name" in df.columns:
            center_su = str(row["su_name"])
        elif center_su_idx is not None and 0 <= int(center_su_idx) < len(SU_DEFS):
            center_su = SU_DEFS[int(center_su_idx)][0]
        else:
            center_su = "Unknown"

        hop1_ms = str(row["hop1_ms"]) if "hop1_ms" in df.columns else None
        hop2_ms = str(row["hop2_ms"]) if "hop2_ms" in df.columns else None

        if hop1_ms is None or hop2_ms is None:
            template_col = None
            if "template_key" in df.columns:
                template_col = row["template_key"]
            elif "chosen_template_key" in df.columns:
                template_col = row["chosen_template_key"]
            elif "chosen_hop1_ms" in df.columns and "chosen_hop2_ms" in df.columns:
                hop1_ms = str(row["chosen_hop1_ms"])
                hop2_ms = str(row["chosen_hop2_ms"])

            if template_col is not None and (hop1_ms is None or hop2_ms is None):
                parsed_center, parsed_h1, parsed_h2 = _parse_template_key(str(template_col))
                if center_su_idx is None and parsed_center is not None:
                    center_su_idx = int(parsed_center)
                if hop1_ms is None:
                    hop1_ms = _format_multiset(parsed_h1)
                if hop2_ms is None:
                    hop2_ms = _format_multiset(parsed_h2)

        if hop1_ms is None:
            hop1_ms = "[]"
        if hop2_ms is None:
            hop2_ms = "[]"

        if "mu_pred" in df.columns:
            mu_pred = pd.to_numeric(row["mu_pred"], errors="coerce")
        else:
            mu_pred = pd.to_numeric(row.get("mu", np.nan), errors="coerce")

        if "pi_pred" in df.columns:
            pi_pred = pd.to_numeric(row["pi_pred"], errors="coerce")
        else:
            pi_pred = pd.to_numeric(row.get("pi", np.nan), errors="coerce")

        if center_su_idx is None:
            continue

        rows.append({
            "node_id": int(node_id),
            "center_su_idx": int(center_su_idx),
            "center_su": str(center_su),
            "hop1_ms": str(hop1_ms),
            "hop2_ms": str(hop2_ms),
            "mu_pred": float(mu_pred) if pd.notna(mu_pred) else np.nan,
            "pi_pred": float(pi_pred) if pd.notna(pi_pred) else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Cannot recognize node CSV schema for visualization.")
    return out


def _build_su_distribution_from_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """从节点表聚合得到完整的 33 类 SU 分布。"""
    counts = nodes_df["center_su_idx"].value_counts().to_dict()
    rows = []
    for su_idx, (su_name, _) in enumerate(SU_DEFS):
        rows.append({
            "SU_Index": int(su_idx),
            "SU_Name": su_name,
            "Total_Count": int(counts.get(su_idx, 0)),
        })
    return pd.DataFrame(rows)


# ===================== 1. Layer3 SU 分布可视化 =====================


def plot_layer3_su_distribution(csv_path: Path | str, output_dir: Path | str | None = None) -> Path:
    """绘制当前逆向结果的 SU 直方图。

    兼容两种输入:
    1. 旧版 `layer3_su_distribution.csv`。
    2. 新版 `layer2/3_nodes_detail.csv` 或 `layer2/3_node_peaks.csv`，会自动聚合统计。
    """
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(csv_path)
    if {"SU_ID", "SU_Name", "Count"}.issubset(df.columns):
        df_plot = pd.DataFrame({
            "SU_Index": df["SU_ID"].astype(int),
            "SU_Name": df["SU_Name"],
            "Total_Count": df["Count"].astype(int),
        })
    else:
        df_plot = _build_su_distribution_from_nodes(_standardize_nodes_df(df))

    _set_plot_defaults()
    fig, ax = plt.subplots(figsize=(18, 6))

    colors = [_node_color(int(idx)) for idx in df_plot["SU_Index"]]
    labels = [f"{idx}" for idx in df_plot["SU_Index"]]
    bars = ax.bar(range(len(df_plot)), df_plot["Total_Count"],
                  color=colors, edgecolor="white", linewidth=0.5)

    # 在柱状图顶部添加数值标签
    for bar, count in zip(bars, df_plot["Total_Count"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f"{int(count)}", ha="center", va="bottom", fontsize=16)

    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_ylabel("Count", fontsize=24)
    ax.set_title("Layer3 Structural Unit Distribution (Center Nodes)", fontsize=24, pad=12)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(False)

    fig.tight_layout()
    out_path = output_dir / "layer3_su_distribution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Layer3 SU distribution figure saved to: {out_path}")
    return out_path


# ===================== 2. Layer3 局部子图可视化 =====================


def _parse_multiset_str(ms: str) -> list[int]:
    """将类似 '[9 13 29]'、'[9, 13, 29]' 的多重集字符串解析为整数列表。"""
    ms = str(ms).strip()
    if not ms or ms == "[]" or ms.lower() == "nan":
        return []
    if ms[0] in "[(" and ms[-1] in "])":
        ms = ms[1:-1]
    ms = re.sub(r"[,\s]+", " ", ms).strip()
    if not ms:
        return []
    return [int(x) for x in ms.split()]


def plot_layer3_local_subgraphs(
    nodes_csv: Path | str,
    output_dir: Path | str | None = None,
    num_examples: int = 16,
) -> Path:
    """从当前可用的节点结果 CSV 中选取若干节点, 绘制局部 3-hop 拓扑示意图。

    - 每个节点一张小子图 (中心 + hop1 + hop2)。
    - 仅展示 SU 类型与层级结构, 不依赖真实分子骨架。
    - 兼容旧版 `layer3_nodes.csv`，也兼容新版 `layer2/3_nodes_detail.csv`
      与 `layer2/3_node_peaks.csv`。
    """
    nodes_csv = Path(nodes_csv)
    if output_dir is None:
        output_dir = nodes_csv.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    df = _standardize_nodes_df(pd.read_csv(nodes_csv))
    if df.empty:
        raise ValueError(f"nodes CSV is empty: {nodes_csv}")

    # 采样若干具有代表性的节点: 优先选择芳香/不饱和等 SU, 再随机补足
    # 芳香&不饱和 SU 索引集合 (0-25 内的碳中心)
    aromatic_like = set(range(5, 14))  # 5~13
    unsat_like = set(range(14, 19))    # 14~18

    def _is_interesting_row(row) -> bool:
        su = int(row["center_su_idx"])
        return (su in aromatic_like) or (su in unsat_like)

    df_interesting = df[df.apply(_is_interesting_row, axis=1)]
    selected_parts = []
    selected_indices: list[int] = []

    # 先选取“有趣”的前 num_examples/2 个
    n1 = min(len(df_interesting), max(1, num_examples // 2))
    if n1 > 0:
        part = df_interesting.sample(n=n1, random_state=0)
        selected_parts.append(part)
        selected_indices.extend(part.index.tolist())

    # 再从剩余中补足
    remaining = num_examples - len(selected_indices)
    if remaining > 0:
        df_rest = df.drop(index=selected_indices, errors="ignore")
        if not df_rest.empty:
            n2 = min(len(df_rest), remaining)
            part = df_rest.sample(n=n2, random_state=1)
            selected_parts.append(part)

    if selected_parts:
        selected_df = pd.concat(selected_parts, axis=0)
    else:
        selected_df = pd.DataFrame()

    if selected_df.empty:
        # 兜底: 直接取前 num_examples 行
        selected_df = df.head(num_examples)

    selected_rows = selected_df.to_dict("records")

    _set_plot_defaults()

    # 布局: 按 4x4 网格排布若干局部子图
    n = len(selected_rows)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for idx, row in enumerate(selected_rows):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        center_su_idx = int(row["center_su_idx"])
        center_su_name = str(row["center_su"])
        hop1_list = _parse_multiset_str(str(row["hop1_ms"]))
        hop2_list = _parse_multiset_str(str(row["hop2_ms"]))
        mu = float(row["mu_pred"])
        pi = float(row["pi_pred"])

        # 构建一个三层同心结构的简单图
        G = nx.Graph()

        # 中心节点 id 使用字符串避免与 hop 索引混淆
        center_id = "C"
        G.add_node(center_id, layer=0, su=center_su_idx)

        # hop1 节点
        hop1_ids = []
        for i, su_idx in enumerate(hop1_list):
            nid = f"H1_{i}"
            G.add_node(nid, layer=1, su=su_idx)
            G.add_edge(center_id, nid)
            hop1_ids.append(nid)

        # hop2 节点：此处不尝试重建真实连接，只将其连接到某个 hop1 或中心
        for j, su_idx in enumerate(hop2_list):
            nid = f"H2_{j}"
            G.add_node(nid, layer=2, su=su_idx)
            # 简单规则: 尽量连到第一个 hop1, 若没有 hop1 则连到中心
            if hop1_ids:
                G.add_edge(hop1_ids[j % len(hop1_ids)], nid)
            else:
                G.add_edge(center_id, nid)

        # 为不同层规划极坐标布局: 中心在 (0,0), hop1 在半径 1, hop2 在半径 2
        pos = {}
        pos[center_id] = (0.0, 0.0)

        # 按角度均匀分布 hop1
        n_h1 = max(1, len(hop1_ids))
        for k, nid in enumerate(hop1_ids):
            theta = 2.0 * np.pi * k / n_h1
            pos[nid] = (1.0 * float(np.cos(theta)), 1.0 * float(np.sin(theta)))

        # hop2 大圆
        h2_ids = [n for n in G.nodes if str(n).startswith("H2_")]
        n_h2 = max(1, len(h2_ids))
        for k, nid in enumerate(h2_ids):
            theta = 2.0 * np.pi * k / n_h2
            pos[nid] = (2.0 * float(np.cos(theta)), 2.0 * float(np.sin(theta)))

        # 颜色与标签
        node_colors = []
        node_labels = {}
        for n_id, data in G.nodes(data=True):
            su_idx = int(data.get("su", -1))
            if 0 <= su_idx < len(SU_DEFS):
                color = _node_color(su_idx)
                label = str(su_idx)
            else:
                color = "#aaaaaa"
                label = "?"
            node_colors.append(color)
            node_labels[n_id] = label

        nx.draw(G, pos, ax=ax, with_labels=False,
                node_color=node_colors, edge_color="#555555",
                node_size=400, width=1.0)

        # 在节点上方加上 SU 索引文字
        for n_id, (x, y) in pos.items():
            ax.text(x, y, node_labels[n_id],
                    ha="center", va="center", fontsize=10, color="black")

        title = f"node {row['node_id']} | SU {center_su_idx} ({center_su_name})\n"
        title += f"mu={mu:.2f}, pi={pi:.2f}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # 关闭多余子图
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        ax.axis("off")

    fig.tight_layout()
    out_path = output_dir / "layer3_local_subgraphs.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Layer3 local subgraph figures saved to: {out_path}")
    return out_path


# ===================== 3. 命令行入口 =====================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize inverse pipeline results with automatic support for the latest Layer2/Layer3 output files."
    )
    parser.add_argument("--result_dir", type=str, default="inverse_result",
                        help="Result directory. Can be a root result folder, a specific layer2_eval/layer3_eval folder, or any directory containing the output CSVs.")
    parser.add_argument("--num_examples", type=int, default=16,
                        help="Number of local subgraph examples to draw")

    args = parser.parse_args()
    result_dir = Path(args.result_dir)

    node_csv_candidates = [
        "layer3_nodes_detail.csv",
        "layer3_node_peaks.csv",
        "layer2_nodes_detail.csv",
        "layer2_node_peaks.csv",
        "layer3_nodes.csv",
    ]
    su_csv_candidates = [
        "layer3_su_distribution.csv",
        *node_csv_candidates,
    ]

    su_csv = _find_latest_matching_file(result_dir, su_csv_candidates)
    nodes_csv = _find_latest_matching_file(result_dir, node_csv_candidates)

    if su_csv is None:
        raise FileNotFoundError(
            f"Cannot find any SU-distribution-compatible CSV under {result_dir}. "
            f"Tried: {su_csv_candidates}"
        )
    if nodes_csv is None:
        raise FileNotFoundError(
            f"Cannot find any node-detail-compatible CSV under {result_dir}. "
            f"Tried: {node_csv_candidates}"
        )

    print(f"[i] Using SU source: {su_csv}")
    print(f"[i] Using node source: {nodes_csv}")

    plot_layer3_su_distribution(su_csv, result_dir)
    plot_layer3_local_subgraphs(nodes_csv, result_dir, num_examples=args.num_examples)


if __name__ == "__main__":
    main()
