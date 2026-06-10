"""Draw a schematic 3-hop induced subgraph.

The figure is intentionally molecule-agnostic: it explains the local topology
used by 3-hop coarse subgraphs before showing extraction from full graphs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge


HOP_COLORS = {
    0: "#D62728",
    1: "#F2A900",
    2: "#1F77B4",
    3: "#2CA02C",
}

RING_FILLS = {
    1: "#FFF2CC",
    2: "#DCEBFA",
    3: "#DFF1DF",
}


def _polar(radius: float, degrees: float) -> tuple[float, float]:
    theta = math.radians(degrees)
    return radius * math.cos(theta), radius * math.sin(theta)


def _build_demo_graph() -> tuple[dict[str, tuple[float, float]], dict[str, int], list[tuple[str, str]]]:
    nodes = {"v": (0.0, 0.0)}
    hops = {"v": 0}

    hop_angles = {
        1: [25, 95, 165, 235, 305],
        2: [5, 55, 105, 150, 200, 250, 300, 340],
        3: [15, 45, 80, 115, 150, 185, 220, 255, 290, 325],
    }
    radii = {1: 1.35, 2: 2.55, 3: 3.7}

    for hop, angles in hop_angles.items():
        for idx, angle in enumerate(angles, start=1):
            name = f"h{hop}_{idx}"
            nodes[name] = _polar(radii[hop], angle)
            hops[name] = hop

    edges = [
        ("v", "h1_1"), ("v", "h1_2"), ("v", "h1_3"), ("v", "h1_4"), ("v", "h1_5"),
        ("h1_1", "h2_1"), ("h1_1", "h2_2"), ("h1_2", "h2_2"), ("h1_2", "h2_3"),
        ("h1_3", "h2_4"), ("h1_3", "h2_5"), ("h1_4", "h2_6"), ("h1_5", "h2_7"),
        ("h1_5", "h2_8"), ("h2_1", "h3_1"), ("h2_2", "h3_2"), ("h2_2", "h3_3"),
        ("h2_3", "h3_4"), ("h2_4", "h3_5"), ("h2_4", "h3_6"), ("h2_5", "h3_7"),
        ("h2_6", "h3_8"), ("h2_7", "h3_9"), ("h2_8", "h3_10"),
        ("h1_1", "h1_2"), ("h1_3", "h1_4"), ("h2_2", "h2_3"), ("h2_4", "h2_5"),
        ("h2_7", "h2_8"), ("h3_2", "h3_3"), ("h3_5", "h3_6"), ("h3_8", "h3_9"),
    ]
    return nodes, hops, edges


def draw_3hop_schematic(output_path: Path, dpi: int = 300) -> Path:
    nodes, hops, edges = _build_demo_graph()

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.set_aspect("equal")
    ax.axis("off")

    annuli = [(0.0, 1.75, 1), (1.75, 3.05, 2), (3.05, 4.25, 3)]
    for inner, outer, hop in annuli:
        wedge = Wedge(
            (0.0, 0.0),
            outer,
            0,
            360,
            width=outer - inner,
            facecolor=RING_FILLS[hop],
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
        ax.add_patch(wedge)

    for radius, hop in [(1.75, 1), (3.05, 2), (4.25, 3)]:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                facecolor="none",
                edgecolor=HOP_COLORS[hop],
                linewidth=1.4,
                linestyle=(0, (5, 4)),
                alpha=0.75,
                zorder=1,
            )
        )

    for u, v in edges:
        x0, y0 = nodes[u]
        x1, y1 = nodes[v]
        ax.plot([x0, x1], [y0, y1], color="#555555", linewidth=1.45, alpha=0.8, solid_capstyle="round", zorder=2)

    for name, (x, y) in nodes.items():
        hop = hops[name]
        radius = 0.19 if hop else 0.26
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=HOP_COLORS[hop],
                edgecolor="black" if hop == 0 else "white",
                linewidth=1.7 if hop == 0 else 1.0,
                zorder=4,
            )
        )
        if hop == 0:
            ax.text(x, y, "v", ha="center", va="center", fontsize=14, color="white", fontweight="bold", zorder=5)

    label_specs = [
        ("hop 0\ncenter", 0.25, 0.0, 0),
        ("1-hop", 1.72, 32, 1),
        ("2-hop", 3.02, 32, 2),
        ("3-hop", 4.22, 32, 3),
    ]
    for text, radius, angle, hop in label_specs:
        x, y = _polar(radius, angle)
        if hop == 0:
            x, y = (0.0, -0.56)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11 if hop else 10,
            color=HOP_COLORS[hop],
            fontweight="bold",
            zorder=6,
        )

    ax.text(
        0.0,
        -4.85,
        r"$G_v^{(3)}$: induced subgraph containing all nodes within graph distance $\leq 3$ from center $v$",
        ha="center",
        va="center",
        fontsize=13,
        color="#222222",
    )
    ax.text(
        0.0,
        4.75,
        "Schematic of a 3-hop subgraph",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#222222",
    )

    ax.set_xlim(-4.7, 4.7)
    ax.set_ylim(-5.15, 5.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a concept figure for a 3-hop induced subgraph.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "3hop_subgraph_schematic.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution.")
    args = parser.parse_args()
    out = draw_3hop_schematic(args.output, dpi=int(args.dpi))
    print(f"Saved 3-hop schematic to: {out}")


if __name__ == "__main__":
    main()
