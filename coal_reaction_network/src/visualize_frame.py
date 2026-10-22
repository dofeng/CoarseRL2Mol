"""Dependency-free SVG visualizations for one analyzed bonds frame."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape


CARBON_BINS = ["C0", "C1-5", "C6-20", "C21-40", "C40+"]
RING_BINS = ["R0", "R1", "R2-5", "R6-10", "R10+"]
OXYGEN_BINS = ["O0", "O1", "O2", "O3+"]


def _color(value: int, maximum: int) -> str:
    if maximum <= 0:
        return "#f1f5f9"
    intensity = int(235 - 170 * value / maximum)
    return f"rgb({intensity},{intensity + 12},{255})"


def _text(x: float, y: float, value: str, size: int = 12, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="#172033">{escape(value)}</text>'


def write_node_grid(data: dict, path: Path) -> None:
    fragments = data["fragments"]
    counts = Counter((f["carbon_bin"], f["ring_bin"], f["oxygen_bin"]) for f in fragments)
    max_count = max(counts.values(), default=1)
    cell = 62
    left, top = 120, 62
    panel_width, panel_height = 5 * cell + 150, 5 * cell + 90
    width = 2 * panel_width + 30
    height = 2 * panel_height + 75
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(_text(20, 28, "Current-frame pseudo-species population grid", 18))
    parts.append(_text(20, 49, f'Fragments: {len(fragments)}; theoretical classes: 5 × 5 × 4 = 100', 12))
    for panel_index, oxygen in enumerate(OXYGEN_BINS):
        px = 20 + (panel_index % 2) * panel_width
        py = 65 + (panel_index // 2) * panel_height
        parts.append(_text(px + 10, py + 18, f"Oxygen class: {oxygen}", 14))
        gx, gy = px + left - 100, py + top - 45
        parts.append(_text(gx + 2.5 * cell, gy - 12, "Carbon class", 11, "middle"))
        parts.append(_text(gx - 42, gy + 2.5 * cell, "Ring", 11, "middle"))
        for col, carbon in enumerate(CARBON_BINS):
            parts.append(_text(gx + col * cell + cell / 2, gy - 2, carbon, 10, "middle"))
        for row, ring in enumerate(RING_BINS):
            parts.append(_text(gx - 8, gy + row * cell + cell / 2 + 4, ring, 10, "end"))
            for col, carbon in enumerate(CARBON_BINS):
                count = counts[(carbon, ring, oxygen)]
                x, y = gx + col * cell, gy + row * cell
                parts.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="4" fill="{_color(count, max_count)}" stroke="#cbd5e1"/>')
                parts.append(_text(x + cell / 2, y + cell / 2 + 5, str(count), 14, "middle"))
        parts.append(_text(gx, gy + 5 * cell + 22, "Cell value = number of fragments", 10))
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_fragment_scatter(data: dict, path: Path) -> None:
    fragments = data["fragments"]
    width, height = 1100, 700
    left, right, top, bottom = 90, 40, 65, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    masses = [f["molecular_mass"] for f in fragments]
    radicals = [f["radical_sites"] for f in fragments]
    max_mass, max_radical = max(1, max(masses, default=1)), max(1, max(radicals, default=1))
    colors = {}
    palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#4f46e5"]
    for index, node in enumerate(sorted({f["node_id"] for f in fragments})):
        colors[node] = palette[index % len(palette)]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(_text(20, 30, "Fragment internal attributes: mass vs estimated radical sites", 18))
    parts.append(_text(20, 51, "Color = pseudo-species node; each point = one real fragment", 12))
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#94a3b8"/>')
    for step in range(6):
        y = top + plot_h - step * plot_h / 5
        value = max_radical * step / 5
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#e2e8f0"/>')
        parts.append(_text(left - 8, y + 4, f"{value:.0f}", 10, "end"))
        x = left + step * plot_w / 5
        value = max_mass * step / 5
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_h}" stroke="#e2e8f0"/>')
        parts.append(_text(x, top + plot_h + 18, f"{value:.0f}", 10, "middle"))
    parts.append(_text(left + plot_w / 2, height - 42, "Molecular mass", 12, "middle"))
    parts.append(_text(22, top + plot_h / 2, "Radical sites", 12, "middle"))
    for fragment in fragments:
        x = left + fragment["molecular_mass"] / max_mass * plot_w
        y = top + plot_h - fragment["radical_sites"] / max_radical * plot_h
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors[fragment["node_id"]]}" fill-opacity="0.72"><title>fragment {fragment["fragment_id"]}; node {fragment["node_id"]}; mass {fragment["molecular_mass"]:.3f}; radicals {fragment["radical_sites"]}</title></circle>')
    legend_x, legend_y = left, height - 18
    for index, (node, color) in enumerate(colors.items()):
        legend_x = left + (index % 5) * 195
        legend_y = height - 18 + (index // 5) * 20
        parts.append(f'<circle cx="{legend_x}" cy="{legend_y - 4}" r="5" fill="{color}"/>')
        parts.append(_text(legend_x + 9, legend_y, node, 10))
    parts.append('</svg>')
    extra_height = max(0, (len(colors) - 1) // 5) * 20
    parts[0] = parts[0].replace(f'height="{height}"', f'height="{height + extra_height}"').replace(
        f'viewBox="0 0 {width} {height}"', f'viewBox="0 0 {width} {height + extra_height}"')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fragments_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    data = json.loads(args.fragments_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.fragments_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_node_grid(data, output_dir / f"{args.fragments_json.stem}_node_grid.svg")
    write_fragment_scatter(data, output_dir / f"{args.fragments_json.stem}_fragment_scatter.svg")
    print(f"wrote visualizations to {output_dir}")


if __name__ == "__main__":
    main()
