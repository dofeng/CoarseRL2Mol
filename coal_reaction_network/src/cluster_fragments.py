"""Dependency-free clustering and SVG plots for fragment distributions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape


def _raw_features(fragment: dict) -> tuple[float, float, float]:
    return (
        float(fragment["element_counts"].get("C", 0)),
        float(fragment["ring_count"]),
        float(fragment["element_counts"].get("O", 0)),
    )


def _standardized_features(fragments: list[dict]) -> tuple[list[list[float]], list[float], list[float]]:
    transformed = [[math.log1p(value) for value in _raw_features(fragment)] for fragment in fragments]
    means = [sum(row[col] for row in transformed) / len(transformed) for col in range(3)]
    scales = []
    for col in range(3):
        variance = sum((row[col] - means[col]) ** 2 for row in transformed) / len(transformed)
        scales.append(math.sqrt(variance) or 1.0)
    standardized = [[(row[col] - means[col]) / scales[col] for col in range(3)] for row in transformed]
    return standardized, means, scales


def _distance_sq(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _initial_centroids(points: list[list[float]], k: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    centroids = [list(points[rng.randrange(len(points))])]
    while len(centroids) < k:
        distances = [min(_distance_sq(point, center) for center in centroids) for point in points]
        total = sum(distances)
        if total == 0:
            candidates = [point for point in points if point not in centroids]
            centroids.append(list(candidates[0] if candidates else points[0]))
            continue
        target = rng.random() * total
        cumulative = 0.0
        selected = points[-1]
        for point, distance in zip(points, distances):
            cumulative += distance
            if cumulative >= target:
                selected = point
                break
        centroids.append(list(selected))
    return centroids


def kmeans(points: list[list[float]], k: int, seed: int = 3512, max_iter: int = 200) -> tuple[list[int], list[list[float]], float]:
    centroids = _initial_centroids(points, k, seed)
    labels = [-1] * len(points)
    for _ in range(max_iter):
        new_labels = [min(range(k), key=lambda cluster: _distance_sq(point, centroids[cluster])) for point in points]
        if new_labels == labels:
            break
        labels = new_labels
        groups = [[] for _ in range(k)]
        for point, label in zip(points, labels):
            groups[label].append(point)
        for cluster, group in enumerate(groups):
            if group:
                centroids[cluster] = [sum(point[col] for point in group) / len(group) for col in range(3)]
    inertia = sum(_distance_sq(point, centroids[label]) for point, label in zip(points, labels))
    return labels, centroids, inertia


def silhouette_score(points: list[list[float]], labels: list[int]) -> float:
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    if len(groups) < 2:
        return -1.0
    values = []
    for index, point in enumerate(points):
        own = groups[labels[index]]
        if len(own) == 1:
            values.append(0.0)
            continue
        a = sum(math.sqrt(_distance_sq(point, points[other])) for other in own if other != index) / (len(own) - 1)
        b = min(
            sum(math.sqrt(_distance_sq(point, points[other])) for other in members) / len(members)
            for label, members in groups.items() if label != labels[index]
        )
        values.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return sum(values) / len(values)


def select_k(points: list[list[float]], minimum: int = 2, maximum: int = 10) -> tuple[int, list[dict]]:
    scores = []
    upper = min(maximum, len(points) - 1)
    for k in range(minimum, upper + 1):
        labels, _, inertia = kmeans(points, k)
        scores.append({"k": k, "silhouette": silhouette_score(points, labels), "inertia": inertia})
    best = max(scores, key=lambda item: (item["silhouette"], -item["k"]))
    return best["k"], scores


def analyze_clusters(data: dict, k: int | None = None) -> dict:
    fragments = data["fragments"]
    points, means, scales = _standardized_features(fragments)
    selected_k, scores = select_k(points) if k is None else (k, [])
    labels, centroids, inertia = kmeans(points, selected_k)
    cluster_order = sorted(range(selected_k), key=lambda cluster: tuple(centroids[cluster]))
    remap = {old: new for new, old in enumerate(cluster_order)}
    labels = [remap[label] for label in labels]
    summaries = []
    for cluster in range(selected_k):
        members = [fragment for fragment, label in zip(fragments, labels) if label == cluster]
        summaries.append({
            "cluster_id": cluster,
            "population": len(members),
            "mean_carbon": sum(_raw_features(item)[0] for item in members) / len(members),
            "mean_ring": sum(_raw_features(item)[1] for item in members) / len(members),
            "mean_oxygen": sum(_raw_features(item)[2] for item in members) / len(members),
            "mean_mass": sum(item["molecular_mass"] for item in members) / len(members),
            "mean_radical_sites": sum(item["radical_sites"] for item in members) / len(members),
            "pseudo_species_distribution": dict(Counter(item["node_id"] for item in members)),
        })
    assignments = [{
        "fragment_id": fragment["fragment_id"],
        "cluster_id": label,
        "carbon_count": int(_raw_features(fragment)[0]),
        "ring_count": int(_raw_features(fragment)[1]),
        "oxygen_count": int(_raw_features(fragment)[2]),
        "molecular_mass": fragment["molecular_mass"],
        "radical_sites": fragment["radical_sites"],
        "pseudo_species_node": fragment["node_id"],
    } for fragment, label in zip(fragments, labels)]
    return {
        "feature_definition": ["log1p(carbon_count)", "log1p(ring_count)", "log1p(oxygen_count)"],
        "standardization_mean": means,
        "standardization_scale": scales,
        "selected_k": selected_k,
        "selection_scores": scores,
        "inertia": inertia,
        "cluster_summaries": summaries,
        "assignments": assignments,
    }


def _svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="#172033">{escape(text)}</text>'


def write_cluster_svg(result: dict, path: Path) -> None:
    points = result["assignments"]
    width, height = 1150, 760
    left, right, top, bottom = 90, 280, 70, 85
    plot_width, plot_height = width - left - right, height - top - bottom
    x_values = [math.log1p(point["carbon_count"]) for point in points]
    y_values = [math.log1p(point["ring_count"]) for point in points]
    max_x, max_y = max(x_values, default=1), max(y_values, default=1)
    max_oxygen = max((point["oxygen_count"] for point in points), default=1)
    palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#4f46e5", "#65a30d", "#ea580c"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(_svg_text(20, 30, "Fragment clustering distribution", 19))
    parts.append(_svg_text(20, 53, f'K-means on standardized log1p(C, rings, O); selected K = {result["selected_k"]}', 12))
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#94a3b8"/>')
    for step in range(6):
        x = left + step * plot_width / 5
        carbon = math.expm1(max_x * step / 5)
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_height}" stroke="#e2e8f0"/>')
        parts.append(_svg_text(x, top + plot_height + 20, f"{carbon:.0f}", 10, "middle"))
        y = top + plot_height - step * plot_height / 5
        rings = math.expm1(max_y * step / 5)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#e2e8f0"/>')
        parts.append(_svg_text(left - 8, y + 4, f"{rings:.0f}", 10, "end"))
    parts.append(_svg_text(left + plot_width / 2, height - 35, "Carbon count (log-spaced axis)", 12, "middle"))
    parts.append(_svg_text(18, top + plot_height / 2, "Ring count", 12, "middle"))
    for point, x_value, y_value in zip(points, x_values, y_values):
        x = left + x_value / max_x * plot_width if max_x else left
        y = top + plot_height - y_value / max_y * plot_height if max_y else top + plot_height
        radius = 3.5 + 6.5 * math.sqrt(point["oxygen_count"] / max_oxygen) if max_oxygen else 3.5
        color = palette[point["cluster_id"] % len(palette)]
        title = f'fragment {point["fragment_id"]}; cluster {point["cluster_id"]}; C={point["carbon_count"]}; R={point["ring_count"]}; O={point["oxygen_count"]}; mass={point["molecular_mass"]:.3f}; radicals={point["radical_sites"]}'
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.62" stroke="white" stroke-width="0.5"><title>{escape(title)}</title></circle>')
    legend_x = left + plot_width + 28
    parts.append(_svg_text(legend_x, top + 5, "Cluster summaries", 15))
    for index, summary in enumerate(result["cluster_summaries"]):
        y = top + 35 + index * 55
        color = palette[summary["cluster_id"] % len(palette)]
        parts.append(f'<circle cx="{legend_x + 5}" cy="{y - 4}" r="7" fill="{color}"/>')
        parts.append(_svg_text(legend_x + 20, y, f'Cluster {summary["cluster_id"]}: n={summary["population"]}', 11))
        parts.append(_svg_text(legend_x + 20, y + 16, f'C={summary["mean_carbon"]:.1f}, R={summary["mean_ring"]:.1f}, O={summary["mean_oxygen"]:.1f}', 10))
        parts.append(_svg_text(legend_x + 20, y + 31, f'mass={summary["mean_mass"]:.1f}, radical={summary["mean_radical_sites"]:.1f}', 10))
    parts.append(_svg_text(left, height - 12, "Point area increases with oxygen count; color is data-driven cluster, not predefined pseudo-species.", 10))
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fragments_json", type=Path)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    data = json.loads(args.fragments_json.read_text(encoding="utf-8"))
    result = analyze_clusters(data, args.k)
    output_dir = args.output_dir or args.fragments_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.fragments_json.stem}_clusters.json"
    svg_path = output_dir / f"{args.fragments_json.stem}_cluster_distribution.svg"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_cluster_svg(result, svg_path)
    print(json.dumps({"selected_k": result["selected_k"], "json": str(json_path), "svg": str(svg_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
