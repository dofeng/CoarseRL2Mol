"""Export a stoichiometry-preserving, mirrored species/reaction SVG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {"C0": "#8491a6", "C1-5": "#36a5b5", "C6-20": "#527fca",
          "C21-40": "#9570c4", "C40+": "#d28c45"}


def _text(horizontal: float, vertical: float, value: str, size: int = 12, anchor: str = "middle") -> str:
    return (f'<text x="{horizontal:.1f}" y="{vertical:.1f}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="#42566f">{escape(value)}</text>')


def write_transition_svg(result: dict, path: Path, max_edges: int = 40) -> None:
    if max_edges < 1:
        raise ValueError("max_edges (reaction channels) must be positive")
    source = result.get("aggregated_reactions", result.get("reaction_channels"))
    if source is None:
        raise ValueError("Full reaction channels are required; projected node edges lose stoichiometry")
    if any("reactant_nodes" not in channel and "reactant_nodes_json" not in channel for channel in source):
        raise ValueError("Legacy channel export has no stoichiometry; regenerate it with the corrected parser")
    channels = sorted(source, key=lambda item: -item["event_count"])[:max_edges]
    channels = [{**channel,
                 "reactants": channel.get("reactant_nodes") or json.loads(channel.get("reactant_nodes_json", "{}")),
                 "products": channel.get("product_nodes") or json.loads(channel.get("product_nodes_json", "{}"))}
                for channel in channels]
    reactants = sorted({species for channel in channels for species in channel["reactants"]})
    products = sorted({species for channel in channels for species in channel["products"]})
    height = max(500, max(len(channels), len(reactants), len(products)) * 58 + 160)
    width = 1280
    def position(index, count):
        return 110 + (index + 0.5) * (height - 180) / max(1, count)
    left = {species: position(index, len(reactants)) for index, species in enumerate(reactants)}
    right = {species: position(index, len(products)) for index, species in enumerate(products)}
    maximum = max((channel["event_count"] for channel in channels), default=1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#fff"/>',
             '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="#859ca4"/></marker></defs>',
             '<g font-family="system-ui,sans-serif">',
             _text(28, 32, "COAL ATLAS / Stoichiometric reaction network", 20, "start"),
             _text(28, 56, f"Channels shown: {len(channels)} / {len(source)}; mirrored species refer to the same category", 11, "start")]
    for horizontal, label in ((260, "REACTANTS"), (640, "REACTIONS"), (1020, "PRODUCTS")):
        parts.append(_text(horizontal, 95, label, 11))
    for index, channel in enumerate(channels):
        middle = position(index, len(channels))
        weight = 0.8 + 2.5 * math.log1p(channel["event_count"]) / math.log1p(maximum)
        for side, positions, source_x, target_x in (("reactants", left, 276, 626), ("products", right, 654, 1004)):
            for species, coefficient in channel[side].items():
                start = positions[species] if side == "reactants" else middle
                end = middle if side == "reactants" else positions[species]
                color = COLORS.get(species.split("_")[0], "#8491a6")
                parts.append(f'<path d="M{source_x},{start} C{source_x+140},{start} {target_x-140},{end} {target_x},{end}" fill="none" stroke="{color}" stroke-opacity=".4" stroke-width="{weight:.2f}" marker-end="url(#arrow)"/>')
                if coefficient > 1:
                    parts.append(_text(297 if side == "reactants" else 980,
                                       (start if side == "reactants" else end) - 7, str(coefficient), 10))
        parts.append(f'<rect x="626" y="{middle-13}" width="28" height="26" rx="5" fill="#edf5f4" stroke="#7caaa5"><title>{escape(channel["reaction_signature"])}</title></rect>')
        parts.append(_text(640, middle + 3, f"R{index+1}", 8))
        parts.append(_text(667, middle + 3, str(channel["event_count"]), 9, "start"))
    for positions, center, label_x, anchor in ((left, 260, 235, "end"), (right, 1020, 1045, "start")):
        for species, vertical in positions.items():
            color = COLORS.get(species.split("_")[0], "#8491a6")
            parts.append(f'<circle cx="{center}" cy="{vertical}" r="15" fill="{color}" fill-opacity=".12" stroke="{color}"/>')
            parts.append(_text(label_x, vertical+4, species, 11, anchor))
    parts.extend([_text(28, height-20, "Edge labels retain coefficients. Width = observed event count, not reaction rate.", 11, "start"), '</g></svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transition_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-edges", type=int, default=40, help="Maximum reaction channels in the displayed subgraph")
    args = parser.parse_args()
    result = json.loads(args.transition_json.read_text(encoding="utf-8"))
    output = args.output or args.transition_json.with_name(args.transition_json.stem + "_network.svg")
    write_transition_svg(result, output, args.max_edges)
    print(output)


if __name__ == "__main__":
    main()
