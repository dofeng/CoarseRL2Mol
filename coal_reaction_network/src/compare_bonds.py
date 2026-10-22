"""CLI for atom-ID lineage matching between two ReaxFF bonds files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fragment_lineage import compare_bonds_files, write_transition_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_file", type=Path)
    parser.add_argument("child_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/transitions"))
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--min-shared-atoms", type=int, default=1)
    parser.add_argument("--min-parent-overlap", type=float, default=0.05)
    parser.add_argument("--min-child-overlap", type=float, default=0.05)
    parser.add_argument("--min-event-conservation", type=float, default=0.90)
    parser.add_argument("--elements", default="1:C,2:H,3:O")
    args = parser.parse_args()
    element_map = {int(item.split(":", 1)[0]): item.split(":", 1)[1] for item in args.elements.split(",")}
    result = compare_bonds_files(
        args.parent_file, args.child_file, args.cutoff, element_map,
        args.min_shared_atoms, args.min_parent_overlap, args.min_child_overlap,
        args.min_event_conservation,
    )
    json_path, csv_path, edge_csv_path = write_transition_outputs(result, args.output_dir)
    print(json.dumps({
        "parent_timestep": result["parent_timestep"],
        "child_timestep": result["child_timestep"],
        "parent_fragments": result["parent_fragment_count"],
        "child_fragments": result["child_fragment_count"],
        "lineage_links": result["lineage_link_count"],
        "events": result["event_count"],
        "reactions": len(result["aggregated_reactions"]),
        "node_edges": len(result["node_edges"]),
        "json": str(json_path),
        "csv": str(csv_path),
        "edge_csv": str(edge_csv_path),
    }, ensure_ascii=False, indent=2))
    print("\nTop reactions:")
    for reaction in result["aggregated_reactions"][:20]:
        print(f"{reaction['event_count']:>5}  {reaction['reaction_signature']}")


if __name__ == "__main__":
    main()
