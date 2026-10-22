"""CLI for extracting molecular fragments from one ReaxFF bonds file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reaxff_fragments import ReaxFFBondFile, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bonds_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--elements", default="1:C,2:H,3:O")
    args = parser.parse_args()
    element_map = {}
    for item in args.elements.split(","):
        atom_type, element = item.split(":", 1)
        element_map[int(atom_type)] = element
    source = ReaxFFBondFile(args.bonds_file, element_map=element_map,
                            bond_order_cutoff=args.cutoff).parse()
    fragments = source.extract_fragments()
    json_path, csv_path = write_outputs(source, fragments, args.output_dir)
    print(json.dumps({
        "timestep": source.timestep,
        "atoms": len(source.atoms),
        "edges": len(source.edges),
        "fragments": len(fragments),
        "json": str(json_path),
        "csv": str(csv_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
