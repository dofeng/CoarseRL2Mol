"""Streaming exports of complete derived data, not a selected screen page."""

import csv


def export_analytics(connection, output_dir):
    queries = {
        "coarse_pair_audits.csv": "SELECT * FROM coarse_pair_audits ORDER BY parent_frame_id,child_frame_id",
        "coarse_channel_flows.csv": """SELECT e.reaction_signature,f.source_species,f.target_species,
            COUNT(*) event_count,SUM(f.atoms) atom_transfers,SUM(f.carbon) carbon_transfers,
            SUM(f.oxygen) oxygen_transfers,SUM(f.mass) mass_transfers
            FROM coarse_event_flows f JOIN events e USING(parent_frame_id,child_frame_id,event_id)
            GROUP BY e.reaction_signature,f.source_species,f.target_species
            ORDER BY e.reaction_signature,f.source_species,f.target_species""",
    }
    exports = {}
    for name, query in queries.items():
        temporary = output_dir / (name + ".tmp")
        rows = connection.execute(query)
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(column[0] for column in rows.description)
            writer.writerows(rows)
        temporary.replace(output_dir / name)
        exports[name.removesuffix(".csv")] = str(output_dir / name)
    return exports
