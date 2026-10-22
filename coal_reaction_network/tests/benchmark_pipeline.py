"""Opt-in synthetic scalability check; not a substitute for a full real-trajectory validation."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pipeline_runtime import process_tree_rss_mib
from coarse_network.queries import coarse_network


def frame_text(atoms: int, split: bool) -> str:
    neighbors = {identifier: [] for identifier in range(1, atoms+1)}
    for start in range(1, atoms+1, 4):
        edges = [(start,start+1),(start+2,start+3)]
        if not split:
            edges.append((start+1,start+2))
        for left,right in edges:
            neighbors[left].append(right)
            neighbors[right].append(left)
    lines = ["# Timestep 0",f"# Number of particles {atoms}"]
    for identifier,bonded in neighbors.items():
        lines.append(" ".join(map(str,[identifier,1,len(bonded),*bonded,1,*([1]*len(bonded)),len(bonded),0,0])))
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames",type=int,nargs="+",default=[100,1000,10001])
    parser.add_argument("--atoms",type=int,default=96)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--output",type=Path,default=Path("/tmp/coal-scalability.json"))
    args=parser.parse_args()
    if args.atoms < 4 or args.atoms % 4 or min(args.frames) < 2:
        parser.error("atoms must be a positive multiple of four; frames >= 2")
    report=[]
    for count in args.frames:
        with tempfile.TemporaryDirectory(prefix="coal-benchmark-") as directory:
            root=Path(directory)
            inputs=root / "inputs"
            inputs.mkdir()
            templates=[frame_text(args.atoms,False),frame_text(args.atoms,True)]
            for frame in range(count):
                (inputs / f"bond{frame}.reaxff").write_text(templates[int(frame % 10 == 5)])
            database=root / "output/network.sqlite"
            progress_path=database.with_suffix(".progress.json")
            command=[sys.executable,"-B",str(Path(__file__).parents[1] / "src/trajectory_pipeline.py"),
                     str(inputs),"--database",str(database),"--workers",str(args.workers),
                     "--max-inflight",str(args.workers*2),"--expected-frame-step","1",
                     "--temporal-mode","annotate","--minimum-lifetime-frames","2",
                     "--reversal-window-frames","1","--no-network-export"]
            started=time.monotonic()
            peaks={}
            with (root / "run.log").open("w") as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    stage="startup"
                    if progress_path.is_file():
                        stage=json.loads(progress_path.read_text()).get("stage",stage)
                    peaks[stage]=max(peaks.get(stage,0),process_tree_rss_mib(process.pid))
                    time.sleep(.15)
            if process.returncode:
                raise RuntimeError((root / "run.log").read_text()[-6000:])
            summary=json.loads((database.parent / "network_summary.json").read_text())
            connection=sqlite3.connect(database)
            duplicates=connection.execute("SELECT COUNT(*) FROM (SELECT track_id,frame_id,COUNT(*) amount FROM fragment_tracks GROUP BY track_id,frame_id HAVING amount>1)").fetchone()[0]
            channels=connection.execute("SELECT SUM(event_count) FROM reaction_channels").fetchone()[0]
            nonpersistent=connection.execute("SELECT COUNT(*) FROM events WHERE event_type NOT IN ('persistence','coarse_persistence')").fetchone()[0]
            audited=connection.execute("SELECT COUNT(*) FROM coarse_pair_audits").fetchone()[0]
            carbon=connection.execute("SELECT COALESCE(SUM(carbon),0) FROM coarse_event_flows").fetchone()[0]
            connection.close()
            if duplicates or channels != nonpersistent or summary["frame_pairs"] != count-1 or audited != count-1 or carbon != 4*nonpersistent:
                raise AssertionError("Network integrity check failed")
            query_started=time.monotonic()
            projection=coarse_network(database,{"coverage":"1","limit":"200"})
            query_seconds=time.monotonic()-query_started
            if projection["total_weight"] != carbon or projection["total_events"] != nonpersistent:
                raise AssertionError("Coarse network projection integrity check failed")
            item=dict(frames=count,atoms_per_frame=args.atoms,seconds=round(time.monotonic()-started,2),
                      peak_process_tree_rss_mib=round(max(peaks.values()),2),
                      stage_peaks_mib={stage:round(value,2) for stage,value in peaks.items()},
                      database_mib=round(database.stat().st_size/2**20,2),
                      fragments=summary["fragments"],events=summary["events"],
                      nonpersistent_events=nonpersistent,completed_pairs=summary["frame_pairs"],
                      audited_pairs=audited,carbon_transfers=carbon,network_query_seconds=round(query_seconds,3),
                      integrity="passed",synthetic=True)
            report.append(item)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2))
            print(json.dumps(item),flush=True)


if __name__=="__main__":
    main()
