"""Short decode timing for one arm: greedy Tetris and code single-stream (per-round sequences kept), and a
four-stream sampled Tetris burst. Reuses runs/draft-block-20261118/probe.py's run()/burst()."""
import json, sys
from pathlib import Path

source = Path("runs/draft-block-20261118/probe.py")
definitions, _ = source.read_text().rsplit("\nmain()", 1)
p = {}
exec(compile(definitions, str(source), "exec"), p)

rows = []
plan = [("run", "tetris greedy", p["TETRIS"], 1024), ("run", "code greedy", p["CODE"], 1024),
        ("burst", "tetris sampled x4", 4, 768),
        ("run", "tetris greedy", p["TETRIS"], 1024), ("run", "code greedy", p["CODE"], 1024),
        ("burst", "tetris sampled x4", 4, 768)]
for kind, label, arg, n in plan:
    row = p["run"](label, arg, False, 1, n) if kind == "run" else p["burst"](label, arg, 810, n)
    rows.append(row)
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1) + "\n")
    print({k: v for k, v in row.items() if k not in ("per_round", "hist")}, flush=True)
    if row.get("failed"):
        sys.exit(1)
