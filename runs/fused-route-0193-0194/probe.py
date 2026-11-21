import json
from pathlib import Path
import sys

# Reuse the established fixed-prompt probe without running its full campaign.
source = Path("runs/draft-block-20261118/probe.py")
definitions, _ = source.read_text().rsplit("\nmain()", 1)
probe = {}
exec(compile(definitions, str(source), "exec"), probe)

rows = []
for repeat in range(3):
    rows.append(probe["run"]("Tetris greedy", probe["TETRIS"], False, 1, 1024))
    rows.append(probe["burst"]("Tetris sampled x4", 4, 810, 768))
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows[-2:]:
        print({k: v for k, v in row.items() if k != "per_round"}, flush=True)
        if row.get("failed"):
            sys.exit(1)
