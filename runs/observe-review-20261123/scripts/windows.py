#!/usr/bin/env python3
"""Benchmark windows and deploy epochs, from the recipe's own records. Writes data/windows.json.

    python3 -I scripts/windows.py

Sources, in order of trust:
  1. every runs/**/{run,status,runner,campaign*,chain,bisect,depth,clocks}.log and started/finished.txt that carries
     UTC stamps at line starts: the window is [first stamp, last stamp] of the file (suites restart the engine per arm);
  2. deploy records (runs/deploy-*/started.txt .. finished.txt, or run.log): the deploy check (exact panel, sampled,
     validate) runs inside them; the epoch starts at `finished`;
  3. runs with no stamps inside their files, placed by file modification time (marked inferred).
Epoch boundaries follow continue.md's adoption table and the deploy records.
"""
import datetime as dt
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
STAMP = re.compile(r"^(2026-11-\d{2}T\d{2}:\d{2}:\d{2})Z")
NAMES = ("run.log", "run1.log", "status.log", "runner.log", "chain.log", "bisect.log", "depth.log", "clocks.log",
         "started.txt", "finished.txt")


def us(s):
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp() * 1e6)


def main():
    wins = []
    files = [f for f in glob.glob(os.path.join(REPO, "runs", "**", "*"), recursive=True)
             if os.path.isfile(f) and (os.path.basename(f) in NAMES or re.match(r"(campaign|status).*\.log$", os.path.basename(f)))
             and "observe-review" not in f]
    per_dir = {}
    for f in files:
        st = []
        try:
            for line in open(f, errors="replace"):
                m = STAMP.match(line)
                if m:
                    st.append(us(m.group(1)))
        except OSError:
            continue
        if not st:
            continue
        d = os.path.relpath(os.path.dirname(f), REPO)
        a = per_dir.setdefault(d, [min(st), max(st)])
        a[0], a[1] = min(a[0], min(st)), max(a[1], max(st))
    for d, (a, b) in sorted(per_dir.items(), key=lambda kv: kv[1]):
        kind = "deploy" if "/deploy-" in "/" + d or d.endswith("/deploy") else "suite"
        wins.append({"name": d, "start": a, "end": b, "kind": kind, "source": "stamps in logs"})
    # no stamps inside the files: placed by modification time (UTC), from the files' own wall_s where recorded
    wins.append({"name": "runs/cache-20261122 (prompt-cache replay on production, 74 requests)",
                 "start": us("2026-11-23T00:58:30"), "end": us("2026-11-23T01:03:45"), "kind": "probe-on-production",
                 "source": "inferred: file mtimes 00:59:56 / 01:00:46 / 01:03:42 UTC less each run's wall_s"})
    epochs = [
        # start (UTC), label, record
        ("2026-11-10T04:55:00", "E0 older images (before DRAFT_BLOCK=6)", "first tensorfold:* sample in OpenObserve"),
        ("2026-11-18T22:21:19", "E1 exl3-tp2-ea522d353626 + DRAFT_BLOCK=6 (ENGRAM_LEAD from Nov 19 02:26)", "runs/draft-block-20261118/deploy, runs/deploy-lead-20261119"),
        ("2026-11-21T07:38:15", "E2 fused kernels image", "runs/deploy-fused-20261121"),
        ("2026-11-22T06:01:26", "E3 fused-next, then KV_FETCH_REUSE 07:10, recipe 10:12, 0264 10:45", "runs/deploy-fnext-20261122 .. deploy-0264-20261122"),
        ("2026-11-22T13:49:49", "E4 DRAFT_BLOCK_KEPT=2", "runs/deploy-kept2-20261122"),
        ("2026-11-22T21:29:05", "E5 image 10c9866", "runs/deploy-10c9866-20261122"),
        ("2026-11-23T02:23:31", "E6 NCCL_MAX_CTAS=4", "runs/deploy-mc4-20261123"),
        ("2026-11-23T05:16:39", "E7 image 2bb052f + TAPS_PACKED + IDX_PICK_WIDE (current production)", "runs/deploy-2bb052f-20261123"),
    ]
    out = {"windows": sorted(wins, key=lambda w: w["start"]),
           "epochs": [{"start": us(a), "label": b, "record": c} for a, b, c in epochs]}
    json.dump(out, open(os.path.join(HERE, "data/windows.json"), "w"), indent=1)
    print("windows", len(wins), "epochs", len(epochs))


if __name__ == "__main__":
    main()
