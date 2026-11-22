#!/usr/bin/env python3
"""Outer idle-gap poller for Step 3 guarded CUDA-graph timings.

Polls http://127.0.0.1:8000/health every 3s for up to 150s total WITHOUT holding gpu.lock
while waiting. Whenever live.connections == 0 with valid integer token counters, acquires
`flock ~/tf-bench/gpu.lock` to run `measure.py --mode timing-burst`, which executes each case
in its own ~0.2-0.7s guarded window and immediately yields gpu.lock if traffic arrives.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import measure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cubin", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--lock-file", default=os.path.expanduser("~/tf-bench/gpu.lock"))
    ap.add_argument("--health-url", default="http://127.0.0.1:8000/health")
    ap.add_argument("--budget-s", type=float, default=150.0)
    ap.add_argument("--poll-s", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    t_start = time.monotonic()
    deadline = t_start + args.budget_s
    polls = 0
    bursts = 0

    while True:
        elapsed = time.monotonic() - t_start
        h = measure.health_now(args.health_url)
        polls += 1
        if measure.check_idle_health(h):
            avail = measure.mem_available_gib()
            if avail < measure.MIN_MEM_AVAILABLE_GIB:
                print(
                    f"[POLL #{polls} +{elapsed:.1f}s] MemAvailable {avail:.2f} GiB < {measure.MIN_MEM_AVAILABLE_GIB:.2f} GiB; aborting",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            bursts += 1
            c = measure.health_counters(h)
            print(
                f"[POLL #{polls} +{elapsed:.1f}s] IDLE connections=0 counters={c} MemAvailable={avail:.2f}GiB -> launching flock burst #{bursts}",
                flush=True,
            )
            cmd = [
                "flock",
                args.lock_file,
                sys.executable,
                str(Path(__file__).with_name("measure.py")),
                "--mode",
                "timing-burst",
                "--cubin",
                str(args.cubin),
                "--json",
                str(args.json),
                "--batch",
                str(args.batch),
                "--reps",
                str(args.reps),
                "--health-url",
                args.health_url,
            ]
            proc = subprocess.run(cmd, check=False)
            if proc.returncode == 0:
                print(f"[DONE +{time.monotonic() - t_start:.1f}s] All guarded cases captured cleanly.", flush=True)
                break
            if proc.returncode != 2:
                print(f"[ERROR] timing-burst exited with {proc.returncode}", file=sys.stderr, flush=True)
                return proc.returncode
        else:
            live = h.get("live") if isinstance(h, dict) else None
            conns = live.get("connections") if isinstance(live, dict) else None
            waiting = live.get("waiting") if isinstance(live, dict) else None
            print(
                f"[POLL #{polls} +{elapsed:.1f}s/{args.budget_s:.0f}s] BUSY connections={conns} waiting={waiting} counters={measure.health_counters(h)}",
                flush=True,
            )

        now = time.monotonic()
        if now >= deadline:
            print(
                f"[BUDGET EXHAUSTED +{now - t_start:.1f}s] Finalizing uncaptured cases with timing_valid=False.",
                flush=True,
            )
            break
        time.sleep(min(args.poll_s, max(0.0, deadline - now)))

    return measure.finalize_state(args.json)


if __name__ == "__main__":
    raise SystemExit(main())
