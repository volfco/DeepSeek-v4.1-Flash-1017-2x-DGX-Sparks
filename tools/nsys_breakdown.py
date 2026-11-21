#!/usr/bin/env python3
"""nsys_breakdown.py - per-step GPU kernel time from an nsys sqlite export, for this engine.

Reads the sqlite that `nsys export --type sqlite` writes for an `nsys profile` of the served process (or a batch
driver run like tf-dsv41-lanes), groups the recorded kernels by this engine's kernel names, segments them into steps by
the kernel that opens a step (`--first`, default tf_ds_clock_kernel) and prints a markdown table of launches/step,
us/step and share of the step's kernel time, plus the mean wall time per step. It only reads the sqlite; it does not run
nsys, and it does not need CUDA.

Modelled on baseten's tools/nsys_breakdown.py (same CLI shape, same output format); the groups are this engine's
kernels. Steps are the stretches between two executions of the boundary kernel: with TF_DS_ROUND_PROFILE=1/2 (or
--profile on the batch driver, with TF_DS_CLOCK_MARKS) the round's start-of-round clock kernel opens each round; with
TF_DS_GRAPHS=1 each stretch of the round gets one, so the script segments stretches. Pass your own `--first` (e.g. the
kernel that opens a stretch you care about). `--first` IS the step boundary: everything before its first execution in
the trace is ignored, and a trailing stretch with no boundary after it is dropped.

This engine runs its rounds eager when TF_DS_GRAPHS=0, so graphNodeId is NOT filtered out by default (an export of an
eager run carries no graphNodeId); --graphs-only applies Baseten's `graphNodeId is not null` filter for a trace that
captured graphs only.

Whether nsys works on GB10 (sm_121) is UNCHECKED; this script has not been run against a real export from a pair run.
The kit's recorded Triton kernels are expected to start with `tri_` (the wrapper names); their recorded names are
unknown until a trace exists, so if a trace shows otherwise, adjust the `tri_` group's entry (or pass --kernels to
inspect them). Its correctness beyond the synthetic fixture is unverified.

export it with, on the pair:

    nsys profile -o run.nsys-rep --capture-range=cudaProfilerApi ...    # or simply: nsys profile -o run <command>
    nsys export --type sqlite -o run.sqlite run.nsys-rep
    python3 tools/nsys_breakdown.py run.sqlite

Exit codes: 0 ok; 1 the export is unusable (missing file, unreadable, no kernel rows, fewer than two boundary kernels);
2 usage error (argparse).
"""

import argparse
import math
import sqlite3
import sys
from pathlib import Path

# (prefixes/substrings, group name). First match wins; everything else lands in "other".
GROUPS = [
    (("tf_ds_scatter_rows", "tf_ds_gather_rows", "tf_gather_rows", "tf_strided_copy", "tf_ds_copy_rows16",
      "tf_ds_iota_vis", "tf_ds_cand_fast", "tf_ds_pool_pick", "tf_ds_apply_pool", "tf_ds_block_max",
      "tf_ds_expert_prune"), "scatter / gather / byte movers"),
    (("tf_ds_topk",), "top-k / selection"),
    (("tf_ds_rms_norm", "tf_ds_rope", "tf_ds_compress2", "tf_ds_bf16_scale", "tf_ds_add2_bf16", "tf_ds_add_bf16",
      "tf_ds_f16_to_f32", "tf_bf16_to_f32", "tf_f32_to_bf16", "tf_ds_vrope", "tf_ds_vrot",
      "tf_ds_hc_mean4"), "norm / rope / cast / compress (exact)"),
    (("tf_ds_argmax_rows", "tf_ds_draw_rows", "tf_ds_accept_chain", "tf_ds_keyed_noise", "tf_ds_spec_clean",
      "tf_ds_gather_cols"), "head / sampler / drafter (our kernels)"),
    (("tf_ds_clock",), "clock (marks)"),
    (("tf_ds_l2_prefetch",), "L2 prefetch"),
    (("dsv41_rdma",), "RDMA gathers"),
    (("nccl",), "NCCL / all-reduce"),
    (("tri_",), "kit Triton kernels"),
    (("embed",), "embed"),
]
OTHER = "other"


def group_of(name: str) -> str:
    for pats, group in GROUPS:
        for p in pats:
            if p in name:
                return group
    return OTHER


def load_kernels(sqlite_path: str, device: int | None, kernels: list[str] | None, graphs_only: bool):
    """All (name, start, end) rows, filtered. Raises RuntimeError with a user-facing message."""
    try:
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        raise RuntimeError(f"cannot open {sqlite_path}: {e}")
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for needed in ("StringIds", "CUPTI_ACTIVITY_KIND_KERNEL"):
            if needed not in tables:
                raise RuntimeError(f"{sqlite_path} has no {needed} table (is this an `nsys export --type sqlite`?)")
        cols = {row[1] for row in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        for needed in ("start", "end", "shortName", "deviceId"):
            if needed not in cols:
                raise RuntimeError(f"CUPTI_ACTIVITY_KIND_KERNEL has no {needed} column (unsupported export)")
        graphs_col = "graphNodeId" in cols
        if graphs_only and not graphs_col:
            raise RuntimeError("--graphs-only: CUPTI_ACTIVITY_KIND_KERNEL has no graphNodeId column (unsupported export)")
        where, params = [], []
        if device is not None:
            where.append("k.deviceId = ?")
            params.append(device)
        if graphs_only:
            where.append("k.graphNodeId IS NOT NULL")
        sql = ("SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
               "JOIN StringIds s ON s.id = k.shortName")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY k.start"
        rows = con.execute(sql, params).fetchall()
    except sqlite3.Error as e:
        raise RuntimeError(f"cannot read {sqlite_path}: {e}")
    finally:
        con.close()
    if kernels:
        rows = [r for r in rows if any(k in r[0] for k in kernels)]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-step GPU kernel time from an nsys sqlite export (this engine's kernels).",
        epilog=(
            "produce the export with: nsys profile -o run.nsys-rep <command> then "
            "nsys export --type sqlite -o run.sqlite run.nsys-rep; this script only reads the sqlite (nsys on "
            "GB10 / sm_121 is UNCHECKED). --first is the step boundary: with TF_DS_ROUND_PROFILE=1/2 (or --profile "
            "plus TF_DS_CLOCK_MARKS) the round's start-of-round clock kernel opens a round; with TF_DS_GRAPHS=1 each "
            "stretch of the round gets one, so the script segments stretches - pass your own --first (e.g. the kernel "
            "that opens a stretch you care about). --graphs-only applies Baseten's 'graphNodeId is not null' filter "
            "(for a trace that captured graphs only); the default keeps an eager round visible. The kit's recorded "
            "Triton kernels are expected to start with 'tri_'; their recorded names are unknown until a trace "
            "exists, so if a trace shows otherwise adjust the 'tri_' group's prefix (or use --kernels tri_ to "
            "inspect them). --kernels also filters the boundary kernel (leave --first's kernel in the filter, or "
            "drop it, to keep steps). --top N prints only the N largest groups by us/step (shares are of the "
            "whole). Exit codes: 0 ok, 1 the export is unusable, 2 usage error."
        ))
    ap.add_argument("sqlite", help="the sqlite written by `nsys export --type sqlite`")
    ap.add_argument("--device", type=int, default=None, help="only kernels on this deviceId (default: all devices)")
    ap.add_argument("--kernels", nargs="+", default=None,
                    help="only kernels whose name contains one of these substrings (applied before grouping)")
    ap.add_argument("--top", type=int, default=None, metavar="N",
                    help="print only the N largest groups by us/step (default: all)")
    ap.add_argument("--first", default="tf_ds_clock_kernel", metavar="KERNEL",
                    help="the kernel that opens a step (default: tf_ds_clock_kernel); this IS the step boundary: "
                         "everything before its first execution is ignored and a trailing stretch with no boundary "
                         "after it is dropped. The clock kernel opens a round when TF_DS_ROUND_PROFILE=1/2 is active; "
                         "with graphs each stretch gets one, so pass the kernel that opens the stretch you care about")
    ap.add_argument("--graphs-only", action="store_true",
                    help="keep only kernels with graphNodeId is not null (a trace that captured graphs only); "
                         "the default keeps an eager round visible")
    args = ap.parse_args()

    try:
        rows = load_kernels(args.sqlite, args.device, args.kernels, args.graphs_only)
    except RuntimeError as e:
        print(f"nsys_breakdown: {e}", file=sys.stderr)
        return 1
    if not rows:
        print("(no kernel running)")
        return 1

    bounds = [i for i, (name, start, _end) in enumerate(rows) if args.first in name]
    if len(bounds) < 2:
        print(f"nsys_breakdown: fewer than two boundary kernels matching --first {args.first!r} "
              f"({len(bounds)}): no complete step", file=sys.stderr)
        return 1
    nsteps = len(bounds) - 1
    walls = [rows[bounds[i + 1]][1] - rows[bounds[i]][1] for i in range(nsteps)]
    mean_wall_us = math.fsum(walls) / len(walls) / 1000.0

    # attribute each kernel to the step it starts in; rows before the first boundary are ignored
    per_group: dict[str, list[float]] = {}   # group -> [launches, us]
    total_us = 0.0
    for gi in range(nsteps):
        lo, hi = rows[bounds[gi]][1], rows[bounds[gi + 1]][1]
        for name, start, end in rows[bounds[gi]:bounds[gi + 1]]:
            if start < lo or start >= hi:
                continue
            us = (end - start) / 1000.0
            slot = per_group.setdefault(group_of(name), [0.0, 0.0])
            slot[0] += 1
            slot[1] += us
            total_us += us

    if total_us <= 0.0:
        total_us = 0.0  # zero-duration kernels only; shares print as 0.0 rather than dividing by zero

    order = sorted(per_group.items(), key=lambda kv: kv[1][1], reverse=True)
    if args.top is not None:
        order = order[:max(0, args.top)]

    print(f"# nsys breakdown: {args.sqlite}")
    print(f"device: {args.device if args.device is not None else 'all'}   "
          f"graphs-only: {'on' if args.graphs_only else 'off'}   "
          f"kernels: {' '.join(args.kernels) if args.kernels else 'all'}   "
          f"boundary (--first): {args.first}")
    print(f"steps: {nsteps} complete steps ({len(bounds)} boundary kernels; "
          f"kernels before the first are ignored)   mean wall/step: {mean_wall_us:.1f} us")
    print()
    print("| group | launches/step | us/step | share |")
    print("| --- | ---: | ---: | ---: |")
    shown_launches = 0.0
    shown_us = 0.0
    for group, (launches, us) in order:
        shown_launches += launches
        shown_us += us
        print(f"| {group} | {launches / nsteps:.1f} | {us / nsteps:.1f} | {(us / total_us * 100) if total_us else 0.0:.1f}% |")
    if args.top is not None and len(order) < len(per_group):
        print(f"| (top {args.top} shown) | {shown_launches / nsteps:.1f} | {shown_us / nsteps:.1f} | "
              f"{(shown_us / total_us * 100) if total_us else 0.0:.1f}% |")
    total_launches = sum(v[0] for v in per_group.values())
    print(f"| total | {total_launches / nsteps:.1f} | {total_us / nsteps:.1f} | 100.0% |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
