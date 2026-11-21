# WP0: the measurement harness and the baselines (do this before any kernel change)

Evidence folder: `runs/fused-next-20261121/wp0/`. No engine commits in this package.

1. **A durable parity + timing harness** in the recipe repo at `tools/fused_bench/` (start from
   `runs/fused-kernel-audit-20261120/{build.sh,probe.py,decode.py}`, `runs/fused-followup-20261121/{sass.sh,sass.py}`
   and `~/tf-bench/followup/` on the head; consolidate rather than rewrite):
   - `build.sh OLD_CUDA_DIR NEW_CUDA_DIR OUT` on the head: cubins for both trees with the engine's flags.
   - `bench.py`: loads old/new cubins and the kit cubins via ctypes + the CUDA driver API, checks MemAvailable and the
     production health endpoint, takes `~/tf-bench/gpu.lock`, and for each fused kernel (0195 indexer, 0197 decode and
     prompt compressor, the five 0198 rope-store kernels, 0199 six-slot prune) runs (a) byte parity of every live output
     between old and new and against the unfused/kit reference, over random, tied, flat, NaN, +-inf, signed-zero,
     partial-block and boundary inputs, and (b) interleaved CUDA-graph timings (>= 20 x 100 replays, median + IQR) at
     the served shapes in the brief, plus the empty-kernel launch floor at each launch shape. `--kernel` and `--shape`
     filters; JSON output plus a readable table.
   - `sass.sh`: per-kernel opcode histogram (LDG/STG by width, LDS/STS, SHFL, BAR, ATOMS, MUFU, BRA), registers, spills,
     shared memory, theoretical occupancy, old vs new.
   - `bw.py`: the copy-kernel bandwidth of this GPU (device-to-device, a few sizes), for rooflines.
   - A `README.md` with one command line per use, and a `just fused-bench` recipe in the recipe repo's `Justfile` that
     syncs the two source trees to the head and runs build + bench + sass.
2. **Prove the harness**: run it with OLD = NEW = `~/Workspace/tfref-fused-next` (must report identical bytes and
   timings within noise), then OLD = production's unfused path where one exists. Seed a deliberate one-bit change in a
   scratch copy of one kernel and show the harness catches it.
3. **Baselines** for every fused kernel on `fused-next` at every served shape: time, launch floor, bytes moved, achieved
   bandwidth versus `bw.py`, the bound each kernel sits at, and the SASS summary. Plus the kit's `_score_keys` launch
   cost at 0195's fold widths (16K-65K columns, width % 4 == 0 only: it faults otherwise), which nothing has priced yet.
   This table decides which later items can pay at all: write it as `wp0/BASELINE.md`, ranked by how much each kernel
   could still gain (current time minus max(launch floor, roofline time)).

Hand back: the harness (uncommitted files in the recipe repo), `wp0/RESULTS.md`, `wp0/BASELINE.md`, and a summary.
