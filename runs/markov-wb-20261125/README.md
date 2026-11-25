# Markov bias cache: sizing a dynamic write-back region (2026-11-25)

Goal (operator): implement a dynamic write-back region for the Markov bias cache — 256 rows (~31.5 MiB) instead of
4,096 static rows (473 MiB) — with a configurable row count and a monitoring signal on evictions that says when to
grow it.

## The blocker: a host-side write-back is impossible

The obvious design — miss → copy the row into the region → update the slot table — cannot be done on the host:

- The Markov loop is `mk.stepsTapped(...)`, **entirely on device**. Each step's token, the `slot` table, `stage` and
  `cache` all live on the GPU.
- The pass's single host read happens **after** the loop (`draft.zig`'s `synchronize` + two `cuMemcpyDtoH`), by which
  time `stage` has been overwritten by later steps. A host-side write-back would only ever see the last step.
- A per-step D2H inside the loop would break graph capture, which is the point of the pass graph.

The existing hit/miss counters (0280) work only because they count **after** the pass from `dr.drafts`, which is fine
for a counter and useless for a write-back.

**So the write-back must be a device kernel.** The Markov `_bias`/`_score`/`_pick` are kit cubins (untouchable), but a
small repo-owned kernel launched after `_bias` is allowed, and it is capture-safe. That needs **nvcc**, and there is
no nvcc on the workstation or the head — the dsv41 kernels compile only inside the image build. So the real cost is a
from-source image build plus a GPU parity gate, not a patch.

## What was built first, and why

Sizing the region is the prerequisite anyway (the row count has to be configurable, and its default has to be chosen
from data), and it costs ~20 minutes instead of a multi-hour build:

- **`TF_DS_MARKOV_TRACE=<path>`** (engine branch `markov-trace`, `6a048d1`, off by default): appends each pass's step
  tokens, taken from the download the pass already pays for. Host only — no launch, no kernel, no bit — so it rides the
  cross-built image. `zig build` 0, two unit tests pass.
  Record: `u32 magic, u32 n, u32 steps, u32 0, i64 toks[n][steps+1]`. Row i is its stream's pending token then each
  step's pick, so **step j reads `toks[i][j]`**.
- **`sim.py`**: replays those real draft chains through a FIFO or LRU region of 64…4096 rows, reporting row hit rate
  and **miss steps a pass** — the unit that matters, since one `_bias` over the 33.1 MB head serves a whole step
  (133–153 µs measured).
- **`build.sh` / `capture.sh`**: the cross-built image `bench-markov-trace` on both ranks (fatbins byte-identical to
  production's, sha256 verified on each), then a sampled (T=1.0) 8-stream capture.

## The decision this run makes

If a 256-row dynamic region moves the row hit rate from the static list's ~49% toward the ~66–73% the offline estimate
suggests, it gets most of what 4,096 static rows buy at **1/15 the memory**, and the device kernel is worth building.
If it lands near the static baseline, the kernel is not justified and the cache stays static.

## Lesson recorded: never run two suites on one pair

The from-source control and the trace capture were started overlapping, and both drive `start.sh` on the same pair.
They fought over the restart: the control's restore failed (`restore restart failed`), the capture aborted, the pair
was left down with an orphan `Created` container, and `start.sh`'s stale `flock` then refused the restore until the
leftover process exited. **One suite at a time on the pair**, always waiting for the previous `restored: validate rc=0`
and its `finished` line before starting the next — which is exactly what `runs/k280/README.md` already says.
