# Dynamic Markov bias region: 256 LRU rows (31.5 MiB) beat the 4,096 static rows just deployed (473 MiB)

2026-11-25 19:12-19:30 UTC. Sizing run only: `TF_DS_MARKOV_TRACE` on image `bench-markov-trace`, a sampled (T=1.0)
workload, production restored to `1421187` afterwards (`validate rc=0`). No kernel was built.

**Trace:** 2,568 passes / 13,432 steps, **one stream a pass** (production's own `streams_a_pass` is 1.42, so close).
Requests did overlap (8 concurrent; each took 2.4 s early, 24 s once queued), the engine simply drafts one stream a
pass for this traffic — so treat the figures as single-stream-pass numbers, not an 8-stream mix.

## Result

Static 256 (the served list) alone: **40.4% of step rows hit**, 3.12 miss steps a pass, ~0.446 ms a pass of `_bias`.
That matches the record's measured 0.44-0.49 ms for a six-step one-stream pass, so the trace and the simulator agree
with reality.

| rows | MiB | policy | row hit% | miss steps/pass | saved ms/pass |
|---:|---:|---|---:|---:|---:|
| 64 | 7.9 | FIFO | 59.8 | 2.10 | 0.145 |
| 64 | 7.9 | LRU | 61.0 | 2.04 | 0.154 |
| 128 | 15.8 | LRU | 66.3 | 1.76 | 0.194 |
| **256** | **31.5** | **LRU** | **72.4** | **1.44** | **0.240** |
| 256 | 31.5 | FIFO | 70.1 | 1.56 | 0.222 |
| 512 | 63.0 | LRU | 79.0 | 1.10 | 0.289 |
| 1024 | 126.0 | LRU | 84.6 | 0.81 | 0.330 |
| 2048 | 252.0 | LRU | 86.4 | 0.71 | 0.344 |
| 4096 | 504.0 | LRU | 86.4 | 0.71 | 0.344 |

## What it means

- **A 256-row dynamic LRU region saves 0.240 ms a pass — more than the 0.20/0.22 ms that the 4,096 static rows just
  deployed deliver — at 31.5 MiB instead of 473 MiB.** The dynamic region is the better cache: it adapts to the
  actual token stream rather than to a fixed frequency list.
- **LRU beats FIFO by 0.018 ms at 256 rows** (and is never worse), so the kernel should use LRU.
- **256 rows is the knee:** 64→256 buys 0.086 ms, 256→512 another 0.049, 512→1024 another 0.041, and above 2,048
  (0.71 miss steps a pass) nothing. So the default should be 256, growth to 512/1024 justified only by the eviction
  signal.
- **Best configuration: static 256 (31.6 MiB) + dynamic 256 (31.5 MiB) = ~63 MiB**, which meets or beats today's
  `TF_DS_MARKOV_CACHE=4096` at 473 MiB and frees ~410 MiB a rank. That is the memory win the operator asked for.
- The 0.24 ms is ~5.7% of a 4.2 ms pass, which matches the offline estimate already in `todo.md` (0.11-0.22 ms) —
  on the high side of it, and now measured on real chains rather than inferred.

## Not done: the kernel

A host-side write-back is **impossible** — see `README.md`: the Markov loop is entirely on device and the pass's only
host read happens after `stage` is overwritten. So the region needs a small repo-owned device kernel launched after
`_bias` (the kit cubins stay untouched), which needs **nvcc** and therefore a from-source image build plus a GPU
byte-parity gate. This run exists to justify that build; it now does.

## Monitoring signal (for the kernel)

Expose per rank, under the existing round profile: `markov_wb_rows`, `hits`, `misses`, `evictions`. The advisory is
`evictions / (hits + misses)`: the curve above says a region at its knee still evicts steadily, so a high ratio alone
isn't the signal — the signal is **a high ratio together with miss steps a pass well above the 0.71 floor** (0.71 at
2,048 rows). Practically: log a one-time warning when `miss_steps/pass > 1.0` with `evictions/lookups > 0.25`, which
the table says corresponds to a region at 256 or below on this traffic, and point at `TF_DS_MARKOV_WB=512/1024`.
