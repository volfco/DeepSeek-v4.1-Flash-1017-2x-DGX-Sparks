# From-source control for adopting 0281/0282: PASSES

2026-11-25 18:47-19:07 UTC. Two arms on production's own image `bench-fused-next-2bb052f` (from-source, GCC 13.3)
under the k280 workload and probe, compared with image `1421187`'s `ctl1`/`ctl2` (cross-built on the workstation).
`1421187` is `2bb052f` + 0280-0283 with every switch off, so the only difference is how the host code was compiled.
Exact panel IDENTICAL in both arms (39 cases); production restored (`validate rc=0`).

| arm | rounds | pass_ms | forward_ms | absorb_ms | rows/round | streams/pass |
|---|---:|---:|---:|---:|---:|---:|
| 1421187 xbuild ctl1 | 7564 | 4.20 | 38.67 | 0.26 | 6.32 | 1.41 |
| 1421187 xbuild ctl2 | 7516 | 4.22 | 38.77 | 0.26 | 6.35 | 1.42 |
| prod 2bb052f a | 7515 | **4.24** | 39.11 | 0.25 | 6.36 | 1.42 |
| prod 2bb052f b | 7516 | **4.22** | 39.08 | 0.25 | 6.35 | 1.42 |

- **`pass_ms` matches within 0.04 ms (0.9%)** — an order of magnitude below the gains being adopted (0281 at 4,096 rows:
  -0.20/-0.22 ms; 0282 `dpf1`: -0.18/-0.17 ms). The deltas transfer.
- `rows_a_round` and `streams_a_pass` match (6.32-6.36, 1.41-1.42), so the same work was done.
- **The forward is ~1.1% slower from source** (39.08-39.11 vs 38.67-38.77). That does not touch the pass, but it means
  absolute forward/throughput figures are not comparable across the two builds, which `runs/k280/README.md` already
  said. If anything it makes the adopted image look ~1% better on the forward for a reason that is not the patches.

**Verdict: the cross-build did not distort the baseline. 0281 `TF_DS_MARKOV_CACHE=4096` and 0282
`TF_DS_DRAFT_PREFETCH=1` may be adopted** (decisions.md 2026-11-25).
