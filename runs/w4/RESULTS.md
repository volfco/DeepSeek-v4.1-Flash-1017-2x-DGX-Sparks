# W4: is the decode MoE bus-bound or trellis-decode-bound? (2026-11-17)

Image `tensorfold-dsv41:exl3-tp2-234f9080b326` (patches 0001-0171 + 0171z), both ranks, no build. Driver
`w4_run.sh` (detached on the head), command builder `w4_cmd.py` (uf_prof.sh's recipe with the spill tier stripped),
numbers `w4_analyze.py` -> `summary.txt`. GPU clocks under the operator's 2400 MHz cap throughout (2359-2385 MHz
under load, max 64 C head / 62 C worker, no throttle reasons; `clocks.log`).

**Eager MoE throughput is consistent with a DRAM bandwidth limit.** Combining two blocking byte-count cells with separate nonblocking MoE timings gives an estimated 221–222 GB/s. Eight streams reuse the four-stream bytes/place. The synthetic scattered and 1 GiB reads reached 231–236 GB/s. This misses the proposed 1.5× headroom criterion for M5. Deprioritize decoded-expert caching on this workload. In e-r1-ctl the MoE took 16.148 ms of a 36.29 ms eager step. Other phases and the graphs-on served path require separate attribution.

## Probe (0142, `--synthetic`; median GB/s, 50 reps, rank 0; worker within 1-2%)

| Shape | ldg sorted / shuffled | ldg + L2 prefetch | cp.async sorted | cp.async + prefetch | repeat-first ldg |
| --- | --- | --- | --- | --- | --- |
| 13 x 6.47 MB contiguous, 9 tiles (decode shape) | 261.7 / 260.6 (rerun 258.3 / 257.6) | 198.1 / 204.2 | 230.2 | 168.2 | 1651 (L2) |
| same, 18 / 36 tiles | 258.3 / 257.4 | 224.1 / 226.1 | 234.6 / 219.6 | 175.6 / 169.6 | 1672 / 1441 |
| 16 x 6.47 MB, 9 / 36 tiles | 250.8 / 252.2 | 179.2 / 219.0 | 219.8 / 210.9 | 156.1 / 153.9 | 1688 / 1555 |
| 13 x 6.47 MB, 4x stride (gapped), 9 tiles | 231.4 / 231.4 | 182.1 | 206.4 | 154.3 | 1673 |
| 16 x 64 MiB (1 GiB), 64 tiles | 234.5 / 234.3 | 205.1 | 234.3 | 176.5 | 2204 |

- Sorted and shuffled order are the same within 1%. Adding blocks does not raise the ldg rate. The 9-tile grid is not
  a launch-shape ceiling.
- The L2 prefetch stream running beside the reads lowers them by 13-30%: the copies compete for the same bus.
  `repeat-first` measures L2 residency, not the bus.
- The relevant DRAM figure is the gapped or 1 GiB case, 231-236 GB/s, because picked experts are scattered in a 95 GB
  buffer. The contiguous 84 MB case is about 10% faster, an address-layout effect (cause not isolated).
- `--spec` was not run. Nothing in 0140 prints device addresses, and the probe opens its own CUDA context, so another
  process's pointers are invalid there.

## Estimated MoE bandwidth (eager `--profile 2`, graphs off, production prune `T13_EXPERT_PRUNE=1`)

| Cell | rows | routed places / layer (device `expert_prune`) | MB / place incl. shared | GB / round (routed + shared) | `moe` ms | GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| e-r1-ctl (no flag) | 3.88 | 12.499 | 6.644 (e-r1-lb) | 3.587 | 16.148 | **222** |
| e-r1-xb | 3.88 | 12.499 | 6.644 | 3.587 | 16.140 | 222 |
| e-r4-xb | 7.46 | 25.987 | 6.484 (e-r4-lb) | 7.000 | 31.567 | **222** |
| e-r8 (r8 ids, no expect_sha) | 13.46 | 41.675 | 6.484 (from r4, inferred) | 11.068 | 50.053 | 221 |

- **Bytes per place (measured, `-lb` cells):** 6.644 MB at 1 stream, +2.7% over the 6.47 MB file average, and
  6.484 MB at 4 streams, +0.2%. The counter includes the shared expert. `Experts.count` is n + 1, with the shared
  expert last and real k2 entries (`weights.zig:517-589` in the engine checkout), so 0140's comment that it counts
  "no bytes" is wrong. A routed place alone is 7.164 / 6.720 MB if the shared place is assumed to cost 0. With
  places x 6.47 MB the rate reads 216 / 221 GB/s, so the average was close.
- **Marginal place (inferred from two `-lb` cells with different request files):** (184.53 - 91.53) MB over
  (27.459 - 12.776) places is 6.33 MB a place. At the regression's 27.2 us a place below, that is **233 GB/s**, inside
  the probe's DRAM 231-236.
- **The counter needs `CUDA_LAUNCH_BLOCKING=1`.** 0140 reads the decode scratch back with `cuMemcpyDtoH_v2` on the
  legacy stream, while the engine's stream is non-blocking (`model.zig:355`), so the read is not ordered after the
  experts. Without blocking, the places disagree with the device counts and between ranks, which run identical routing:
  14.84 / 15.39 at 1 stream and 27.77 / 28.37 at 4 streams, where 13.50 and 26.99 are expected. With blocking on rank
  0, places equal `distinct_after + 1.000` exactly (the shared place) in both cells. So bytes come only from the `-lb`
  cells, and times only from cells without blocking (blocking inflates `moe` to 21.4 / 38.3 ms and moves the rows
  slightly: 3.98 / 8.12).
- The timing difference was below the resolution of this comparison: `moe` is 16.140 with it and 16.148 without. Every cell with `TF_DS_EXPERT_BYTES=1` ends with
  `free(): invalid pointer` (exit 133, both ranks) after all output is printed. The control without the flag exits 0.
  This is a 0140 teardown bug.
- Not counted: whatever runs outside the 40 decode windows a round. There are 43 layers and 40 windows, and why 3
  layers have none was not checked. The shared expert's bytes are counted, though the L2 prefetch behind `wo_b`
  already pulls its trellises, so some of them may be L2 hits.
- The MoE's 222 GB/s beats the probe's cp.async kernel on gapped ranges (206). That probe form is not a ceiling; the
  ldg form is.

## Rows against places at fixed depth (0139 `TF_DS_FIX_K`)

| Cell | rows | places / layer | `moe` ms |
| --- | --- | --- | --- |
| k3-r1 (1 stream, K=3) | 3.99 | 13.247 | 16.795 |
| k0-r4 (4 streams, K=0) | 4.00 | 18.122 | 22.215 |
| k3-r1-np (K=3, prune unset) | 3.98 | ~16.0 (inferred: k3-r1's `distinct_before`; no prune line with prune off) | 19.768 (`all_equal: false`, probably because the expect_sha values were recorded with prune on; inferred) |

Least squares over the five non-blocking cells with measured places (e-r1, e-r4, e-r8, k0-r4, k3-r1):
`moe = 1.54 + 0.23 x rows + 1.088 x places-a-layer` ms, residuals 0.1 ms or less. Places alone fit within 0.26 ms. Rows
alone miss by up to 3.4 ms. One routed place costs **27.2 us**. A row adds ~0.23 ms a round beyond its places, and the
intercept is ~1.5 ms (shared expert, launches). The regression is on device-counted places, not bytes: fresh bytes
exist for only the two `-lb` cells. At the 6.33 MB marginal place above, it reads at ~233 GB/s, the probe's DRAM rate. k0-r4 against k3-r1 is the clean separation: the same 4 rows with 37%
more places cost 32% more `moe`.

## Prefill phases (0141, item 3: partial)

One cell, prompts of 263 / 505 / 999 / 1,978 / 7,866 / 31,388 ids, 24 chunks. 0141 prints one average over all chunks,
so sizes cannot be told apart from one cell (one cell per size would be needed for M6). Average a chunk (ms):
`moe` 369 (`e_gu` 224, `e_down` 117), `attn` 197, `engram` 158, dense `d_gemm` 90 / `d_unpack` 21 / `d_rot` 19,
gathers 46 + 47.

Downtime: the core (probe + 5 cells) finished at 683 s. The fixed-depth cells, r8 and prefill added ~10 min.

## Production

Stopped 17:06:43Z (idle, 0 running) and healthy again 17:30:00Z: **23 min 17 s down**. `./start.sh restart` (head
`.env`, nothing exported) returned rc=1 on the known first-request smoke failure ("the smoke test request failed"),
and the retried chat answered "OK". Image, command and TF_*/TENSORFOLD_*/HF_HUB env MATCH on both ranks
(`pre.rank*.json` against `post.rank*.json`). `validate.py --compare runs/matrix/base.validate.json`: all passed (8 of 8
equal). Spill store unchanged (points 1/1, bytes equal before and after; the batch cells ran with spill stripped).
