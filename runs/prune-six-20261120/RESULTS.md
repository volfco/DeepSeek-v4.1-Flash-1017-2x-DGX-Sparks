# Six-slot decode pruning on DGX Spark

Patch 0199 selects a one-thread-per-row kernel when `TF_DS_FUSED_ROUTE=1`, the applied rule is 0.85/3/kept, the cap is off, and each row has six routed slots plus one shared slot. It works with triage on or off. Both stats settings use the specialization. The original kernel handles other rules and shapes. [Recipe patch](../../zig/patches/0199-dsv41-six-slot-prune.patch)

The recipe series applied in order to the pinned engine `fbbc9713`. I moved the location of an existing forward declaration in patch 0195 so it applies with the declaration already present from 0190z. Both the native server and rank executable built for `sm_121` in the Spark container. The Zig test for eligible and fallback dispatch passed. Production continued serving its existing image. [Build log](build.log)

## Timings

Median µs per kernel node on the GB10 head. The table uses randomized synthetic weights and expert IDs, with occasional missing picks and no dead-row mask. Each cell has 16 measured 256-node graph replays. Modes rotate across passes, inputs reset outside the timed region, and every node gets a fresh slice of input.

| Decode rows | Existing kernel, stats on | Earlier six-slot prototype, stats off | 0199, stats off | 0199, stats on |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.444 | 1.794 | 1.520 | 2.216 |
| 7 | 9.782 | 2.133 | 1.819 | 2.547 |
| 28 | 10.716 | 2.400 | 1.976 | 2.736 |
| 64 | 10.924 | 2.480 | 2.036 | 2.776 |

At seven rows the extra work saved over the earlier stats-off specialization was 0.314 µs per layer (14.7%). With stats enabled, the saving over the current generic kernel was 7.235 µs per layer. Forty such layers on the critical path would save 0.29 ms in a 42 ms round, about 0.69%. This is an extrapolation from the isolated kernel. The previous service experiment measured the stats toggle on the generic kernel. [Earlier service A/B](../prune-stats-20261120/RESULTS.md)

Already-ordered rows take a shorter path. At seven rows the final stats-off kernel took 1.274 µs with descending weights and 1.811 µs with the same full six-weight rows in generated order. The actual frequency of weight-ordered rows in serving was not measured. With missing picks and a synthetic 25% dead mask the final medians were 1.980 µs without counters and 2.587 µs with counters. [All timing samples](timings.csv)

## Arithmetic and validation

- Carry each slot's identity in one bit. Lower original slot wins a tie between equal rounded shares. Test the order first, then run the six-item sorting network if needed. The prefix retains the reference addition order. Only ranks four through six require a drop decision.
- Accumulate retained weights and dropped mass in original slot order. Dead rows contribute to stats, then all seven slots are cleared.
- With counters enabled, each in-range expert ID sets a bit in either the retained set or the discarded set. Their union gives distinct experts before pruning, including IDs that occur in both sets. The warp sums integer counters and publishes six totals.
- Compared raw picks and FP32 weight bits with the generic CUDA kernel at 1, 7, 28, 32, 33, and 64 rows. Compared all six counters separately for each of 256 windows per size and dataset. Inputs included missing picks, ties, repeated experts, nonpositive and nonfinite totals, subnormals, and threshold cases. CUDA graph timings used PDL off. Service throughput, long replies, and PDL replay with this specialization remain unmeasured. [GPU verification and compiler output](verification.log)

A follow-up correctness check replayed the same 256-node graph twice with fresh inputs and a shared counter buffer seeded near a 32-bit carry. At all six window sizes, each replay matched the generic eager reference for picks and weight bits. The six counters accumulated the expected totals with stats on and retained their seeds with stats off. All five datasets passed on GB10 with PDL off. Append `--verify-only` to the dataset argument to run these checks without collecting timings. [Replay verification](graph-verify.log)

To reproduce, run the following from the recipe root with the patched engine checkout as the argument to `source.py`. CSV columns are dataset, rows, pass, mode, and µs per node. Modes 0–3 correspond to the table columns in order.

```sh
python3 runs/prune-six-20261120/source.py /path/to/patched/engine > /tmp/prune-six.cu
nvcc -O3 -arch=sm_121 --fmad=false --ftz=false -Xptxas=-v /tmp/prune-six.cu -o /tmp/prune-six
for inputs in edges mixed full sorted masked; do
    /tmp/prune-six "$inputs"
done
```
