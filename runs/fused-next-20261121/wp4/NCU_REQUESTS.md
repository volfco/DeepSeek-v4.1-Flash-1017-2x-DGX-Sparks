# ncu requests for WP4 (no counters were available: ncu needs root)

Nothing in WP4's decisions depends on counters: item 1 is a launch-floor comparison, item 2 was decided on chain
timings plus SASS wait placement and delayed-producer canaries. If the supervisor wants counter confirmation of the
item 2 mechanism (whether the successor's pre-wait work actually overlaps the predecessor), these are the runs:

```bash
# The split's overlap, decode chain, PDL on, old vs new cubin. Look at:
#   - the SM's active-warps during the predecessor's tail (does the dependent's CTA have its weight load issued
#     before ACQBULK? `smsp__inst_executed_op_global_ld` split by kernel),
#   - `gpu__time_duration.sum` per kernel in the chain (the compressor's and key's own durations must NOT grow),
#   - `lts__t_sectors_srcunit_tex_op_read.sum` for the weight line (one extra sector per launch, 128 B).
ncu --target-processes all --kernel-name regex:"tf_ds_(compress2x_norm512|fused_idx_key)_kernel" \
    --metrics gpu__time_duration.sum,smsp__inst_executed_op_global_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum \
    --replay-mode application python3 tools/chain.py --old bin --new bin-cand \
    --kit /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --mode chain --shape "R=48 nb=8192 k=2048" --no-guard
```

What it would decide: whether the 0.2-0.35 µs chain gain is the hidden weight-load latency (the metric above) or the
earlier dependent launch (the active-warps profile), which changes where a further split would pay. Neither outcome
would revert the commit: the chain timing and the parity are the decision.
