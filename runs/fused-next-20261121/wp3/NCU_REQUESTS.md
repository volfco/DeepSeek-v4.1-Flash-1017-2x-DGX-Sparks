# ncu requests for WP3 (the supervisor runs these; no decision in this package was blocked on counters)

Nothing in this package needed hardware counters: every keep/drop was decided by byte parity, SASS and CUDA-graph
timings. Two measurements would still settle open questions for a later pass.

## 1. Executed SHFL count for the latent's warp guard (item 1)

The guard's *static* histogram is unchanged (20 `SHFL.BFLY`, 10 `WARPSYNC.COLLECTIVE`/`ENDCOLLECTIVE` pairs), because
it is a branch around the same block; the 9% decode-row win says the executed count dropped, but the exact figure is
unmeasured. This confirms the guard skips 14/16 latent warps' low-half exchange and 6/16 high-half.

```
ncu --target-processes all --kernel-name regex:tf_ds_fused_rope_store_kernel \
    --launch-count 1 --launch-skip 20 \
    --metrics smsp__inst_executed_op_shfl_pred_on.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum \
    python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --kernel latent --shape "rope latent R=48"
```

Decides: the executed SHFL/instruction ratio for the guarded kernel against `b28310e`'s (build the old tree's cubin as
a second run). Expected: SHFL ≈ 6/32 of the old count at R=48.

## 2. The prompt latent's body at R=1024 (6.39 µs against a 0.99 µs floor)

Which limiter the 5.4 µs body sits on, and whether the 16-lane redundant E4M3 scale computation (every lane of a group
divides by six and converts to E4M3, one lane writes the byte) is worth replacing with a lane-0 computation plus a
broadcast.

```
ncu --target-processes all --kernel-name regex:tf_ds_fused_prompt_latent_kernel \
    --launch-count 1 --launch-skip 20 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_elapsed,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__inst_executed.sum,l1tex__t_bytes.sum \
    python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --kernel prompt-latent --shape "rope prompt latent R=1024"
```

Decides: if issue-active is high and the instruction count is dominated by the per-element chain, only cheaper code
(`sm_121a` intrinsics) helps; if long-scoreboard stalls dominate, the group-scale redundancy is not the problem and the
fusion/broadcast idea is closed.
