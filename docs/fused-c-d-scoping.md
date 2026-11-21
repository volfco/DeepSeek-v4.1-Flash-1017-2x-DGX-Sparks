# Fused candidates C and D: scope, launch inventory and bit-identity risk

First pass of 2026-11-19, for the two remaining fusible candidates in [continue.md](../continue.md) item 4: (c) the
ratio-2 compressor chain and (d) Engram / norm-RoPE glue. Both are queued to be gated on the Spark exact suite **after**
0193/0194 (the fused route) and after an A/B arm, exactly like 0195/0196. Nothing here is deployed; every switch is off
by default.

The engine is `/home/colum/Workspace/tfref-next` branch `fused-route` (0193/0194 + 0195 + 0196 + these two). The kit's
Triton source is public (`bertholomus/TensorFold` branch `deepseek-v41-tp2`,
`src/tensorfold/families/deepseek_v41/cuda/kernels.py`); see [kernel-kit-provenance.md](kernel-kit-provenance.md).

## 1. What the model's layer mix is

`config.zig` (the served checkpoint's `compress_ratios` and source lists): 40 layers — 2 uncompressed (ratio 0), 18 at
ratio 2, 20 at ratio 1; `kv_source_layer_ids` = {2, 8, 14, 20}; `index_source_layer_ids` = {2, 8, 14, 20, 24, 28, 32, 36};
2 Engram layers; 3 draft blocks. The compressor and the norm-RoPE-store block exist only on the KV-source layers
(`weights.zig` sets `comp_wkv`/`idx_wk` only for `kv_sources`), so the real per-round site counts are far smaller than
the layer count suggests:

| Candidate | Sites per round | Launches saved |
| --- | --- | --- |
| (c) ratio-2 compressor chain | 3 KV-source layers at ratio 2 (2, 8, 14) | 3 (decode), plus the prompt chunk's chain |
| (d) norm-RoPE glue, indexer key | 4 KV-source layers | 4 × 2 = 8 |
| (d) norm-RoPE glue, latent | 4 KV-source layers | 4 × 1 = 4 |
| (d) indexer query prep (`_rope_heads`+`_fp4qd_p2`) | 8 index-source layers | 8 |
| (d) Engram glue | 2 Engram layers | 2 |

So the honest ceiling for this pair is roughly 3 launches (C) and 12–22 launches (D) a round — on the order of 0.05–0.15%
of a 30–50 ms round at ~2–3 µs a launch, i.e. well inside `continue.md`'s "under 1%" / "0.5–1%" as ceilings, not
expected values. `continue.md`'s "~20 launches/round" for (c) describes the **unfused** chain (7 launches × 3 layers ≈ 21),
which patch 0127 already fuses — so (c)'s remaining gain is the single trailing norm launch a layer, not 20.

Both are only worth keeping if the PDL gap profiler (`TF_DS_PDL`, item 2 of continue.md) shows real device-side launch
gaps; a captured graph hides host launches, so these are graph-visible only.

## 2. (c) the ratio-2 compressor chain — already half built

**Patch 0127 (`TF_DS_FUSED_COMPRESS2`) already fuses the chain's scatter ×2, memset and gather ×2 with `compress2` into one
kernel** (`tf_ds_compress2x_kernel`, `zig/kernels/cuda/dsv41_torch.cu:126`; launcher `Exact.compress2x`,
`exact.zig:106`; call site `round.zig`'s `compressorChain`, the `rd.fused_compress2` arm). So the decode chain today is
**two** launches, not seven:

| Arm | Launches |
| --- | --- |
| unfused | `scatterRows`, `scatterRows`, `memset`, `gatherRows`, `gatherRows`, `compress2`, `_rmsnorm` (7) |
| 0127 | `compress2x`, `_rmsnorm` (2) |
| (c), this pass | one fused `compress2x`+norm launch (1) |

The only remaining launch is the trailing RMSNorm. The prompt path's chain is unfused but already repo-owned:
`Exact.compress2` + `Exact.rmsNorm` (both `dsv41_torch.cu`, byte-checked against torch's ops), so folding those two is
repo+repo.

**Bit-identity risk.** The decode path's norm is the kit Triton `_rmsnorm` (`tri_basic.rmsnorm`), **not** torch's
`ops.rms_norm`; `Exact.rmsNorm` reproduces torch's reduction order (`rmsLanes`), which is a different implementation.
Folding the decode norm therefore means replicating `_rmsnorm`'s `tl.sum(x*x, axis=0)` order — the same class of risk as
0193's `_rowmm_parts`/`_route_parts`, and unproven on a GB10. The prompt fold needs no such replication.

## 3. (d) Engram / norm-RoPE glue

**norm-RoPE (decode).** Per compressed layer, after the indexer's `wk` linear the engine runs `_rmsnorm` → `_rope_heads` →
`_fp4_store` for the indexer key, and `_rope_heads` → `_fp4_store` for the latent. All three are kit Triton kernels.
`_rope_heads` is pure elementwise (no reduction → bit-safe to replicate). `_fp4_store` is elementwise quantisation plus a
`tl.max(tl.abs(...))` over the block — a max, which is order-independent → bit-safe. `_rmsnorm` carries the reduction
risk above. The prompt path uses repo `Exact.rmsNorm` + repo `Exact.rope` + kit `_fp4_store`, so its fold only needs the
`_fp4_store` replication.

**Engram (decode).** The only fusible pair is `add2Bf16` (repo, `tf_ds_add2_bf16_kernel`) followed by `_engram_gate` (kit
Triton), which has three `tl.sum` reductions over `BLOCK=1024` plus `tl.exp`/`tl.sqrt`. Two launches a round for a
reduction-order risk of the same class as `_rmsnorm` — a poor trade unless it comes free with the `_rmsnorm` work.

## 4. The constraint that decides every fold

**Bits are per call site, not per operation.** The repo has torch-parity CUDA kernels (`tf_ds_rms_norm_kernel`,
`tf_ds_rope_kernel`) that reproduce *torch's* `ops.rms_norm` / `ops.rope_`. The decode path instead launches *Triton*
kernels (`_rmsnorm`, `_rope_heads`, `_fp4_store`, `_engram_gate`), which are different implementations. A fold may not
swap one for the other: each fused kernel must replicate exactly the kernel its call site launches today, op for op,
including its reduction order. Any other choice changes bits and fails the exact suite.

## 5. Deliverables of this pass

Two off-by-default patches, built with the same discipline as 0193–0196 (switch parse that refuses bad values, `fuseOk`
bounds with an exact unfused fallback, host reference and tests, a CPU emulation harness of the exact kernel source, and
an independent review before the record calls them ready):

- **0197** `TF_DS_FUSED_COMPRESS2_NORM` — the ratio-2 compressor chain in one launch, decode and prompt.
- **0198** `TF_DS_FUSED_ROPE_STORE` — the norm-RoPE-store glue in one launch per site (indexer key, latent, and the
  indexer query's `_rope_heads`+`_fp4qd_p2`), stacked on 0197 (it reuses the replicated `_rmsnorm`). Engram's
  `add2Bf16`+`_engram_gate` fold is evaluated in this pass and kept only if it is clean.

Neither is justified ahead of the 0193/0194 Spark exact gate; the order stays 0193/0194 → A → B → C → D, one arm at a
time, with the full sibling set stacked, exactly as for 0195/0196.

## 6. References

- `continue.md` item 4 (c)/(d) and item 2 (PDL, the gate that decides whether any of this is worth keeping).
- `zig/patches/0127` (`TF_DS_FUSED_COMPRESS2`, the chain's first half), `zig/patches/0193`/`0194` (the fused route, the
  reduction-replication precedent), `0195`/`0196` (A and B).
- `zig/kernels/cuda/dsv41_torch.cu` (`tf_ds_rms_norm_kernel`, `tf_ds_compress2_kernel`, `tf_ds_compress2x_kernel`,
  `tf_ds_rope_kernel`), `zig/src/families/dsv41/exact.zig` (their launchers and the torch-parity claim),
  `zig/src/families/dsv41/tri_basic.zig` / `tri_norm.zig` / `tri_attn.zig` (the Triton launches being folded),
  `zig/src/families/dsv41/round.zig` `compressorChain`/`attention`/Engram stretch, `zig/src/families/dsv41/prompt.zig`
  compressor/indexer blocks.
- Kit Triton source: `bertholomus/TensorFold` @ `deepseek-v41-tp2`, `src/tensorfold/families/deepseek_v41/cuda/kernels.py`
  (`_rmsnorm`, `_rope_heads`, `_fp4_store`, `_engram_gate`).
- `docs/kernel-kit-provenance.md` (why the kit's binaries are public but a rebuild still owes the Spark gate).
