# WP3: the 0198 rope-store kernels (decode key/latent/query, prompt key/latent)

Run 2026-11-21 on the GB10 head (`sm_121`, unified LPDDR5X), engine tree `~/Workspace/tfref-fused-next` branch
`fused-next`. The "old" arm is `b28310e` (the WP2 tip) throughout; the new arm is this package's tip. CUDA 13.0,
`nvcc -O3 -arch=sm_121 --fmad=false --ftz=false` (never fast math). Every timing block held `~/tf-bench/gpu.lock`,
waited for `health.live.connections == 0` and verified the served-token counters did not move across it; timings are
CUDA-graph medians of 20 repetitions of 100 replays (µs per launch) with the empty-kernel floor at the same launch
shape. No commit in this package changed a default: every change rides `TF_DS_FUSED_ROPE_STORE`, and the unfused
fallbacks are untouched.

Harness: the WP0 `tools/fused_bench` (parity + timing + SASS), extended for this package in the recipe repo only:
`--variants` candidate arms in `wp3/vbench.py` (with `wp3/variants.cu`), the query's per-row-count geometry, the
prompt query rows (128, 1024), the latent's shard-target (`latent-owner`) parity case, `REDUX` in the opcode
histogram and the pair kernel's launch block. The engine tree carries only the kernels and their launchers.

## Commit order

Item 2 landed before item 0 (commits `0240` then `0241`), because item 0's geometry only pays with item 2's body:
with the old body the one-element query form is 2.81 µs at R=32 against the kit chain's 2.71, and the pair form is
2.57 µs only once the scale no longer costs an FP32 division by six plus a per-element division. Items 3, 4 and 5
are closed with measurements and carry no commit.

| WP3 item | verdict | commit |
|---|---|---|
| 2. E8M0 scale from BF16 bits, exact reciprocal, warp integer max | landed (latent keeps the shuffle max: measured slower) | `cfa54ab` (0240) |
| 0. the query's geometry at R=32/48 | landed (per-row-count dispatch) | `50a70e1` (0241) |
| 1. skip partner shuffles on wholly unrotated warps | landed for the latent chain only (key and query measured neutral-to-slower) | `9d4232a` (0242) |
| 3. CUDA FP4 conversion intrinsics | closed, no change (exact with a zero-sign mask, but slower on the pinned `sm_121`) | none |
| 4. skip latent packing on non-owner rows | closed, no change (the latent is per-block latency bound at decode rows, so skipping rows saves nothing) | none |
| 5. larger fusion (compressor + latent) | closed, no change (the boundary is one launch floor plus ~0.05 µs of L2 traffic at decode rows) | none |

## Item 2 — E8M0 scale from BF16 bits, exact reciprocal multiply, warp integer maximum (commit 0240)

**What changed** (`zig/kernels/cuda/dsv41_torch.cu`): `tf_ds_pow2_exponent`'s `f = max(amax, 6*2^-126) / 6` and its
exponent/mantissa extraction become `tf_ds_scale_byte_bits`, which reads the byte off the BF16 exponent and the
mantissa's comparison with 1.5 (`byte = exp - 2 + (mantissa > 1.5's), floored at 1`; an infinite magnitude keeps the
kit's infinite scale). The per-element `__fdiv_rn(x, scale)` in the key packing and the query becomes
`__fmul_rn(x, tf_ds_recip_bits(byte))`, where the reciprocal is `2^(127-byte)` built from the exponent (zero for an
infinite scale). The 32-element group maximum becomes `__reduce_max_sync(0xffffffffu, ...)` over sign-cleared BF16
bits with NaNs mapped to zero, replacing four shuffle/maximum stages.

Both transformations are exact by construction: `amax` is always a BF16 value (the rotated rows are bf16-rounded), so
the byte is a function of its bits; multiplying by an exact power of two rounds identically to dividing by its
reciprocal (the same exact scaling, including overflow and subnormal results).

**Parity** (`tools/fused_bench`, `--kernel key,latent,query,prompt-key,prompt-latent`, `wp3/i2.json`, `wp3/i2b.json`,
`wp3/i2c.json`): every case passes with 0 differing bytes against the kit chain (`_rmsnorm`+`_rope_heads`+`_fp4_store`,
`_rope_heads`+`_fp4qd_p2`) — decode key/latent/query at R=1/7/32/48 plus the NaN/±inf/±0 stress rows, prompt key and
prompt latent at R=128/1024 plus stress, and **all 65,536 BF16 encodings through the packed latent store**
(`0198-latent all 65,536 bf16 encodings: 0 differing bytes`). `LOCAL=0 STACK=0` everywhere.

**Timing** (old `b28310e` → new, µs; floor is the empty kernel at the same launch shape):

| kernel | shape | old | new | floor | change |
|---|---|---:|---:|---:|---:|
| 0198-key | R=1 | 1.772 | 1.623 | 0.457 | −8.4% |
| 0198-key | R=7 | 1.768 | 1.624 | 0.457 | −8.1% |
| 0198-key | R=32 | 1.771 | 1.625 | 0.457 | −8.2% |
| 0198-key | R=48 | 1.788 | 1.645 | 0.457 | −8.0% |
| 0198-prompt-key | R=128 | 4.270 | 1.909 | 0.496 | −55% |
| 0198-prompt-key | R=1024 | 6.174 | 3.087 | 1.111 | −50% |
| 0198-query | R=1 | 1.440 | 1.379 | 0.457 | −4.2% |
| 0198-query | R=7 | 1.563 | 1.481 | 0.518 | −5.2% |
| 0198-query | R=32 | 2.957 | 2.853 | 0.928 | −3.5% |
| 0198-query | R=48 | 4.041 | 3.796 | 1.215 | −6.1% |
| 0198-latent | R=1/7/32/48 | 1.542/1.522/1.524/1.523 | 1.543/1.526/1.541/1.542 | 0.457 | 0 (see below) |

The prompt key's large win is the division removal in a kernel whose block is only 64 threads: its per-element chain
was 2 dependent FP32 divisions (each a MUFU.RCP plus Newton steps) long, and it is latency-bound, not issue-bound.

**The latent keeps the shuffle maximum.** With the half-warp masked `__reduce_max_sync` the latent measured
1.541 → 1.644 µs at R=1/7/32/48 (IQR 0.001–0.004), a 0.10 µs regression at every served row: the two half-warp
maxima are not a whole-warp reduction, ptxas duplicated the masked REDUX into three code paths (6 static REDUX), and
the masked form is slower than the four shuffle stages it replaced. Reverting only the latent's maximum restored
1.542/1.522/1.524/1.523 (and its SASS is instruction-for-instruction identical to `b28310e`'s).

**SASS and resources** (`cuobjdump -sass`, `-res-usage`, `wp3/sass.json`; static counts):

| kernel | arm | insns | SHFL | REDUX | MUFU | BRA | REG | SHARED | spills | occupancy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| idx_key | old | 496 | 21 | 0 | 6 | 30 | 30 | 2080 | none | 100% |
| idx_key | new | 344 | 11 | 2 | 2 | 12 | 29 | 2080 | none | 100% |
| prompt_idx_key | old | 632 | 13 | 0 | 6 | 39 | 34 | 4112 | none | 100% |
| prompt_idx_key | new | 480 | 3 | 2 | 2 | 21 | 30 | 4112 | none | 100% |
| rope_qd | old | 368 | 12 | 0 | 4 | 33 | 26 | 1024 | none | 100% |
| rope_qd | new | 168 | 1 | 1 | 0 | 12 | 22 | 1024 | none | 100% |
| rope_store | old | 488 | 20 | 0 | 2 | 31 | 34 | 1024 | none | 100% |
| rope_store | new | 496 | 20 | 0 | 2 | 33 | 34 | 1024 | none | 100% |

The removed instructions are the FP32 division sequence (MUFU.RCP + 4-6 FFMA/FMUL steps per element) and the four
shuffle/FMNMX stages per group; what replaced them is one REDUX, one shift-and-compare byte derivation and one FMUL.
The latent's static histogram is unchanged because its guard is a branch around the same instructions (see item 1).

**Bound.** The key, latent and query kernels are launch-plus-latency bound at every served decode row count: their
bodies are 0.9–2.6 µs against a 0.46 µs launch floor, they move 0.01–0.8 MB (a 0.01–0.8 µs roofline at 954 GB/s),
and one CTA a row means at most 48 CTAs on 48 SMs. The prompt key at R=1024 (3.087 against a 1.111 floor and a
0.33 MB roofline) is now within 2 µs of the floor, i.e. the remaining cost is the per-element chain, not traffic.

## Item 0 — the decode query's geometry at R=32/48 (commit 0241)

**What changed** (`dsv41_torch.cu`, `exact.zig`): the query's one-element-a-thread kernel now takes its grid as
`(rows, heads)` instead of `rows * heads` (the old form divided `blockIdx.x` by the runtime head count in every
block's prologue — 15 instructions and a serial divide before the first load), and a second geometry is launched from
`Exact.pair_rows = 32` upward: `tf_ds_fused_rope_qd_pair_kernel`, a warp a head, walking the head in 64-element chunks
with the rotary pair `(2l, 2l+1)` loaded into one lane as a 4-byte word. That form has no partner shuffle at all (the
pair is a register), the 32-element scale groups are the two half-warps (`__reduce_max_sync` with a half-warp mask),
and the load is one coalesced 128-byte transaction a warp a chunk. `heads % 8` or a head dim that is not a multiple of
64 keeps the one-element form; the launcher's `warpShapesOk` gate is unchanged.

**Parity** (`wp3/i0.json`): R=1/7/32/48, the prompt row counts 128 and 1024, and the NaN/±inf/±0 stress row all pass
with 0 differing bytes against the kit `_rope_heads`+`_fp4qd_p2` chain (`new_vs_ref = 0`, `old_vs_ref = 0`).

**Timing** (old `b28310e` → new; the kit chain is the two-launch reference; floor is the empty kernel at the new
launch shape):

| R | kit chain | old fused | new fused | floor | new vs kit |
|---:|---:|---:|---:|---:|---|
| 1 | 2.141 | 1.442 | 1.336 | 0.456 | −38% |
| 7 | 2.177 | 1.563 | 1.405 | 0.516 | −35% |
| 32 | 2.710 | 2.975 | 2.587 | 0.498 | −4.5% |
| 48 | 2.996 | 4.020 | 2.812 | 0.517 | −6.1% |
| 128 | 5.024 | 9.242 | 5.801 | 0.703 | +15% |
| 1024 | 51.767 | 68.900 | 36.942 | 2.747 | −29% |

The WP0 regression is gone: at R=32 the fused launch is 2.587 against the kit chain's 2.710 (it was 2.975), and at
R=48 2.812 against 2.996 (it was 4.020). Every served decode row count is now faster than the chain it replaces.

**The prompt row counts.** R=1024 improves 47% (68.9 → 36.9, and it is 29% faster than the kit chain), but R=128
remains 15% slower than the two-launch chain (5.80 against 5.02) — a pre-existing gap this package narrows (9.24 →
5.80) rather than closes. Six geometries were measured at R=64/128/256 (`wp3/q7.json`: 2/4/8/16 warps a block,
3.65–6.17 µs against the chain's 3.53–4.97), and none beats the chain there; the chain's `_fp4qd_p2` launch is simply
efficient at those row counts while degrading above them (at R=1024 it is 51.8 µs, the fused form 36.9). The dispatch
threshold therefore sits at decode's largest window rather than at the prompt rows.

**SASS and resources** (`wp3/sass.json`): the pair kernel is 232 instructions, `SHFL=0`, `REDUX=1`, `REG=26`,
`SHARED=1024`, no spills, 100% theoretical occupancy at (256,1); the one-element form is 168 instructions with
`SHFL=1` (was 368 with `SHFL=12`, `MUFU=4`). Neither kernel has a 16-byte global load (`LDG.128=0`): the widest is the
pair form's 4-byte `LDG.E`.

**Bound.** At R=48 the pair kernel is 2.81 µs against a 0.52 µs launch floor and a 0.82 µs roofline (0.79 MB moved at
954 GB/s): issue/latency bound at ~1.9 µs of body, 2.3× the roofline. Its instructions are dominated by the E2M1
threshold chain (7 dependent comparisons an element), which item 3 could only shorten with `sm_121a` codegen.

## Item 1 — skip the partner exchange on wholly unrotated warps (commit 0242)

**SASS first.** `tf_ds_latent_element` exchanged the partner value before testing `i < start`; at the served shapes
`start = D - RD` puts 14 of 16 latent warps (D=512, RD=64) and 2 of 4 key warps (D=128, RD=64) wholly inside the
unrotated prefix, so those exchanges are source-level dead. `tf_ds_fused_rope_store_kernel`'s old SASS: 20 `SHFL.BFLY`
with 10 `WARPSYNC.COLLECTIVE`/`ENDCOLLECTIVE` pairs.

**What changed.** `tf_ds_latent_element` takes a warp-uniform `rot` and returns the value untouched when no element the
warp holds rotates; the test that used to follow the shuffle stays where it is, so a mixed warp still runs the
full-mask exchange with every lane in it. `tf_ds_latent_row` computes the predicate from the warp's last element index
(`(i & ~31) + 31`), which is warp-uniform because `blockDim.x % 32 == 0` and every stride steps in whole warps — the
invariant `rope_store.warpShapesOk` and its tests pin.

**Parity**: the latent's cases (R=1/7/32/48, the stress row, all 65,536 BF16 encodings, the four `latent-owner`
shard-target cases) and every other rope case pass with 0 differing bytes (`wp3/final-timing.json`, 29/29; full run
146/146 three times).

**Timing** (old `b28310e` → new): 1.540 → 1.401 (R=1), 1.544 → 1.424 (R=7), 1.522 → 1.399 (R=32), 1.543 → 1.417
(R=48), and the prompt latent 4.209 → 1.664 (R=128) / 10.268 → 6.393 (R=1024). The static histogram is unchanged
(20 `SHFL.BFLY`) because the guard is a branch around the same block; the executed exchanges drop as the measurement
shows (−9% at every decode row, −4% at the prompt rows).

**The key and the query keep their unconditional exchange.** The same guard on the key path measured 1.623 → 1.686 µs
at R=1 (and the prompt key 1.909 → 1.953), and on the query's one-element form it was neutral (1.336 vs 1.338): both
are latency-bound on the chain around the exchange, and the extra warp-uniform predicate costs more than the skipped
shuffle saves there. They were reverted; only the latent chain carries the guard.

## Item 3 — the CUDA FP4 conversion intrinsics: CLOSED, no change

**Parity first.** `__nv_cvt_float2_to_fp4x2`/`__nv_cvt_float_to_fp4` with `tf_ds_clamp6` in front (which maps a NaN to
−6, the kit's code) and `__nv_cvt_fp4_to_halfraw` for the dequantization. `__nv_cvt_fp4_to_bf16raw` does not exist in
CUDA 13.0; the half raw round trip is exact for the E2M1 magnitudes. The intrinsic keeps the sign of a negative value
that rounds to zero (code `0x8`) where the kit drops it (code `0x0`), so it needs one mask
(`c &= (c & 7) ? 0xf : 0x7`). With that mask, `wp3/vbench.py --kind fp4` reports **0 differing codes or output bytes
over all 16,711,680 (BF16 input, reachable E8M0 byte) pairs** — 65,536 inputs × 255 bytes, above the audit's 8,257,536
— on both the `sm_121` and the `sm_121a` builds.

**SASS and timing.** On the engine's pinned `-arch=sm_121` the intrinsics are *emulated*: the query body grows from
168 to 184 instructions (`v_q_bits` vs `v_q_bitsi2`) and measures 1.297 → 1.362 (R=1), 1.462 → 1.608 (R=7),
2.527 → 3.075 (R=32), 3.386 → 4.123 (R=48). With `-arch=sm_121a` the same source emits the real instruction
(`CVT`/`F2FP`, 37 of them in the file) and the intrinsic wins: 1.296 → 1.173, 1.461 → 1.337, 2.547 → 2.259,
3.385 → 2.975 (`wp3/i3a.json`, `wp3/i3b.json`). The engine's flags are pinned to `sm_121`, so the intrinsic stays out;
switching the build to `sm_121a` would be worth ~10–15% on the query body and is recorded as a follow-up, not a WP3
change.

## Item 4 — skip latent packing on non-owner decode rows: CLOSED, no change

**What was tried.** The sharded latent targets `rows_idx[r] = trashFirst + i` for a non-owner row (past the part's
rows), whose bytes nothing reads. The kernel took `skip_from` (the trash area's first row) and returned before the
rotation and packing when `rows_idx[r] >= skip_from`, and `round.zig` passed `Shard.trashFirst(ratio)`.

**Parity**: exact — the new `latent-owner` cases (owned rows `0..own-1`, the rest on trash rows) compare the packed
bytes whole and pass, including `own = 0` (every row a trash row).

**Why it was reverted.** At decode row counts the latent kernel is *per-block latency* bound, not throughput bound: one
CTA a row, at most 48 CTAs on 48 SMs, each with a ~0.9 µs dependent chain. Skipping half the rows frees half the SMs
and shortens nothing:

| case | old (writes every row) | new (skips non-owner rows) | floor |
|---|---:|---:|---:|
| R=48, own=24 | 1.546 | 1.582 | 0.457 |
| R=48, own=1 | 1.542 | 1.564 | 0.457 |
| R=32, own=0 | 1.542 | 0.706 | 0.459 |

Only the `own = 0` case (every row skipped) gains; with one owned row left the critical path is unchanged, and the
extra `rows_idx` load the branch depends on costs 0.02–0.04 µs. Reverted; the harness's `latent-owner` parity cases
stay as the shard-target regression check.

## Item 5 — larger fusion (compressor + latent RoPE/FP4): CLOSED, no change

The boundary between 0197 and 0198 is `rd.lat`: the compressor must write it (the index-key projection reads it
before the latent packing), so a fusion can only save the latent launch and its read of `rd.lat`, not the write.
Measured against the WP0/WP2 tables: the latent launch floor is 0.457 µs at decode rows and its `rd.lat` read is
49 KB at R=48 D=512 (0.05 µs at the 954 GB/s cache roofline) — 0.51 µs, i.e. one launch floor plus noise, and the
latent's own body (0.95 µs) is latency, which a fusion would not remove. At the prompt rows the read is 1 MB
(~1.0 µs) plus the 0.99 µs floor, but the prompt latent's 5.4 µs body dominates and the prompt compressor is already
at the bandwidth roofline (857–894 GB/s cache, 264 GB/s DRAM), so fusing would remove ~2 µs of a ~17 µs pair. Under
the item's own gate ("only if the boundary costs more than a launch floor") the answer is no for the decode path and
marginal for the prompt path; no change.

## Final position of every kernel against the launch floor and the roofline

Final tip (0242), old = `b28310e`. Bytes are the launch's live traffic (inputs read plus outputs written, from the
argument shapes); the roofline is those bytes at the measured 954 GB/s cache / 224 GB/s DRAM copy rates (WP0's
`bw.json`); the floor is the empty kernel at the same launch shape. `wp3/final-timing.json`, `wp3/sass.json`.

| kernel | shape | us | floor | MB | roof us | basis | bound |
|---|---|---:|---:|---:|---:|---|---|
| 0198-query | R=48 pair | 2.812 | 0.517 | 0.79 | 0.82 | cache | issue/latency, 1.9 µs body |
| 0198-query | R=32 pair | 2.587 | 0.498 | 0.52 | 0.55 | cache | issue/latency, 1.6 µs body |
| 0198-query | R=7 one-element | 1.405 | 0.516 | 0.11 | 0.12 | cache | latency (per-block chain) |
| 0198-query | R=1 one-element | 1.336 | 0.456 | 0.02 | 0.02 | cache | latency (per-block chain) |
| 0198-prompt-key | R=1024 | 3.086 | 1.110 | 0.33 | 0.35 | cache | latency, 2.0 µs body |
| 0198-prompt-key | R=128 | 1.912 | 0.497 | 0.04 | 0.04 | cache | latency |
| 0198-prompt-latent | R=1024 | 6.393 | 0.993 | 1.34 | 1.41 | cache | issue/latency, 5.4 µs body |
| 0198-prompt-latent | R=128 | 1.664 | 0.497 | 0.17 | 0.18 | cache | latency |
| 0198-key | R=48 | 1.647 | 0.457 | 0.02 | 0.02 | cache | latency |
| 0198-latent | R=48 | 1.417 | 0.457 | 0.06 | 0.07 | cache | latency |

Every decode-row kernel is at the launch/latency bound WP0 predicted: 0.9–2.4 µs above a 0.46 µs floor with well
under 1 MB moved, one CTA a row, and no 16-byte global loads. The two prompt-row kernels that are still above the
floor (prompt latent R=1024 at 6.39, prompt key R=1024 at 3.09) are issue/latency bound on the per-element E2M1/E4M3
chain; the remaining lever there is cheaper conversion code, which needs `sm_121a` (item 3) rather than another
geometry or a fusion. Nothing in this package is at a bandwidth bound, and nothing is within 0.2 µs of its floor
except the decode key/latent at R=1–7, where the floor is 0.46 and the kernels are 1.40–1.42.

## Reproduce

Everything runs on the GB10 head; the workstation has no GPU or nvcc. Sources: `~/tf-bench/fused-next/wp3/src-old`
(`b28310e`'s two `.cu` files), `wp3/src-new` (this tip's), `wp3/variants.cu` (the candidate and probe kernels),
`wp3/vbench.py` (the candidate harness).

```bash
# build both trees with the engine's flags, plus the floor/bandwidth nodes
cd ~/tf-bench/fused-next && bash tools/build.sh wp3/src-old wp3/src-new wp3/bin
# build the candidate and probe kernels (both arches; sm_121 is the engine's)
cd wp3 && /usr/local/cuda/bin/nvcc -O3 -arch=sm_121  --fmad=false --ftz=false -include cuda_fp4.h -cubin variants.cu -o bin/variants.cubin
          /usr/local/cuda/bin/nvcc -O3 -arch=sm_121a --fmad=false --ftz=false -include cuda_fp4.h -cubin variants.cu -o bin/variants_a.cubin
# parity + timings per kernel (each run takes ~/tf-bench/gpu.lock and waits for an idle production)
cd ~/tf-bench/fused-next && python3 tools/bench.py wp3/bin wp3/bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --kernel key,latent,query,prompt-key,prompt-latent --json wp3/final-timing.json
# the full byte-parity matrix (146 cases, all kernels) — the three runs reported above
python3 tools/bench.py wp3/bin wp3/bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --parity-only --json wp3/final-parity-1.json
# SASS/resources/occupancy
bash tools/sass.sh wp3/bin --json wp3/sass.json
# the query geometry sweep and the FP4 intrinsic probe
cd wp3 && python3 vbench.py --kind query --rows 1,7,32,48,64,128,256,1024 --arms v_q_bits,v_q_p8,v_q_p16 --json q7.json
          python3 vbench.py --kind fp4 && python3 vbench.py --kind fp4 --variants bin/variants_a.cubin
```

`NCU_REQUESTS.md` (below) lists the counter commands for a later run; no decision in this package needed one.

## What is left open

- **R=128 prompt query**: 5.80 µs against the kit chain's 5.02. Six geometries were measured; none wins at those row
  counts. Closing it needs a cheaper per-element chain (`sm_121a` codegen, item 3's follow-up), not another geometry.
- **`sm_121a` build**: the FP4 conversion intrinsics are exact with the zero-sign mask and 10–15% faster on the query
  body when the build targets the family-specific arch. The engine's flags are pinned to `sm_121`, so this is a
  build-flag decision above this package.
- **The prompt latent's 5.4 µs body at R=1024** is the largest single body left in the rope-store family; its E4M3
  scale needs a real division per group (the scale is not a power of two), and it computes the scale byte in all 16
  lanes of a group. Doing it once a group and broadcasting is the next measurement.
