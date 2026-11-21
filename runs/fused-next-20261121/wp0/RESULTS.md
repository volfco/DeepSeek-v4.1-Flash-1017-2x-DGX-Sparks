# WP0: the measurement harness and the baselines

Run 2026-11-21 on the GB10 head (`sm_121`), engine tree `~/Workspace/tfref-fused-next` branch `fused-next` at
`3c54a9d`, CUDA 13.0. **No engine commits**: this package only adds `tools/fused_bench/` and a `just fused-bench`
recipe to the recipe repo, and writes this evidence. The engine tree was not modified.

## What was built

`tools/fused_bench/` (recipe repo, uncommitted):

| file | what it does |
|---|---|
| `build.sh OLD_CUDA_DIR NEW_CUDA_DIR OUT` | compiles both trees' `dsv41_ops.cu`/`dsv41_torch.cu` with `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false`, plus `floor.cubin` (`empty_kernel` launch floor, `copy_kernel` bandwidth probe) |
| `bench.py` | ctypes + CUDA driver API: loads old/new/kit cubins, checks MemAvailable (≥3 GiB) and the production health endpoint, takes `~/tf-bench/gpu.lock`, runs byte parity of every live output against the unfused/kit producer, and interleaved CUDA-graph timings (A B A B, ≥20 graphs × 100 replays, warmed, median+IQR) with an empty-kernel floor at the fused launch shape; `--kernel`/`--shape` filters, `--json`, `--parity-only`, per-block idle-window guard |
| `sass.sh` / `sass.py` | per-kernel opcode histogram (LDG/STG by width, LDS/STS, SHFL, BAR, ATOMS, MUFU, BRA, FFMA/FMUL/FADD), registers/spills/shared/local, theoretical occupancy at the served block, old vs new |
| `bw.py` | copy-kernel bandwidth (device-to-device, 1/8/64/256 MiB) for the rooflines |
| `report.py` | bench JSON + bw + sass → ranked baseline table with roofline, achieved bandwidth, bound and headroom |
| `README.md` | one command line per use |

The harness is the consolidation of `runs/fused-kernel-audit-20261120/{build.sh,probe.py,decode.py}` and
`runs/fused-followup-20261121/{sass.sh,sass.py}`, with the current tree's signatures (0195's `idx_fused` has no
bmax/pidx scratch arguments; 0198's key/prompt-key kernels have no dead rotated-row argument), the launch floor, the
bandwidth roofline, the 0199 prune family and the kit `_score_keys` pricing added.

## Proof runs

1. **OLD = NEW = `~/Workspace/tfref-fused-next`.** `build.sh src-old src-new` produced md5-identical cubins
   (`0aa40aee…` ops, `70d020a9…` torch). The full run (`wp0.json`) reports **0 differing bytes over all 88 parity rows**
   and `failures: []`. Old and new timings (same cubin, separate modules and output allocations) agree to a median
   |old−new| of **0.001 µs** over 48 timed rows; the largest deltas are allocation/address placement on the two largest
   DRAM-bound shapes (0195 R=2048: 1115.340 vs 1119.294 µs, 0.35%; compress-prompt R=1024 D=2048: 141.871 vs
   143.302 µs, 1.0%). No code difference exists between the arms, so this is the harness's own noise floor.
2. **OLD = production's unfused path, where one exists.** Every parity case compares the fused launch against the
   unfused chain or the kit producer in the same process, and all pass with 0 differing bytes:
   - 0195 `idx_fused` vs `block_max` + `topk_indices` + `pool_pick` (mask and block list; decode and prompt shapes);
   - 0197 decode `compress2x_norm` vs `compress2x` + kit `_rmsnorm`; prompt `compress2_norm` vs `compress2` + `rms_norm`;
   - the five 0198 kernels vs the kit's `_rmsnorm`, `_rope_heads`, `_fp4_store`, `_fp4qd_p2`;
   - 0199 `prune_six` vs `expert_prune` (pick, weights and all six counters).
3. **Deliberate one-bit change.** A scratch copy of the new tree (`src-seeded`) drops the `^ 0x8000000000000000` in
   `tf_ds_score_key` (the known 0206 encoding difference, one instruction). Built as `bin-seeded` and run against the
   pristine `bin`:
   ```
   FAIL  0195-idx  idx R=7 nb=4096 width=32765 k=2048 list random  {'new_vs_ref': 11384, 'old_vs_ref': 0, 'new_vs_old': 11384}
   FAIL  0195-idx  idx R=7 nb=4096 ... special                     {'new_vs_ref': 11698, 'old_vs_ref': 0, ...}
   EXIT=1
   ```
   The harness catches it, attributes it to the new arm (old_vs_ref stays 0) and exits non-zero. `seeded.log`.

## Parity coverage (all 0 differing bytes; `wp0.json`)

| kernel | rows | cases |
|---|---:|---|
| 0195 idx | 15 | R=1/7/32/48 nb=8 k=8; R=7 nb=2049/4096/8192; R=32 nb=8192; R=128/1024/2048 nb=4096; mask and block list; random, tied, flat; NaN/±inf/−0 rows; width 61/16389/32765/65533 |
| 0197 decode compressor | 25 | D=128/512, R=1/7/32/48 mixed; unaligned, cached, cross, invalid and stress layouts at R=1/7/33/65; stride padding, both rings and the invalid flag compared |
| 0197 prompt compressor | 6 | D=512/2048, R=128/1024; special (NaN/±inf/subnormal/±0) |
| 0198 key | 5 | R=1/7/32/48; special; codes and scales |
| 0198 latent | 6 | R=1/7/32/48; special; **all 65,536 bf16 encodings** through the packed store |
| 0198 query | 5 | R=1/7/32/48; special; dequantized output |
| 0198 prompt key | 3 | R=128/1024; special |
| 0198 prompt latent | 3 | R=128/1024; special |
| 0199 prune | 8 | R=1/7/32/48; dead mask R=7/48; ties/NaN/inf/negative shares; all six counters |

The rotated intermediates are not compared for the key/latent/prompt kernels because the fused path keeps them on chip
(their stores have no reader); codes/scales (or the dequantized query) are the live outputs and are compared.

## Timing methodology and the idle-window guard

Every timing block waits for `127.0.0.1:8000/health` to report `"connections": 0`, records
`prompt_tokens_total`/`generation_tokens_total`, runs the block, and re-reads them; if they moved, the block is rerun.
The reported run has `timing_retries: 0`, `timing_clean: true`, and identical before/after counters — no request was
served during any timed block. The run held `~/tf-bench/gpu.lock`. `MemAvailable` was 6.1 GiB at start; the largest
case (0195 R=2048, 268 MB score + outputs) stays well under 1 GiB of device allocations, and each family is freed
before the next.

Timings are medians of 20 repetitions of 100 graph replays, old/new/floor interleaved. The full per-shape table,
rooflines and bounds are in `BASELINE.md`.

## SASS and resources

Old and new cubins are md5-identical on this tree, so the old-vs-new resource diff is empty. The baseline SASS table
(registers, spills, shared, occupancy, opcode histogram) is in `BASELINE.md`. Headline facts: no spills
(`STACK=0 LOCAL=0`) anywhere; **no `LDG.E.128` in any kernel** (only scalar/64-bit global loads); `idx_fused` has
`BAR=8`, `SHFL=20`, `LDS=12`, `STS=14` at 66.7% occupancy; `prune_six` is a single-CTA kernel with `ATOM=6`, `STG=64`
and no SHFL.

## Bound each kernel sits at

See `BASELINE.md` for the per-shape table. In short: 0195 is DRAM-bandwidth-bound at R≥1024 (240–247 GB/s, at the copy
roofline) and issue/latency-bound at R≤128 (up to 18.4 µs of headroom, one CTA per row, radix tie passes); the 0197
prompt compressors are at the bandwidth roofline; the 0197 decode compressor, the 0198 decode key/latent/query and 0199
prune are launch/latency-bound 1.0–2.8 µs above the empty-kernel floor; 0198 prompt latent/key at R=1024 have 5.1/2.4 µs
over the roofline. The kit's `_score_keys` is priced at 216–220 GB/s (DRAM) for R=128, 16K–65K columns.

## Hardware counters

No decision in WP0 needed counters, so no `NCU_REQUESTS.md` is raised. If a later item needs the executed-instruction
or shared-atomic contention mix, the request is:
`ncu --set full --kernel-name tf_ds_idx_fused_kernel --launch-count 1 --launch-skip 0 python3 tools/bench.py ...`
(single kernel, graphs off) — recorded here so the supervisor can run it without root.

## Reproduction

```bash
# on the head, from ~/tf-bench/fused-next (tools + src-old/src-new synced from the recipe repo)
bash tools/build.sh src-old src-new bin
python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --json wp0.json --wait 0
bash tools/sass.sh bin --json sass.json
python3 tools/bw.py bin --json bw.json
python3 tools/report.py wp0.json bw.json sass.json --md BASELINE-table.md

# seeded-change proof
cp -r src-new src-seeded
python3 - <<'PY'
from pathlib import Path
p = Path("src-seeded/dsv41_ops.cu"); s = p.read_text()
s = s.replace("    return key ^ 0x8000000000000000ull;", "    return key;")
p.write_text(s)
PY
bash tools/build.sh src-old src-seeded bin-seeded
python3 tools/bench.py bin bin-seeded <kit> --kernel idx --shape "idx R=7 nb=4096" --parity-only   # exits 1

# or from the workstation: syncs the two trees, builds, benches, sasses, and pulls the artifacts into this folder
just fused-bench
# overrides are environment variables (the recipe inherits them), e.g. one kernel and shape:
KERNEL=idx SHAPE="idx R=7 nb=8 width" JSON=one.json just fused-bench
```

## Left open

- The kit `_score_keys` cubin choice: `df85d0…` is the `HAS_TMAX=1` variant used by the pruned path; `e1e8b0…` is the
  no-tmax variant. Priced with `df85d0…`. Both were checked to run with the recorded grid `(rows, cdiv(n,1024))`; an
  earlier harness revision launched grid `(rows,)` and under-priced it (only the first 1,024 columns), so the pricing in
  `BASELINE.md` uses the recorded grid.
- The fused 0198 query is **slower** than the unfused two-launch kit chain at R=32/48 (2.98 vs 2.73 µs; 4.02 vs
  3.02 µs). Recorded as a baseline finding; the later geometry item should decide whether the fused query dispatches
  to the two-launch form at those rows.
- `_score_keys` is only a baseline here (this tree has no 0206 fold); the fold's cost is consistent with the previous
  campaign's 316.6 µs vs 38.5 µs at R=128 width 32,768.
- No engine commits in WP0, so the host suite was not re-run for this package (nothing in the engine changed).
