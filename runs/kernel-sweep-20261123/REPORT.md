# Kernel pattern sweep — report (2026-11-23)

Three patches landed on `kernel-sweep-0290` (`~/Workspace/tfref-ksweep`, base `2bb052f`), each off by default, each
with a CPU emulation of the old and new kernel source that compares bytes and a clean local gate. One Metal change is
staged as an unverified diff. **Nothing was timed and nothing was compiled by nvcc or Apple's compiler**: every
statement below about speed is what the same pattern measured somewhere else. Nothing was pushed; nothing in the
recipe repository was committed; no other worktree was written.

## Landed

| Patch | Commit | Switch | Pattern | What it does | Where it runs |
|---|---|---|---|---|---|
| 0290 | `6571e56` | `TF_DS_DRAW_CUT=1` | P1 | The two sampled selectors (`tf_ds_topk_select_kernel`, `tf_ds_draw_rows_kernel`) stop their radix select at a complete bin instead of running all eight digits | once a sampled round; both ranks (select) or rank 0 (the fallback draw) |
| 0291 | `c58b41e` | `TF_DS_QD_PAIR_SHFL=1` | A4 | The pair query's two half-warp scale-group maxima by four width-16 shuffles instead of a half-warp masked `__reduce_max_sync` | per indexer query at 32 rows and up: prompt chunks, wide decode windows |
| 0292 | `c5c257f` | `TF_DS_BLOCK_MAX8=1` | P5 | The pool's block maxima read a block's eight scores in two aligned 16-byte loads (0213's body in the grid kernel) | once a candidate-source layer a round at deep context (≥ 6,144 blocks) |

Each new kernel is a new symbol that `load` looks up only when its switch is on, so the off path is launch-identical
to `2bb052f` and runs against today's fatbin. With a switch on, the launches, grids and arguments are unchanged; only
the function differs. The three switches change how a rank computes the same bytes, so the ranks need not agree.

| Patch | Harness (`h<NNNN>/run.log`) | Local gate (`gate-<NNNN>/gate.txt`) |
|---|---|---|
| 0290 | 1,310 cases, 11,790 launches, 3 lane orders, 11.5 MB of output compared, **0 failures**; the select's list also equals an independent sort | `zig build test` 0, `zig build` 0, `check.sh` identical to base, `check_all.sh` identical (331 lines) |
| 0291 | 498 cases, 5,976 launches, 3 lane orders, 460 MB compared across four arms (including the one-element kernel), all 65,536 bf16 encodings, **0 failures** | `zig build test` 0, `zig build` 0, `check.sh` identical apart from line numbers; plain set +3 lines and 1 reworded, **all stub gaps in the edited function**; strict pass identical |
| 0292 | 890 cases, 2,670 launches, 42 MB compared, 5.2 M blocks through the new form, **0 failures** | `zig build test` 0, `zig build` 0, `check.sh` identical apart from line numbers; plain set +4 `float4` lines (the gap the base shows on the same statements elsewhere); strict pass identical |

Work removed, as the emulation counted it (not a time): 0290 runs 2.4–2.5 digit passes of 8 on Gaussian-like rows at
64,640 columns, 6.4 on heavily tied rows, and all 8 on degenerate rows. 0292 turns eight loads a block into two.
**0291 removes no work**: it trades one masked reduction for four shuffles, and rests on a single measurement of
that trade on a different kernel (latent, +0.10 µs for the masked form). It may be neutral or worse; see its report.

## Staged, not landed

| File | What | Why not landed | Evidence |
|---|---|---|---|
| `unverified/0001-glm-index-select-cut.patch` | `GLM_INDEX_CUT=1`: `glm_index_select` (Metal, GLM) stops its four-digit select at a complete bin. P1. Kernel plus six host lines (`forward.zig`, `engine.zig`); the struct has one binder | No local check exists for Metal, and the GLM family's host Zig is not compiled on Linux | `hmetal-glm/`: the old and patched Metal text, mechanically rewritten to C++, 573 cases, 3 lane orders, 51 MB compared, **0 failures**; `git apply --check` passes against `2bb052f` |

The switch is a field of the kernel's argument struct, not a second kernel (the file is 531 of the lint's 600 lines),
so with it off the kernel still does one extra shared store and one never-taken branch a digit. GLM has no banner convention; none
was added.

## Catalogue and survey

- **`PATTERNS.md`**: 8 patterns with a measured gain (P1, P1b, P2, P3, P5, P6, P7, P8), 10 anti-patterns (A1–A10), and
  the bit-exactness rules. Three entries of the brief's list did **not** survive the record: P4 (0278's gain is
  "about 1,400 times the work removed" and unexplained; 0260 and 0269 are neutral), the grid half of P7 (0264 was
  re-scored neutral), and the warp integer maximum outside whole-warp groups.
- **`SURVEY.md`**: every kernel file, 68 sites.

| Family / backend | Files | Sites | Landed | Staged | Does not transfer | Transfers, not applied |
|---|---:|---:|---:|---:|---:|---:|
| dsv41, CUDA | 6 | 25 | 3 patches (4 sites) | 0 | 3 | 5 |
| other CUDA (Nemotron's lane, shared `torch_ops`) | 30 | 11 | 0 | 0 | 1 | 6 |
| GLM, Metal | 41 | 4 | 0 | 1 | 0 | 3 |
| Nemotron, Metal | 87 | 4 | 0 | 0 | 1 | 3 |
| Flash Next, Metal (incl. its `prefill/` files) | 57 | 15 | 0 | 0 | 1 | 14 |
| `ops/`, `core/`, Qwen 3.5, Kimi, the MLX-replica `prefill/` files, Metal | 49 | 9 | 0 | 0 | 0 (P1: zero sites) | 9 |

(The remaining sites are "already in the pattern", "never launched" or "no catalogued pattern". Files: 36 CUDA sources and headers, 234 Metal.)

## Sites rejected, and why

- **`tf_ds_topk_i64_kernel`** (P1): it writes the k-th key last; a partial prefix cannot, and the consumer is a kit
  cubin with no local source.
- **`tf_ds_idx_select`'s tile exit** (the audit's 0279 idea): the newest block is pinned to +inf, so the k-th pick is
  always in the last occupied tile.
- **`tf_ds_fused_rope_qd_kernel`'s unrotated-warp guard**: the record already measured it neutral on this kernel.
- **`torch_ops/topk.cu`** (P1): the compaction's above-then-tied order depends on the exact k-th key, and the consumer
  reads positions.
- **The estimate selects** (`tf_ds_radix_kth_row` under `TOPK_FAST`; Flash Next's sample loop): a partial prefix
  changes the candidate count and which path runs.
- **`tf_topk_probs`** and every router top-k on Metal: arg-max rounds whose order is consumed, not a radix.

## Transfers, not applied (ranked in `SURVEY.md` section 9)

1. **The gate-weight widen** (`prompt.zig:1710`, P3): 11.8 MB of traffic a layer a bulk prompt chunk for a constant.
   Caching costs about 315 MB a rank when its switch is on.
2. **The 8-wide bf16/fp16 widen and round** (`pointwise.cu`, `tf_ds_f16_to_f32_kernel`, P5), same prefill stage.
3. `tf_ds_block_argmax_v` (P5), once a greedy round.
4. `sample.cu` `top_keys` (P1) and `tf_strided_copy_kernel` (P5/P7): Nemotron's CUDA lane.
5. Flash Next's `q4_idx_select` (P1): a generated text whose generator is not in the tree.

All five were left for time, not for want of a check: `check_all.sh` gives every `.cu` the same syntax gate. They
are owed work.

## Hardware gates each landed patch still owes

All three: an **nvcc build** for `sm_121` (three new symbols; a switch set against today's fatbin fails at load),
`LOCAL=0 STACK=0` and register counts, **GPU parity** of the new symbol against the old on the same inputs, and the
exact 39/39 panel. Then:

- **0290**: selector-only timing; the audit's three arms on sampled requests (full-logit fallback, shipped 0261, 0261
  with the stop) at 1 and 4 streams on the GPU clock; a histogram of the digit the stop fires at on real logits;
  racecheck on a tied row; the seeded sampled panel.
- **0291**: standalone timing of both forms at 32 / 48 / 128 / 1,024 rows, and the SASS (is the masked REDUX
  duplicated here as in the latent?). Adopt only on a standalone win.
- **0292**: standalone timing at 1 and 7 rows, 8,192 / 16,385 / 65,536 blocks, alone and in the chain (0215's
  6,144-block crossover was set with the scalar kernel); a deep-context decode suite if that wins.

## Everything unverified

- **No timing of anything.** No patch is known to be faster.
- **No device compile.** No `.cu` change has been through nvcc; no Metal text through Apple's compiler; the GLM
  host Zig in the staged patch has not been through the Zig compiler either (Linux does not build that family).
- **The emulation checks C source**, in IEEE arithmetic with the host's libm and host forms of the bf16 conversions.
  It is not codegen, not the device's math library, not a race detector.
- **The launchers' new lines never ran.** `zig build test` has no GPU: the symbol lookups, the per-launch dispatch
  in `Ops.blockMax` and the banners are compiled and type-checked only. The gates they call (`block_max8.ok`,
  `draw_cut.selectKernel`, `qd_pair_shfl.kernel`) are host-tested.
- **`check.sh` parses less than it looks.** At the base, every error it prints for `dsv41_torch.cu` is inside the
  function 0291 edits. The strict pass (`stubs-extra/extra.h`) is this sweep's addition, not a repository gate.
- **Passes on real logits** (0290) are unknown; the pass counts above are for synthetic rows.
- **Stacking with 0280–0284** is untested. This branch touches `ops.zig` (imports, the `Ops` struct, `load`, three
  launchers), `exact.zig`, `dsv41.zig`, the two `.cu` files and three new modules; it does not touch `round.zig`,
  `draft.zig`, `model.zig`, `tri_markov.zig`, `markov_tokens.zig` or `l2Prefetch`. A conflict in `ops.zig`'s struct
  or `load` list is possible.
- **Engagement on a pair**: the switches are not in `pairSwitches`' cross-rank check, by design.
- **`tools/zig/lean_check.py`** (not part of the gate) counts 47 more findings on the branch than on the base (5,487
  against 5,440): 46 multi-line comments and one `/*bsize*/`, all in the dsv41 family's existing style (that family
  carries 3,761 of the base's findings; `topk_merge.zig` alone has 21). None is past 120 columns. The staged GLM
  patch adds none (checked on the patched files).
- **Survey findings from the four reading agents** are reported as they gave them unless `SURVEY.md` says a site was
  re-read (its section 8 lists the open ones, including agent C's bf16-key claim and the files the agents skipped).
  `dsv41_torch.cu` lines 280–680 were read in outline against the audit, not line by line.
- **Metal throughout**: whether Apple's compiler already merges adjacent scalar loads, Metal's contraction default,
  and the cost of a launch are all unknown, so no Metal row has a price.

## Files (`runs/kernel-sweep-20261123/`)

`PATTERNS.md`, `SURVEY.md`, `report-0290.md`, `report-0291.md`, `report-0292.md`, `REPORT.md`;
`h0290/`, `h0291/`, `h0292/`, `hmetal-glm/` (each: `build.sh`, sources, `run.log`); `emu/` (`emu.h`, the fiber
emulation, and `extract.py`); `gate.sh`, `check_all.sh`, `stubs-extra/extra.h`; `base/` and `gate-0290…0292/` (logs);
`unverified/0001-glm-index-select-cut.patch`. The worktree's Zig caches (`.zig-cache`, `zig-out`, 1.4 GB) were
removed after the last gate; `git status` there is clean.
