# report-0277 — TF_DS_DRAFT_RING_BATCH: NOT BUILT (premise failed: the brief's own stop condition; no commit)

**Status: premise failed.** The brief's premise — that `kvNormRope`'s kernel is ours to reuse — does not hold, and
the item is not worth building at its parked value (~0.02% of a round). Stopped before any code. Worktree `~/Workspace/tfref-k0277` (branch `kernel-0277`) stays at base
`10c9866`, clean. **No commit, no switch, no harness** (h0277/ untouched). The brief's stop condition fired:
`kvNormRope` launches a **kit cubin**, and the brief says "if it is a kit cubin, stop and write that as the report:
we never replace kit kernels".

## The premise check (BRIEF-0277's own stop condition)

**Which kernel does `kvNormRope` launch, and whose is it?** `_kv_norm_rope`, a Triton **kit** cubin. The full chain,
verified in the tree at `10c9866`:

| Step | Location | What it does |
| --- | --- | --- |
| call sites | `zig/src/families/dsv41/draft.zig:509` (`absorb`), `:535` (`absorbMany`) | `tri_norm.kvNormRope(e.t, ...)` per DSpark stage |
| wrapper | `zig/src/families/dsv41/tri_norm.zig:68` (`kvNormRope`) | `t.run("_kv_norm_rope", .{ u(rows), 1, 1 }, ...)` |
| dispatch | `zig/src/families/dsv41/tri.zig:22` (`Tri.run`) | `t.set.?.run(stream, name, grid, args, consts)` — the captured set only; no repo-kernel fallback anywhere |
| set | `zig/src/cuda/aot.zig:1` (`Set`), loaded at `zig/src/families/dsv41/model.zig:428` | "the recorded Triton set (aot.json and its cubins)" from `kit_dir/aot` = `TF_DS_KIT`/`TENSORFOLD_CUDA_KERNELS` |
| provenance | `docs/kernel-kit-provenance.md` (DeepSeek repo root) | `kit/aot/cubins/*.cubin` = "the Triton AOT kernel set", source of truth `bertholomus/TensorFold` @ `deepseek-v41-tp2`, `kernels.py` — the enumeration of kit kernels names **`_kv_norm_rope`** explicitly |

There is **no repo-owned implementation** of this kernel to reuse: `grep -rn "kv_norm\|norm_rope" zig/kernels/cuda`
finds nothing; the name `_kv_norm_rope` occurs only in `tri_norm.zig`. The recorded launch fixtures
(`zig/src/families/dsv41/fixtures/launches.json`) carry shapes/consts only (`QUANT=1, D=512, RD=64`, num_warps 4,
eps 1e-20, grid `rows×1×1`, phases `absorb` and `draft_batch_capture:s1..s4`) — **no tensor bytes** to pin values
against (0263's finding, re-verified). No kit cubin, SASS, `cuobjdump` or `nvdisasm` exists on this machine
(filesystem search for `aot.json`/`cubins/` outside `runs/` comes up empty).

So the batched kernel this brief sketches would replace the launches of a kit cubin with a repo CUDA kernel
re-deriving its arithmetic — exactly what the brief rules out ("we never replace kit kernels"; common brief: "Never
edit kit cubins or anything under the kit"). Per the brief: stop, report, no commit.

## Independent second blocker: "bit-identical, same float order" is not achievable from available material

Even setting the kit rule aside, the brief demands the ring bytes be **bit-identical with the same float order**,
"reuse the exact arithmetic sequence of the existing kernel". The existing kernel is compiled Triton. Its float
order is codegen, not source:

- `runs/kernel-round-20261122/kit-kv-norm-rope.md` (the spec fetched after 0263 stopped) pins the **algorithm**
  (split even/odd RMSNorm sums, `re = ne*cs - no*sn` etc., per-32-element FP8 E4M3 quant-dequant with a pow2 scale)
  but its own closing note says the split reduction is a **bit-exactness risk**: `tl.sum`'s cross-warp tree order
  (num_warps 4) is unspecified.
- The 0197 lesson (continue.md register): a kit RMSNorm replica needed the exact SASS sequence
  (`MUFU.SQRT` then `MUFU.RCP`) to be bit-exact; `sqrtf` + `__fdiv_rn` left 9 bf16 bytes differing. For
  `_kv_norm_rope` there is no local cubin to read that sequence from, so any replica would ship an unverified
  float order and could corrupt every drafter ring the moment the switch is on (0263's assessment, still true).

## `dr.ao`: no reader on either path (the brief asked to prove or refute)

**Proven unused.** Whole-tree grep: `dr.ao` appears exactly four times in `draft.zig` — the field
(`:325`, its comment: "the keys' normed rows (kv_norm_rope's out, not read)"), the allocation (`:479`), and the two
`kvNormRope` OUT arguments (`:509`, `:535`). No read anywhere in `zig/src`. The batched kernel could safely drop
the 21 KiB-a-rank scratch writes (7 rows × 512 × 2 B × 3 stages).

## Launch counts at the served shape (for the record)

Per decode round the served path runs `Drafter.absorbMany` once (`model.zig:1787`, drafting windows' rows as items;
5–6 rows a stream, up to 7, 1–4 streams) and each call loops the 3 DSpark stages (`draft.zig:534`):

- **Before:** 3 `_kv_norm_rope` launches per round (one per stage, `rows×1×1`, D 512, RD 64). Plus 3 per prompt
  chunk's `absorb` (`draft.zig:508`, up to 128 rows).
- **After (if built):** 1 launch per round / per chunk — a saving of 2 launches per round.
- **Value of the saving:** `kit-kv-norm-rope.md` prices the whole item at ~10 µs of a 4.5–7 ms drafter pass,
  ≈ **0.02% of a round** — parked at that priority on 2026-11-22, and nothing since has raised it.

The recorded captures confirm `absorbMany` runs under graph capture (`phase: draft_batch_capture:sN`), so the
graph-safety constraint in the brief is real but was never reached.

## History considered

- **0263 (`runs/kernel-round-20261122/report-0263.md`)** attempted this same todo item and stopped: the kit's FP8
  step was unknown then. The spec has since been fetched (`kit-kv-norm-rope.md`), but 0263's other findings stand
  (no local cubin/SASS/tensor bytes; the harness cannot adjudicate an unverifiable float order), and 0263's stop
  report explicitly listed "the served `_kv_norm_rope` source or its SASS" as an unlock that was never supplied.
- **Tension noted for the brief owner:** the tree does contain repo kernels that replicate kit-chain arithmetic
  (`tf_ds_triton_rmsnorm_row`, `tf_ds_key_store_row` in `dsv41_torch.cu`; switches `TF_DS_FUSED_ROPE_STORE=1`,
  `TF_DS_FUSED_COMPRESS2_NORM=1` in production), and the 0193/0194 fused-route work replicated
  `_rowmm_parts`/`_route_parts` on CPU against recorded arithmetic (`docs/kernel-kit-provenance.md`). If the owner
  wants this item treated as a qualified replica (same gate: CPU pinning + the 39-case exact suite on a Spark),
  the brief must say so explicitly; as written ("we never replace kit kernels"), it forbids it. That is a decision
  for the brief owner, not for this run — this run followed the brief.

## Decision log

- **Decision:** stop, no commit (brief-0277's "if it is a kit cubin, stop" + the unsatisfiable bit-identity clause).
  **Alternatives considered:** (a) build the batched replica from `kit-kv-norm-rope.md` following the
  `rope_store`/`compress2_norm` precedent — rejected: contradicts the brief's explicit rule and would ship an
  unverified float order at 0.02% of a-round upside; (b) batch the launches while keeping the kit kernel
  arithmetic per stage (e.g. three kit launches into one stream-ordered group) — no such CUDA-side mechanism
  exists in `Tri`, and it would not remove launches; (c) drop only the `dr.ao` write — the write is inside the
  kit cubin; the only lever is the `out` pointer, and passing a dummy is behaviourally identical to today
  (no reader) but saves nothing measurable.

## Gate results (baseline only; no code changed)

- `zig build test` on the untouched base: **pass** (exit 0). The first invocation hit the known timing flake under
  parallel agents' load and was rerun per the common brief's flake rule.
- `zig build`: **pass** (exit 0).
- `cuda-syntax-stubs/check.sh zig/kernels/cuda`: the known **9 stub false positives** on the base
  (`float4`, `__clz`, `__nv_bfloat162`, `tf_ds_triton_rmsnorm_row` overload in stubs). Unchanged set — trivially,
  since no file changed.
- CPU-emulation harness: not applicable (no new kernel).

## What would unblock this item (for the owner's queue, not this round)

1. The served `_kv_norm_rope` SASS (or a kit cubin + `nvdisasm` on this machine) to pin the split-reduction float
   order, **and** a brief revision granting replica status (the 39-case exact gate would still be mandatory); or
2. Real kit bytes (Y, W, POS, COS/SIN, resulting RING rows) to validate a replica against on CPU; or
3. Accept the item stays parked at ~0.02% of a round.

## Files touched

None. `git status` clean at `10c9866`. No commit hash — none was made.
