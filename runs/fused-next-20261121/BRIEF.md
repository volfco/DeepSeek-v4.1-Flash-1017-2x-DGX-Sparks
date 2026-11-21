# Brief: fused-kernel work on the DeepSeek-V4.1 TP2 engine (read all of it before acting)

You are implementing and optimizing fused CUDA kernels for a production inference engine. The goal is to fuse as much
as possible and to make every kernel optimally implemented for the hardware: keep going on an operation until you
can show, with measurements and instruction-level evidence, that there is no better way to do it on this GPU. Every
claim you make must come from a measurement you ran or SASS you read, never from intuition.

## Where things are

- **Engine source (edit here):** `~/Workspace/tfref-fused-next`, git branch `fused-next`. It is production's source
  (`prune-six`: reworked 0193/0194, 0195/0197/0198, 0199) plus follow-ups 0200-0212 except 0206-0208. Kernels:
  `zig/kernels/cuda/dsv41_ops.cu` (0195 indexer `tf_ds_idx_fused_kernel`, 0199 prune) and
  `zig/kernels/cuda/dsv41_torch.cu` (0197 `tf_ds_compress2_norm_kernel`, 0198 rope-store kernels). Host side:
  `zig/src/families/dsv41/` (`idx_fused.zig`, `compress2_norm.zig`, `rope_store.zig`, `exact.zig`, `round.zig`,
  `prompt.zig`, `ops.zig`).
- **Recipe repo (evidence goes here):** `~/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`. Read `todo.md` (the
  open items), `runs/fused-kernel-audit-20261120/` and `runs/fused-followup-20261121/` (the existing GB10 harnesses:
  `build.sh`, `probe.py`, `decode.py`, `sass.sh`, `sass.py`, and their RESULTS.md). Write your evidence under
  `runs/fused-next-20261121/<work package>/`.
- **The GPU:** an NVIDIA GB10 (sm_121, unified LPDDR5X memory) on the head node, `ssh -o BatchMode=yes
  colum@100.95.49.21`. CUDA 13.0 at `/usr/local/cuda/bin` (`nvcc`, `cuobjdump`, `nvdisasm`, `compute-sanitizer`),
  Python 3 with NumPy. The kit's reference cubins: `/nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins`. Previous
  harness work: `~/tf-bench/followup/`. Use `~/tf-bench/fused-next/` for yours. There is no GPU and no nvcc on the
  workstation; build and run every CUDA harness on the head.
- **Engine build flags for these kernels:** `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false`. Never use fast math.
- **Host tests:** `cd ~/Workspace/tfref-fused-next && ~/.local/zig-0.17.0/zig build test --summary all` (on the
  workstation; ~5 min cold). Baseline on this branch: 794/796, the two failures pre-existing and not yours to fix
  (`prompt_cache.test.0178 (TF_DS_PROMPT_ENDSTATE)` and `round.test.TF_DS_EXPERT_HOT_CACHE: hotPick`). Your commits
  must keep every other test passing. A `lane_host` prefill-interleave timing test is known to be flaky under load:
  rerun before believing it.

## Hard rules (breaking one ends the work)

1. **Production is serving on the head and the worker, and it is also the model you are running on.** Never touch
   docker, containers, `start.sh`, `stop.sh`, `just`, `.env`, the head's checkout `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`,
   or anything under `/nfs1`, `/hf` or `~/.cache/dsv41-*`. Never ssh to the worker. Never use `sudo`. Read-only use of
   the kit cubins is fine.
2. **Memory:** the head has only ~5 GiB available while serving. Every harness checks `MemAvailable` first and refuses
   below 3 GiB; keep each harness process under 1 GiB of device allocations.
3. **GPU timing discipline:** wrap every timed run in `flock ~/tf-bench/gpu.lock`. Before timing, check
   `curl -s 127.0.0.1:8000/health` on the head shows `"connections": 0`; record it. If production is busy, wait and
   retry; never report timings taken while it served a request (rerun them).
4. **Bit-exactness is non-negotiable.** A change is acceptable only if every live output byte is identical to the path
   it replaces, across random, tied, flat, NaN, +-inf, signed-zero, partial-block and boundary inputs at every served
   shape. For any output a kit kernel consumes (for example `_score_keys`' keys read by `topk_select_pruned`), the
   reference is the kit producer's bytes, not an engine helper. That gap is exactly how 0206 passed standalone
   (it matched `idx_fused.zig:scoreKey`) and failed on the pair (the kit's encoding differs in the top bit).
5. **Git:** commit only in `~/Workspace/tfref-fused-next` on `fused-next`, one item per commit, message
   `NNNN dsv41: <what changed> (todo item)`, numbering from the number your work package gives. Do not push. Do not
   commit in the recipe repo (write files there; the supervisor commits). Never print or copy API keys or `.env`
   contents.
6. **No new default-on behaviour.** Every change rides an existing fused switch (`TF_DS_IDX_FUSED`,
   `TF_DS_FUSED_COMPRESS2_NORM`, `TF_DS_FUSED_ROPE_STORE`, `TF_DS_FUSED_ROUTE`) or a new opt-in one; unfused fallbacks
   stay.

## How to measure (the standard every item is held to)

- **Microbenchmarks:** CUDA graphs (the engine serves from graphs), the old and new kernel interleaved in the same
  process (A B A B ...), at least 20 repetitions of 100+ graph replays each, warmed, report median and IQR in µs.
  Always also time an empty-kernel graph node at the same launch shape: that is the launch floor, and a kernel within
  ~0.2 µs of it cannot get faster by changing its body.
- **Served shapes:** decode R = 1, 7, 32, 48 rows; prompt R = 128, 1024 (and 2048 where the path takes it); widths 128
  and 512; 0195 at 4,096 and 8,192 blocks and its 16K-65K column window; 100K+ where an item targets it.
- **Instruction level, for every kernel you touch:** SASS from `cuobjdump -sass`; an opcode histogram (LDG/STG widths,
  LDS/STS, SHFL, BAR, ATOMS, MUFU, branches), registers and spills from `cuobjdump -res-usage` / `-Xptxas -v`, theoretical
  occupancy, and bytes moved per launch versus a measured copy-kernel bandwidth on this GPU (memory roofline). State
  which bound the kernel is at (launch floor, latency, bandwidth, issue) and why the new form is closer to it.
- **Hardware counters:** you cannot run `ncu` (it needs root). If a decision truly needs counters, write the exact
  `ncu` command line and what it should decide into `NCU_REQUESTS.md` in your evidence folder and continue with SASS
  and timing; the supervisor runs it later.
- **Keep or drop:** keep a change only if parity is exact and it is faster at the served shapes that matter (decode
  for decode paths), outside the IQR. Otherwise revert it and record "closed, no change" with the numbers. A slower
  served shape is a drop even if another shape wins, unless the kernel dispatches per shape.

## What to hand back

For each item: a commit (or "closed, no change"), and in its evidence folder a `RESULTS.md` with the parity cases and
result, the timing table (old / new / launch floor, per shape), the SASS and resource diff, the bound it now sits at,
and the exact commands to reproduce. End your session with a short summary listing each item, its verdict, its commit,
and anything left open.
