# The kernel kit: what is in it, and where its source is

Audit of 2026-11-19 (verification of `continue.md`'s "Do not touch: the kit cubins (experts GEMMs, attention core, head,
Engram lookups) and the all-gathers — no source, no safe replication"). That summary was **stale**: every piece of the
kit's content has public source. The binding constraint is not the absence of source; it is that a rebuilt kit is a
different binary, and only the served pair's exact gate can say whether the bits match (`decisions.md`).

## What the served build loads at run time

`TF_DS_KIT` (`TENSORFOLD_CUDA_KERNELS=<kit>/aot`) is fetched by `scripts/prepare.sh` at `KIT_REF` (`abe07c7`) from
`https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10.git`, a public repo, checked against its
`kit/MANIFEST` (~230 cubins, sha256sum format) on every prepare. The image (`zig/Dockerfile`, `docs/zig-engine.md`)
contains only `tensorfold-native`; the kit is mounted from the HF cache over NFS. Layout:

| Part | What it is | Source of truth |
| --- | --- | --- |
| `kit/aot/cubins/*.cubin` | the Triton AOT kernel set, packed from recorded launches of the Python engine (`tools/dsv41-zig/rec`, `aot_pack.py`) | **public**: `bertholomus/TensorFold` @ `deepseek-v41-tp2`, `src/tensorfold/families/deepseek_v41/cuda/kernels.py` (`_rowmm_parts`, `_route_parts`, `_sparse_attn_part`, `_sparse_attn_merge*`, `_hc_*`, `_engram_gate`, `_collapse_norm`, `_collapse`, `_topk_*`, `_index_*`, `_fp4_*`, `_kv_norm_rope`, ...) |
| EXL3 extension cubins (cuobjdump of `tensorfold_exl3_experts_v19`, `tensorfold_exl3_linear_v7`) | the experts GEMMs and EXL3 linears (`rot_in`, `gateup_epilogue`, `grouped_cp`, `decode_prep`-adjacent) | **public**: same branch, `src/tensorfold/cuda/exl3/` (`experts.cu`, `linear.cu`, `experts_grouped.cuh`, ...) |
| `kit/rope-*.f32` | RoPE tables; made by `make_rope.py` from the checkpoint's config | config-derived, made in the image |
| `kit/engram.json` | Engram tables' fixture/revision hashes (used in the rank fingerprint) | recorded fixture, made by `zrec_fixtures.py` (CPU); the primes themselves are derived at run time in `engram.zig` |
| `kit/vision/torch_fmha_sm120.cubin` | PyTorch's mem-efficient attention, sm_120, from `libtorch_cuda.so` | binary from NVIDIA's container; PyTorch is public (BSD-3/CUTLASS-BSD) but this is a build-specific binary |
| `kit/vision/gate_bias_vl.safetensors` | the gates' VL bias | original-checkpoint tensors (MIT) |

`tools/dsv41-zig/build_kit.sh` rebuilds the kit from source; it states plainly: **"An extension rebuilt from source
is not byte-identical (nvcc, build to build)"** — the published kit stays the record; a rebuild must pass
`tools/dsv41-zig/kit/verify_kit.sh` and is a *different artifact* requiring the exact gate.

## The individually "untouchable" items, mapped

| Item | What it actually is | Source | Replication blocker |
| --- | --- | --- | --- |
| experts GEMMs | EXL3 extension cubins (runtime `dlopen` via `exl3_linear.zig` / `exl3_experts.zig`; `fingerprint.zig` tracks them as `kit_cuda = 7`) | public (above) + our local port `zig/kernels/cuda/exl3_experts.cu` (v1-era lineage, native Zig build) | rebuilt ≠ byte-identical; decode-bit routing confirmed only in-place |
| attention core | Triton `_sparse_attn_part`/`_sparse_attn_merge*` in the recorded AOT set (`tri_attn.sparseAttn` → `tri_basic`-style launches in the Zig line) | public (above) | same build-to-build issue; `tfref-next` has no CUDA port of it |
| head | Triton `_collapse_norm`/`_collapse` + EXL3 linear cubins for the vocab projection (`prompt.grouped`) | public (above) | same |
| Engram lookups | GPU: `_engram_gate` (Triton, AOT set) + `engram_wkv` projection (EXL3 cubins) + all-gather of partials; host: row preads on NVMe (`engram_io.zig`, repo-owned, unchanged) | public (above); host side is repo-owned | same |
| all-gathers | (i) decode-size gathers: the RDMA ring — **repo-owned source**, ported from the GLM branch, `tfref-next: rdma.zig`, `rdma_gather.cu`, `ring2d.zig`, `link.zig`; production sets `TF_DS_RDMA_CPUS=15`. (ii) prompt-size / fallback: `ncclAllGather` via runtime-linked libnccl | (i) ours; (ii) NCCL is upstream NVIDIA's binary (source public, not built here) | NCCL is a linked binary, not buildable/byte-patchable from the repo; the ring is delicate but not untouchable by "no source" |

## `../TensorFold` (ashhart/TensorFold) — what it is and is not

A **parallel native implementation** of the same model family (its own Zig/CUDA/Metal — `zig/src/families/deepseek_v41/`,
`zig/kernels/cuda/deepseek_v41/*.cu` (~4.8 K lines), `zig/kernels/metal/deepseek_v41/*.metal`; ~8 K lines of CUDA/cuh;
last touched 2026-11-13). It is upstream of the native engine lineage (TensorFold 1.0.2), not the served build's source,
and predates/parallels the `tfref-next` port. It is genuine source and a feasible reference for re-deriving matching
kernels, but it is not drop-in, and nothing in it proves bit-identity with the served kit on a GB10 either. It does **not**
contain the served kit's sources: no `src/tensorfold/` (the Python family), no `_route_parts`/`_rowmm_parts`, no recorded
launches.

## What this changes about the "do not touch" rule

The rule stands, but for the correct reason: **a rebuilt or replaced kit is a new artifact whose bits the served pair's
exact gate must adjudicate — not because the source is unavailable.** The 0193/0194 fused-route work already replicates
`_rowmm_parts`/`_route_parts` bit-for-bit on CPU against the recorded arithmetic and is unproven on GB10. Any future
kit-replacement candidate inherits the same gate: build → 39-case exact suite on the Sparks → adopt or drop; there is
no "close enough" arm. NCCL and the published kit binary stay as-is; the RDMA ring is ours to reason about (carefully).

References: this file (audit), `docs/zig-engine.md` (image composition, KIT_REF), `tfref-next`'s
`tools/dsv41-zig/build_kit.sh` + `kit/verify_kit.sh`, `KIT-LICENSES.md` (in the kit repo; ExLlamaV3-MIT, DeepSeek-MIT, PyTorch-BSD, Triton-MIT, CUTLASS-BSD, Apache-2.0),
`continue.md`, `decisions.md`.
