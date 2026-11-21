# Runtime LoRA for the DSpark drafter (TF_DS_LORA)

`zig/patches/0118` (`drafter: TF_DS_LORA ...`) adds a runtime PEFT adapter on the DSpark blocks' attention
projections. It replaces the offline route (merge the adapter into the checkpoint and re-quantize/splice the 196 GiB
pack) with an additive term applied per draft pass. Operator direction: runtime adapter, `todo.md` "Runtime LoRA for
the DSpark drafter".

## What it does

`TF_DS_LORA=<folder>` loads ONE PEFT adapter folder (`adapter_config.json` + `adapter_model.safetensors`) at model
load, for the DSpark stages (`mtp.0`-`mtp.<draft_layers-1>`, 3 on this checkpoint), and applies

    y += (alpha / r) * B * (A * x)

to the projection's output `y` after the EXL3 glinear has written it, where `x` is the projection's own input:

| target module | x (input) | y (output) | notes |
|---|---|---|---|
| `wq_a` | the mixed block rows (`dr.x`) | `dr.qa` bf16 | replicated on both ranks |
| `wq_b` | the normed q rows (`dr.qr`, pre-rotation) | `dr.q` bf16 | the term is roped with the same tables/pos as wq_b's own glinear epilogue |
| `wkv` | the mixed block rows (`dr.x`) | `dr.ykv` bf16 | replicated |
| `wo_b` | wo_a's output, this rank's groups (`dr.u`) | `dr.pa` fp32 | accumulates into the fp32 attention partial |

The EXL3 trellis kernels are untouched: the term is two cuBLAS GEMMs into a scratch, plus the family's
`tf_ds_add_bf16` kernel (`rope_heads` for `wq_b`). `wo_a`'s grouped slices and the MoE experts are NOT targeted (a
`target_modules` entry naming them fails the load with `UnsupportedTargetModule`).

Tensor names must end in `.mtp.<j>.attn.<module>.lora_A.weight` / `.lora_B.weight` (any prefix, e.g.
`base_model.model.`). A and B are validated against the plan: dtype BF16 only (PEFT's own save format), A
`[r, k_full]`, B `[n_full, r]`, r in 1..128, alpha >= 0. `use_rslora` and `use_dora` are refused.

## Switch semantics (off by default)

- Unset or empty: no adapter, no allocation, no launch. The pass is bit-identical to base.
- A zero adapter (`r` 0, an empty `target_modules`, or `alpha` 0): also nothing allocated, nothing launched.
- A module the adapter does not target: `lora.apply` returns before any launch (a no-op).
- A missing folder, a missing `adapter_config.json` / `adapter_model.safetensors`, a bad rank/alpha, an unknown target
  module, a wrong shape or a non-BF16 dtype **fails the model open** (`std.log.err` names the folder and the reason),
  not the first request.
- Every rank reads the same folder and applies its own slice, so both ranks must see the same adapter.

## TP=2 shard convention

The adapter is saved for the unsharded model. Each rank uploads only its own slice, by the same `plan.Split` its
weights were loaded with (the DSpark blocks are TP2 inside the pair, `plan.wants`' `tp2`; `pair` is null):

- `wq_b`: B's rows = the rank's output columns, `s.wqB(c)` (rank 0 heads 0-31, rank 1 heads 32-63 on the 2-node pair).
  A is replicated (its input, `q_lora`, is not split).
- `wo_b`: A's columns = the rank's groups' input rows, `s.range(s.groups(c) * o_lora)`. B's rows are the full hidden
  width (each rank contributes its own groups' partial to the hc post).
- `wq_a`, `wkv`: replicated projections, so the whole adapter goes to both ranks.

The host test "shard: the ranks' slices tile the full matrix" pins this: rank 0 + rank 1 are disjoint and complete,
the replicated ones are whole. Nothing else (no new plan format) is involved.

## Capture path

Every launch the epilogue issues is a stream-ordered launch on the engine's stream inside `Drafter.body`, which is
exactly what `PassGraphs.run` captures (`cuda.graph.beginCapture` around `body`). The two GEMMs are cuBLAS, already
captured inside `body` for the confidence head; `rope_heads` is the kit's Triton cubin already in the AOT set; the add
is the family's `dsv41_ops` fatbin kernel. **No new AOT/fill-set entry is needed and replay does not drop the
adapter.** The graphs are keyed by `(streams, steps, keyed)` only, which is sound because the adapter is fixed for the
process lifetime (hot reload would need the key to carry it - deferred).

## Byte-cost warning

Draft rounds are priced by bytes read, not by launches: each targeted module adds a `[rows, n]` bf16 scratch write
plus a second GEMM over `n` columns (wq_b's `n` is 16384 on a rank), and `wo_b` accumulates into a 2 MiB fp32
buffer. Two extra GEMMs per module per stage can cost more than the acceptance the adapter buys. Measure
`acceptance_rate` and `tokens_per_round` per class before and after (to-benchmark.md), and prefer the narrowest
`target_modules` set that moves acceptance.

## LIVE GATES (on the pair; none of these ran here - no GPU on the dev host)

1. `prepare.sh` / `just restart-head` builds the image with patch 0118 (`just engine-smoke` already proves the
   patch's host tests).
2. Put the adapter under the `/nfs1` checkpoint volume (visible in the container as `/hf/...`), then
   `just engine-restart-lora lora=/hf/adapters/<name>`; both ranks must log the load and neither may error.
3. **Graph replay keeps the adapter** (first gate): with the adapter on, drafts must differ from the plain run
   (acceptance_rate moves) - if the numbers are identical to the switch-off arm, replay dropped the epilogue.
4. Hardware smoke: `just restart-head` + one `/v1/chat/completions`, then `tools/exact.py` / `tools/validate.py`:
   **drafted replies must equal plain** - a drafter adapter may never change reply bits. Any difference is a bug in
   the epilogue (most likely wq_b's rope sign, the one piece not verifiable on the host).
5. Zero-adapter bit-identity: `TF_DS_LORA=` (unset) and `TF_DS_LORA=/hf/adapters/<zero>` (r=0) must both reproduce
   the base arm's `exact.py` record byte for byte.
6. `acceptance_rate` / `tokens_per_round` per class for the adapter arm next to the base arm in `to-benchmark.md`.

## Deferred (named, not built)

- Multiple adapters (one folder only), and per-stage/per-class adapter selection.
- Hot reload (the captured graphs would need the adapter in their key).
- Per-class acceptance tuning of the adapter's scale.
- Quantized adapter storage (BF16 only today), and F32 adapters (re-save in BF16).
- `wo_a` (grouped slices) and the MoE experts as targets; `use_rslora`/`use_dora`.
