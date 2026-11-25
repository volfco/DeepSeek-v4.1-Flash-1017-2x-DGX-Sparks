# Kernel survey against the pattern catalogue (sweep 2026-11-23)

Engine tree: `~/Workspace/tfref-ksweep` (branch `kernel-sweep-0290`, base `2bb052f`). Pattern ids are
`PATTERNS.md`'s. Paths below are under `zig/` of the engine tree unless they start with `runs/`. **Nothing was timed.**
"Frequency" is how often the kernel is launched, read from the host call site; "provable" means a CPU emulation of the
old and new source can compare bytes.

## 1. What was read, and by whom

| Slice | Files | Lines | Read by |
|---|---:|---:|---|
| CUDA, dsv41 (`dsv41_ops.cu`, `dsv41_torch.cu`, `dsv41_shard.cu`) + `sample.cu`, `torch_ops/{topk,argmax,argmax_f64,movement,indexing}.cu`, headers | 12 | 4,570 | this agent, line by line, except `dsv41_torch.cu` 280–680 (the 0197/0230/0250 compressor and norm bodies), read in outline with `docs/fused-kernel-rebenchmark-audit.md`'s line-level findings |
| CUDA, everything else (`gdn`, `qmm_*`, `experts*`, `prefill_attention`, `lane_gemv`, `scan_rows`, `nemotron_ops`, `probe`, `dsv41_port`, `dsv41/rdma_gather`, the other nine `torch_ops/*.cu`) and every non-dsv41 CUDA call site | 25 | 3,440 | survey agent C |
| Metal `flashnext/` (50), `decode/` (2) | 52 | 11,040 | survey agent FN |
| Metal `prefill/` (14), `ops/` (10), `core/` (4), `qwen3_5/` (17), `kimi/` (8), `nax.h` | 54 | 14,360 | survey agent PF |
| Metal `nemotron/` (77), `glm/` (35), 16 top-level files | 128 | 14,420 | survey agent NG |

146 CUDA kernels in 31 `.cu` files; 234 Metal files. The agents read one file of each generated family in full and
diffed the rest; their own "not read" lists are in section 8. Every site this document turns into a patch or a diff
was re-read by this agent first.

## 2. What counts as a local check (decided before any edit)

| File class | Local check that exists | Decision |
|---|---|---|
| `dsv41_ops.cu`, `dsv41_torch.cu`, `dsv41_shard.cu` | `cuda-syntax-stubs/check.sh` (clang, stub headers, first 20 lines of errors a file); CPU emulation of the extracted source | **land** behind a switch |
| every other `.cu` | none in `check.sh` (it names three files). The same clang + stubs command parses all 31 files with 331 stub false positives at the base; `runs/kernel-sweep-20261123/check_all.sh` runs it over every file with no error limit and prints a line-number-free set (`base/check_all.txt`) | landable on the same footing (syntax by `check_all.sh`, bytes by emulation); **no such patch was landed in this sweep** (section 7) |
| stub gaps | the stock stubs lack `__nv_bfloat162`, `float4`, `max`, `__clz`; all nine errors `check.sh` prints for `dsv41_torch.cu` sit in one function (`tf_ds_qd_pair_chunk`), which clang therefore does not really parse | `stubs-extra/extra.h` supplies the four declarations as a second, strict pass (`EXTRA=… check_all.sh`): 225 lines at the base, 0 in `dsv41_ops.cu`, 2 in `dsv41_torch.cu`. The gate for a patch is: `check.sh` unchanged apart from line numbers, the plain set diffed and every changed line explained, the strict set **identical** |
| `tools/zig/check_*_ops.py`, `tools/zig_gate.py` | need an nvcc-built library, MLX or a GPU | not usable here |
| `tools/dsv41_shard_emulate/emulate.cpp` | host emulation of `dsv41_shard.cu`'s fetch/select/union | exists; no shard kernel was changed |
| **Metal (`.metal`, `nax.h`) and the Metal families' host Zig** | **none.** `build.zig` reaches Metal only under `.macos` (`xcrun metal`); the rest is compiled at run time by `newLibraryWithSource`. Linux `zig build test` is `zig/build/cuda.zig`'s `hostTests`: no `.metal` text is embedded in any Linux module, and `families/{glm,flashnext,qwen3_5,kimi_k3}` are not compiled. The host references (`core/lanes/gpu_rule.zig`, `gpu_full.zig`) sort the row; they mirror results, not the kernels' digit loops. `tools/zig/lean_check.py` is a line-length lint | **not landed: no local check.** Diffs go to `unverified/` |

## 3. dsv41, CUDA (the served family)

Production switches (`continue.md`): `IDX_FUSED`, `FUSED_COMPRESS2_NORM`, `FUSED_ROPE_STORE`, `FUSED_ROUTE`,
`KV_FETCH_REUSE`, `CONF_PACK`, `TOPK_MERGE`, `PLACE_BOUND_ALL`, `TAPS_PACKED`, `IDX_PICK_WIDE`.

| id | File:lines (base) | Kernel | Pattern | Finding and change | Frequency | Provable | Outcome |
|---|---|---|---|---|---|---|---|
| D-1 | `kernels/cuda/dsv41_ops.cu:1670–1724` | `tf_ds_topk_select_kernel` (0261) | P1 / A6 | Eight unconditional digit passes over a rank's 64,640 draw keys. Consumer: `key >= prefix` compaction, then a rank-count sort: a set test and a re-sort. Stop at a complete bin. | both ranks, once a device-drawn (sampled) round | yes | **landed 0290** |
| D-2 | `dsv41_ops.cu:1549–1662` | `tf_ds_draw_rows_kernel` (fallback) | P1 / A6 | The same eight passes over the whole 129,280-column row on rank 0. Same consumer shape. | rank 0, once a sampled round when `TOPK_MERGE` does not engage (greedy rows return before the select) | yes | **landed 0290** |
| D-3 | `kernels/cuda/dsv41_torch.cu:979–981` | `tf_ds_fused_rope_qd_pair_kernel` (0241) | A4 | The pair query takes each 32-element scale group's maximum with a **half-warp masked** `__reduce_max_sync`, the form that cost the latent +0.10 µs. Four width-16 xor shuffles over the same integer bits. | once an indexer-query launch at 32 rows and up: prompt chunks (128–2,048 rows) and multi-stream decode windows | yes | **landed 0291** |
| D-4 | `dsv41_ops.cu:914–934` | `tf_ds_block_max_kernel` | P5 | Eight scalar fp32 loads a block; `tf_ds_block_max_t<8>` (0213) already has the two-load body, but only inside the fused kernel. A grid kernel with that body for bsize 8 and aligned rows. | decode: once a candidate-source layer a round in the small-row (≥ 6,144 blocks, ≤ 7 rows) and wide (> 16,384 blocks) arms; the unfused chain | yes | **landed 0292** |
| D-5 | `dsv41_ops.cu:60–110` | `tf_ds_topk_i64_kernel` | P1 | Eight unconditional digits. The output is the k largest keys "in no particular order", but the kernel writes the keys above the k-th first and the k-th **last**; a partial prefix cannot keep that placement. The consumer is the kit's `_topk_finish` (a cubin, no local source). | the kit's unfused `topk_select` path (`tri_index.zig:126–141`) | no: the consumer's order-blindness cannot be shown | **rejected** |
| D-6 | `dsv41_ops.cu:275–351, 353–420` | `tf_ds_radix_kth_row` / `_shared` under `tf_ds_topk_indices_fast_kernel` (`TF_DS_TOPK_FAST`) | P1 | The first select's result is an **estimate** (it sizes the candidate set and picks the path): it must stay exact. The final select feeds `key >= kth` and a column sort, so it could stop early, but it scans at most 4,096 keys in shared memory. | off in production (`IDX_FUSED` takes `topk_indices_cut`) | yes, for the final select only | transfers in part; **not applied** (unserved, shared-memory work only) |
| D-7 | `dsv41_ops.cu:147–251` | `tf_ds_topk_indices<false>` | P1 | The off arm of 0254. | n/a | n/a | already done (0254) |
| D-8 | `dsv41_ops.cu:1081–1151` | `tf_ds_idx_select` decode arm (0195/0279) | P7 / A1 | The audit's "exit after the tile that emits the k-th pick". The row's newest block is pinned to +inf (`tf_ds_block_max_t`), so it is always picked and sits in the row's last occupied tile: the exit cannot fire before the end for the newest row. | once a candidate-source layer a round | yes | **does not transfer** |
| D-9 | `dsv41_ops.cu:1213–1269, 428–438` | `tf_ds_block_argmax_v` (`tf_ds_argmax_rows_kernel`, `_pair`) | P5, A1 | One 1,024-thread CTA a row, scalar fp32 loads at a 1,024-column stride over 129,280 (or 64,640) logits. `torch_ops/argmax.cu:31–43` already reads bf16 rows 16 bytes a load with four loads in flight. Four columns a thread (`float4`) keeps the answer: the reduction is a total order on (value, column) with NaN skipped. P8 (a tiled first stage) is a redesign, and the switch that would use it (`GREEDY_ARGMAX`) measured slower. | once a greedy round | yes (P5 form) | transfers; **not applied** (once a round; ranked 4 in section 9) |
| D-10 | `dsv41_ops.cu:24–28, 1345–1358`; `torch_ops/pointwise.cu:18–28` | `tf_ds_f16_to_f32_kernel`, `tf_bf16_to_f32_kernel`, `tf_f32_to_bf16_kernel`, `tf_ds_add_bf16_kernel` | P5 | One element a thread, 2-byte loads and 4-byte stores. An 8-wide form (one 16-byte load, two 16-byte stores, the **same conversion intrinsic** per element, so NaN handling cannot move). | prefill: `prompt.zig:1709–1710` every layer of every bulk chunk (up to 2,048 × 5,120 values widened, 384 × 5,120 gate weights), plus `:776, 809, 1029, 1201, 1528, 1688` | yes | transfers; **not applied** (ranked 2 in section 9). At these sizes the kernel streams from DRAM: 0255's gain was at L2-resident sizes and may not carry |
| D-11 | `families/dsv41/prompt.zig:1710` | host: `f16ToF32(lay.gate_w → ch.gate_f)` | P3 | The constant 384 × 5,120 fp16 gate weights are re-widened to fp32 (3.9 MB read, 7.9 MB written) on every layer of every bulk chunk; the drafter widens its own once at init (`draft.zig:399–400`). Caching 40 layers costs about 315 MB a rank. | 40 a bulk prompt chunk | by construction (the same kernel's output, kept) | transfers; **not applied, for time** (an off-by-default switch would make the 315 MB a rank opt-in; the head's low-water mark is 4.9 GiB); ranked 1 in section 9 |
| D-12 | `dsv41_ops.cu:959–966` | `tf_ds_apply_pool_kernel` | (none catalogued) | A mask byte and an integer division by the runtime block size per score. | prompt arm with a pool mask | yes | no measured pattern; not applied |
| D-13 | `dsv41_torch.cu:933–961` | `tf_ds_fused_rope_qd_kernel` | P7 | The unconditional partner shuffle on unrotated warps. 0242's guard was tried on exactly this kernel and measured neutral (1.336 vs 1.338 µs): A9. | per indexer query below 32 rows | n/a | **does not transfer** (the record's own result) |
| D-14 | `dsv41_torch.cu:99–122, 233–270` | `tf_ds_compress2_kernel`, `tf_ds_rope_kernel`, `tf_ds_hc_mean4_kernel` | (none catalogued) | A 64-bit divide and modulo per element; every float op is an explicit rounding point that must stay. | the unfused arms (production takes the fused ones), `hc_mean4` once a drafter absorb | n/a | none: no catalogued gain, rounding points pinned |
| D-15 | `dsv41_torch.cu:280–680` | the 0197/0230/0250 compressor and norm kernels | P2, P6, A5 | Where rounds 0197–0258 worked. `expf` duplication closed by SASS (register), `isfinite` (0207) already reverted, PDL split landed (0250). | per ratio-2 layer a round (3 layers) | n/a | nothing new |
| D-16 | `kernels/cuda/dsv41_shard.cu:22–146` | `tf_ds_kv_fetch_kernel`, `tf_ds_select_rows_kernel`, the union kernels | P5, P8 | Already 16-byte words, sixteen lanes a row, over a 2D grid. The fetch's blocks past `kk` still have to write −1 (not a no-op grid). | fetch: 30 of 38 exchanges a round before 0259; 0259 reuses it | n/a | already in the pattern |
| D-17 | `dsv41_shard.cu:198–234` | `tf_ds_ring_pack` / `_unpack` | (none catalogued) | An unsigned divide per 16-byte word; four loads in flight on the unpack already. | every sharded exchange | yes | no measured pattern; not applied |
| D-18 | `families/dsv41/prompt.zig:1099–1104` | host: `ridx` upload + `tf_gather_rows_kernel` | A8, P5 | The same index list is uploaded and the invalid flag cleared on every layer of a chunk; the gather moves ≤ 127 rows × 1,024 B a byte a thread. | once a layer a bulk chunk | yes | neutral class (0269's); not applied |
| D-19 | `dsv41_ops.cu:567–705, 765–864` | `tf_ds_expert_prune_kernel`, `tf_ds_prune_six_kernel` | P2, P7 | 0199 and 0211 are these. | per layer | n/a | already done |
| D-20 | `dsv41_ops.cu:889–908, 1046–1075, 1202–1208, 114–128` | `cand_fast8`, `block_max_t<8>`, `copy_rows16`, `scatter_rows` | P5 | Already aligned 16-byte forms. | | n/a | already done |
| D-21 | `dsv41_ops.cu:1306–1412` | the vision tower's seven kernels | (none) | Per-element, each torch op its own rounding point. | once an image | n/a | none |
| D-22 | `dsv41_ops.cu:1435–1522` | `tf_ds_keyed_noise_kernel`, `tf_ds_spec_clean_kernel`, `tf_ds_gather_cols_kernel` | (none) | `SPEC_SAMPLE` / `T26` paths, off; fp64 `log`/`exp` per element is the contract. | off | n/a | none |
| D-23 | `dsv41_ops.cu:1851–1872`; `draft.zig` `pass`/`body`; `tri_markov.zig` | `tf_ds_conf_pack_kernel`, the Markov steps | | Off limits (the 0280–0284 agent's region). | | | not surveyed for change |
| D-24 | `kernels/cuda/dsv41/rdma_gather.cu` | stage / collect / collect_into | P5, P8 | 16-byte words over a grid; fences and waits are the protocol; no per-call host upload. | every `gatherF32` (attention and MoE gather every layer every round) | n/a | none |
| D-25 | `kernels/cuda/dsv41_port.cu`, `probe.cu` | port and probe kernels | | Opt-in / probe binary only. | never served | n/a | none |

## 4. Other CUDA (Nemotron's CUDA lane; `flashnext`, `glm`, `kimi_k3`, `qwen3_5` have no CUDA references)

| id | File:lines | Kernel | Pattern | Finding and change | Frequency | Provable | Outcome |
|---|---|---|---|---|---|---|---|
| C-1 | `kernels/cuda/sample.cu:148–209` | `top_keys` (`tf_draw`, `tf_draw_ids`) | P1 | Four unconditional key digits (then up to three id digits when the cutoff key ties). `sample_full` consumes the result only through `Mark` set tests (`mass`, `extend`, `race`, the PROB mass). `first_block` also uses `above` as a slot base, which a partial prefix would break unless both branches share one fill counter (the bitonic sort that follows makes the slot order invisible). | Nemotron only (`families/nemotron/cuda_forward.zig:43`, `cuda_mtp.zig:178`); reached at `1,024 < top_k < V`, or when no near window holds the candidates | yes | transfers with a consumer edit; **not applied** (ranked 5; needs the Nemotron family's switch convention and its own harness) |
| C-2 | `kernels/cuda/torch_ops/topk.cu:63–106`, `families/nemotron/cuda_torch_ops.zig:94–106` | `tf_topk_f32_*` | P1 | Four histogram + choose launch pairs, host-driven. The compaction writes keys above the exact k-th in column order, then the ties: a partial prefix moves that split, and the consumers (`_keyed`'s VALS and IDS, a Triton cubin) read positions. The README requires ATen order. | once a greedy draft level (≤ 15) a round | no | **does not transfer** |
| C-3 | same | same | P1b | A survey-agent finding: every input is an exact bf16 widening (`cuda_mtp.zig:182`), so the key's low 16 bits are a function of its high 16 (checked by the agent over all 65,536 encodings); the last two digit pairs (4 of 14 launches) could be replaced by an OR in `choose_digit`. | same | yes | plausible; **not applied, not re-derived by this agent** (unverified claim, section 8) |
| C-4 | `kernels/cuda/torch_ops/movement.cu:66–79` | `tf_strided_copy_kernel` | P5, P7 | A byte a thread with two 64-bit divisions a byte; both host callers pass `middle = 1`, where the quotient is 0 and the remainder is `i`. A `middle == 1` word path. | Nemotron: every attention block of every prompt chunk (≤ 2,048 rows × 8,192 B). dsv41: only at odd `kk` (`prompt.zig:1133`) | yes | transfers; **not applied** (ranked 6) |
| C-5 | `movement.cu:39–52` | `tf_gather_rows_kernel` | P5 | A byte a thread; `tf_ds_scatter_rows_kernel` is the word form of the same mover. | dsv41 production: `prompt.zig:1104` only (D-18); the other two sites are off (`FUSED_COMPRESS2_NORM`, `CONF_PACK`) | yes | transfers; not applied (about 130 KB a launch) |
| C-6 | `movement.cu:22–29`, `torch_ops/indexing.cu:7–27, 96–99` | `tf_copy_spans_kernel`, `tf_select_axis_kernel`, `tf_widen_indices_kernel`, `tf_narrow_indices_kernel` | P5 | Byte movers. | resolved by no host code | n/a | never launched |
| C-7 | `torch_ops/argmax.cu:31–43` | `tf_row_max_column` | P5 | Already 16-byte loads, four in flight. | once a stream a greedy round | n/a | already in the pattern |
| C-8 | `families/nemotron/cuda_forward.zig:281–283` | host: head GEMM + draw per prompt chunk | P7 | Only the last chunk's sample is read; the others are overwritten. | chunks − 1 times a prompt past 2,048 tokens | by construction | transfers (host only); not applied (another family's serving path, no local Nemotron gate) |
| C-9 | `kernels/cuda/qmm_prefill.cu:101–117` | `prefill_kernel` at M = 1 | P7 | Seven of eight 16-row tiles are zero rows and still take `ldmatrix`/`mma`. | once a prompt chunk | no (tensor-core ops are not emulable; a kit copy whose SASS compare would differ) | not applied |
| C-10 | `gdn.cu`, `qmm_group.cu`, `qmm_frag.cuh`, `lane_gemv.cu`, `experts*.cu`, `experts.cuh`, `prefill_attention.cu`, `scan_rows.cu`, `nemotron_ops.cu` | | | fma chains and summation trees are pinned; `experts.cu`'s `__match_any_sync` builds a stable member rank, not a count (A10 does not apply); `experts_prefill.cu` already skips dead tiles. | Nemotron per layer / per chunk; `gdn.cu` test-only | n/a | none |
| C-11 | `torch_ops/{norm_rotary,nucleus,nemotron_sampler_core,nemotron_limits,logprob,grammar_mask,cast_f64}.cu` | | | Not in the fatbin list (`zig/build/cuda.zig:13–37`). | never built | n/a | none |

## 5. Metal

No catalogue number is a Metal number, and there is no Metal launch floor on record: "applies" means the work is
present. Every row is **not landed: no local check**.

### GLM (`glm/`, `glm_*.metal`)

| id | File:lines | Kernel | Pattern | Finding and change | Frequency | Provable | Outcome |
|---|---|---|---|---|---|---|---|
| M-GLM-1 | `kernels/metal/glm_glue.metal:377–400` | `glm_index_select` | P1 | Four unconditional digit passes over the block scores. Consumers (`:406–431`): `key > prefix`, then the first `need` keys equal to it in block order, written in block order. At a complete bin `need` is the bin's population, so every key at or above the partial prefix is taken: the same blocks in the same order. Thread 0 publishes the flag; every thread breaks after the barrier. No consumer change. | per MLA layer per forward for rows past 2,048 keys (`families/glm/forward.zig:356–359`, `prompt.zig:379`) | yes (integer) | **staged: `unverified/0001-glm-index-select-cut.patch`**, with a CPU emulation of the old and new Metal text (`hmetal-glm/`) |
| M-GLM-2 | `kernels/metal/glm_sparse_nax.metal:59` | `glm_sparse_nax` | P7 | Dense rows carry −1 padding to 2,051 keys. Bounding the loop needs `fast::exp(0) == 1` and `nax.h` internals. | prompt rows in the first 2,048 positions | no | not written |
| M-GLM-3 | `families/glm/forward.zig:281–287, 238, 425–429`; `mtp.zig:37–39, 57–58` | RMS / layer-norm / scale / add / embed chains | P2 | One launch for the kernels reading one row. Re-emits float code: only source-order identity is provable, and Metal's contraction default is not established. | per MLA layer per forward | source order only | not written |
| M-GLM-4 | `families/glm/prompt.zig:370–375` | `glm_dense_indices` | P3 | Written per MLA layer; could be once a chunk. | prompt chunks in the first 2,048 positions | yes | negligible; not written |
| — | `glm/qmv_*`, `qmvp_*`, `moe_*`, `sparse_attention*`, `kda_rows*`, `hc_expand_10`, `glm_attn`, `glm_kda_prompt`, `glm_absorb_nax`, `glm_ep` | | P5 form only | Scalar byte loads the compiler may already merge (not assessable here); float accumulation fixed. | | | none |

### Nemotron (`nemotron/`, `nemotron_*.metal`)

| id | File:lines | Kernel | Pattern | Finding and change | Frequency | Provable | Outcome |
|---|---|---|---|---|---|---|---|
| M-NM-1 | `kernels/metal/nemotron_sample.metal:92–114`; `nemotron/sample.metal:124–146`; `nemotron/sample_ids.metal` | `top_keys` | P1 | The Metal original of C-1, digit for digit (same four key digits, descending bin walk, three id digits on ties, same `{key, above, need, idcut}`). `tf_sample_full` consumes it through `Mark` set tests only; `first_block` needs the one-counter fill. | per sampled draw on the radix path | yes | transfers; **not written**: two of the three copies are generated files whose generator (`gen_nemotron_kernels.py`) is not in the tree, and C-1 should land and be gated on CUDA first |
| M-NM-2 | `nemotron_sample.metal:118`, `sample_ids.metal:151` | id digits | P1b | Start at shift 8 when V ≤ 65,536 (the draft vocabulary is 49,152). | draft-head draws with ties | yes | not written (agent finding, not re-derived) |
| M-NM-3 | `nemotron_tree.metal:345–398` | `tf_topk_probs` | P1 | Not a radix: four argmax passes with exclusion; rank order and values are consumed. | per MTP level | n/a | **does not transfer** |
| M-NM-4 | `nemotron_glue.metal:153–162` + `nemotron/add_norm_xs.metal` | combine → norm | P2 | A fused sibling; float code re-emitted. | per Mamba and attention layer | source order only | not written |
| — | `nemotron/coop_*` (33), `up_relu2_*`, `group_norm_xs_*`, `xsum_*`, `router`, `route*`, `attn_partial_*`, `attn_merge`, `expert_*`, `nemotron_head`, `nemotron_round`, `nemotron_experts` | | | Generated tensor-unit projections (constants differ), already-fused heads, six-round argmax routing whose rank order is consumed. `nemotron_glue.metal:60–63`'s `isnan` is the NaN-first argmax spec, not A5. | per layer | | none |

### Flash Next (`flashnext/`, `decode/`, and the Flash Next files of `prefill/`)

| id | File:lines | Kernel | Pattern | Finding and change | Frequency | Provable | Outcome |
|---|---|---|---|---|---|---|---|
| M-FN-1 | `kernels/metal/flashnext/q4_idx_select_4092b8bd389cc7ed.metal:106–119, 155–168` | top-512 block select | P1 | Every digit pass unconditional. Output: the selected blocks' key ids in ascending block id; the k-th value is never exported; the consumer reads an ordered list, and the order is block order, which the exit keeps. `cut_bin` publishes the flag. | 12 a round + 1 per MTP step past 2,048 tokens; 12 a prompt chunk | yes (integer) | transfers; **not written**: a recorded text of the Python engine (`flashnext_checked.py`), which is not in the tree; a hand edit is overwritten by regeneration and reported as DIFFERS by `flashnext_check_dump.py` |
| M-FN-2 | same, `:69–80` | the sample loop | P1 | Its result `thr` is an estimate (it sizes C and picks the arm). | same | | **does not transfer** (the same reason as D-6) |
| M-FN-3 | `families/flashnext/replay.zig:2087–2104, 2224–2250` | scores + select launches | P7 (0216's analogue) | Skip both while no row can pass TOP blocks. | 2 × (12 + MTP steps) a round until a request passes 2,048 tokens | yes | host only; not written (launch cost on Metal unknown) |
| M-FN-4, -5, -6 | `q4_attn_parts_*:217–227`; `lane_qmm_bytes_grouped_*:9–13, 69`; `q4_idx_scores_*:202, 210` | attention parts, lane kernels, block scores | P5 | Scalar bf16 loads that are consecutive; 8- or 16-byte loads. Whether Apple's compiler already merges them is not established. | 12–98 a round | yes | not written (generated texts) |
| M-FN-7 | `q4_gdn_pipe_*`, `q4_gdn_step_*` | DeltaNet step | P5, P4, A1 | All three are already fixed in the hand-written `decode/fn_gdn.metal` behind `FZ_GDN`. | 36 a round | | no edit: measure `FZ_GDN=2` on a Mac |
| M-FN-8, -9, -10 | `qa_expert_gateup_*:285`; `replay.zig` chains; `engine.zig:650–651` | | P3, P2/P4, A8 | Per-simdgroup `simd_topk` recomputation; foldable second launches; one command buffer per first draft. | per layer / per round / per request | partly | not written |
| M-PF-1…5 | `prefill/fn_prompt.metal:5–9, 101–122`, `fn_attn.metal:44–46, 115–125`, `nax.h:23–29` | prompt gather/scatter, `frag_get`, `tf_sattn_nax`, `tf_idx_scores_nax`, `pf_hc_normed` | P4, P5, A5, P7 | Row copies of rows × 51,200 B per layer; eight scalar fragment loads; a bounds test on every load; a per-thread recomputation of a per-row sum. | per layer per prompt chunk | integer/copy rows: yes | not written |

### `ops/`, `core/`, `qwen3_5/`, `kimi/`, the MLX-replica half of `prefill/`

| id | File:lines | Kernel | Pattern | Finding | Outcome |
|---|---|---|---|---|---|
| M-CO-1 | `core/moe_route.metal:33–49` | `tf_route_logits` | P7 | Accumulates four rows and discards the padded ones (75% of the multiply-adds at one row). | not written (needs row-count variants; a dynamic bound invites A9) |
| M-CO-2 | `core/affine_mm.metal:46–53` | `am_codes` | P5 | 16 byte loads as two 8-byte loads. | not written |
| M-OP-1 | `ops/route.metal:14–26, 51–54` | `tf_route_topk` | A5 | Four `isnan` per pair in an n² rank; one ordered-integer key a value. | not written |
| M-QW-1, -2, -3 | `families/qwen3_5/state.zig:64–71`, `backend.zig:160–166`, `forward.zig:127–128` | host commit, host draw, `qmm_16_2048` ×2 | A8/P4, A5, P2 | An 18 × 1 MB copy per forward; a two-pass host draw; two launches that could be one. | not written |
| M-KM-1, -2, -3 | `kimi/mla.metal:51–155`, `norms.metal:44–59, 170–194` | `k3_mla_uv<16>`, `k3_res_norm`, `k3_argmax` | P7, P3, A1/P8 | Sixteen `simd_sum` whatever the row count; Σb² recomputed at both residual calls of every layer; one threadgroup a row over 163,840 logits. | not written |
| — | every selector in this slice (`pf_route`, `tf_route_select`, `tf_route_topk`, `k3_router`, `sort_count`, `scan`, `argmax`, `k3_argmax`, `fz_sel_meta`) | | P1 | Arg-max rounds or one counting pass; the order is consumed as an order. | **P1: zero sites** |
| — | `prefill/{attention_nax,conv,gemm_nax,gemv,glue,scan,sort,qmm_nax}.metal`, `ops/{argmax,gemv,nax_gemm,sdpa}.metal`, `qwen3_5/qmm_*`, `kimi/{dense,dense_mma,experts,kda}.metal` | | | MLX replicas with pinned arithmetic; `ops/{argmax,gemv,nax_gemm,sdpa}` are never launched; the Qwen and Kimi matmuls already load 8 and 16 bytes. | none |

## 5b. Risk, per site

"Low" means a byte mover or an integer rule with a mechanical proof; "medium" means the proof needs a consumer or a
layout to stay as it is; "high" means float code is re-emitted or a contract outside the tree is involved.

| Sites | Risk |
|---|---|
| D-1, D-2 (0290) | Low for bytes (unique keys, set test + sort). The open risk is codegen: the template's two instantiations and a new early exit in a 1,024-thread CTA (registers, `LOCAL`/`STACK`). |
| D-3 (0291) | Low for bytes (an integer maximum). **High for value**: it may be neutral or slower; the only evidence is the latent's. |
| D-4 (0292) | Low for bytes. Medium for engagement: the gate needs a 16-byte-aligned base and a stride of whole words (decode passes `bucket / ratio` and a 256-byte-aligned arena block; a bucket capped at an odd pool size, or the prompt site's live count, falls back). |
| D-5 | High: the consumer is a kit cubin. Rejected for that. |
| D-6 | Low for the final select, **high if applied to the estimate select** (it changes the path taken). |
| D-8, D-13 | None taken: the pattern does not transfer. |
| D-9 | Low for bytes (a total order on (value, column)); medium for codegen (four columns a thread changes the register set of the shared argmax every greedy round uses). |
| D-10 | Low if each element keeps the same conversion intrinsic; **high if `bits << 16` replaces `__bfloat162float`** (a signalling NaN's quiet bit is the device instruction's to decide). The grid shrinks eightfold when on. |
| D-11 | Low for bytes (the same kernel's output, kept). Medium for memory (315 MB a rank) and for cache behaviour: the GEMM then reads a cold per-layer buffer instead of scratch just written. |
| D-12, D-14, D-17 | Low; no catalogued gain to justify them. |
| D-18 | Low; neutral class. |
| C-1 | Medium: `first_block`'s slot base must change with it (one fill counter); the `Mark` consumers need nothing. |
| C-2 | High: order is consumed by a cubin. Rejected. |
| C-3 | Medium: holds only while every caller's input is an exact bf16 widening; an fp32-native caller must pass the flag off. Not re-derived here. |
| C-4, C-5 | Low (byte movers; the word path needs the same alignment test `tf_ds_scatter_rows_kernel` has). |
| C-8 | Low to medium, host only: a skipped launch whose outputs are overwritten; the dump path must keep it. |
| C-9 | Medium: a kit copy whose SASS comparison would then differ; tensor-core work cannot be emulated. |
| M-GLM-1 (staged) | Low for the rule (integer; emulated). Unknown for the compile: the break is taken from threadgroup memory after a barrier, which is the legal form, but no Metal compiler has seen it. |
| M-GLM-2 | High (`fast::exp(0) == 1` and `nax.h` internals). |
| M-GLM-3, M-NM-4, M-FN-9, M-QW-3 | High: float code re-emitted; Metal's contraction default is not established. |
| M-GLM-4 | Low; negligible work. |
| M-NM-1 | Medium (as C-1), plus generated copies that cannot be regenerated. |
| M-NM-2 | Low. |
| M-FN-1 | Low for the rule; the file is a recorded text that regeneration overwrites. |
| M-FN-3 | Low for bytes (both kernels return early on such rows); the saving is launches, whose Metal cost is unknown. |
| M-FN-4, -5, -6, M-PF-2, -3, M-CO-2 | Low for bytes; alignment is evidenced to 8 bytes only, and the compiler may already merge the loads. |
| M-FN-7 | None: adopt or measure the existing `FZ_GDN` path. |
| M-FN-8 | Medium: adds a launch a layer (A7). |
| M-FN-10, M-QW-1 | Medium, host only: buffer identity across forwards (`keep`, `qualify.zig`). |
| M-PF-1 | High: a P4 change with no measured gain behind it, 419 MB of row copies a layer at 8,192 rows. |
| M-PF-4, M-PF-5, M-OP-1, M-QW-2, M-KM-1 | Low. |
| M-CO-1 | Low; a dynamic bound would invite A9, so it needs row-count variants. |
| M-KM-2 | Medium to high: a new plane and the cluster stage handover. |
| M-KM-3 | Low: keep k3's comparator (a NaN never wins, all −inf gives 0). |

## 6. Generators (where an edit has to go)

- **Flash Next** `flashnext/*.metal` (49) and `sources_gen.zig`: recorded from the Python engine by
  `tools/zig/flashnext_dump.py` and written verbatim by `flashnext_checked.py`. The Python engine is not in this tree.
  `decode/fn_lane.metal` and `fn_gdn.metal` are hand-written.
- **Nemotron** `nemotron/*.metal`, `kernels.zig`, and the top-level `nemotron_expert_{up,down}.metal`: headed as output
  of `tools/zig/gen_nemotron_kernels.py` / `gen_nemotron_metal.py`. **Neither script exists in the tree or its
  history.** The other top-level Nemotron files are hand-written.
- **GLM** `glm/*`: `tools/zig/gen_glm_kernels.py`, whose inputs are absent. `glm_*.metal` at the top level are
  hand-written (this is why M-GLM-1 is the one Metal diff written).
- **Qwen 3.5** (all but `layout.metal`): "Generated by tools/zig/gen_qwen35_kernels.py; do not edit"; the script is absent.
- **Prefill MLX replicas**: bodies hand-written, entry points and `kernels.json` from `check_prefill_ops.py --write`
  (needs MLX on a Mac). **Kimi**, `ops/`, `core/`: hand-written.
- Load-time text edits a hand edit must survive: `families/flashnext/replay.zig:2478–2481` (a string that must occur
  exactly once), `split.zig:27–47`, `families/nemotron/kernels.zig:47–58, 192–207`, `families/glm/kernels.zig:207–248`.

## 7. Why only dsv41 files were landed

The three landed patches are all in `check.sh`'s three files. The `torch_ops` and `sample.cu` sites (C-1, C-4, C-5,
D-10's `pointwise.cu` half) are landable on the footing section 2 describes, and were left for time, not for lack of a
check. They are ranked in section 9.

## 8. Not established (carried from the survey agents, not re-derived)

- C-3's claim that a bf16-widened key's low 16 bits follow from its high 16 (agent C checked it; this agent did not).
- M-NM-2, M-FN-3…10, M-PF-1…5, M-CO-1/2, M-OP-1, M-QW-1…3, M-KM-1…3: sites as the agents reported them, re-read only
  where this document says so (M-GLM-1; the sampler's `top_keys` against `sample.cu`).
- Whether Apple's compiler already merges adjacent scalar loads (decides every Metal P5 row), Metal's contraction
  default outside the `#pragma clang fp contract(off)` helpers, and buffer alignment past 8 bytes.
- Not read by the agents: `families/flashnext/replay.zig` 380–880 and most of 2330–3388, `tp.zig`, `marks.zig`,
  `snapshot.zig`; the bodies of `prefill/{attention_nax,gemm_nax,gemv,conv,qmm_nax}.metal`, `ops/qmv.metal`,
  `ops/sdpa.metal`; `glm/gemv_t_*` and `moe_down`'s first branch.
- How often Nemotron's radix path runs (it needs no near window to hold, or `top_k` above 1,024), and whether
  production drafter passes reach `draft.zig:815`'s `toF32`.

## 9. Where a hardware gate is most worth running

Per-layer or per-round work on the served stack first; the record says microseconds once a round are invisible.

1. **D-11** (not applied): the gate-weight widen, 11.8 MB of traffic a layer a bulk chunk, the largest byte count found
   on the production stack. Owed work: a per-layer fp32 cache behind a switch (315 MB a rank when on).
2. **D-10** (not applied): the 8-wide widen and round on the same prefill stage (prompt routing is 0.90% of the head's
   chat-prime window, the ceiling for both).
3. **0290** (landed): once a sampled round, both ranks. The audit already asks for the three-arm timing (full-logit
   fallback, shipped 0261, 0261 with the exit); `runs/k280/topk-run.sh` is staged for the first two.
4. **0292** (landed) and **D-9**: once a candidate-source layer a round at deep context (`a_idx` is 0.8–2.8 ms of a
   round and grows with depth); once a greedy round.
5. **0291** (landed): per indexer-query launch at ≥ 32 rows, so prefill chunks and multi-stream decode; one-stream
   decode (about 6 rows) never takes the pair form. Standalone first: the pair kernel has no timing of either form.
6. **C-1, C-4** (not applied): Nemotron's CUDA lane only.
7. **M-GLM-1** (staged) and **M-FN-1**: a Mac, and the first Metal timing of any catalogue pattern.
