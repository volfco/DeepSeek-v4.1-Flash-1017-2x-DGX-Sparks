// Six narrow read-only reviewers for the fused-next 0213-0251 round.
// Each writes its own findings file early and refines it; hard 45-min cap.

const { recipeRepo, engineRepo, reviewDir } = args;

const common = `
You are one of six parallel reviewers auditing a CUDA kernel optimization series on a production
DeepSeek-V4.1 TP2 inference engine. This is a READ-ONLY review: do not edit, commit, or rebuild the
engine; do not run anything on the GPU. Allowed: read files, read-only git, grep. The only file you
write is your own findings file.

TIME BUDGET — THIS IS THE MOST IMPORTANT INSTRUCTION.
You have 45 minutes. Your FIRST action must be to write your findings file (a skeleton plus any
finding you already have). Then refine it as you learn. Update it again at least every 10 minutes.
Never leave it empty or stale. At minute 30, stop reading and finalise the file with what you have,
marking anything unfinished as "not verified". A complete short report beats an unfinished perfect one.

PATHS
  your findings file: ${reviewDir}/findings-<FOCUS>.md
  the pre-cut diff you must review: <DIFF>          (read this first; do not re-read the whole series)
  engine repo (read-only): ${engineRepo} branch fused-next
  evidence: ${recipeRepo}/runs/fused-next-20261121/wp{0,1,2,3,4}/ (RESULTS.md, SUPERVISOR.md, JSON)
  the rules the work was held to: ${recipeRepo}/runs/fused-next-20261121/BRIEF.md (rule 4 = bit-exactness,
    rule 6 = no new default-on behaviour, the measurement standard)
  the review assignment: ${recipeRepo}/runs/fused-next-20261121/REVIEW-BRIEF.md
Read the two briefs first (they are short), then your pre-cut diff, then only the evidence sections
your focus needs. Do not read every file in every wp directory.

OUTPUT FORMAT for findings-<FOCUS>.md (max 120 lines):
  # Findings: <focus>
  ## Verdict            (3-5 lines: is this series sound in your area? what must be fixed?)
  ## Blockers           (numbered: file:line, the concrete failing input/shape/scenario, confidence high/med/low)
  ## Should-fix         (same shape)
  ## Notes
  ## Not verified       (what you did not get to, and whether it matters)
Rank: blocker = can change a served byte, deadlock, race, OOB read, or crash on a served shape;
should-fix = wrong on an unserved/edge shape, a broken fallback, or a claim the evidence does not
support; note = cosmetic/maintenance. Verify every file:line you cite. Prefer few well-evidenced
findings over many suspicions. Say plainly when you could not verify something.

Then reply with <=12 lines: your verdict and the blocker/should-fix headlines only.
`;

const alreadyChecked = `
Already checked by the coordinating reviewer (do not redo; confirm or refute, and go past it):
- 0230's inline decode-norm reduction order (w0+w2)+(w1+w3) and tf_ds_kit_squares4 are identical to the
  replaced tf_ds_compress2_decode; removing the "tid < 128" guard is valid only because the 128 kernel is
  launched with exactly 128 threads (exact.zig block = d).
- 0240's E8M0 byte matches the kit formula max(amax,6*2^-126)/6 at mantissa 0x40, subnormal/zero (->1)
  and inf (->255); tf_ds_recip_bits(255) = 0.0 preserves the sign of a zero result; byte 254 unreachable.
- warpShapesOk already forces d%64==0, rd even and (d-rd) even, so the 0241 pair geometry's rotary pairs
  sit inside one lane; hd%64==0 is already required before the pair dispatch.
- 0231's slot formula (raw_pos0 + j - raw_first) % ring_len equals (start + j) % raw_rows, the removed
  scatter's mapping; ch.gpos/ch.groups have no readers outside prompt.zig.
- 0216's bypass bound n_comp_end <= index_topk == 512 equals round.zig's decode identity bound
  ("nb > 512 -> notPorted"); iotaVis matches tf_ds_topk_indices_kernel's ascending bitonic order and
  "cols < vis ? cols : -1" masking. pooled is false only because candidate_block*candidate_blocks
  (8*2048) > 512, which is config-dependent, not proven in code.
- 0251's idxArm/compressArm/keyArm/latentArm/queryArm are called ONLY from the host test (grep-verified);
  the call sites call fused_trace.note() directly with their own inline conditions.
- The engine and harness agree on the renamed kernels (compress2x_norm128/512, fused_rope_qd_pair).
`;

const tasks = [
  {
    key: "bitexact-select",
    label: "Audit selection parity for 0213/0214/0215",
    task: `${common}
<FOCUS> = bitexact-select
<DIFF> = ${reviewDir}/ctx/bitexact-select.diff   (commits 7d32cab 0213, 338be16 0214, 0b78bec 0215)
${alreadyChecked}
YOUR FOCUS: can 0213/0214/0215 change a byte of the mask (prompt arm) or the block list cblk (decode arm)?
Read the new tf_ds_idx_select / tf_ds_pool_cut / tf_ds_tile_prefix / tf_ds_block_max_t<8> against the
removed tf_ds_pool_kth and the removed inline compaction, side by side.
Questions to answer concretely:
1. tf_ds_pool_cut scans only digits 7-4 (the score bytes) and resolves the cutoff bin's ties during
   compaction. Prove or disprove that the picked set is identical to the old 8-digit scan for: random,
   all-tied (flat), NaN, -inf, +-0, and a bin whose want_s is 0 versus partial. Does the old code's
   want_s ever reach 0 with a bin of more than one column (which would change the set)?
2. tf_ds_tile_prefix's warp_end[32] zeroing: "threadIdx.x < 32 && threadIdx.x >= (blockDim.x >> 5)".
   Check it for blockDim 1024 and any other block the call sites use. Is warp_end[31] the block total in
   every case, including a block whose warp count is not a power of two?
3. The mask arm's want==0 branch: picked = in && (u >= prefix) with cmask/prefix covering only the score
   bytes. Is that the same predicate as the old "key >= kth" for columns above the bin, in the bin, and
   below it? What about a column whose score bytes equal the bin but whose column digits put it below
   the old cutoff?
4. tf_ds_block_max_t<8>: the guard is "bsize == 8 && (ss & 3) == 0", then two float4 loads at
   row + b*8. Is 16-byte alignment guaranteed for every r and b? What if ss % 4 == 0 but the row base is
   not 16-byte aligned? Is the NaN/greater comparison order identical to the scalar loop, including a
   partial last block and width not a multiple of 8?
5. 0215's small-row form: tf_ds_idx_pick_kernel stages block_max's global maxima and reuses
   tf_ds_idx_select with mask = nullptr. Is that byte-identical to topk_indices + pool_pick (the unfused
   chain) including the -1 padding past every, and to the one-launch fused form? Check the k == nb case
   and k > nbk.
6. Anything the harness never covered: width < 8, nbk == 1, rows 0, every[r] == 0, vis[r] == 0, k == 1.`
  },
  {
    key: "bitexact-encode",
    label: "Audit FP4/E8M0 and query-geometry parity for 0240/0241/0242",
    task: `${common}
<FOCUS> = bitexact-encode
<DIFF> = ${reviewDir}/ctx/bitexact-encode.diff   (commits cfa54ab 0240, 50a70e1 0241, 9d4232a 0242)
${alreadyChecked}
YOUR FOCUS: can 0240/0241/0242 change a byte of the 0198 outputs (codes, scales, dequantized bf16)?
1. 0240: tf_ds_abs_bits / tf_ds_fp4_amax_bits / tf_ds_scale_byte_bits / tf_ds_recip_bits versus the
   removed tf_ds_fp4_amax + tf_ds_pow2_exponent + __fdiv_rn. Enumerate the inputs that could differ:
   a value whose BF16 bits are not what the float max would see (is every amax input really bf16-exact,
   including the unrotated prefix and the prompt path?), a NaN, a bf16 subnormal, a group whose max is
   exactly 6*2^-126, and an inf. Is the claim "multiplying by the exact reciprocal rounds identically to
   dividing" true for every x, including overflow to inf and subnormal results (--ftz=false)?
2. 0240's __reduce_max_sync(0xffffffffu, ...) in tf_ds_key_store_row and the one-element query kernel:
   are all 32 lanes of the warp always active where it is called? Check the dead-lane handling
   ("live ? srow[i] : zero"), the loop bounds (half, HD) and the partial-warp tail. A full-mask reduce
   executed by a subset is undefined behaviour.
3. 0241's pair kernel: is it byte-identical to the one-element form for every HD, RD and head index?
   Check the rotary branch condition (i0 >= HD - RD), the cos/sin index j, the half-warp
   __reduce_max_sync masks, the 4-byte load/store alignment, the "h >= H" early return (does it skip a
   full-mask primitive?), and the chunk loop at HD % 64 == 0 but HD < 64.
4. 0242: the "rot" guard skips the rotary partner exchange on wholly-unrotated warps. Is "wlast >= start"
   the correct predicate for a warp-uniform decision, given that lane i's partner is i^1 and start may be
   inside the warp? Is the guard applied where the warp is uniform, and is the key kernel's unconditional
   exchange still correct? Does skipping the shuffle change anything for a mixed warp?
5. Anything the harness never covered: RD == HD (no unrotated prefix), RD == 2, HD == 32 with block 128,
   a head count not a multiple of 8, rows 1 with the pair threshold at 32.`
  },
  {
    key: "bitexact-compress",
    label: "Audit compressor parity for 0230/0231",
    task: `${common}
<FOCUS> = bitexact-compress
<DIFF> = ${reviewDir}/ctx/bitexact-compress.diff   (commits 90052a0 0230, b28310e 0231)
${alreadyChecked}
YOUR FOCUS: can 0230/0231 change a byte of the 0197 outputs (the latent row, both raw-ring buffers, the
invalid flag, the prompt key/latent codes and scales, the rotation positions)?
1. 0230: the split into tf_ds_compress2x_norm128/512 kernels. Is the body's arithmetic identical for both
   widths to the removed shared kernel (shared row staging, kit_squares4, warp_sum, the final multiply)?
   Check the "partner slots read before the current row" reordering: does it change any value, or only
   the schedule? Check the invalid flag (atomicExch) and the D != 128/512 refusal.
2. 0231: tf_ds_compress2_norm_kernel now writes the raw-ring tails itself and takes an extra block for an
   odd trailing row. Check: the slot arithmetic for ring wrap, raw_first/raw_pos0/n_rows bounds, the
   one-row chunk (full == 0) path, the "2*g+1 >= n_rows" early return, and whether the trailing block
   writes only the raw slot and no latent. Is there any row the old scatter wrote that the kernel now
   does not, or vice versa (keep < n, raw_rows < n, start + n crossing the ring twice)?
3. 0231: the prompt stores now derive gid = base_group + r and pos = gid * ratio instead of reading
   ch.gpos/ch.groups. Prove that equals the removed upload for every chunk (including a chunk whose start
   is not a multiple of ratio, and one that spans a ratio boundary). Check the ch.groups/ch.gpos staging
   condition (idx_key_fused/lat_fused) covers every consumer.
4. The host-side argument plumbing in exact.zig: n_rows, raw_first, raw_len, raw_pos0 types and the
   "n_rows < rows*2 or n_rows > rows*2+1" check. Is any of it truncating a 64-bit value?
5. Anything the harness never covered: D = 2048 prompt rows, D = 128 prompt, an odd n, keep == 0, a
   chunk whose start wraps the ring, R = 1 decode with an invalid partner.`
  },
  {
    key: "concurrency",
    label: "Audit PDL ordering, shared memory and warp-mask safety",
    task: `${common}
<FOCUS> = concurrency
<DIFF> = ${reviewDir}/ctx/concurrency.diff   (commits f013282 0250, d01975d 0217, 0b78bec 0215, 90052a0 0230)
${alreadyChecked}
YOUR FOCUS: races, ordering, OOB and UB. Read the kernels adversarially.
1. 0250 PDL: pdl_launch/pdl_wait are now split. For every kernel that uses the split, name the first byte
   it reads after the wait and the last byte its predecessor wrote. Is "w[threadIdx.x]" (the norm weight)
   really written by no kernel in the round on either rank? Check the weight is not a dequant/LoRA target
   and that the prompt half was left unsplit for a reason. Also: is griddepcontrol.launch_dependents legal
   and safe before griddepcontrol.wait, and does the macro's "if (pdl)" branch keep every thread in the
   block executing the same instructions?
2. 0217: shared-memory sizing. Recompute the request per launch (idx_fused at nbk*4 up to 65,536 bytes,
   idx_pick the same) against the static shared in the same kernels (hist[256], warp_end[32], prefix/mask/
   want) and the opt-in limit the code sets. What happens when allowDynamicShared is not set (a load path
   that skips it, or a second context)? Is the occupancy claim in the evidence right, and does a
   1024-thread block with 64 KiB of dynamic shared actually launch?
3. 0215: the two-phase form is two launches on one stream. Is stream ordering enough, or does anything
   need an explicit fence/event? Check the 6144/6145-block switch boundary and the bmax row stride.
4. 0231's ring writes inside the compress2_norm kernel: can a later launch in the same stream read the
   ring before the write is visible? Is there any cross-block hazard inside the kernel (two blocks writing
   the same slot)? Check the modular slot arithmetic for aliasing when raw_len < n.
5. Edge shapes: partial blocks, odd tails, R = 1, nbk = 1, rows 0. Look for OOB or uninitialized reads in
   tf_ds_tile_prefix (warp_end uninitialized before the first write?), tf_ds_idx_select's
   __syncthreads inside conditional regions, and the ballot/prefix under divergence.
6. Warp primitives: every __reduce_max_sync / __shfl_xor_sync / __ballot_sync mask against the set of
   lanes that actually execute it. A full-mask primitive executed by a subset is UB; a __syncthreads in
   divergent code deadlocks. Name the launch shape for every race or OOB you claim.`
  },
  {
    key: "hostside",
    label: "Audit host dispatch, fallbacks and graph capture keys",
    task: `${common}
<FOCUS> = hostside
<DIFF> = ${reviewDir}/ctx/hostside.diff   (the whole host-side diff: exact.zig, fused_trace.zig, idx_fused.zig,
       ops.zig, prompt.zig, round.zig)
${alreadyChecked}
YOUR FOCUS: host-side correctness. The harness byte-checks kernels, not the dispatch.
1. 0216's prompt bypass (prompt.zig indexer, "n_comp_end <= c.index_topk"): enumerate what the skipped
   path would have done that the bypass does not. Is ch.cidx layout, sh.kk, ch.pool/sh.pool state and the
   TP2 split/allGather result identical? Does anything downstream read ch.iq/ch.iq4/ch.iw/ch.wl/ch.keys/
   ch.score/ch.tmax for a layer that took the bypass? Is the bypass correct for the KEYED layers (before
   the candidate source), where the full path uses the kit's topk_select_pruned rather than the engine's
   topk_indices? Is "pooled is false" guaranteed for every config, or only this checkpoint?
2. 0215's 6144-block switch and 0217's max_nbk and 0241's row threshold: are the boundaries reachable and
   right (R=7/8, blocks=6143/6144/6145, nbk=16383/16384/16385)? Does every call site that can select the
   small-row form also have idx_pick_fn loaded, and is it loaded under the same switch?
3. Graph capture keys: find the capture key for each fused arm and check it includes every parameter that
   selects a kernel variant: rows, width, blocks, d, heads, the pair/single choice, pdl. A missing key
   replays the first captured shape for another shape. Check both ranks agree on the decision.
4. Unfused fallbacks: prove each is still reachable with the switch off and still correct after the
   refactor. Flag any new default-on behaviour (BRIEF rule 6), dead code left behind, or a switch read
   that moved from load time to call time.
5. 0251's trace: can it change behaviour or timing? Is fused_trace.enabled()/note() safe if the two ranks
   are threads in one process (unsynchronised statics)? Does the host test prove the CALL SITES take the
   arms, or only that the idxArm/compressArm/... model functions do?
6. exact.zig's new arguments: check every launcher's argument order and types against the .cu signature
   (raw_kv/raw_score/ring_len/raw_pos0/raw_first/n_rows; base_group/ratio). A swapped argument is silent.`
  },
  {
    key: "evidence",
    label: "Audit claims versus measured evidence per commit",
    task: `${common}
<FOCUS> = evidence
<DIFF> = none — you review the CLAIMS in the commit messages against the evidence.
Commits: git -C ${engineRepo} log --oneline 3c54a9d..fused-next  (13 commits, 0213-0251)
Evidence: ${recipeRepo}/runs/fused-next-20261121/wp0..wp4/ (RESULTS.md, SUPERVISOR.md, JSON, logs)
${alreadyChecked}
YOUR FOCUS: does the evidence support each stated claim? Do not read every file: read wp0/RESULTS.md
(the harness and its own proof runs), then the section of wp1/wp2/wp3/wp4/RESULTS.md for each commit,
then open the raw JSON/log ONLY where a number decides something.
Per commit, give a one-line verdict (supported / partly supported / not supported) plus the file and the
number that decides it. Check specifically:
1. Is the parity reference the RIGHT one? For anything a kit kernel consumes, the reference must be the
   kit producer's bytes, not an engine helper (BRIEF rule 4; 0206 is the precedent that passed standalone
   and changed 4 of 39 replies). For each of 0213/0214/0215 (mask, cblk), 0240/0241/0242 (codes, scales),
   0230/0231 (latent, rings, codes): name the reference actually used.
2. Were the parity cases actually run: random, tied, flat, NaN, +-inf, signed-zero, partial-block,
   boundary, at every served shape (decode R=1/7/32/48; prompt R=128/1024/2048; widths 128/512; 0195 at
   4096/8192 blocks and the 16K-65K window; 100K+)? Name the classes the harness does NOT cover.
3. Is the timing methodology what BRIEF.md demands: CUDA graphs, old and new interleaved A B A B in one
   process, >=20 reps of 100+ replays, warmed, median + IQR, and an empty-kernel launch floor at the same
   launch shape? Was flock held and was production idle (connections: 0)? Check the JSON for the actual
   rep counts and the clean/retry flags, not just the prose.
4. Is each stated speedup outside the IQR, and is every served shape that matters faster? A slower served
   shape is a drop unless the kernel dispatches per shape (0215, 0241 do). Check 0217's "unchanged
   occupancy" and "2.5-3.4x" claims, 0241's R=32/48 crossover, 0230's 11-13% at D=128, 0240's -55% prompt.
5. Is each bound claim (launch floor / latency / bandwidth / issue) supported by SASS and resource data
   actually in the folder, or asserted? Find at least one claim that is asserted but not measured.
6. 0216 cannot be byte-checked by the harness — is it covered another way (the 39-case exact suite)? Did
   that suite actually exercise a short prompt that takes the bypass? 0251's "prove engagement" — does it?
7. Did any commit smuggle a second change, and are dropped items recorded honestly as "closed, no change"
   with numbers?
8. Cross-check the supervisor runs: wp1/SUPERVISOR.md, wp4/SUPERVISOR.md and supervisor-parity.log claim
   151/151 and 39-case exact suites. Are those runs against the right tips, and do they cover the bytes
   the individual items changed?`
  }
];

const results = await runs.all(
  tasks.map((t) => ({
    key: t.key,
    label: t.label,
    task: t.task,
    agent: "reviewer",
    options: { timeoutMs: 2700000 },
  }))
);
return results;
