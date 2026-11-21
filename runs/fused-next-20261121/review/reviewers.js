// Four parallel read-only reviewers for the fused-next 0213-0251 round.
// Each writes its own findings file under runs/fused-next-20261121/review/.

const { recipeRepo, engineRepo, reviewDir } = args;

const common = `
CONTEXT
You are one of four parallel reviewers auditing a CUDA/kernel optimization series on a production
DeepSeek-V4.1 TP2 inference engine. Read these first, fully:
  ${recipeRepo}/runs/fused-next-20261121/REVIEW-BRIEF.md   (the review assignment, read-only rules)
  ${recipeRepo}/runs/fused-next-20261121/BRIEF.md          (the rules the work was held to)
  ${recipeRepo}/runs/fused-next-20261121/wp0..wp4/*.md     (evidence: RESULTS.md, SUPERVISOR.md)
  ${recipeRepo}/tools/fused_bench/                          (the harness)

ENGINE REPO (read-only): ${engineRepo}, branch fused-next.
  Diff under review: git log --oneline 3c54a9d..fused-next  and  git diff 3c54a9d fused-next
  Files touched: zig/kernels/cuda/dsv41_ops.cu, zig/kernels/cuda/dsv41_torch.cu,
    zig/src/families/dsv41/{exact,fused_trace,idx_fused,ops,prompt,round}.zig
  Commits: 0213 0214 0215 0216 0217 0230 0231 0240 0241 0242 0250 0251.

HARD BOUNDARIES (this is a review, not implementation)
- Do NOT edit, commit, or rebuild the engine. Do NOT run anything on the GPU.
- Allowed: read files, git log/show/diff/blame (read-only git), grep/ripgrep.
- The ONLY file you may write is your own findings file: ${reviewDir}/findings-<FOCUS>.md
- You may run \`zig build test\` in a scratch COPY of the engine if you truly need it (~5 min cold,
  baseline 794/796 with two known pre-existing failures: prompt_cache.test.0178 and
  round.test.TF_DS_EXPERT_HOT_CACHE: hotPick). Copy, do not build in place. Skip it if not needed.
- Every bash command needs a timeout.

WHAT A GOOD FINDING LOOKS LIKE
- file:line (real line numbers from the branch, verify them), the concrete failing input or
  scenario (shape, values, flag combination, launch ordering), what breaks, and your confidence
  (high / medium / low). Rank each as blocker / should-fix / note.
- A blocker = can change a served byte, deadlock, race, OOB read, or crash on a served shape.
- A should-fix = wrong on an unserved/edge shape, a fallback that no longer works, a claim that
  the evidence does not support, or a maintenance trap.
- A note = cosmetic, comment-level, or a risk to watch.
- Prefer a small number of well-evidenced findings over a long list of suspicions. If you could
  not verify something, say so explicitly rather than asserting it.

DELIVERABLE
Write ${reviewDir}/findings-<FOCUS>.md, at most 200 lines:
  # Findings: <focus>
  ## Summary            (3-6 lines: is this series sound in your area? what must be fixed?)
  ## Blockers           (numbered, each with file:line, scenario, confidence)
  ## Should-fix
  ## Notes
  ## Not verified / open questions
Then return a SHORT reply (<=15 lines): your verdict for your focus, plus the blocker and
should-fix headlines only.
`;

const tasks = [
  {
    key: "bitexact",
    label: "Audit bit-exactness of every changed kernel path",
    task: `${common}
YOUR FOCUS: BIT-EXACTNESS.
The production engine requires every changed path to be byte-identical to the path it replaces, and
any output consumed by a kit kernel must match the kit producer's bytes, not an engine helper (the
0206 failure mode: it matched idx_fused.zig:scoreKey and then changed 4 of 39 replies on the pair).

For each commit, read the new code and the code it replaces side by side and ask: can any input make
the bytes differ? Cover reduction order, rounding points, FMA contraction (--fmad=false), NaN / inf /
signed-zero propagation, comparison ties and total orders, saturating vs wrapping integer math, and
shift/mask edge cases. Specifically:
- 0213, 0214, 0215: the static/vectorized phase 1, mask-only arm, tied-column resolution during
  compaction, and the two-phase small-row form. Check tie handling and total order in selection
  against the replaced path; check partial blocks and boundary rows.
- 0240: FP4 / E8M0 scale encoding from BF16 bits with an exact reciprocal and a warp integer
  maximum. Check exponent edge cases (denormals, 0, inf/NaN inputs), the reciprocal being exact for
  all group maxima, and whether the kit's encoder is the reference.
- 0241: the query geometry dispatch (one element a thread below decode's largest window, a warp a
  head above). Check that both geometries produce identical bytes for every row count, including
  the switch boundary R=32/48 and rows outside the harness matrix.
- 0242: the shuffle guard that skips the rotary partner exchange on wholly unrotated warps. Check
  what "wholly unrotated" means at partial warps / odd tails and whether skipped lanes were needed.
- 0230, 0231: the decode compressor per served width and the prompt source step folding raw-ring
  tails and position setup into the fused launches. Check the ring wrap, odd trailing row, and
  one-row chunk cases byte-for-byte.
- 0250's PDL entry split must not change results; 0217's wider pool must not change results.
Name every shape or input class the harness never covered that could break parity, and say whether
you can prove safety from the code alone.`
  },
  {
    key: "concurrency",
    label: "Audit concurrency, PDL ordering, and memory safety",
    task: `${common}
YOUR FOCUS: CONCURRENCY AND MEMORY SAFETY.
Read the new kernel code as an adversarial reviewer. Specifically:
- 0250 PDL: nothing produced by a predecessor may be read before griddepcontrol.wait. Find every
  producer/consumer pair across the split launches (norm weights, indexer key) and check that each
  read sits after the wait, that the launch is actually marked PDL, and that graph capture preserves
  the dependency. Check the decode compressor and the indexer key paths separately.
- 0217: shared-memory sizing and the opt-in dynamic shared limit. Recompute the shared-memory
  request per launch shape; check it against the per-block limit and the opt-in ceiling, check what
  happens when the opt-in attribute is not set (fallback path / launch failure), and check whether
  occupancy collapses to 1 block at large max_nbk.
- 0215: the two-phase form. Check cross-block and cross-launch dependencies: is there a global
  barrier (cooperative launch? grid sync?) and is it valid at the launch grid size? Check the
  6144-block switch boundary.
- 0231: ring writes. Check for cross-block hazard on the ring buffer, missing __threadfence or
  syncwarp, and whether a consumer launch can start before the producer's writes are visible.
- Edge shapes: out-of-bounds or uninitialized reads at partial blocks, odd tails, R=1, and the
  smallest/largest served shapes. Check bounds arithmetic for overflow on large column windows.
- Warp primitives: __reduce_max_sync and full-mask shuffles under the new guards — verify the mask
  argument matches the participating lanes (a full-mask shuffle executed by a subset is UB), and
  that divergence before a __syncwarp/bar.sync cannot deadlock.
State the concrete failing launch shape for every race or OOB you claim, and your confidence.`
  },
  {
    key: "hostside",
    label: "Audit host-side dispatch, fallbacks, and capture keys",
    task: `${common}
YOUR FOCUS: HOST-SIDE CALL SITES AND DISPATCH.
Read every host file in the diff and the surrounding call sites:
zig/src/families/dsv41/{round,prompt,idx_fused,exact,ops}.zig plus compress2_norm.zig, rope_store.zig,
fused_trace.zig and the CUDA graph capture code. Specifically:
- 0215's 6144-block switch and 0217's max_nbk and 0241's row threshold: are the thresholds
  reachable, do they match the kernel's own assumptions, and is every served shape on the intended
  side? Check the exact boundary values (R=31/32/33, 47/48/49, blocks=6143/6144/6145).
- 0216's prompt bypass: prompt chunks whose compressed keys all fit skip projection, scoring and
  selection. The harness cannot byte-check this, so read the host condition and the kernel's
  equivalent condition and prove they agree; a mismatch silently changes selection. Check what
  happens when the chunk is exactly at the fit boundary and when only some keys fit.
- 0231's ring wrap / odd trailing row / one-row chunk: check the host computes the same chunking
  the kernel assumes, and that the position setup matches the unfused path.
- Graph capture keys: does the capture key include every parameter that selects a kernel variant
  (row count, width, blocks, flags)? A missing key means the first shape captured is replayed for
  another shape. Check both ranks and whether both ranks agree on the dispatch decision.
- Unfused fallbacks: prove each fallback is still reachable (flag off) and still correct after the
  refactor; check for dead code, changed defaults, or a new default-on behaviour (BRIEF rule 6).
- 0251's fused-arm trace: check it cannot change behaviour or timing in production, and whether the
  host test it adds actually proves engagement.
Name the concrete call site and input for each finding.`
  },
  {
    key: "evidence",
    label: "Audit claims versus measured evidence per commit",
    task: `${common}
YOUR FOCUS: CLAIMS VERSUS EVIDENCE.
For each of the 13 commits, take the commit message's claim and check it against the evidence in
${recipeRepo}/runs/fused-next-20261121/wp{0,1,2,3,4}/ (RESULTS.md, SUPERVISOR.md, JSON, logs) and the
harness in ${recipeRepo}/tools/fused_bench/. Do not accept a summary; open the raw JSON/log where a
number matters.

Check, per claim:
- Is the parity reference the RIGHT one? For any output a kit kernel consumes, the reference must be
  the kit producer's bytes (see rule 4 in BRIEF.md), not an engine helper. 0206 is the precedent.
- Were the parity cases actually run: random, tied, flat, NaN, +-inf, signed-zero, partial-block,
  boundary, at every served shape (decode R=1,7,32,48; prompt R=128,1024,2048; widths 128,512;
  0195 at 4096/8192 blocks and the 16K-65K window; 100K+ where targeted)?
- Is the timing methodology what BRIEF.md demands: CUDA graphs, old and new interleaved in the same
  process (A B A B), >=20 reps of 100+ replays, warmed, median + IQR, and an empty-kernel launch
  floor at the same launch shape? Was flock held and was production idle (health connections: 0)?
- Is the stated speedup outside the IQR, and is every served shape that matters faster (a slower
  served shape is a drop unless the kernel dispatches per shape)?
- Is the bound claim (launch floor / latency / bandwidth / issue) supported by the SASS and
  resource data actually present, or asserted from intuition?
- Anything asserted but not measured: occupancy, "no spills", "at the launch floor", bandwidth
  headroom, SASS opcode claims with no dump in the folder.
- 0216's prompt bypass cannot be byte-checked by the harness — was it checked another way?
- 0251's "prove engagement" host test: does it actually observe the fused arm being taken?
Also check that no commit smuggles a second change, and that dropped/reverted items are recorded
honestly as "closed, no change" with numbers.
Give, per commit, a one-line verdict: supported / partly supported / not supported, with the file
and the number that decides it.`
  }
];

const results = await runs.all(
  tasks.map((t) => ({ key: t.key, label: t.label, task: t.task, agent: "reviewer" }))
);
return results;
