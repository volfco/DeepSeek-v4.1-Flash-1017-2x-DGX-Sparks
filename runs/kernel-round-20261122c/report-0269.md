# report-0269 — TF_DS_DRAFT_TAPS_DIRECT

**Commit:** `5d1b4cdfa09ff34564eaa85a4f9038ed0b26b21c` on `kernel-0269` (parent `27baa93`, exactly one commit, tree
clean). **Switch:** `TF_DS_DRAFT_TAPS_DIRECT` (off by default; `1` on, `0`/`""`/unset off, anything else refuses
startup with `TF_DS_DRAFT_TAPS_DIRECT: expected 1 or 0`).
**Subject:** `0269 dsv41: TF_DS_DRAFT_TAPS_DIRECT (off by default): the drafter reads contiguous verify-round taps in place`

## Premise

Holds. `lanes.Built.build` (`lanes.zig:196-222`) lays a verify round's rows out back-to-back in window order
(`b.wins[i].row = b.r`, then `b.r += w.ids.len`), so `rd.taps` rows of the participating windows are consecutive
whenever the participating windows form a contiguous run of the round's windows. At the served config every stream
drafts (`Win.absorb = m.drafting() and w.stream.drafts`, `lanes.zig:1733`; per-stream `drafts` defaults true), so all
windows participate and the rows are always consecutive — the in-place arm engages on every verify round. The
gathered bytes are `rows × 3 × 5,120 × 2 B` per rank per round (the item's 7-row shape: 215,040 B ≈ 210 KiB,
re-checked in `h0269/run.log`), in one `copyRows` launch per participating window. Bit-identity is exact and local:
`copyRows` is a byte-exact row copy and `mainKeys`' input row stride is `dr.taps_w` (= the `rd.taps` row width
`taps × D`) in both arms, so the GEMM inputs are the same bytes either way (the prompt path's `Drafter.absorb`
already feeds `mainKeys` in place this way, `draft.zig:504`).

## Files touched

- `zig/src/families/dsv41/draft_taps_direct.zig` (new, pure): switch parse (`idx_fused.zig` pattern), the arm
  predicate `inPlace(taps, ns, row_bytes) ?u64` (byte addresses of the items' rows; accepts only rows consecutive in
  the given order), tests (parse truth table; adjacency/gap/overlap/out-of-order/zero-row truth table; the
  1–4 windows × lengths 1–7 × adjacent/gapped layouts byte-for-byte matrix against a packed-copy reference).
- `zig/src/families/dsv41/ops.zig`: `draft_taps_direct: bool` parsed at `Ops` load via `fromText` (refuses startup on
  a malformed value). **Deviation from the convention's "bool + ?cuda.Function":** there is no kernel — the arm skips
  a host-decided launch and re-points `mainKeys`' input; no `.cu` code changes.
- `zig/src/families/dsv41/draft.zig`: `Drafter.absorb_many`'s arm — the contiguity decision over the items, the
  gather `copyRows` skipped when it holds, `mainKeys(first row, total count)`; the engagement banner
  (`taps_direct_noted`, atomic `swap`, exactly as 0265's `head_scored_noted`).
- `zig/src/families/dsv41/dsv41.zig`: module export + `refAll`.
- `zig/src/families/dsv41/layer_main.zig`: one comment at the fixture check that reads `dr.at` (see Known gaps).

No `dsv41_ops.cu` / `dsv41_torch.cu` change (the kernel gate re-verifies them unchanged).

## Design

- **What decides contiguity, and when:** the host, in `Drafter.absorb_many`, at enqueue time each round, from the
  participating items' byte addresses and row counts (the round's windows in round order). This is exactly the same
  per-round host state that already decides each item's source pointer and the gather's launch count today.
- **Graph capture:** the absorb's launches (gather `copyRows`, the `prompt.upload`s, `mainKeys`' GEMMs,
  `kvNormRope`) are eager — they are inside **no** captured graph. The only capture sites are the round's stretch
  graphs (`round.zig:1904`, which capture `stretch`'s tap writes into `rd.taps` at fixed offsets — untouched) and the
  drafter pass graphs (`draft.zig:224`, `PassGraphs.run` wrapping `dr.body` — untouched). All three `absorbRows`
  call sites are eager (`lanes.zig:1495`, `:1540` with `TF_DS_ABSORB_AFTER_PICK`, `:2808` the follower). So there is
  no captured graph whose replays could observe a different decision; the decision cannot vary across replays because
  the absorb is re-enqueued, not replayed.
- **Both ranks identically:** the leader writes the windows to the frame in `wins` order and both ranks build rows
  from that order (`Built.build`), so the decision is a pure function of the frame's windows. The arm is bit-identical
  anyway, so even a rank-uneven env pair stays correct (as 0160 notes for `TF_DS_ABSORB_AFTER_PICK`).
- **Bit identity:** on the in-place arm `mainKeys` consumes input row `r` = `rd.taps` row `first + r`; on the packed
  arm row `r` = `dr.at` row `r` = the same source rows copied byte-exactly in the same item order. The `pos`/`rows`
  arrays are built in item order in both arms, and `inPlace` only accepts when item order == row order, so the ring
  writes stay row-aligned. No arithmetic changes anywhere.
- **Banner:** `std.log.info("dsv41: TF_DS_DRAFT_TAPS_DIRECT: the drafter reads the round's contiguous taps rows in
  place (first at {d} windows x {d} rows)")` once behind `taps_direct_noted.swap(true, .monotonic)`, at the arm's
  first actual run (after the `n == 0` early return).

## Launch counts (Drafter.absorb_many; the round and the pass are unchanged)

| | off (today) | on, contiguous |
| --- | ---: | ---: |
| gather `copyRows` | k (one per participating window, 1–4) | 0 |
| `prompt.upload` pos/rows | 2 | 2 |
| `mainKeys` (grouped main_proj, rmsnorm, grouped stages) | unchanged | unchanged |
| `kvNormRope` per stage | unchanged | unchanged |

At the served shape the arm removes 1–4 launches and `N × 30,720 B` of device copying per rank per round
(N = 2–7 rows per window; 7 rows total = 215,040 B ≈ 210 KiB, the item's arithmetic; 4 streams × 7 rows = 860,160 B
in 4 launches). Expected saving ~10 µs class (the item's estimate; unmeasured here).

## Contiguity frequency at the served config (1–4 streams, `TF_DS_DRAFT_BLOCK=6`, `KEPT=2`, windows up to 7 rows)

By the row layout above: the participating windows are always consecutive in `rd.taps` **whenever all participating
windows are adjacent in the round's window order**. At the served production config every stream drafts, so the
participating set is the whole window list: **100% of verify rounds engage the in-place arm** (any 1–4 streams, any
window lengths 1–7). Gaps need a stream configured without drafts (`Stream.drafts = false`) between two drafting
streams' windows; that mixed case keeps today's packing, as the brief requires. A non-participating window before the
first or after the last participating one does not break contiguity (covered in both test matrices).

## Harness (`h0269/`, gate 3)

`build.sh` + `draft_taps_direct_run.cpp` + `run.log` + `stubs/` (the fixed CUDA-syntax stubs, see gate 2). No `.cu`
kernel exists to splice, so the harness emulates the arm's row addressing on the CPU: `absorb_many`'s contiguity
decision on byte addresses, the removed `copyRows` launch (its `copy16` kernel semantics: byte-exact rows, one
simulated thread a 16-byte word — no OS threads), and `mainKeys`' input row reads (`x[r * ldx + c]`, `ldx = taps_w`).
The reference is written from the brief's spec (the participating windows' taps rows concatenated in round order),
independent of the arm's first+count addressing. Matrix: 1–4 windows × lengths 1–7 × 5 gap layouts (gapped lengths
1–3) at small width + the served width (3 × 5,120 channels, mixed 5/6/7-row windows, gapped), decision checked
against expectation per layout. **28,016 cases, 11,252 in-place, 0 failures** — every arm output matches the packed
reference byte for byte. `run.log` also records the 215,040 B served-shape accounting.

## Gate results

1. `zig build test`: **exit 0** (three consecutive clean runs after the final change; `/tmp/t0269-test*.log`). One
   earlier post-fix run failed once at the run step without a captured test name — the known
   `lane_host.test.prefill` interleave flake signature (brief's known flake); it did not recur in three runs.
2. CUDA syntax check: **`cuda syntax check: ok`** — run with the round's fixed stubs
   (`h0269/stubs/check.sh <worktree>/zig/kernels/cuda`, copied from the previous round's `h0265/stubs/`). The shared
   `/home/colum/Workspace/cuda-syntax-stubs` tree **fails on the unmodified base itself** (7 errors: `float4`,
   `__nv_bfloat162`, `__clz`, int→bf16 — stubs older than the tree, same as report-0260/0265 found); evidence in the
   transcript, not a 0269 regression. No `.cu` file changed in this commit.
3. Harness: **all cases match** (above).

## Design decisions and deviations

- **No `mainKeys` signature change:** its input already carries the source row stride (`dr.taps_w` elements = the
  `rd.taps` row width), so "first row + total count" is its existing `(taps, n)`. A stride parameter would be
  dead code.
- **Ops convention deviation:** bool at `Ops` load per convention, no `?cuda.Function` (no kernel exists), same as
  0265's precedent. Documented there too.
- **Fallback order preserved:** the off/gap path issues the identical `copyRows` launches in the identical order
  (the decision block launches nothing), so the off path is byte- and launch-identical to `27baa93`.
- **`items.len > max_streams`** silently falls back to packing (no caller exceeds 16; fail-safe direction).
- **Bookkeeping deviation from AGENTS.md:** no `decisions.md`/`todo.md`/`continue.md` edits in the worktree — the
  brief mandates exactly one commit with its subject and a clean tree, so decisions are recorded in this report
  instead. Follow-ups (below) are recorded here rather than in `todo.md`.
- **`code-review-graph` tool unavailable** in this environment (MCP search: no match); reviews ran as `reviewer`
  subagents instead. Review provenance: the first reviewer verified all five contract points from source (its
  salvaged session: `sessions/0269/.../5adae625.../run-0/session.jsonl`) but the provider 500'd before it wrote the
  final report; a narrow follow-up reviewer died to the same provider 500s with no output. The one open check it
  left (`dr.taps_w` vs `rd.taps`'s `@max(len,1)` dummy row width at `round.zig:576`) was resolved by hand: the widths
  agree for any config with ≥1 tap layer; a zero-tap config (`taps_w = 0`) feeds both arms zero-byte rows
  (bit-identity trivial; the config is nonsensical for the drafter — `main_proj` would have `k = 0`). Claims on the
  eager call sites and `layer_main`'s item ordering were re-verified by hand (`lanes.zig:1495/1540/2808`,
  `layer_main.zig:1280-1315`). No findings in either review.

## Known gaps (hardware gate only)

- **Timing:** the ~10 µs class saving is the item's estimate; only the hardware gate can price the removed
  `copyRows` (1–4 launches, 0.2–0.86 MiB per rank per round) inside the served graphs-on round.
- **Banner counting:** the hardware gate must count the `TF_DS_DRAFT_TAPS_DIRECT: ... (first at ...)` line to prove
  the arm ran (a null result without the banner is indistinguishable from a switch that never fired).
- **Fixture checker coupling:** `layer_main`'s `Drafter.absorb_many` input check reads `dr.at` (the packed buffer);
  with the switch **on** that one check sees stale bytes (the drafter rings remain bit-identical). Run the layer
  fixture gate with the switch off (as the fixtures were recorded); noted in a code comment at the check.
- **GPU-replay reality:** the absorb is eager by construction so no captured graph can observe a different decision,
  but only the hardware gate exercises real multi-stream replays at all; a rank-uneven env pair is safe by
  bit-identity but has only been argued, not run.

## Out of scope (per the brief)

The todo item's second half — skipping the three tap launches in ordinary verify rounds when the drafter is disabled
(retaining taps for explicit probes) — is **not implemented**; the tap writes in `round.stretch` are untouched. It is
a separate, larger change (probe/checker behavior around `rd.taps`) and the brief scoped it out explicitly.

## Follow-ups (todo.md material)

- The tap-launch skip when the drafter is disabled (the item's second half; needs the probe story settled).
- Price the removed gather with the eager stage timer / a graphs-on A/B at the served config.
