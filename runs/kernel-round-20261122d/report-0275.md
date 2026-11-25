# report-0275 — TF_DS_COPY_MIN

**Commit:** `42a2c2d59600aae1889a0a38233817021cf61768` on `kernel-0275` (parent `27baa93`, one commit).
**Switch:** `TF_DS_COPY_MIN=<n>` — unset/`""`/`"0"` off (today's hard-coded minimum), integer 2..7 on, anything else
refuses startup with `TF_DS_COPY_MIN: expected 0 or an integer 2..7` on stderr.
**Subject:** `0275 dsv41: TF_DS_COPY_MIN (unset: today's minimum): the copy draft's minimum match length`

## Premise

Holds. The hard-coded gate exists exactly where the brief says, modulo a path drift: the tree has no
`zig/src/families/dsv41/windows.zig`; the gate is `zig/src/core/lanes/windows.zig:141`
(`if (copied.len < 2 or p.lastMatch() < e.cfg.enter_match) return &.{};`) — line number matches, the file lives in
family-agnostic core (`lanes.windows.copyProposal`, shared with nemotron). The source patch's other hunk site maps to
`zig/src/families/dsv41/lanes.zig` `facts()` (the `Model` wiring). The saving the source claims (78.4→89.8 tok/s
geomean at n=3 on repetitive text, a different quant) is a hardware question and is not needed for the premise: the
switch changes only which drafts a round proposes, and the code path to change exists and is reachable at the served
shape (drafted streams, suffix-lookup proposer, copy widths 2..6 tokens).

## What the source patch's `n` is, mapped explicitly

The source replaces `copied.len >= 2` with `copied.len >= e.cfg.copy_min` in its `copyProposal` — **the same
quantity as our `copied.len < 2`**: the token count of the copied continuation (`copied.len`, the span proposed
after the earlier matched occurrence, capped by the round's copy width). It is **not** the match length behind the
proposal (`lastMatch`); that keeps its own unchanged `Config.enter_match` (8) gate. (The commit subject's phrase
"minimum match length" means the copy draft's minimum length; the context match is a separate, unchanged gate.)

- **n=2 (unset)** = today exactly: a copy of ≥2 tokens whose match is ≥8 takes the round from the draft head.
- **n=3** = a 2-token copy no longer takes the round; it falls to the head's queued drafts (or a plain round).
  3-token-and-longer copies are unchanged. This is the source's benchmark setting.
- Our served copy widths (TF_DS_DRAFT_BLOCK=6 → `win_rows` 7 → `first_copy` 6): copies are 1..6 tokens. So n=3 only
  flips 2-token copies; n=7 (the parse ceiling) disables every copy at the served shape (equivalent to
  `TF_DS_NO_COPIES=1` there; at `TF_DS_DRAFT_BLOCK=7` a 7-token copy can still qualify).

Differences from the source's semantics, deliberate: the source clamps `@max(2, m.copy_min)` in `Config.init` (values
< 2 act as 2) and sets `copy_min = if (drafting) self.copy_min else 2`; this port **refuses** < 2 at startup (the
brief: "integer within a sane range, else refuse startup") and mirrors the drafting guard as
`if (drafting) self.copy_min else null` (without a draft head there is no round to take from the head).

## Lossless by construction (why)

Drafts are proposals only; nothing the switch touches feeds state:

1. **Verification is unchanged.** Every round's window — host-copied rows or head drafts — goes through the same
   verify forward and the same accept rule (`accept.acceptPath` / chain path), which keeps the longest prefix of rows
   whose sampled continuation matches. A refused short copy just makes the round carry the queued head drafts (or a
   plain row) instead; KV rows are written only for kept rows either way.
2. **Sampling is draft-invariant.** `sampling.zig` keys every Gumbel draw by `(seed, position, token id)` through
   splitmix64 ("a verify row draws as the serial run at its position"); greedy rows are argmax. The token sampled at
   a position does not depend on which window or round verified it, nor on draw order, and the drafter's own noise is
   keyed the same way. So the committed token stream equals the serial run's whatever the drafts are — the switch
   changes only how many rows each round verifies (speed), never which tokens land.
3. Residual GPU-only caveat (see Known gaps): per-row logits in the verify forward must be row-count invariant. The
   same caveat already applies to today's plan mix (2..7-row windows round to round); the exact panel settles it.

## Files touched

- `zig/src/families/dsv41/copy_min.zig` (new, pure — `idx_fused.zig` pattern): `env_name`, `default_min = 2` (today's
  gate), `max_min = 7` (the widest a copy can be: `window_rows − 1`, `draft.max_draft_block = 7`; above it no copy can
  ever qualify, so it refuses rather than silently disables copies — `TF_DS_NO_COPIES=1` is that switch), `parse` /
  `fromText` (`error.BadCopyMin`), host reference `takes(len, set)`, and 3 `std.testing` tests (parse matrix,
  refusal matrix, gate truth table).
- `zig/src/families/dsv41/dsv41.zig`: module export + `refAll(copy_min)` (tests run in `zig build test`).
- `zig/src/families/dsv41/native.zig`: env doc line; parse at `open` beside `TF_DS_NO_COPIES`:
  `st.lanes.copy_min = if (getenv("TF_DS_COPY_MIN")) |v| try copy_min.fromText(v) else null;`
- `zig/src/families/dsv41/lanes.zig`: `Lanes.copy_min: ?u32 = null`; `facts()` passes
  `.copy_min = if (drafting) self.copy_min else null`.
- `zig/src/core/lanes/config.zig`: `Model.copy_min: ?u32 = null`, `Config.copy_min: ?u32 = null`, init passthrough
  (the optional idiom mirrors `solo_head_overhead_ms`).
- `zig/src/core/lanes/windows.zig`: the gate `copied.len < (e.cfg.copy_min orelse 2)` (`orelse 2` = the old literal)
  and the first-engagement banner (below).

## Off-path parity

Unset ⇒ `cfg.copy_min == null` ⇒ `orelse 2` ⇒ the exact comparison of today. Zero new launches, zero changed bytes,
no banner, no log lines: byte- and launch-identical by construction. The one changed line in core is the threshold
expression itself; nemotron's `Model` leaves `copy_min` null and sees today's behaviour untouched.

## Banner (engagement)

One-shot behind a file-level `std.atomic.Value(bool)` swap guard (0265's `head_scored_noted` pattern), at the arm's
run site (`windows.copyProposal`, after `propose`, before the gate):

```
dsv41: TF_DS_COPY_MIN: a copy takes the round at <N> tokens or more (first at stream <id>, <k>-token proposal)
```

Engagement = the first time the configurable gate evaluates a proposal while the switch is set — including an empty
or rejected proposal — so a run with the switch set always prints exactly one banner even when the setting changes
nothing (a null result with no banner cannot happen). With `TF_DS_NO_COPIES=1` the arm never runs (`copyAllowed`
returns before the gate), so no banner prints — by design. Unset: no banner.

## Launch counts at the served shape (TF_DS_DRAFT_BLOCK=6 TF_DS_DRAFT_BLOCK_KEPT=2, graphs on, TP2, ≤4 streams)

| | unset (today) | n ≥ 3 |
| --- | ---: | ---: |
| new/removed launch sites | 0 | 0 |
| per-round backend calls | verify 1 + keep (unconditional in shared rounds, conditional in lone ones) + draft 1 per drafted stream | identical |

The switch adds no kernel and changes no launch. The round loop's backend sequence is plan-kind independent: the
head's draft pass runs every round for every drafted stream whether the round's window is a copy or head window
(`roundStreams` heads = all drafted streams; `familyRound` likewise), `probabilitiesFn` is host-only (survivals from
`l.confs`), and both window kinds enter the same verify. What changes with n is the plan mix — a round whose
2-token copy is refused carries the queued head drafts instead — i.e. window row contents (≤7 rows), not launch
counts. At the served shape, on repetitive text, n=3 therefore buys nothing per launch; it changes what each round
verifies (the source's throughput claim is about accepted tokens per round, hardware-measured).

## CPU-emulation harness (`runs/kernel-round-20261122d/h0275/`)

`build.sh` extracts `copy_min.zig` + `proposer.zig` from the worktree and runs `zig test copy_min.zig` (3/3 pass)
and `zig run harness.zig` (exit 0; `run.log`). The emulation side is the real `SuffixLookup.propose` (served params:
`min_match = 4`, `max_extension = 64`, `enter_match = 8`, `max_draft = 6`) plus the real gate (`copy_min.takes`); the
independent reference is written from the brief's spec (brute force over every earlier match end, latest-ties-win,
`n orelse 2` + `enter_match` gate — no n-gram index, not copied from the kernel). Sequential host execution, no
threads. Coverage: 14 constructed cases (2/3/4/5/6/1-token copies via periodic sequences, a word-repeat prose
sequence, a strictly ascending non-repetitive sequence, match-7/match-8/match-9 anchors for the `enter_match`
boundary, a short context, `max_draft` 2 and 0 caps) × every growing prefix × n ∈ {unset, 2, 3, 4, 5, 6, 7}:

- proposals and match lengths match the reference on every prefix (both the incremental-index and the
  index-rebuild paths agree), 0 mismatches;
- gate decisions match the reference for every n;
- **unset == today** (`len >= 2`) holds at every prefix;
- the parse matrix (12 rows: unset/`""`/`"0"`→2, `"3"`→3, `"7"`→7, `"1"`/`"8"`/`"65"`/`"-1"`/`"3.0"`/`"abc"` refuse)
  behaves as designed.

The end-of-run table shows the n=2→n=3 mapping directly: exactly the 2-token-copy cases flip from "copy round" to
"head round".

## Gate results

1. `zig build test` — **exit 0**. (Run 1 of 8 failed once in one test binary without capturing the name before the
   rerun; the brief's known flake `lane_host.test.prefill` is the only known intermittent here and the patch touches
   nothing in its path. Runs 2–8 all clean, including the final recorded gate run.)
2. `cuda-syntax-stubs/check.sh` — base `27baa93`: 9 errors (the known stub false positives, `float4` etc.);
   this tree: the **same 9 errors**, same messages, same line numbers (no `.cu` file is touched);
   normalized error-set diff: identical.
3. Harness `h0275/` — **exit 0**, `run.log` kept with `build.sh` and sources.

## TF_DS_NO_COPIES / TF_DS_SOLO_OVERHEAD (the brief's question)

Both exist in our tree (`zig/src/families/dsv41/lanes.zig` `Lanes.no_copies` / `Lanes.solo_overhead`, parsed in
`native.zig` `open`).

- **`TF_DS_NO_COPIES=1` dominates:** `facts()` sets `copy_proposals = false`, `copyAllowed` returns before the
  threshold is consulted, so `TF_DS_COPY_MIN` is inert and its banner never fires. Set only one of the two.
- **`TF_DS_SOLO_OVERHEAD=1` is orthogonal:** it only prices a lone drafted stream's head windows with the fixed
  draft allowance in the allocator (`solo_head_overhead_ms`). No code interaction with the gate. Indirect effect:
  raising n moves rounds from copy windows to head windows, and under `TF_DS_SOLO_OVERHEAD` those head rounds are
  priced with the fixed allowance — cost accounting shifts as the plan mix shifts, nothing more.

## Design decisions (decision log)

- **Parse range 2..=7, `"0"`/unset/`""` off, `"1"` refused.** The brief says "integer within a sane range, else
  refuse startup"; the range bottoms at today's minimum (the source clamps < 2 to 2, so < 2 has no meaning) and tops
  at the widest a copy can be in this build (`window_rows − 1 = 7`); above that no proposal can ever qualify and
  `TF_DS_NO_COPIES=1` is the honest off switch. `"0"` keeps the switch convention's off value.
- **`?u32` through Model/Config (null = unset)** rather than a `u32` + clamp: it preserves set-ness for the banner
  rule ("banner when set") and matches `solo_head_overhead_ms`'s optional idiom.
- **Parse in `native.zig` `open`, not `ops.zig`.** This switch has no kernel (the convention's
  "bool + `?cuda.Function`" does not apply, as 0265's report also noted for its kernel-less switch) and its value
  flows through `lanes.Model`/`Config` exactly like the source patch and exactly like its siblings `TF_DS_NO_COPIES`
  and `TF_DS_SOLO_OVERHEAD`, which parse at the same spot.
- **Banner in `core/lanes/windows.zig`** (the arm's run site). Core logs nowhere else, but the arm lives there and
  the rule asks for the banner when the arm first runs; the line only prints when dsv41 armed the switch
  (`cfg.copy_min != null`).
- The brief's `zig/src/families/dsv41/windows.zig` path does not exist in our tree (see Premise); the change was
  re-derived against `27baa93` as instructed.

## Known gaps (hardware-only or out of scope)

- **Output bit-identity on the on path** rests on the verify forward's per-row logits being row-count invariant
  (GPU-side rounding cannot be checked here). The sampler argument is from code and the accept rule is unchanged;
  the exact panel on hardware settles the rest.
- **The throughput premise** (78.4→89.8 tok/s geomean at n=3 on repetitive text) is only measurable on hardware,
  on their quant; locally we show which proposals flip, not what the flips earn.
- **Banner count** on hardware proves the arm ran; at the served shape a2-token copy must occur in the workload for
  n=3 to change anything (the banner fires regardless, on first gate evaluation).
- `tf-dsv41-lanes` (`lanes_main.zig`) does not read `TF_DS_COPY_MIN` (like it reads none of the sibling switches);
  only the served `native.zig` path arms the switch.
