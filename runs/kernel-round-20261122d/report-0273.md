# report-0273 — two bug-fix ports: the failed-stretch capture and the tool-call gate

**Commits** (branch `kernel-0273`, base `27baa93` = recipe-0264, two commits in the brief's order):

| # | Hash | Subject |
| --- | --- | --- |
| 0273 | `7c8fee275ea1158532878dc4af2efe8e5c7e0c05` | `0273 dsv41: a failed stretch capture joins its forks and owes the next capture no join` |
| 0274 | `0dfeaaa35961414bc51c4bd71201b7c61fc665c5` | `0274 core: the tool-call gate commits the drawn token as the first token` |

**Switch name:** none — both are bug-fix ports (BRIEF-common: no switch; the contract is "every reply that does
not hit the fixed case keeps its bits"). **Launch counts before/after at the served shape:** unchanged. 0273 adds
waits and an `endCapture` only on a path that already returned an error from a dead capture (zero launches on any
successful stretch; the off/success path is byte- and launch-identical). 0274 is host-only. Working tree clean
after both commits.

This report covers both brief items (BRIEF-0273.md is the two-port brief; there is no BRIEF-0274.md, so no
separate report-0274.md is written — the 0274 section below is self-contained).

---

## 0273 — a failed stretch capture joins its forks and owes the next capture no join

### Premise: holds

`zig/src/families/dsv41/round.zig` `Graphs.run` (was :1905, `git show 7c8fee2^:...`) ended the dead capture and
freed the graph but never cleared `rd.side_out` (set in `mix`, :2141-2143) or `rd.pf_out` (set in `prefetch`, :865).
The next capture's first `mix` calls `rd.joinSide` (:2140) and waits `rd.join` — an event recorded only in the dead
capture (a captured record never executes when the graph is discarded). CUDA refuses that wait (INVALID_VALUE) and
invalidates the capture (901 on everything after), and the refused wait leaves the flag set: every later capture
fails until a restart, while graphs already captured keep replaying (the recipe's issue #65; also in our findings
register: "Bug confirmed in our tree ... round.zig:1905"). The harness's negative control reproduces the cascade
against the capture model, and `tf-cuda-test capture-fork` (ported) reproduces it on a GPU.

### What the fix does (the failure path only)

`Graphs.run`'s catch now logs a warning and calls `leaveCapture(StretchCapture{ ... })` (new, next to `Graphs`):

1. **joins what the stretch forked** — every flagged fork, in fork order: the mixes' side work (`joinSide`), the
   paced prefetch (`joinPrefetch`), `TF_DS_COMPRESS_SIDE`'s chain (`cs_join`), and the KV-shard overlap stream's
   early query stage (`ss_jq`), whatever each call returns;
2. **ends the capture** and frees the graph it returns (the captured work never ran; the round is already failing);
3. **clears every fork flag** (`side_out`, `pf_out`, `cs_out`, `ss_q_out`) whatever the calls returned — a stretch
   starts with nothing forked, so every flag can only refer to this capture;
4. **logs a warning** (the brief's bug-fix banner rule), once per failed capture, with the error name:
   `dsv41: a stretch failed inside its capture (<err>): joined what it forked, ended the capture, cleared the fork
   flags; the next capture owes no join`.

### Every fork flag set inside `stretch`, and its fate

Line numbers refer to the pre-fix tree (`27baa93`; they drift below the 0273 insertions in the committed tree).

| Flag / event | Set at | Joined at | Fate in the failure path |
| --- | --- | --- | --- |
| `side_out` (`rd.join`) | `mix`'s defer (:2141) | next `mix` (:2140) and the stretch tail (:2096) | joined + cleared (the source patch's pair) |
| `pf_out` (`pf_join`) | `prefetch` (:865) | the stretch tail (:2097) | joined + cleared (the source patch's pair) |
| `cs_out` (`cs_join`) | the compressor chain's fork at `a_in` (:1173-1178) | `_kv_idx`'s read (:1237-1240) | joined + cleared — **our tree's flag, the source tree lacks it**. Stale `cs_out` is safe by ordering alone (the only wait it gates is always behind its fork's fresh record in the same layer), but clearing keeps the round's state honest |
| `ss_q_out` (`ss_jq`) | `sendQueries`' early stage (:1543-1547) | `ownerRing` (:1660-1662) | joined + cleared. The source patch relies on `forward`'s errdefer (:1966-1969, still present) clearing it; the fix clears it in both places, harmlessly |
| `ss_jo` (own pass, **no flag**) | `ownerRing`'s fork (:1655-1658) | `ownerRing` (:1674), unconditional | **not joined (safe)**: its wait is always preceded by its record in the same `ownerRing` call (`rd.ss` is fixed after open), so no stale event can be waited across captures. A failed capture with it open ends with one 904 (accepted; documented in `leaveCapture`'s comment). A ring send left open makes every later ring transfer refuse (`ShardRingOrder`), which ends the pair |
| `rd.sink.fork` (Sinkhorn/dots) | internal to `hcPre2`'s side protocol | covered by `side_out`: `mix`'s defer records `rd.join` on `rd.side` after the side work | via `side_out` |
| `PhaseTimer` events | `rd.mark` records on `e.s` itself | n/a (same stream, eager profiles only) | not a fork; nothing to clear |

Ring sends (`rd.rcur` / Ring's `RowPairing`) are not fork flags: `forward`'s errdefer resets the cursor, and a sized
send left open already ends the pair by design.

### Tests (the brief's "fake capture plus the flags on a real Round")

- `round.zig`, `test "a failed stretch joins what it forked, ends its capture, and owes the next capture no join"`:
  a fake capture asserts the call order (every join before the end, the graph freed, the clear last), that a failed
  join stops nothing after it (each of the four joins, and the end), that an invalidated capture's refused calls
  still clear the flags, and the four flags cleared on a real `Round`. **Mutation-checked:** commenting out
  `cap.forgetForked()` makes exactly this test fail (evidence in the run record).
- `tf-cuda-test capture-fork` (GPU, optional port from `joseyarashio-0062`): a capture with work forked to a second
  stream (as the round's side/prefetch streams), an unjoined end (904 refused, yet the next capture works), and the
  dead-capture join wait (refused every time, 901 cascade, while captured graphs still replay). Ported because it
  fits our `tf-cuda-test` cleanly (every API checked against our `cuda` module; `zig build` compiles it). It is
  compile-checked only here (no GPU): run it at the hardware gate.

### CPU-emulation harness (`runs/kernel-round-20261122d/h0273/`)

Neither patch has a `.cu` kernel, so the harness splices the fix's **Zig** source instead (the `splice.py` trick
adapted: `build.sh` extracts `StretchCapture` + `leaveCapture` verbatim from round.zig into
`leave_capture.inc.zig`, concatenated after `capture_harness.zig` into `gen_capture_run.zig`), executes it against
an independent CPU model of CUDA stream-capture semantics (capture-local event dependencies, INVALID_VALUE + 901
invalidation on a dead-capture wait, STREAM_CAPTURE_UNJOINED/904 on an unjoined end, a freed graph never running),
and compares it to a reference written from BRIEF-0273's prose only (`refTrace`/`refFlags`).

Case matrix — 7 fork states (`none`, each of the four alone, `side+prefetch`, all four) × 7 outcome sets (all calls
ok, each join failing, the end failing, everything failing) = 49 cases: **all match the reference** (joins in fork
order, the end, the free when it returns a graph), all flags clear in every case, and the next capture completes
with zero refusals and zero 904s in every case. Plus: the negative control (today's path leaves `side_out` set and
poisons every later capture — reproduced), the unflagged `ss_jo` case (one 904 in the dead capture, yet the next
capture's record-then-wait is clean), and the call-order robustness matrix. `run.log` holds all of it.

## 0274 — the call gate commits the drawn token as the first token

### Premise: holds, reproduced on our toolchain

`zig/src/core/lanes/engine.zig` `started` (was :195) read the first token through the same `feed` it assigned:

    if (s.gate != null and forced == null) feed = .{ .value = try e.correctCall(s, try e.readFeed(feed)) };

Zig builds the literal at `feed`'s address, so the tag becomes `.value` before `readFeed(feed)` runs and the read
takes the union's payload (the draw handle, `Feed = union(enum) { handle: u64, value: u32 }`) as the value.
**Reproduced exactly** on our zig 0.17 with the shipped test: `engine_test.zig "a disarmed gate commits the drawn
first token"` failed before the fix with `[0]` (the fake backend's first draw handle) where `[68]` (the drawn token)
was owed — the same handle-as-token mechanism the source observed live (506, 607, … over 600 requests). The fix:
read into a temporary first, then assign. `h0273/feed_gate.zig` models the two statement shapes against the brief's
contract and prints the same verdict independently.

### Which request shapes reach the changed line

The line runs at `started` — a prompt pass ended and its first token is drawn — when the stream has a call gate
(`tool_choice: required`) **and** nothing is forced at the first draw (`forcedNext` empty: no thinking-budget close,
no score/fix force):

- **thinking on** (the prompt ends inside a think block, so `Gate.afterPrompt` starts it disarmed; `cut` leaves the
  token alone): the **fixed case** — the drawn token is now committed (the handle was). This is the only shape whose
  output changes.
- **thinking off / armed gate**: reaches the line but is bit-identical before and after — `correctCall`'s `cut`
  fixes the first token to the call opener over either value (proven by `test "an armed gate still forces the
  opener"`, which passed on the buggy line too).
- `tool_choice: auto`/none: no gate, the line is not reached. Forced-first requests (score streams, a close that
  starts at the first draw): `forced != null`, the line is not reached.

Everything else keeps its bits (the fix changes evaluation order only, never a value computed).

### Tests

`engine_test.zig`: `test "a disarmed gate commits the drawn first token, not the backend's handle"` (fails on the
old line — the premise reproduction) and `test "an armed gate still forces the opener"` (the unchanged-shape guard).
Both run in `zig build test` on the fake backend.

## Local gate (all on the committed tree)

| Gate | Result |
| --- | --- |
| `/home/colum/.local/zig-0.17.0/zig build test` | exit 0 on every run after the first (the very first baseline run, before any edit, failed once — the failing test's name was not captured; the brief's known flake is `lane_host.test.prefill` interleave — rerun green) |
| `cuda-syntax-stubs/check.sh zig/kernels/cuda` | same 9-error set as base `27baa93` (stub false positives, identical modulo line numbers) |
| `h0273/build.sh` (CPU-emulation harness) | all cases matched their references (49-case matrix, negative control, `ss_jo` case, call-order matrix, feed-gate shapes); `run.log` kept |
| `zig build` (default, compiles `tf-cuda-test`) | exit 0 — the ported `capture-fork` compiles |
| mutation check (`forgetForked` removed) | the new round.zig test fails as designed |

Incidental flake seen once during the mutation run (not reproducible on 4 clean runs): `rank_wire.test.rank 0 and a
follower agree, restore, and fail closed over a loopback link` — loopback timing, unrelated to these patches.

## Design decisions (the round's decision log)

1. **Two commits, one report.** BRIEF-0273.md is one brief with two ports ("one agent, two commits, in this
   order"); the user instruction names report-0273.md. Summary: 0273 and 0274 land as separate commits with the
   brief's exact subjects; this file reports both; no report-0274.md exists because no BRIEF-0274.md does.
2. **The failure path joins and clears all four flagged forks, not the source's two.** The source tree lacks
   `cs_out`; leaving it stale is safe-by-ordering but a foot-gun, and a `cs`/`ss_q` fork left unjoined would still
   end the capture with 904 — the noise the fix exists to remove. `ss_q_out`'s clear duplicates `forward`'s
   errdefer harmlessly. Rejected alternative: source-verbatim two joins (smaller diff, leaves our tree's forks
   unjoined and the audit as documentation only).
3. **The warning lives in `Graphs.run`'s catch**, not inside `leaveCapture` — the catch is the fixed path's entry
   and holds the error name; `leaveCapture` stays pure and quiet under test.
4. **GPU `capture-fork` ported** (brief marks it optional): it fits our `tf-cuda-test` cleanly and is the only
   direct evidence of the 901/904 mechanics; compile-checked here, runnable at the hardware gate.
5. **Harness shape:** no kernel to splice in these ports, so `h0273` splices the fix's Zig source (adapted splice
   trick) and models CUDA capture semantics; the independent references are written from the briefs' prose. The
   real 0274 regression tests live in `engine_test.zig` (in the gate), with `feed_gate.zig` as the standalone
   evaluation-order probe.
6. **No shared-file writes** (`continue.md`, `todo.md`, `decisions.md` live in the shared Sparks dir/worktree and
   agents 0275/0276 are running concurrently; the worktree must end clean). Follow-ups are recorded below for the
   round driver to merge.

## Known gaps (hardware gate only)

- **`tf-cuda-test capture-fork` runs on a GPU only.** The 901/904 cascade and its recovery are modeled and
  documented, not GPU-verified here. The new warning line's production observability (that it appears when a
  capture fails and is countable) is also a hardware item.
- **Bit-identity:** 0273 touches only a path that already returned an error (no reply can reach it); 0274 changes
  only `tool_choice: required` first-token commits, and only where the old code committed a handle. Both are
  host-side; no GPU rounding points are involved, so bit-identity is established locally by reading + tests.
- **Eager-stretch failures** (probe/eager rounds) leave `side_out`/`pf_out`/`cs_out` set with the events recorded
  on real streams; a later capture then fails **once** (its first mix waits an uncaptured event inside the capture)
  and self-heals through `leaveCapture`. Pre-existing and outside "only the failure path changes"; a full fix would
  join (not just clear) in `forward`'s errdefer. Follow-up candidate.
- Both ranks behave identically (no rank-conditional code); whether one rank's failed capture alone (its peer
  healthy) trips `pair_fatal` is unchanged behavior the hardware gate observes.

## Files touched

- `zig/src/families/dsv41/round.zig` (0273: `Graphs.run`'s catch, `StretchCapture` + `leaveCapture`, the host test)
- `zig/tests/cuda/main.zig`, `zig/tests/cuda/runtime_tests.zig` (0273: the optional `capture-fork` GPU test)
- `zig/src/core/lanes/engine.zig`, `zig/src/core/lanes/engine_test.zig` (0274: the one statement + two tests)
- Harness (outside the repo): `runs/kernel-round-20261122d/h0273/{build.sh, capture_harness.zig, feed_gate.zig,
  leave_capture.inc.zig, gen_capture_run.zig, run.log}`
