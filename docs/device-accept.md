# Device accept for greedy chain windows (TF_DS_DEVICE_ACCEPT)

Status: implemented on branch `decode-step-structure` (patch 0124), **off by default**. Not measured on the pair yet;
not committed here — this note rides along untracked.

Switch: `TF_DS_DEVICE_ACCEPT=1` (or `on`; anything else or unset: off). Every rank reads it the same, but an uneven
pair is still bit-identical: the switch changes only which side compares, and rank 1's copy never reads the keep.

## What Baseten does

In [basetenlabs/deepseek-ultrafast](https://github.com/basetenlabs/deepseek-ultrafast) at `76766a9`:

- The accept is a CUDA kernel, `spec_accept` (`crates/kernels/cu/ops.cu:591`): it walks the chain of drafted tokens
  and keeps the longest prefix whose tokens match the target's tokens, on the device, so no host round trip happens
  between verify and the next draft.
- Draft, verify, accept and the next step's setup are one captured CUDA graph (`capture_spec_blk`,
  `crates/engine/src/model.rs:3956`); the server launches steps ahead and reads the committed tokens from a snapshot
  two steps back (`crates/server/src/engine.rs:320-345`).
- With the confidence cut, the graph picks its verify depth itself through a conditional node
  (`capture_spec_vadapt`, `crates/engine/src/model.rs:4004`).

## What this engine does today (the base this patch extends)

A round on rank 0's serving lane is two primitives with host state between them:

1. `verifyFn` (`zig/src/families/dsv41/lanes.zig`) builds the windows (the rows of the pending token plus the held
   drafts), then `m.verify(rows, ...)` runs the forward. For a greedy/device-draw round it calls `m.roundArgmax(R)`
   (`zig/src/families/dsv41/model.zig`): a kernel (`ops.argmaxRows`, or `drawRows` under `TF_DS_T27_DEVICE_DRAW`)
   writes the per-row picks into `m.amax`, a sync follows, and `R` u32s are memcpy'd D2H into `m.amax_host`.
2. The engine core (`zig/src/core/lanes/engine.zig` `conclude`) reads the picks and runs the accept rule on the
   host: `accept.acceptPath` (`zig/src/core/lanes/accept.zig`) walks the window's rows in row order and keeps the
   longest first-matching-child prefix. For a chain window (`parents == null`) the path is always the identity
   prefix `0..keep`, where `keep` is the longest prefix r ≥ 1 with `tokens[r] == picks[r - 1]` (keep ∈ [1, rows]).
3. Only then is the draft pass issued (`draftFn`). So between verify and the pass there is a sync plus a D2H of the
   picks, all of `conclude` on the host, and — on rank 0 only — the round profile attributes none of it separately
   from `sample_ms`.

Trees and the full first-match rule stay on the host. The forced windows never take this path.

## What this patch changes

For a round in which **every** window is a chain (`Window.parents == null`, i.e. the ordinary greedy/device-draw
head windows) and the picks come from the device (argmax or device draw — never a `host_draw` round, which has no
picks kernel):

1. `Lanes` reads `TF_DS_DEVICE_ACCEPT` at init (every rank alike) into `accept_chain`.
2. `verifyFn`, after the path decision and before the pick, writes one u32 row0 and one u32 row count per window
   into a device tab (`Model.setAccept`, an HtoD copy into the arena slot `m.accept_tab`) and sets `m.accept_rows`.
3. `roundArgmax`, right after the pick kernel, launches one more kernel, `tf_ds_accept_chain_kernel`
   (`zig/kernels/cuda/dsv41_ops.cu`): one block per window (32 threads; R ≤ 16, single-threaded walks, deliberately
   un-parallelized) computes exactly the chain rule above into the u32s right after the picks in `m.amax`
   (keeps land at `amax + R * 4`).
4. The same D2H then carries `R + nwin` u32s; `verifyFn` stores each window's keep into the matching `Verified`'s
   `device_keep` (`zig/src/core/lanes/backend.zig`).
5. `conclude` takes `device_keep`; for a non-forced window with an in-range keep and a chain parent array it uses
   the identity prefix instead of calling `acceptPath`. Out of range, a tree parent array, or a absent keep: the
   host rule runs, unchanged. Bytes on the wire are identical either way — the device computes the same integer
   the host would.

The `amax` slot and `amax_host` grew by `max_streams` (16) keep slots, not by the design sketch's +4: a shared
round can carry up to `max_streams` windows (48 rows + 16 keeps = 256 B, one small arena take; `arena.take` is a
bump allocator with no guard, so the +4 sizing would have let the kernel write past the slot on wide rounds). The
memcpy stays `(R + nwin) * 4`; off rounds memcpy exactly `R * 4` as before, and an unset switch launches no kernel.

## Why the one-frame verify+pass stretch still does not exist here

Baseten's one-graph step works because the accept kernel feeds the next draft directly. The same is **not** built
or claimed here; this patch is the foundational piece (the accept's keep, computed on the device), not the stretch:

- The depth rule (`headDepth`) is rank 0's host state: it prices each stream's next draft depth from measured costs.
- The cuts (the thinking budget, the loop guard, the call gate) are `Stream` host state evaluated on the committed
  tokens inside `conclude`; when one fires, the window's path is cut and the draft pass must see the cut.
- The T26 keyed-draft keys exist only inside the pass frame; nothing about them survives a frame boundary.
- Trees would need the parent array uploaded (chains need only row0/count); deliberately out of scope.
- Baseten reads tokens from a snapshot two steps back, which maps onto Baseten's one-stream loop, not onto this
  engine's two-frame round (verify frame → forward → host `conclude` → pass frame → drafter pass) with rank 1 a
  follower of every frame. The depth and cut decisions are what force the host round trip; `spec_accept` alone does
  not remove it.

## What it does NOT do

- No D2H elision: the picks and the keeps are still read back every round (`m.amax_host`); the host still decides.
- No trees: a tree window (`parents != null`) sends every row's parent upload through a different shape; those
  rounds keep the host rule, even with the switch on.
- No one-graph / one-frame round: the depth rule, the cuts and the T26 keys stay host-side between the two frames.
- No rank-1 accept: rank 1 never decides; the keep is a rank-0 host input. An uneven pair is bit-identical.
- No picks-D2H of the keeps when the switch is off: unset, nothing changes (no launch, no memcpy of the extra
  words, replies byte for byte the base's).

## Gates on the pair

1. `tools/exact.py` IDENTICAL expected: the device keep equals the host keep by construction (same integer), and
   the switch changes only which side compares. If it is not identical, the kernel is wrong — do not ship it.
2. `tf-dsv41-lanes --profile` on rank 0 (and the served process's `TF_DS_ROUND_PROFILE` line for `send_ms`,
   `logits_ms`, `sample_ms` shares of `step_ms`, from 0123): this patch does not, by itself, move those shares much —
   it removes the host `acceptPath` walk, not the sync. If the profile shows `sample_ms`/`logits_ms` small, stop:
   the full stretch is not worth it. If they are large, the remaining levers are the ones in "What it does NOT do".
3. Hardware smoke (one chat completion on the pair with `TF_DS_DEVICE_ACCEPT=1` on both ranks) is still owed —
   this host cannot run the CUDA path; the image's nvcc build is the first real compile of the new kernel.

## The check

There is no `TF_DS_DEVICE_ACCEPT_CHECK` env (an earlier sketch had one; it was dropped). Equivalence is covered by
(a) host tests: `accept.zig`'s property test pins a chain's `acceptPath` to the identity prefix over the matching
drafts (all 0/1 token windows up to 6 rows), and `engine.zig`'s `chainPath` test pins the identity prefix by
length; (b) `conclude`'s fallback to `acceptPath` on any out-of-range keep or non-chain parent array; (c) the
pair gate (`exact.py`) for the bits. If the device kernel is wrong on the pair, `exact.py` catches it.
