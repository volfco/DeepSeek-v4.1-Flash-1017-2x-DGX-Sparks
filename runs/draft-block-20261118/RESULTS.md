# A larger draft block (`TF_DS_DRAFT_BLOCK`), November 18

**Result.** Letting the drafter offer six or seven drafts a round, instead of the checkpoint's five, raises tokens per
round on Baseten's Tetris prompt from 4.76 to 5.13–5.38 (Baseten reports 5.6). Throughput rises 2–3% on that prompt and
up to 9.5% on one code prompt, at one or two streams only. Replies are unchanged. Production has run the refined
version at block 6 since 22:21 UTC ([adoption](#adoption-refined-6-in-production)).

## Why the block

Baseten's 5.6 tokens a step is measured on a GPU profile that overrides the checkpoint's draft block to 6
(`profile::dspark_block()` returns 6 for `Sm103Tp4`/`Sm103Tp8`, applied in `load_config`; their 30-run range is 5.1–6.1,
which a five-draft block cannot reach). The README does not state the block size; this is read from their source.
This engine ran the checkpoint's own block of five, so a round delivers six tokens at most. On Tetris 61% of rounds
already delivered six. [Per-round counts on production](tetris-rounds-prod.json)

## The switch

[`draft-block.patch`](draft-block.patch) applies to the pinned image's source (`TF_REF` plus the 117 patches in the
image). It is off by default.

- `TF_DS_DRAFT_BLOCK=6` or `7` lets a drafter pass run that many block rows; the lane window is one row wider. A pass
  asked for five drafts or fewer runs exactly what it ran before.
- A stream's pass is widened only after a round that kept at least `TF_DS_DRAFT_BLOCK_KEPT` rows (default 4) and
  only when a round has room for the extra drafts. `TF_DS_DRAFT_BLOCK_KEPT=0` widens every pass.
- The kernel kit has two drafter kernels compiled only for 5, 10 and 15 rows. At other row counts the expert gate
  takes the cuBLAS path that passes over 16 rows already use, and the stream mix takes a tile size the kit has.
- Both ranks must agree on the block; a mismatch refuses the start.

Host tests of the measured build, clean cache: 758 of 760 pass. The two failures (`prompt_cache` 0178, `round` hot-pick) also fail on the
unpatched pinned source, which reads 757 of 759; the extra test is the switch's parser.

Implemented but never exercised on the pair:

- The refusal when the two ranks are given different blocks.
- `TF_DS_DRAFT_BLOCK_KEPT=0` on the final image. It is still bounded by room in the round, so at four streams it is
  not the first version.
- The measured builds had no host test for the held-draft bug described below; GPU validation found it. The adopted
  patch has one ([adoption](#adoption-refined-6-in-production)).

## Measurements

One image per version, both ranks, private spill directories, production's `.env` otherwise. Each arm is a fresh
process. Five switch-off arms across the three images agree within ±0.3% on every cell and produce identical hashes.
Single-stream rates are the engine's own per-request figure. Reproduce the table with [`summary.py`](summary.py).

| Cell (thinking off) | Control | Always wide, 6 | Always wide, 7 | Refined, 6 | Refined, 7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tetris greedy, tokens a round | 4.76 | 5.17 | 5.38 | 5.13 | 5.32 |
| Tetris sampled, tokens a round (3 seeds) | 4.77 | 5.14 | 5.35 | 5.09 | 5.22 |
| Tetris greedy, tok/s (2 passes) | 118.0 | +1.1 to +1.5% | +0.6 to +0.9% | +2.5 to +2.7% | +2.2% |
| Tetris sampled, tok/s (3 seeds) | 112–120 | −0.4 to +3.5% | −0.5 to +1.4% | +0.4 to +2.7% | −0.7 to +1.2% |
| Python module, tok/s | 114.1 | +5.5% | +9.5% | +3.1% | +9.5% |
| JSON array, tok/s | 100.5 | +7.2% | +3.9% | +2.5% | +1.9% |
| Prose, tok/s (3 runs) | 60–80 | −0.9 to −1.5% | −2.7 to +0.2% | 0.0 to +0.1% | −0.2 to +0.1% |
| Tetris sampled × 2, aggregate tok/s | 159.6 | +2.3 to +2.4% | −0.3 to −0.2% | +2.0% | −0.2 to −0.1% |
| Tetris sampled × 4 | 208.3 | −2.8 to −2.7% | −2.3 to −2.1% | −0.3 to +0.8% | −0.5 to +1.0% |
| Tetris sampled × 8 | 216.3 | 0.0% | −0.5% | −0.3% | +0.3% |

"Always wide" is the first version ([`draft-block-v1.patch`](draft-block-v1.patch), image
`bench-draft-block-6a718995cdee`); "refined" is the final patch (image `bench-draft-block-0fd3edcba57d`). The always-wide losses on prose and at four streams come from running the wider
block in rounds that cannot use it; the refinement removes them and gives up part of the JSON and code gain at 6.

Gates, for all four switched-on arms: all 15 probe hashes equal to control; `validate.py` 8 of 8 saved replies and
8 of 8 concurrent-to-solo; `exact.py` IDENTICAL on 39 cases. The probe hashes at one and two streams are the
evidence for the wide path. In the refined build a pass widens only there and only after a good round, so the
39 cases and the eight-way validation rarely or never enter it; read those two as no regression.
[Status](status.log) · [First run](status-v1.log)

The pair served benchmark binaries with empty private caches from 18:16 to 20:35 UTC, apart from 19:05–19:12
and 19:56–20:04 when production was back between runs. Every check before a restart found no connected clients and
only loopback requests.

## Limits

- One prompt per class, thinking off, short context, one or two repetitions. The default threshold of 4 was chosen
  on these same traces.
- At four or more streams a round's 16 rows leave no room for a sixth draft, so the switch cannot help there. On
  eight code streams 98% of drafts are accepted with one draft a stream.
- The remaining gap to Baseten at an equal block (5.17 against 5.6) is unexplained. Candidates are this pair's
  2.9 bpw target with expert pruning, and their sampling defaults.
- A lower threshold was not measured: the arm for it ran on a build with a held-draft bug (below).

## What went wrong on the way

The second build narrowed a stream's pass but left the stream holding fewer drafts than the round loop had asked
for. Code-like requests then failed mid-reply with the switch on, and `validate.py` caught it (6 of 8) while
`exact.py` still read IDENTICAL. The final patch keeps every asked draft held and gives the ones the pass did not
run a zero chance. That image (`bench-draft-block-fb3709799501`) is deleted from both ranks.
[Failed run](status-v2.log) · [Superseded patch](draft-block-v2-heldmismatch.patch)

## Adoption: refined 6 in production

- **Testable fix.** The narrowing rule and the held drafts moved into two pure functions that the pass calls, with
  host tests: the held count equals the asked count, a draft the pass did not run has a survival of exactly 0, and the
  round loop's allocation never grants one. With the earlier bug put back the new test fails. Clean-cache host suite:
  760 of 762 on the pinned series (the same two known failures) and 772 of 772 on this checkout's 128-patch series.
- **Gate.** One block-6 arm on the rebuilt binaries before production changed: validation 8 of 8, the 39 cases
  identical, and all 15 probe rows equal to the earlier refined-6 arm in rounds, accepted drafts and hash.
  [Gate status](status.log) · [Probe](e6/probe.log)
- **Promotion.** The patch joined the head's `zig/patches` (as `0192`, the name it has in this checkout; it applies
  last either way), which moved the recipe's image hash from
  `234f9080b326` to `ea522d353626`. The gated executables were tagged as that image on both ranks, with the patch in
  `/opt/tf-patches` and the hash as the label. The executables' hashes equal the gated ones, both ranks hold the same
  image, and the weight-file tool is the pinned image's. [Checks](promote/checks.txt) · [Script](promote.sh)
- **Restart, 22:21 UTC.** `TF_DS_DRAFT_BLOCK=6` was added to `.env` and production restarted with no image or setup
  override. Setup found the image built and every file ready. The first request failed as it does after every start;
  one retry answered. Validation passed 8 of 8 both ways, and the probe reproduced the gated arm on all 15 rows.
  [Start log](deploy/restart.log) · [Validation](deploy/validate.log) · [Probe](deploy/probe.log)
- **Depth.** Every wide round before this came from prompts of a few dozen tokens. On production, a 111,101-token
  prompt asking for a 3.2 KB excerpt back ran 112 rounds for 768 tokens alone, 108 of them at seven tokens
  (137.7 tok/s). Four copies of the same request at once, which stay on the five-draft block, gave the same output
  hash as the solo run. [Depth check](deploy/depth-check.json)
- **Spill.** The new image label is a new spill identity: production started with an empty namespace. The earlier
  namespace is still on disk and not read: 2.3 GB on the head, measured through a container, against the 2.1 GB the
  tier reported for its 71 points. A plain `du` before the restart read 13G for the directory; that difference is
  not explained.

The image's executables are a workstation cross-build layered over the pinned image, not a from-source build by the
recipe's Dockerfile. `./start.sh` with `--rebuild` would build the same source on the head; that build has not been
run or gated.

## State left behind

- Production: image `exl3-tp2-ea522d353626`, `TF_DS_DRAFT_BLOCK=6` and the token cache on both ranks.
- To turn the switch off, remove the `.env` line and restart; the image can stay. Switch-off was not run on the
  pair with this exact binary. To return to the pinned image `exl3-tp2-234f9080b326`, which is still on both ranks,
  also remove `zig/patches/0192-…` on the head.
- Not run: a from-source build of this recipe on the head, and the refusal when the ranks get different blocks.
- Both ranks keep the always-wide benchmark image and the gated one the production image is built from. The
  experiment's spill directories and package folder are removed.
- Engine source for the patch: `~/Workspace/tfref-draftblock`, branch `draft-block` over `pinned`. This checkout
  carries the patch as `zig/patches/0192`.
