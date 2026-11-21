# Draft ladder (5/6/7), November 19

**Result.** The ladder is built and not worth adopting. On 21 code-like prompts it decodes 0.4% faster than
production's six-draft block, with a spread of ±0.5% across prompts, and plain block 7 does exactly the same. The
six-draft block itself is confirmed: 3.2% faster than five, slower on none of the 21 prompts.

Every reply the ladder produced equals production's: validation 8 of 8 both ways and 25 of 25 probe replies. The
39-case exact panel was not run on it and is owed if it is ever adopted.

Nothing was deployed. Production ran test arms from 05:25 to 05:48 and from 05:51 to 06:17 UTC and is back on its
image and `.env` on both ranks.

## What was built

`TF_DS_DRAFT_LADDER` (rank 0, off by default, under `TF_DS_DRAFT_BLOCK`): the block's last draft is offered to a
round only when this pass's survival through the drafts before it, times the stream's running share of last drafts
that were right once the earlier ones had landed, reaches a bar (default 0.43). The share is scored from the next
round's own tokens, so it needs no trial rounds. `TF_DS_DRAFT_LADDER_LOG` prints every scored pass.

- Source: branch `draft-ladder` (commit `8b1e07c`) in `~/Workspace/tfref-draftblock`, on top of the production
  patch. [Patch against the pinned series](draft-ladder.patch) · [Increment on 0192](ladder-on-0192.patch)
- Host tests: 763 of 765 on a clean cache, the same two failures as the pinned series. Two mutants (a gate that
  leaves later drafts on offer, a score that ignores an earlier wrong draft) each fail a new test.
- Image: `tensorfold-dsv41:bench-draft-ladder-0bd63fda950f`, a workstation cross-build layered over the pinned
  image, on both ranks. Its embedded kernels equal production's. It is not in `zig/patches` or the head's recipe.

This is a different rule from the one first described (a per-stream ladder with trial rounds at seven). The per-round
gate was chosen because the November 19 suite data suggested block 7 lost only where the seventh draft was verified
in rounds whose first six would not land. The first arm below showed that reading was wrong.

## Arms

Every arm: the bench image, production's `.env` (six-draft block, token cache, lead reader), a private empty cache,
validation both ways, then a probe of 22 solo prompts and three bursts.
[Plan and keep rule, written before each step](PLAN.md) · [Status](status.log) · [Probe](ladder_probe.py) ·
[Table](final.py) · [Replay](replay.py)

| Arm | Switches | Proved by |
| --- | --- | --- |
| Six (control) | none | all ten solo rows of the deployed production probe: rounds, accepted drafts, hash |
| Five | `TF_DS_DRAFT_BLOCK=0` | its five shared rows equal the draft-block campaign's switch-off arm |
| Seven, seventh unused | block 7, ladder bar `1.0`, logged | banner; no pass of the 7,574 logged left more than six drafts on offer |
| Seven | block 7 | its nine shared rows equal the November 19 suite's block-7 arm |
| Ladder | block 7, ladder at 0.43 | banner; verified drafts differ from plain seven on 20 of 22 rows |
| Production, fresh | restored production | all 22 solo rows equal the control's |

All 25 replies in every arm equal the control's, and validation passed 8 of 8 both ways in every arm.

Counts (rounds, verified drafts) repeat exactly and are compared with the control. Rates are compared with the
fresh production run, because the control's probe ran in a process about ten minutes old after an accidental full
suite probe. Predicted time is the November 19 fit: 24.7 ms a round plus 3.58 ms a verified draft.

## Results: 21 code-like solo prompts, summed

| Arm | Rounds | Verified drafts | Tokens a round | Predicted decode time | Measured decode time | Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Five | +8.3% | −3.7% | 4.80 | +3.16% | +3.16% | 116.2 tok/s |
| **Six (production)** | 5,119 | 26,368 | 5.20 | — | — | 119.9 tok/s |
| Seven, seventh unused (logged) | +0.4% | +1.6% | 5.18 | +0.92% | +1.53%, with the log's cost | 118.1 tok/s |
| Seven | −4.1% | +4.6% | 5.43 | −0.40% | −0.43% | 120.4 tok/s |
| Ladder at 0.43 | −3.7% | +4.0% | 5.40 | −0.37% | −0.38% | 120.4 tok/s |

Per-prompt differences against six, mean ± standard error:

| Arm | Predicted | Measured | Prompts slower than six, by counts |
| --- | ---: | ---: | ---: |
| Five | +3.32 ± 0.38% | +3.29 ± 0.39% | 21 of 21 |
| Seven, seventh unused (logged) | +1.04 ± 0.34% | +1.67 ± 0.34%, with the log's cost | 18 of 21 |
| Seven | −0.39 ± 0.50% | −0.41 ± 0.50% | 10 of 21 |
| Ladder at 0.43 | −0.43 ± 0.51% | −0.40 ± 0.50% | 10 of 21 |
| Ladder against seven, paired | −0.03 ± 0.14% | +0.01 ± 0.14% | |

- **Six against five:** 3.2% faster in total. By rounds and verified drafts five is slower on all 21 prompts; by
  measured rate on 20, with the README prompt equal. This is the first measurement of the production block on more
  than four code-like prompts.
- **A seven-row pass is not free.** With the seventh draft never offered, the first six land slightly less well and
  the round loop verifies 1.6% more of them: 0.9% slower by the counts. That arm's measured rate also carries its
  log (7,574 lines to stderr): the three unlogged arms' measured and predicted times agree within 0.03 points, and
  plain seven runs the same pass, so the extra 0.6 points are the log's and not the pass's.
- **The seventh draft earns about that much back.** Seven ends 0.4% faster than six, inside two standard errors of
  both zero and the 1% line.
- **The gate changes nothing.** It verified 140 fewer drafts than plain seven and ran 23 more rounds. The engine's
  own rule already prices the seventh draft about right: on the logged arm, bars from 0.30 to 0.50 replay within
  0.1% of each other.
- **Per prompt, seven swings from 6.1% faster to 3.5% slower**, and not by language. The Python cache prompt gains
  6.1% greedy and 5.3% sampled, while the pandas prompt loses 1.4%. The Go prompt gains 1.8% greedy and is flat
  sampled. Docker loses 3.5%, SQL 2.2%, sampled Tetris 2.1%.

| Prose and bursts, rate against fresh production | Five | Seven, seventh unused (logged) | Seven | Ladder |
| --- | ---: | ---: | ---: | ---: |
| Prose (59.7 tok/s) | +0.3% | +0.3% | +0.3% | +0.5% |
| Sampled Tetris × 2 (161.4) | −1.0% | −0.7% | −1.3% | −1.8% |
| Sampled Tetris × 4 (210.6) | −1.5% | −1.4% | −1.3% | −1.3% |
| Python and Go prompts at once (157.1) | −2.2% | +0.8% | +3.8% | +2.7% |

At four streams every arm is 1.3–1.5% under six, five included, and seven delivers the same tokens per stream per
round as five there (3.68 against six's 3.70). A reading, not a test: sampled Tetris is one of the prompts seven
loses on solo, and the Python and Go prompts are two it gains on.

## From the logged arm

- The pass's own survival through six drafts is close to calibrated: 0.50 → 52% landed, 0.70 → 70%, 0.97 → 93%.
- Once six had landed, the seventh was right 53–91% of the time by prompt. The pass's own confidence for it is
  about ten points high but ordered (0.95 and over → 87% right, 0.50–0.70 → 58%).
- Choosing per prompt the better of six and the gated seven would have reached 0.9% under six: a width ladder that
  picked perfectly would still miss the 1% line. [Replay output: `python3 -I replay.py m2-shadow`](replay.py)

## Keep rule

| Condition | Ladder |
| --- | --- |
| Predicted time at least 1% under control | No: 0.37% under, ±0.5% |
| Measured time under both controls | Yes, by 0.4% and 0.6% |
| No solo prompt slower than control beyond the 0.2% gap between the controls | No: five prompts 1.3–3.6% slower |
| No burst below control beyond that gap | No: sampled Tetris −1.8% and −1.3% |
| No worse than plain seven | Yes: equal |

Not met. The 0.37% is inside two standard errors of the 1% line, which the plan reads as undetermined, not a pass.

## Earlier statements this corrects

- "Block 7 loses only where the seventh draft is verified in rounds whose first six will not land." Wrong: the cost
  is the seven-row pass's effect on the first six drafts.
- "A seventh draft pays on Python- and Go-like output" ([suite results](../suite-20261119/RESULTS.md)). Over-read
  from one run per prompt: the effect is per prompt, and one prompt's result swings by a few percent with its
  trajectory.

## Follow-up

One question is left open: why a seven-row pass makes the round loop verify 1.6% more of the first six drafts. If
that cost were removed, seven would be about 1.3% faster than six.

## Exposure and state

No request from anywhere but the head appears in any arm's engine log or during the fresh production probe.
Production was restored and reproduced its saved replies after each campaign (05:48 and 06:17 UTC); both ranks are
on `exl3-tp2-ea522d353626` with the six-draft block, the lead reader and production's cache. The arms' five private
cache directories are removed from both ranks. The bench image remains on both ranks.
