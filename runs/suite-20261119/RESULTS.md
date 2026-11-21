# Narrow run: lead reader, draft-widening threshold and seven-draft block, November 19

**Result.** The lead reader raised cold prefill on natural text by 4.5–5.0% at 8K to 100K with decode and replies
unchanged. Neither widening threshold raised output beyond the gap between controls. The seven-draft block gained
on three code prompts and lost on two others and at two streams, so its total equals control.

This is one block per arm on fresh processes, cut down from the planned 18-block suite to fit in 40 minutes. Nothing
was deployed by the run. Production was restored at 02:06 UTC and reproduced its saved replies (8 of 8 both ways) at 02:09.

**Afterwards.** The lead reader was turned on in production at 02:22 UTC as a provisional default, with its full
test owed at the next full benchmark round. [Deploy record](../deploy-lead-20261119/deploy.log) ·
[What is owed](../../to-benchmark.md#current-evaluation-order)

## Setup

Every arm ran the production image `exl3-tp2-ea522d353626` with production's `.env` (six-draft block, token cache)
and an empty private cache. [Status](status.log) · [Gate](gate.py) · [Tabulation](summary.py) ·
[Probe](suite_probe.py) · [Planned full schedule](schedule.full.json)

| Block | Arm | Switch | Cells |
| --- | --- | --- | --- |
| s01 | base | none | validation, warm-up, natural-text prefill at 2K–100K, code and p125 prefill, decode at 1, 4 and 8 streams, narrow probe |
| s02 | lead | `TF_DS_ENGRAM_LEAD=1` | validation, warm-up, natural-text prefill at 2K–100K, code prefill, decode at 1 and 4 streams |
| s03 | threshold 3 | `TF_DS_DRAFT_BLOCK_KEPT=3` | validation, narrow probe |
| s04 | seven-draft | `TF_DS_DRAFT_BLOCK=7` | validation, narrow probe |
| s05 | threshold 2 | `TF_DS_DRAFT_BLOCK_KEPT=2` | validation, narrow probe |
| after | production, restored | none | validation, narrow probe |

The narrow probe is nine solo prompts (Tetris greedy and sampled, Python, JSON, prose, Go, TypeScript, SQL, YAML)
and sampled Tetris at two and four streams. Block s01 was started as the full suite's first block and stopped
during its long-context cell, which is discarded; its probe was then run by hand on the same process and the block
closed with [finalize_s01.py](finalize_s01.py).

**Gates.** All five blocks pass: saved replies 8 of 8 both ways, and every probe reply hash equal to production's
(11 of 11 in each probe). Each drafting arm changed rounds or accepted drafts on all nine solo rows, so each switch
took effect. The lead block's banners show the reader on at both ranks.

## Lead reader

| Cold prefill on natural text, tok/s (median of repetitions) | Base | Lead | Change |
| --- | ---: | ---: | ---: |
| 2K (3 repetitions) | 1,501 | 1,534 | +2.2% |
| 8K (3) | 1,893 | 1,986 | +4.9% |
| 32K (3) | 1,994 | 2,093 | +5.0% |
| 100K (2) | 1,961 | 2,049 | +4.5% |
| 8K code (3) | 1,896 | 1,928 | +1.7% |

- Repetitions within a block span 0.2–0.4% at 8K to 100K. There is no second base block for prefill; the only
  block-to-block control is decode, where the same two blocks agree within 0.1%. At 2K the lead block's
  repetitions span 1,427–1,605, so that row shows no clear change.
- The first request of each size after the restart moved −0.2%, +2.5%, +5.1% and +4.9% at 2K, 8K, 32K and 100K.
- Decode is unchanged: one stream +0.1% and four streams 0.0%, paired by prompt; all 11 decode replies equal
  base's.
- The two blocks prefilled different passages of the same corpus, and every run reported zero cached tokens.
- The gain is about half of the screen's 7.5–11.6% on random-word prompts, and matches the +5–6% natural text
  gave at 8K on November 12.

## Drafting arms

Two controls bracket the arms: the base block, and production after the restore. Production ran 0.1–1.4% above
base on the solo rows (0.7% in total), so differences of about 1% a row are inside the control gap. The base probe
ran at the end of a ten-minute block; the arms and the closing control ran on processes about a minute old.

| Decode rate, tok/s | Base | Production after | Threshold 3 | Threshold 2 | Seven-draft |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tetris, greedy | 120.4 | 121.0 | 120.1 | 119.0 | 120.1 |
| Tetris, sampled | 123.0 | 123.9 | 123.6 | 124.8 | 120.9 |
| Python | 117.0 | 117.9 | 117.9 | 118.2 | 124.7 |
| JSON | 102.5 | 103.1 | 103.3 | 105.6 | 102.3 |
| Prose | 59.5 | 60.1 | 59.7 | 59.2 | 59.7 |
| Go | 131.1 | 133.0 | 131.9 | 132.9 | 134.6 |
| TypeScript | 128.3 | 129.1 | 128.4 | 127.3 | 131.4 |
| SQL | 115.6 | 116.2 | 115.2 | 115.5 | 113.6 |
| YAML | 129.1 | 129.2 | 128.7 | 130.6 | 128.9 |
| All nine (tokens over decode seconds) | 113.1 | 113.9 | 113.4 | 113.8 | 113.9 |
| Sampled Tetris × 2, aggregate | 162.3 | 162.3 | 163.2 | 162.6 | 159.2 |
| Sampled Tetris × 4, aggregate | 209.6 | 208.8 | 209.9 | 210.0 | 207.6 |

| Tokens a round (exact counts) | Base | Threshold 3 | Threshold 2 | Seven-draft |
| --- | ---: | ---: | ---: | ---: |
| Tetris, greedy | 5.133 | 5.159 | 5.133 | 5.319 |
| Tetris, sampled | 5.347 | 5.418 | 5.447 | 5.461 |
| Python | 5.389 | 5.466 | 5.486 | 6.071 |
| JSON | 4.129 | 4.185 | 4.315 | 4.243 |
| Prose | 1.790 | 1.794 | 1.786 | 1.790 |
| Go | 6.095 | 6.095 | 6.124 | 6.598 |
| TypeScript | 5.766 | 5.792 | 5.792 | 6.124 |
| SQL | 5.000 | 5.020 | 5.039 | 5.039 |
| YAML | 5.766 | 5.792 | 5.872 | 6.009 |

- **Threshold 3:** tokens a round rose 0–1.4%. Six rows sit between the controls or above them by at most 0.2%;
  greedy Tetris, SQL and YAML are 0.2–0.9% below both. No gain.
- **Threshold 2:** four rows are above both controls: JSON by 2.4–3.0% (4.5% more tokens a round), and sampled
  Tetris, Python and YAML by 0.3–1.5%. Four are below both: greedy Tetris, TypeScript, prose and SQL, by 0.1–1.7%.
  The total equals the closing control.
- **Seven-draft block:** tokens a round rose 0.8–12.7% on the code-like prompts, and each round took 2.6–5.7%
  longer. Python gained 5.8–6.6%, Go 1.2–2.7% and TypeScript 1.8–2.4%, all outside the control gap. SQL lost
  1.7–2.2%, sampled Tetris 1.7–2.4% and two streams 1.9%, all below both controls; greedy Tetris, JSON, YAML and
  four streams are 0.2–1.0% below both. The total equals the closing control. Its keep rule (most code-like prompts gain and no cell falls below the control gap) is not met.

Rates are the engine's own decode rate for each request (0.1 tok/s resolution). Burst tokens a round vary a little
between runs of one configuration (3.68–3.70 at four streams across the two controls).

**Later the same day.** The seven-draft block and a draft ladder were measured on 21 code-like prompts: seven is
0.4% faster than six (±0.5% across prompts), and the gains here on the Python, Go and TypeScript prompts did not hold
as a per-language pattern. [Ladder results](../draft-ladder-20261119/RESULTS.md)

## What this run does not cover

- One block per arm and one repetition per probe prompt: no pass-to-pass spread.
- No exact panel (39 cases), no thinking cells, no eight-stream cells for the drafting arms, no 300K prompt.
- No soak: the lead reader was on during the November 13 slowdown, and its behaviour in a long-lived process with
  deep sessions is still unmeasured. It is not cleared to deploy by this run.

## Exposure and state

The pair ran test arms on private caches from 01:28 to 02:04 UTC. No block's saved engine log, and no request
to production since the restore, came from anywhere but the head. Production is back on its image, `.env` and
cache on both ranks; its closing probe matches the deployed probe in hash, rounds and accepted drafts on all 11
rows. The 18 private cache directories are removed from both ranks.
