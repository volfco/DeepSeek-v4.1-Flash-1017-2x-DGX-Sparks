# Screen: pruning off, milder expert caps and the Engram lead reader, November 18–19

**Result.** Two of the three are worth a full benchmark round. The lead reader alone raised cold prefill 7.5–11.6%
with identical replies and no decode cost. Expert caps of 32 and 24 raised output 8% and 13% at four mixed streams and
5% and 15% at eight Tetris streams, with single-stream likelihood inside the margin, but replies then depend on what
else is in the round. Turning pruning
off did not raise draft acceptance and cost 11–13% of round time, so it does not explain the gap to Baseten.

This is a screen: one fresh process per arm, each under ten minutes old, thinking off, short prompts. Nothing was
deployed and production was restored and validated at 00:21 UTC.

## Setup

Six arms on the production image `exl3-tp2-ea522d353626` with production's `.env` (six-draft block, token cache),
private spill directories and `TF_DS_SCORE=1`: control, `TF_DS_T13_EXPERT_PRUNE=0`, `TF_DS_T13_EXPERT_CAP=32,1`,
`24,1`, `TF_DS_ENGRAM_LEAD=1`, control. The two controls agree within 0.5% on every cell, and their replies and all
80 likelihood scores are identical. The first control matched production's deployed probe on all six shared solo cells
(rounds, accepted drafts and hash) and on the eight-stream Tetris hashes, so the scoring switch is speed-neutral. [Status](status.log) · [Tabulation](summary.py) · [Probe](screen.py)

The bars were set before the results were read.

## Lead reader alone: the bar is met

Bar: 8K and 32K cold prefill both outside the gap between the controls, with no decode cell below it.

| Cold prefill, tok/s (same prompts in every arm) | Control A | Control B | Lead | Change |
| --- | ---: | ---: | ---: | ---: |
| 8K | 2,110 | 2,101 | 2,264 | +7.5% |
| 32K | 2,151 | 2,134 | 2,390 | +11.6% |
| 100K (one repetition) | 2,122 | 2,099 | 2,323 | +10.1% |

- The lead arm's three repetitions span 0.2% at each size, and every prefill run in all three arms reported zero
  cached tokens on identical prompt lengths.
- Lossless: validation 8 of 8 both ways, all 80 likelihood scores and every solo reply equal to control.
- Decode: solo cells −0.2% to 0.0%, four mixed streams +0.2 to +0.3%, eight Tetris streams +0.2%. One of two
  eight-stream mixed bursts had a round time 0.5% above the slower control; the other sat inside the controls.
- The prompts are the benchmark's random words. Natural text gave +5–6% at 8K on November 12.
- Four-stream decode is unchanged here. That fits the earlier short-decode cost of the three-switch Engram bundle
  coming from the mapped scales and not from the lead reader; this screen did not test that directly.

Still open: whether the gain and the neutral decode hold in a process that has run for hours, on natural text, at
300K, and on the 1–2K uncached tail of a deep warm turn, which is where most prefill time sits when 98% of input is
reused.

## Milder expert caps: the speed is there, with a behaviour change

Bar: round time at eight mixed streams clearly outside the two controls. That cell did fall far outside, but it
turned out not to be comparable (below), so the two comparable cells carry the result. Both are far outside the
controls' ±0.3%.

| Cell | Controls | Cap 32 | Cap 24 |
| --- | ---: | ---: | ---: |
| Mixed prompts × 4, aggregate tok/s | 115.7–116.7 | 125.7 (+8%) | 131.3 (+13%) |
| Mixed prompts × 4, round time | 67.6–68.3 ms | 59.3 ms (−13%) | 53.2 ms (−22%) |
| Mixed prompts × 4, tokens per stream per round | 2.72 | 2.62 | 2.50 |
| Tetris × 8, aggregate tok/s | 215.5–216.2 | 226.3 (+5%) | 248.5 (+15%) |
| Tetris × 8, round time | 71.7–72.0 ms | 68.5 ms (−5%) | 62.5 ms (−13%) |
| Mixed prompts × 8, round time (not comparable) | 90.0–90.3 ms | 71.0 ms | 63.5 ms |

At four mixed streams part of the round-time gain is given back in fewer tokens per round. At eight mixed streams
the arms' replies differ in length and pace, so neither rate is a clean comparison: one stream stops early at a
different point in each arm, and under cap 24 the story prompt's reply ran at 3.46 tokens a round against 1.41 in the
controls, in both bursts. The screen kept hashes, not text, so whether that reply is merely different or degraded is
unknown. Solo prose is untouched (same reply, same speed). Solo Tetris changed its reply and lost 1–5%: the cap binds
within one stream's window on that prompt, and fewer drafts landed. The solo code prompt gained 1.5–2.1%.

| Quality signal | Cap 32 | Cap 24 | Cap 12 (earlier campaign) |
| --- | ---: | ---: | ---: |
| Likelihood change, original 20 sources | +0.0031 [−0.0007, +0.0070] | +0.0005 [−0.0065, +0.0074] | +0.0267 [+0.0059, +0.0474] |
| Likelihood change, independent 20 sources | −0.0001 [−0.0029, +0.0027] | +0.0087 [−0.0002, +0.0176] | +0.0394 [+0.0107, +0.0680] |
| Saved solo replies reproduced | 8 of 8 | 7 of 8 | 4 of 8 |
| Concurrent reply equals solo reply | 2 of 8 | 1 of 8 | 0 of 8 |

Likelihood is in nats per token against the first control, paired by source, with a 95% interval; the earlier
acceptance line was an upper bound under +0.020. The panel is single-stream. The cap did bind there (34 to 40 of 40
excerpts changed score), but it binds much harder at eight streams, where this screen has no quality reading.
"Concurrent equals solo" counts determinism, not quality: under a cap a reply depends on the other streams in its
round. The current rule keeps 8 of 8.

Still open, in this order: whether replies that vary with load are acceptable at all; then task quality at eight
streams (the earlier cap lost the one solved SQL case there), with the replies read.

## Pruning off at a six-draft block: the bar is not met

Bar: Tetris tokens per round above the spread of the control's seeds.

| Tetris, five sampled seeds | Control | Pruning off |
| --- | ---: | ---: |
| Tokens per round, mean [range] | 5.15 [4.79–5.35] | 5.02 [4.97–5.07] |
| Drafts accepted | 78.6% | 76.3% |
| Round time | 42.3–43.4 ms | 47.3–47.8 ms |
| Tok/s, mean | 119.8 | 105.4 |

Acceptance did not rise, so the lossy prune is not what separates this pair from Baseten's 5.6. Round time rose
11–13% at one, four and eight streams. Likelihood improved by 0.006 and 0.010 nats per token, the same as the
earlier campaign found.

## Exposure and state

The pair ran the screen from 23:26 to 00:21 UTC with empty private caches; the three lossy arms ran from 23:37 to
00:00. Every pre-restart check found no connected clients and only loopback requests. Production is back on its
image and `.env` with 8 of 8 saved replies, and the screen's spill directories are removed from both ranks.

## Disposition, November 19

- **Lead reader:** queued for a full round. [Protocol](../../to-benchmark.md#current-evaluation-order)
- **Expert caps:** parked as a future idea. The speed is real, but replies then vary with load and quality at
  eight streams is unread. Reopen with the eight-stream task panels and saved reply text, for `32,1` first.
- **Pruning off:** closed. It does not explain the gap to Baseten.

## Reading of the engine source on two variants of the cap

Source read: the production image's tree with the draft-block patch (`~/Workspace/tfref-draftblock`). Nothing was
built or run for this section.

**A cap per stream instead of per round.** Production's rule is per token, so streams do not affect each other
today. The cap's kernel (`tf_ds_expert_prune_kernel`, `zig/kernels/cuda/dsv41_ops.cu`) sums expert weight over
every row of the round, which is where the cross-stream effect comes from. A per-stream cap needs the kernel to
take each row's stream id and rank within a stream, the same change in the host reference (`pruneWindow`,
`expert_prune.zig`) with tests, a scope on `TF_DS_T13_EXPERT_CAP` in the fingerprint the ranks compare, the stream
ids uploaded each round the way the dead-row mask is (rank 0 has them, `lanes.zig:219`; the follower's side was not
checked), and one rebuilt CUDA module. About a day of work, as an estimate.

It would not pay at caps 24–32. With one stream the round-wide cap already is a per-stream cap, and this screen's
solo cells measured it: prose unchanged, code +1.5–2.1%, Tetris −1 to −5%. With more streams each stream has fewer
rows: at eight streams about 1.7 rows and about 5 distinct experts a layer, against the 41.7 a layer the whole
round reads ([W4](../w4/RESULTS.md)). The round-wide cap is faster because it cuts that 41.7 to 24–32 across
streams. A per-stream cap also counts a stream's draft rows, and the draft window shrinks under load, so replies
could still differ between solo and loaded operation.

**A reduced-precision first pass to choose experts.** Choosing experts (the gate projection, the top six and the
prune) is 2.4–3.2% of a solo decode round and about 2% at four streams in the eager profile
([stage windows](../overnight-opt/followthrough02/stage-windows.csv)). The gate matrix is 3.9 MB a layer, 157 MB a
round, against 3.6 GB of expert weights at one stream and 11 GB at eight. A free first pass would save 2–3% at
most. It could be made exact with a candidate set and an exact rescore, but that replaces the kit's routing kernel,
whose source is not in the image or any local checkout. It would not make pruning less lossy: the loss is that
chosen experts are not run, and the scores are already exact.
