# Draft ladder: plan and keep rule, written before any pair run (November 19)

**Switch.** `TF_DS_DRAFT_LADDER` under `TF_DS_DRAFT_BLOCK=7`: the pass runs seven drafts as block 7 does, and the
seventh is offered to a round only when this pass's survival through six drafts, times the stream's running share of
seventh drafts that were right once six had landed, reaches the bar. The sixth draft and the 5-or-wider rule are
production's. The share is scored from the next round's tokens, whether or not the seventh was verified.

**Measure first** (campaign `m`): the new image with production's switches must reproduce production's probe row for
row (rounds, accepted drafts, hash). Then a seven-draft pass whose seventh draft is never offered (`=1.0`), logged:
its rounds and accepted drafts against the first arm show whether a seven-row pass changes the first six drafts, and
its round time shows what the wider pass costs. If that arm is worse than production's six by more than the control
gap, the free scoring fails and the design falls back to a pass-width ladder.

**Bar.** 0.43, from the November 19 fit (3.58 ms a verified draft against 7.4 to 9.7 ms a token). It is changed
before the timed arms only if the logged survival through six is off from the realised share by more than a tenth.

**Timed arms** (campaign `t`): ladder, plain block 7, and the new image with production's switches as the closing
control; the measure-first arm is the opening control. Each: validation both ways, the 22-prompt probe with three
bursts. The ladder arm also records the 39-case exact panel.

**Gates.** Validation 8 of 8 both ways, the exact panel identical, every probe reply hash equal to the control's.

**Metrics.** Primary: rounds and verified drafts, which repeat exactly, through the fit
(24.7 ms a round + 3.58 ms a verified draft) as predicted decode time. Confirmation: the engine's decode rate.

**Keep rule.** Adopt the ladder for single- and two-stream code-like output only if all of:
- its predicted decode time summed over the code-like solo prompts is at least 1% under the control's;
- its measured decode time over those prompts is under both controls';
- no solo prompt's measured rate is below the slower control by more than the gap between the controls;
- neither sampled Tetris burst nor the two-prompt burst is below the slower control by more than that gap;
- its summed predicted and measured times are no worse than plain block 7's.

## After the measure-first arms (written 05:51 UTC, before the timed arms)

- The new image with production's switches reproduced production: the ten solo rows of the deployed probe in rounds,
  accepted drafts and hash, and the sampled Tetris bursts' tokens a round at two and four streams.
- **The free-scoring premise failed.** A seven-row pass with the seventh draft never offered is not production's six:
  over the 21 code-like solo prompts it ran 0.4% more rounds and verified 1.6% more drafts, +0.92% predicted decode
  time and +1.28% measured, 18 of 21 prompts slower. Every reply was identical.
- The pass's own survival through six drafts is close to calibrated (0.50 → 52%, 0.70 → 70%, 0.97 → 93%). Its own
  confidence for the seventh is about ten points high but ordered (0.95+ → 87% right).
- Replayed on that arm's log, offering the seventh at a bar of 0.43 saves 1.4% of decode time against never offering
  it: −0.52% against production in total. The engine's own rule replays at −0.48%, so the gate adds nothing. Choosing
  per prompt the better of production and the gated seven would reach −0.92%.
- So no form of the ladder is expected to meet the keep rule (1% under control). The bar stays 0.43.

**Timed arms, amended.** On the same 22 prompts and three bursts: the built ladder at 0.43, plain block 7, and the
checkpoint's five-draft block (`TF_DS_DRAFT_BLOCK=0`), against the measure-first control. They measure what the
replay predicts and put 5, 6 and 7 on one prompt set. No exact panel unless the ladder meets the keep rule, which is
unchanged.

## How the timed arms will be read (written 05:54 UTC, before any timed arm's results were opened)

- **Timing control.** The measure-first control's probe ran in a process about ten minutes old, after an accidental
  full suite probe; every other arm's probe starts about a minute after its restart. Rounds and verified drafts are
  unaffected, but its rates are not like-for-like. After the campaign the same probe runs on restored production
  (six-draft block, a fresh process), and every measured-rate statement uses that run.
- **Each arm is proved from its counts.** Plain seven's nine shared solo rows must equal the November 19 suite's
  block-7 arm in rounds, accepted drafts and hash. Five's shared rows must equal the draft-block campaign's
  switch-off arms. The ladder must differ from plain seven in verified drafts on some rows.
- **Sums, with a spread.** One prompt's result swings by a few percent with its trajectory (the replay put the Go
  prompt 0.5% slower under seven where the suite's block-7 arm measured it 2.7% faster). So results are read as the
  sum over the 21 code-like solo prompts, with the per-prompt differences' mean and standard error beside it, and
  the ladder is paired against plain seven prompt by prompt.
- **The 1% line.** A sum within two standard errors of 1% under control is undetermined, not a pass.
- **Five against six** is judged on measured rate against the fresh control. The fit has no five-draft run, and a
  five-row pass takes the kit's tuned kernels where six takes a fallback, so predicted time alone would mislead.
