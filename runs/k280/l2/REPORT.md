# k280 L2 screen: no knob beats the kit's served defaults; keep 3000 ns / 230 GB/s / 16 MiB

Window 2026-11-25 08:34:33Z → 09:14:08Z. Image `tensorfold-dsv41:bench-fused-next-1421187`, 8 arms, production
restored at the end (`restored: validate rc=0`, `all passed`). This is a **screen**, not a confirmation: it ranks
values, and only a value beating the controls' spread deserves an A/B/B confirmation.

## Gates — all pass

| Gate | Result |
|---|---|
| Exact panel IDENTICAL every arm | **yes**, 8/8 |
| Drafts unchanged (`gate_drafts.py`) | **yes**, 3/3 gated rows, exit 0 |
| Banner 1 per rank in knob arms, 0 in controls | **yes** — ctl1/ctl2 0/0, all six knob arms 1/1 |
| Production restored and validated | **yes**, `validate rc=0` |

The knobs took effect: `TF_DS_L2_RATE_GBPS=180/270`, `TF_DS_L2_DELAY_NS=0/10000`, `TF_DS_L2_BUDGET_MIB=8/32` are
each in that arm's `.armed.txt` and absent from the controls.

## Timings — controls 4.58 / 4.55 ms `pass_ms`, 40.43 / 40.31 ms `forward_ms`; spread 0.66% / 0.30%

| Arm | `pass_ms` | vs ctl | `forward_ms` | vs ctl | Verdict |
|---|---:|---:|---:|---:|---|
| rate180 | 4.59 | +0.88% | 40.55 | +0.22% | no win |
| rate270 | 4.62 | +0.88% | 40.37 | +0.22% | no win |
| delay0 | 4.60 | +0.22% | 40.33 | +0.36% | no win |
| delay10k | 4.55 | +0.22% | 40.70 | +0.36% | no win |
| bud8 | 4.59 | +0.77% | 41.61 | **+2.9%** | **worse** |
| bud32 | 4.61 | +0.77% | 40.97 | **+1.3%** | **worse** |

Nothing clears the 0.66% / 0.30% control spread in the favourable direction. The rate arms move `pass_ms` in the
wrong direction by 0.88%; the delay arms are neutral; **both budget arms are worse on the forward** — 8 MiB by
+2.9% and 32 MiB by +1.3%, so 16 MiB is the sweet spot and the kit's default is already it.

## Decision

**Adopt nothing.** The kit's served 3000 ns / 230 GB/s / 16 MiB stays. This closes 0283 as a screen with no
candidate, and removes the last "the kit's numbers were never swept" gap for the forward's L2 prefetch.

## Caveats

- The screen ran `work.py` (the k280 workload), not the live 155K shape, and only 1- and 4-stream cells. A knob
  that only matters at 8 streams or at long contexts would not show here.
- The `bud` arms' `forward_ms` penalty is the one signal with a plausible mechanism (a larger or smaller budget
  than the resident set thrashes the L2), so it is worth one confirmation run at 8 streams before calling 16 MiB
  optimal for the live shape. Everything else can be closed.

## Artefacts

`runs/k280/l2/` — 8 arms with `.json`, `.probe.json`, `.cmp.txt`, `.armed.txt`, `.banner.txt`, `.profile.txt`.
Replay: `bash runs/k280/l2-run.sh` from the recipe root with production idle.
