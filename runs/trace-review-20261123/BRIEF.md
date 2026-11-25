# Trace review — brief (2026-11-23)

**Goal (the operator's words, paraphrased).** Review the traces collected so far and identify and explore areas for
improvement that nobody has looked at yet.

Read-only analysis on this workstation. No ssh, nothing on the serving pair, no engine changes. Write scripts and
results only under `runs/trace-review-20261123/` in the recipe repository
(`/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`). Do not commit.

## Step 1: inventory what was collected

"Traces" is meant broadly. Start by listing every trace-like artifact under `runs/` and saying what each records:
- **Ring traces** (0272 `TF_DS_RING_TRACE`, one JSON line per sampled ring transfer per rank, GPU-clock nanoseconds:
  `prev`, `stage`, `doorbell`, `collect`, `flag`, `copy`, with `op`, `layer`, `rows`, `bytes`, `stream`, `graph`,
  `round`, `seq`; plus `proxy` and `drain` records):
  `runs/overnight-20261122/k272/tr16{a,b}.trace.r{0,1}.jsonl` (N=16; taken with the round profile on rank 0 only,
  which skews the ranks: see round B2), `runs/sched-20261122/b2/mech/{fx1L,fx1Lb,np1L,ns1L}.trace.r{0,1}.jsonl`
  (N=4, profile off on both ranks; `np` = `PREFETCH=0`, `ns` = `SIDE=0`) and
  `runs/sched-20261122/b2/rowsplit/{k0x1,k0x4,k6x1,k6x4,fx1,fx4}.trace.r{0,1}.jsonl`.
  Timestamp definitions: `runs/kernel-round-20261122c/report-0272.md`; kernel: `zig/kernels/cuda/dsv41/ring_trace_rec.cuh`
  and `zig/src/families/dsv41/ring_trace.zig` in `~/Workspace/tfref-kstack4`; summarizers: `h0272/summarize.py`,
  `runs/sched-20261122/b2/skew.py`.
- Other timing records: `*.clocks.jsonl` under `runs/engine-followup/`, the stage windows and meters under
  `runs/overnight-opt/` (for example `followthrough02/stage-windows.csv`), the `round_clock_us` / `round_launches` /
  `phases_ms` lines in `*.profile.txt` and the phase runs in `runs/overnight-20261122/phases/`, the fused-arm
  engagement traces (`runs/fused-next-20261121/wp4/engagement-trace.txt`, suite `s05`), the SASS operation counts
  (`runs/fnext-sass-20261122/`), the nsys attempt (`runs/sched-20261122/b2/`: it captured only OS-runtime calls).
- Literal stack traces, panics or error dumps in any saved log (`grep -ril 'panic\|stack trace\|backtrace\|segfault'`
  over `runs/`, excluding `sessions/`). Report what they are, or that there are none.

## Step 2: learn what has already been explored, so you do not redo it

`runs/overnight-20261122/k272/RESULTS.md`, `runs/sched-20261122/RESULTS-A.md`, `runs/sched-20261122/b2/RESULTS.md`,
`runs/overnight-20261122/ringprobe/SCHEDULING.md`, `todo.md` ("K272 performance diagnosis"), and in `continue.md` the
sections "Audit review and re-score of rounds 0259–0279", the B2 and k272 bullets, and "Kernel, instruction and
hardware findings register". In short, already known: the ranks' producer skew and its prefetch cause, the proxy's
cost, the ring's protocol cost, the per-gather waits, the MoE being memory-bus bound, and that the drafter pass
(4.15 ms, about 10% of a one-stream cycle) has **no** breakdown anywhere.

## Step 3: mine the traces for what is new

Use your own judgment; these are leads the lead session thinks the ring traces can answer and nobody has asked:
1. **A drafter-pass breakdown from existing data.** The drafter's gathers are traced as `op: "other"`: its hidden
   gathers carry `layer` 0–2 and the pass's block rows, its Markov-step gathers carry `layer` −1 and `rows` = streams
   (`draft.zig:663,670,194`). `prev` is the previous transfer's copy-done on the same stream, so `stage − prev` is the
   compute between two gathers. That gives the time of each drafter segment: attention of stage j, MoE of stage j,
   the head plus the first Markov step, each later Markov step, and the hand-off into the forward's first gather.
2. **The cost of a Markov bias miss.** Each Markov step either hits the 256-row bias cache or recomputes against a
   33 MB head (miss rate 52–53%). If the per-step interval is bimodal, the gap between the modes is the miss cost,
   which prices patch 0281 without a hardware run. Say plainly if the data cannot separate them.
3. **The pass-to-forward hand-off.** k267 showed a change inside the pass moving the next forward by about 0.25 ms.
   Compare the producer interval of the forward's first layers with later layers of the same kind, and with and
   without prefetch (`np1L` against `fx1L`), to see whether the forward starts cold.
4. **A per-layer producer profile.** Producer time by layer index for attention and MoE: which layers are outliers
   (Engram layers, KV-source layers 2/8/14/20, index-source layers 24/28/32/36, the first and last layers) and what
   the source says they do differently. Where does prefetch help most and least, layer by layer?
5. **Tails and periodicity.** p99 / p99.9 of each interval; whether slow rounds recur on a period (the hot plan
   rebuilds every 32 rounds; graph captures; Engram reads; spill writes); the one-off 70–100 ms host stalls and the
   0.5–0.9 ms-a-round slow cells the re-score found; time between rounds.
6. **The other transfers:** `kv`, `engram` and `head` records, and the proxy's records, at one and four streams.
7. Anything else the data shows that the documents above do not mention.

For each finding give: what was measured (script, file, numbers, sample sizes), what is inferred, the ceiling as a
share of the 42.8 ms one-stream cycle (38.3 ms forward + 4.15 ms pass), and whether the engine source explains it
(read `~/Workspace/tfref-kstack4`, family code under `zig/src/families/dsv41/`; cite file:line). Sampling matters:
N=4 or N=16 means one transfer in N is recorded, but `prev` still refers to the true previous transfer.

## Deliverables

`runs/trace-review-20261123/RESULTS.md`: the inventory, what was already known, the new findings ranked by ceiling,
each with its evidence and a proposed next step (an experiment with its arms, or a patch brief with the switch and the
parity argument), and a list of what the traces cannot answer and what extra trace would. Keep the analysis scripts
beside it. Mark every number as measured or inferred.
