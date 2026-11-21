# Width-eight expert pruning and independent quality confirmation

Retain the current uncapped prune for general serving. Cap `12,1` improved measured generation throughput but exceeded the predeclared +0.020 nats/token quality margin in two disjoint source cohorts from the same repository. Turning pruning off improved mean likelihood slightly, reduced generation throughput, and produced fewer executable SQL answers against both current controls. [Frozen protocol](PROTOCOL.md) · [Source and task pairing](analyze.py)

## Held-out likelihood

Each cohort contains two excerpts from each of ten code files and ten prose files. The second cohort uses 20 source files excluded from the first. The table gives the mean paired change against the opening current configuration, in nats per target token, with a two-sided 95% interval across the 20 source means. A positive change means lower likelihood. The 40 target excerpts contain 3,362 tokens in the original cohort and 3,352 in the confirmation. Scoring uses the single-stream teacher-forcing endpoint. It cannot measure interactions between concurrent request routes. [Original item set](../../engine-followup/runtime01/items.jsonl) · [Independent item set](width8-02/confirmation-items.jsonl)

| Configuration | Original 20 sources | Independent 20 sources |
| --- | ---: | ---: |
| Prune off | −0.00643 [−0.01749, +0.00463] | −0.01012 [−0.01994, −0.00030] |
| Cap `12,1` | +0.02665 [+0.00593, +0.04736] | +0.03935 [+0.01073, +0.06796] |

The cap increased mean NLL on 14 of 20 original sources and 13 of 20 independent sources. On the independent cohort its mean changes were +0.02304 for the ten code files and +0.05565 for the ten prose files. The cap fails the criterion that the upper bound fall below +0.020. Each interval still includes +0.020. The data support a loss of likelihood and failure of the stated acceptance test without establishing that the true loss must exceed +0.020. [Original cap scores](width8-02/cap12.score.json) · [Independent cap scores](width8-02/cap12.confirmation.json)

## Real eight-request task batches

Eight independently seeded requests entered each SQL batch, and each tool episode made four successive calls. SQL success required an executable answer with the correct result on all six databases. Tool success required the actual ticket, quotes, constrained allocation and receipt. Reasoning used the same output budget as the answer. HTTP completion records and a 120 ms margin at both ends of the measured decode overlap bound the interval in which all eight streams were active. [SQL execution grader](sql.py) · [Tool task and grader](plans.py) · [Client and measurements](hard.py)

| Configuration | SQL passed | Tool episodes passed | SQL / tool emitted tok/s | SQL / tool physical rounds observed at width eight |
| --- | ---: | ---: | ---: | ---: |
| Current, opening | 1/8 | 2/8 | 151.63 / 126.53 | 732 / 100 |
| Prune off | 0/8 | 2/8 | 143.23 / 115.17 | 757 / 82 |
| Cap `12,1` | 0/8 | 2/8 | 209.99 / 165.65 | 779 / 130 |
| Current, return | 1/8 | 2/8 | 156.27 / 127.58 | 730 / 99 |

Both current runs solved SQL task 2. The other seven reached 1,536 tokens before emitting any query. Both alternative policies reached the limit without a query on all eight tasks. Each alternative therefore recorded one paired SQL loss and zero wins. All four policies completed tool episodes 3 and 7. Each solved the first two calls in all eight episodes. The 900-token limit cut off 7/32 tool turns in each current run and with pruning off, and 9/32 with cap `12,1`. The return reproduced all eight SQL and all 32 tool token hashes, all 80 held-out excerpt NLLs, and both task score vectors. [Per-case outcomes and source-paired intervals](width8-summary.json) · [Opening raw requests](width8-02/current-a.jsonl) · [Closing raw requests](width8-02/current-b.jsonl)

Within the all-eight-active intervals the SQL burst took 90.08 and 88.16 ms per physical round in the opening and returning controls, 96.04 ms with pruning off, and 60.04 ms with the cap. The four tool turns together measured 77.77 and 76.31 ms in the controls, 87.43 ms with pruning off, and 52.34 ms with the cap. These ratios divide observed active seconds by the change in the served KV fetch-round counter and exclude the concurrent ramp-up and drain. [Active-window measurements](width8-summary.json)

The sentinel-adjusted native counters measured 40.76/38.41 distinct expert places per SQL/tool routing window after the current rule, versus 15.14/15.32 with the cap. It dropped 38.95%/40.05% of routing weight versus 10.88%/11.32% for the current rule. The `12,1` policy retains a per-row floor in addition to the window cap. The counters include each whole task group. Every SQL burst and every tool step also contained a measured all-eight-active interval. In the independent validation, solo greedy replies matched the frozen outputs 8/8 on both current arms, 2/8 with pruning off, and 4/8 with the cap. Solo and concurrent greedy outputs matched 0/8 under the cap. [Per-window counters](width8-summary.json) · [Cap semantics](../../../zig/patches/0053-expert-prune-cap.patch) · [Cap validation](width8-02/cap12.validate.log) · [Prune-off validation](width8-02/off.validate.log)

A tool call can fail early and change the history for the next turn. The first user requests are content-identical across arms. Later trajectories and emitted lengths can diverge. The reported task rates count tokens emitted over total batch wall time, including those differences. A token cap reached without a query is a failed request under this budget. It does not distinguish an incorrect final SQL query from one the model would eventually have produced with more reasoning time.

## Disposition of smaller widths

The harder pilot exercised width eight with one seed. Prior width-one and width-four tests used easier tasks that all four configurations solved. The two independent NLL failures close the general-serving cap decision. For a workload with a different quality allowance, a smaller-width follow-up can measure the first four frozen tasks per family at both widths, comparing identical seeds and limits under two settings. Higher limits such as 2,304 SQL and 1,200 tool tokens would address the observed SQL truncation and define a different workload from this pilot. The campaign supports this via `--widths 1 4 --seeds 6100 --item-count 4 --sql-budget 2304 --tool-budget 1200 --behavior-only`. The smaller-width harder comparison remains unmeasured. [Previous width coverage](../../engine-followup/RESULTS.md#concurrent-pruning-tasks) · [Campaign](campaign.py)
