# Deploy: production onto `bench-fused-next-1421187` + `MARKOV_CACHE=4096` + `DRAFT_PREFETCH=1`

2026-11-25 19:10 UTC. Authorised after the from-source control passed (`runs/k280/fromsource/REPORT.md`).

| | |
|---|---|
| Image | `tensorfold-dsv41:bench-fused-next-1421187` = `2bb052f` + 0280–0283, both ranks |
| Switches | `TF_DS_MARKOV_CACHE=4096`, `TF_DS_DRAFT_PREFETCH=1` |
| Exact panel | **IDENTICAL 39/39** vs `runs/deploy-10c9866-20261122/exact.json` |
| `validate.py` | **rc=0**, `all passed` (8/8 replies identical to `runs/matrix/base.validate.json`) |
| MemAvailable | **7.4 GiB head / 8.2 GiB worker** (memguard floor 2.5 GiB; 4,096 rows cost 473 MiB a rank) |
| Health | ok, batch 8, 0 connections, 0 waiting |

Not adopted, deliberately: 0282 `dpf2` (a wash on forward+pass) and 0283's L2 knobs (no winner; the kit's
3000 ns / 230 GB/s / 16 MiB stay).

Undo: `cp ~/dsv41-env-backups/.env.bak-20261125-pre-mwb .env && ./start.sh restart`. The previous image
`bench-fused-next-2bb052f` is still on both ranks.

Measured gains being adopted (within image `1421187`, k280 main, drafts IDENTICAL 19/19):

- **0281 at 4,096 rows:** `pass_ms` −0.20/−0.22 ms (−5.0%), Markov bias cache miss 52.2% → 24.2%.
- **0282 `dpf1`:** `pass_ms` −0.18/−0.17 ms (−4.2%), forward unchanged.
- Together ≈ −0.4 ms a pass, which at ~1.41 streams a pass is ~−0.55 ms a round, about −1.2% of a ~45 ms one-stream
  round — the largest adopted decode change since 0259.
