# Observability gaps: engine per-request records now in the store (2026-11-25 04:45–05:05 UTC)

Queue item 3. Production was not restarted and no engine code changed. Both ranks' telemetry collectors were
reconfigured, so the changes are live on `gx10-eb0e` and `gx10-235f` now.

## What was wrong (two of the three recorded causes were not the recorded cause)

| Recorded in `runs/observe-review-20261123/COVERAGE.md` | Actual, verified |
|---|---|
| `tensorfold:requests_total` is "scraped but not stored", dropped by a collector processor or OpenObserve ingest | **Never scraped, because it is never rendered.** `metrics.zig:131` prints the family only `if (requests.len > 0)`, and that list is filled only by `server.zig:299`'s reply hook — which is installed only `if (conn.auth_enabled)`. Production runs with **no API keys**, so `auth_enabled` is false, `countReply` never runs, and the family does not exist. `curl localhost:8000/metrics \| grep -c requests_total` = **0**. The collector's metrics pipeline has no `filter`/`transform` at all, so nothing was dropped on the way. |
| `done` lines "do not reach OpenObserve" | True. The engine logs to the docker **json-file** driver (`/var/lib/docker/containers/<id>/<id>-json.log`), and the head's `filelog/std` receiver only includes `/var/log/**log`. |
| "the worker's collector scrapes its own `localhost:8000` every 30 s and fails every time" | True. Rank 1 is the follower and serves no `/metrics`. |

Also verified: `log.zig:77` returns an empty key label when auth is off, so **every `done` line carries no client
identity**. Client separation is not a plumbing problem; it needs API keys enabled (operator decision, `todo.md`).

## What changed

- **`filelog/engine` receiver on both nodes**, `include: [/var/lib/docker/containers/*/*-json.log]`, `start_at: end`,
  `json_parser` + `move body.log → body`, exported to the `default` stream. Every engine line is shipped, not only
  `done`: about 2 MB a day a rank, and rank 1's warnings and errors were invisible before. No prompt or reply text is
  in them. (A `filter` keeping only `^done ` was tried first and passed everything — the engine prefixes lines with
  `[tensorfold] ` — so the filter was dropped rather than debugged: the volume is trivial.)
- **The worker's `tensorfold` scrape job removed.** It had been `up=0` on all 33,237 scrapes since 2026-11-11 18:12 and
  wrote a warning twice a minute, shipped twice (syslog and journald). The head's job is unchanged and still `up=1`.
- **The collector runs as root** (`User=root`, drop-in `/etc/systemd/system/otel-collector.service.d/10-engine-logs.conf`).
  Required, not cosmetic: `/var/lib/docker/containers` is `0710 root:root`, so a glob needs **readdir** and group root
  is not enough (`SupplementaryGroups=root` was tried first and failed). It also clears the pre-existing
  `/var/log/boot.log: permission denied` on `filelog/std`.

## Verified

- `otelcol_receiver_accepted_log_records{receiver="filelog/engine"}` on the head rises with each request; refused 0.
- The store answers a query for them, with every field intact:
  `[tensorfold] done chatcmpl-… prompt=182565 cached=180224 thinking=True effort=none tokens=744 sha=… finish=stop
  rounds=243 accepted=501/785 prefill_s=1.641 ttft_s=1.675 decode_s=9.3 s_per_token=0.0125 tok_per_s=80.2 accept=63.8%
  tokens_per_round=3.1 render_s=0.005 encode_s=0.004 engram_host_ms=73.889 queue_s=0.020 restore_s=0.000`
- Rank 1 writes no per-request `done` line (its log holds startup banners and warnings only), so the worker receiver
  carries rank-1 diagnostics, not per-request records. Rank-1 per-request numbers still arrive only through rank 0's
  relayed `spill_*` / `kv_shard_rank_*` series.
- Both collectors `validate rc=0`, both `active`.

## What this buys

`queue_s`, `restore_s`, `render_s`, `encode_s`, `thinking`, `effort`, `finish`, `accepted/drafted`, `rounds`,
`tok_per_s`, `tokens_per_round` per request, now queryable instead of log-only. That is the instrument the
2026-11-24 attribution said was missing for the live acceptance gap (80–100% prompt-dependent was a bound, not a
measurement) — with no engine change and no restart.

## Still owed (engine change, fold into the next image build)

1. **`requests_total`** — install the reply hook unconditionally and label `unauthenticated` when auth is off
   (`server.zig:299`, one line). Without it the store still cannot separate good replies from errors. Enabling API keys
   is what actually buys per-client separation, and that is an operator decision.
2. **Rank 1's gauges** (device memory, Engram read histogram, request histograms, pair gauges) — relay them in the
   stats reply the way the tier counters already are (`COVERAGE.md`, "Everything from rank 1").
3. `Status.decode_tokens_per_second` / `prefill_tokens_per_second` / `warming` and the round profile
   (`forward_ms`, `pass_ms`, `rows_a_round`) are still log-only. Four counters and four gauges would make the cycle
   model checkable on live traffic including 8 streams.

## Files and undo

- `otel-config.head.before.yaml` / `.after.yaml`, `otel-config.worker.before.yaml` / `.after.yaml` (the configs as
  fetched and as applied).
- Reapply: `just observe-engine-logs`. Undo, per node: `sudo cp ~colum/otel-config.yaml.bak-20261125
  /etc/otel-config.yaml && sudo rm /etc/systemd/system/otel-collector.service.d/10-engine-logs.conf && sudo systemctl
  daemon-reload && sudo systemctl restart otel-collector`.
- **`/etc/otel-config.yaml` is Ansible-managed** (`# Managed by Ansible (opentelemetry_collector)`), and the role's
  source is not in any checkout on this workstation, so a later Ansible run will revert this. The role needs the same
  two receivers, the job removal and the `User=root` drop-in.
- Reachability note: the head's tailnet address `100.95.49.21` is **offline** (as during the k296 suite); both nodes
  were reached with `ssh -J ai-node colum@10.50.0.10` and `-J ai-node,10.50.0.10 colum@10.50.0.11`.
