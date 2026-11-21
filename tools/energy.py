#!/usr/bin/env python3
"""Energy per request from a pair's TF_DS_ENERGY lines: rank 0's done lines and rank 1's follower frames.

    python3 -I tools/energy.py HEAD_LOG WORKER_LOG [--streams 1|4] [--json OUT]

Rank 0's done line ends with energy_j=, idle_j=, round_from= and round_to= (TF_DS_ENERGY=1). Rank 1's follower logs
`energy step=N mj=M` per frame while the switch is on, and rank 0's ordinals bracket a request's frames exactly (the
pair is lockstep: rank 1 refuses a frame whose step is not the next one). The worker GPU's figure is the counter's rise
between the frame at round_from and the frame after round_to; a request whose window rank 1 does not cover is reported
and makes the run exit 1, never averaged.

energy_j is the request's own window (gross: it holds whatever else the head's desktop and containers drew); idle_j is
the counter's rise over an equal window just before it, scaled from the gap since the last finished request ("-" when
no gap is at least as long as the request). The pair figure is rank 0's gross plus rank 1's delta.

Stdlib only. `--selftest` checks the parsing and the join on synthetic lines.
"""
import json
import re
import statistics
import sys

DONE = re.compile(
    r"done (?P<id>\S+) prompt=(?P<prompt>\d+) cached=(?P<cached>\d+) .*?tokens=(?P<tokens>\d+) sha=(?P<sha>\S+) "
    r"finish=(?P<finish>\S+).*?rounds=(?P<rounds>\d+) accepted=(?P<accepted>\d+)/(?P<drafted>\d+).*?"
    r"accept=(?P<accept>\S+) tokens_per_round=(?P<tpr>\S+) energy_j=(?P<energy>\S+) idle_j=(?P<idle>\S+) "
    r"round_from=(?P<from>\S+) round_to=(?P<to>\S+)"
)
FRAME = re.compile(r"energy step=(?P<step>\d+) mj=(?P<mj>\d+|-)")
# the old done line's fields, which the energy ones come after: a parser of it still matches
OLD_FIELDS = ("prompt=", "cached=", "tokens=", "sha=", "finish=", "rounds=", "accepted=", "prefill_s=")


def num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def done_lines(path):
    out = []
    for line in open(path, errors="replace"):
        m = DONE.search(line)
        if m:
            out.append(m.groupdict())
    return out


def frame_lines(path):
    out = []
    for line in open(path, errors="replace"):
        m = FRAME.search(line)
        if m:
            out.append((int(m.group("step")), None if m.group("mj") == "-" else int(m.group("mj"))))
    out.sort()
    return out


def worker_delta(frames, first, last):
    """The worker GPU's millijoules over frames [first, last]: the counter at `first` (before it runs) to the one at the
    first frame after `last` (after it ran). None when either is missing."""
    start = next((mj for step, mj in frames if step >= first), None)
    end = next((mj for step, mj in frames if step > last), None)
    if start is None or end is None:
        return None
    return end - start


def join(head, worker, streams=None):
    frames = frame_lines(worker)
    rows, missing = [], []
    for d in done_lines(head):
        ident = d["id"]
        energy, idle = num(d["energy"]), num(d["idle"])
        first, last = num(d["from"]), num(d["to"])
        if energy is None:
            missing.append((ident, "rank 0 has no energy figure (the switch off, or no NVML counter)"))
            continue
        wj = None
        if first is None or last is None:
            missing.append((ident, "rank 1 window not found for request %s (no round_from/round_to: no rank pair)" % ident))
        else:
            mj = worker_delta(frames, int(first), int(last))
            if mj is None:
                missing.append((ident, "rank 1 window not found for request %s (rounds %d..%d)" % (ident, int(first), int(last))))
            else:
                wj = mj / 1000.0
        tokens, prompt = int(d["tokens"]), int(d["prompt"])
        decode = max(tokens - 1, 1)
        pair = energy + (wj or 0.0)
        rows.append({
            "id": ident, "prompt": prompt, "cached": int(d["cached"]), "tokens": tokens,
            "rank0_j": energy, "rank1_j": wj, "pair_j": pair,
            "idle_j": idle,
            "j_per_1k_prompt": pair * 1000.0 / prompt if prompt else None,
            "j_per_1k_decode": pair * 1000.0 / decode,
            "accept": num((d["accept"] or "").rstrip("%")),
            "tokens_per_round": num(d["tpr"]),
            "finish": d["finish"], "streams": streams,
        })
    return rows, missing


def fmt(v, places=2):
    return "-" if v is None else f"{v:.{places}f}"


def report(rows, missing, streams):
    if not rows and not missing:
        print("no done line with TF_DS_ENERGY fields: is TF_DS_ENERGY=1 on rank 0?", file=sys.stderr)
        return 1
    print("| request | prompt | decode | J (pair) | J rank0 | J rank1 | idle J | J/1K prompt | J/1K decode | accept | tok/round |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['id']} | {r['prompt']} | {r['tokens']} | {fmt(r['pair_j'])} | {fmt(r['rank0_j'])} | {fmt(r['rank1_j'])} | "
              f"{fmt(r['idle_j'])} | {fmt(r['j_per_1k_prompt'], 1)} | {fmt(r['j_per_1k_decode'], 1)} | "
              f"{fmt(r['accept'], 1)}% | {fmt(r['tokens_per_round'], 1)} |")
    if rows:
        med = lambda k: statistics.median([r[k] for r in rows if r[k] is not None]) if any(r[k] is not None for r in rows) else None
        idle = [r["idle_j"] for r in rows if r["idle_j"] is not None]
        print(f"\nstreams={streams or '?'} requests={len(rows)} "
              f"J/req median={fmt(med('pair_j'))} J/1K prompt median={fmt(med('j_per_1k_prompt'), 1)} "
              f"J/1K decode median={fmt(med('j_per_1k_decode'), 1)} accept median={fmt(med('accept'), 1)}% "
              f"tok/round median={fmt(med('tokens_per_round'), 1)}")
        print(f"gross {fmt(sum(r['pair_j'] for r in rows))} J; idle baseline measured on {len(idle)}/{len(rows)} requests "
              f"(median {fmt(statistics.median(idle) if idle else None)} J): the marginal figure is gross minus it")
    for ident, why in missing:
        print(f"rank 1 window not found for request {ident}: {why}", file=sys.stderr)
    return 1 if missing else 0


def selftest():
    import io
    head = (
        "2026-11-14T10:00:00Z [tensorfold] done chatcmpl-a prompt=2000 cached=0 thinking=False effort=none tokens=101 "
        "sha=abc finish=stop rounds=10 accepted=20/40 prefill_s=1.4 ttft_s=1.4 decode_s=2.0 s_per_token=0.02 "
        "tok_per_s=50.0 accept=50.0% tokens_per_round=10.1 energy_j=6.412 idle_j=5.123 round_from=100 round_to=139\n"
        "2026-11-14T10:01:00Z [tensorfold] done chatcmpl-b prompt=500 cached=0 thinking=False effort=none tokens=1 "
        "sha=def finish=length rounds=0 accepted=0/0 prefill_s=0.3 ttft_s=0.3 decode_s=0.0 s_per_token=- tok_per_s=- "
        "accept=- tokens_per_round=- energy_j=1.500 idle_j=- round_from=200 round_to=200\n"
    )
    worker = (
        "2026-11-14T10:00:00Z info: energy step=100 mj=1000000\n"
        "2026-11-14T10:00:00Z info: energy step=139 mj=1005000\n"
        "2026-11-14T10:00:01Z info: energy step=140 mj=1006000\n"
        "2026-11-14T10:01:00Z info: energy step=200 mj=1009000\n"
        "2026-11-14T10:01:01Z info: energy step=201 mj=1009500\n"
    )
    head_p, worker_p = "/tmp/energy-selftest-head.log", "/tmp/energy-selftest-worker.log"
    open(head_p, "w").write(head)
    open(worker_p, "w").write(worker)
    rows, missing = join(head_p, worker_p, 1)
    assert len(rows) == 2 and not missing, (rows, missing)
    a, b = rows
    assert a["rank0_j"] == 6.412 and abs(a["rank1_j"] - 6.0) < 1e-9, a
    assert abs(a["pair_j"] - 12.412) < 1e-9 and a["idle_j"] == 5.123, a
    assert abs(a["j_per_1k_prompt"] - 12.412 * 1000 / 2000) < 1e-6, a
    assert abs(a["j_per_1k_decode"] - 12.412 * 1000 / 100) < 1e-6, a
    assert a["accept"] == 50.0 and a["tokens_per_round"] == 10.1, a
    # a one-token reply: decode tokens count as one, the idle figure is "-"
    assert b["idle_j"] is None and abs(b["j_per_1k_decode"] - 2.0 * 1000) < 1e-6, b
    # the old line's fields are still there, before the energy ones
    for field in OLD_FIELDS:
        assert field in head.splitlines()[0], field
    assert head.splitlines()[0].index("tokens_per_round=") < head.splitlines()[0].index("energy_j=")
    # rank 1's window is not covered: loud, never averaged
    open(worker_p, "w").write("2026-11-14T10:00:00Z info: energy step=100 mj=1000000\n")
    rows, missing = join(head_p, worker_p, 1)
    assert len(rows) == 2 and len(missing) == 2, (rows, missing)
    assert all("rank 1 window not found" in why for _, why in missing), missing
    assert report(rows, missing, 1) == 1
    # a switch-off log has no energy fields at all
    open(head_p, "w").write("2026-11-14T10:00:00Z [tensorfold] done chatcmpl-c prompt=1 tokens=1 rounds=0\n")
    open(worker_p, "w").write("")
    assert report(*join(head_p, worker_p), 1) == 1
    print("selftest: ok")


def main(argv):
    if "--selftest" in argv:
        selftest()
        return 0
    args = [a for a in argv[1:] if not a.startswith("--")]
    streams = None
    out = None
    for i, a in enumerate(argv):
        if a == "--streams":
            streams = int(argv[i + 1]) if argv[i + 1].isdigit() else None
        if a == "--json":
            out = argv[i + 1]
    if len(args) < 2:
        print(__doc__.splitlines()[2].strip(), file=sys.stderr)
        return 2
    rows, missing = join(args[0], args[1], streams)
    if out:
        json.dump({"streams": streams, "requests": rows, "missing": missing}, open(out, "w"), indent=1)
    return report(rows, missing, streams)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
