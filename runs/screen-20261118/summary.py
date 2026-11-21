"""Tabulate the screen's arms against its two controls. Run from the recipe root:

  python3 -I runs/screen-20261118/summary.py

Solo rates are the engine's own (the done line). A burst's round time is the streams' summed decode seconds over their
summed rounds, which holds whatever text an arm produced; its tok/s is only comparable when every stream ran to its limit.
"""
import json
import math
import os
import statistics as st

S = "runs/screen-20261118"
CTL = ["ctl-a", "ctl-b"]
ARMS = ["prune-off", "cap32", "cap24", "lead"]


def load(arm, name):
    p = f"{S}/{arm}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def cells(rows):
    out, seen = {}, {}
    for r in rows or []:
        k = (r["cell"], r.get("seed", r.get("seed0")))
        seen[k] = seen.get(k, 0) + 1
        out[k + (seen[k],)] = r
    return out


screens = {a: cells(load(a, "screen.json")) for a in CTL + ARMS}
have = [a for a in CTL + ARMS if screens[a]]
ctl = [a for a in CTL if screens[a]]
print("arms with data:", have)

print("\n== solo cells: tokens/round, ms/round, tok/s (change against the controls' mean)")
for k in screens[ctl[0]]:
    r0 = screens[ctl[0]][k]
    if "width" in r0:
        continue
    cs = [float(screens[a][k]["tok_per_s"]) for a in ctl if k in screens[a]]
    line = f"{k[0]:15s} s{k[1]}: ctl {r0['tokens_per_round']:.2f} tok/rd {r0['ms_per_round']:.1f} ms {min(cs):.1f}-{max(cs):.1f} tok/s"
    for a in ARMS:
        r = screens[a].get(k) if screens[a] else None
        if r and r.get("rounds"):
            same = "" if r["sha"] == r0["sha"] else "*"
            line += f" | {a} {r['tokens_per_round']:.2f} {r['ms_per_round']:.1f}ms {float(r['tok_per_s']):.1f} ({100 * (float(r['tok_per_s']) / st.mean(cs) - 1):+.1f}%){same}"
    print(line)
print("(* = a different reply from the control's)")

print("\n== sampled Tetris, tokens a round over the seeds an arm ran")
for a in have:
    v = [r["tokens_per_round"] for k, r in screens[a].items() if k[0] == "tetris sampled"]
    acc = [r["accept_pct"] for k, r in screens[a].items() if k[0] == "tetris sampled"]
    tps = [float(r["tok_per_s"]) for k, r in screens[a].items() if k[0] == "tetris sampled"]
    if v:
        print(f"{a:10s} n={len(v)} mean {st.mean(v):.3f} [{min(v):.2f}-{max(v):.2f}]  accept {st.mean(acc):.1f}% [{min(acc):.1f}-{max(acc):.1f}]  tok/s {st.mean(tps):.1f} [{min(tps):.1f}-{max(tps):.1f}]")

print("\n== bursts: round time (streams' decode seconds over rounds), tokens/stream/round, aggregate tok/s")
for k in screens[ctl[0]]:
    r0 = screens[ctl[0]][k]
    if "width" not in r0:
        continue
    cm = [screens[a][k]["ms_per_round"] for a in ctl if k in screens[a]]
    ct = [screens[a][k]["agg_tok_s_wall"] for a in ctl if k in screens[a]]
    line = f"{k[0]:10s} #{k[2]}: ctl {min(cm):.2f}-{max(cm):.2f} ms {r0['tokens_per_stream_round']:.2f} tok/rd {min(ct):.1f}-{max(ct):.1f} tok/s"
    for a in ARMS:
        r = screens[a].get(k) if screens[a] else None
        if r and r.get("ms_per_round"):
            tok = f"{r['agg_tok_s_wall']:.1f} ({100 * (r['agg_tok_s_wall'] / st.mean(ct) - 1):+.1f}%)" if r.get("comparable") and r0.get("comparable") else f"{r['agg_tok_s_wall']:.1f} (lengths differ)"
            line += f" | {a} {r['ms_per_round']:.2f}ms ({100 * (r['ms_per_round'] / st.mean(cm) - 1):+.1f}%) {r['tokens_per_stream_round']:.2f} {tok}"
    print(line)

print("\n== held-out likelihood against ctl-a, paired by source (nats a token; 95% interval over 20 sources)")
for name in ("score.json", "confirmation.json"):
    ref = load(ctl[0], name)
    if not ref:
        continue
    refi = {r["id"]: r for r in ref["items"]} if isinstance(ref.get("items"), list) else ref["items"]
    for a in have[1:]:
        d = load(a, name)
        if not d:
            continue
        it = {r["id"]: r for r in d["items"]} if isinstance(d.get("items"), list) else d["items"]
        diffs = {}
        for k, v in it.items():
            diffs.setdefault(k.rsplit("-", 1)[0], []).append(v["nll_mean"] - refi[k]["nll_mean"])
        means = [st.mean(x) for x in diffs.values()]
        delta = st.mean(means)
        se = st.stdev(means) / math.sqrt(len(means))
        same = sum(it[k]["nll_mean"] == refi[k]["nll_mean"] for k in refi)
        print(f"{name:18s} {a:10s} delta {delta:+.5f} [{delta - 2.093 * se:+.5f}, {delta + 2.093 * se:+.5f}]  identical excerpts {same}/{len(refi)}  sources worse {sum(m > 0 for m in means)}/{len(means)}")

print("\n== cold prefill, tok/s (the same prompts in every arm)")
for a in have:
    p, q = load(a, "prefill.json"), load(a, "prefill100k.json")
    if p:
        s = dict(p["summary"], **(q["summary"] if q else {}))
        print(f"{a:10s} 8K {s.get('prefill.8000.tok_s', float('nan')):.1f}  32K {s.get('prefill.32000.tok_s', float('nan')):.1f}  100K {s.get('prefill.100000.tok_s', float('nan')):.1f}"
              f"   reps 8K {[round(r['prefill_tok_s'], 1) for r in p['runs'] if r.get('target_tokens') == 8000]} 32K {[round(r['prefill_tok_s'], 1) for r in p['runs'] if r.get('target_tokens') == 32000]}")

print("\n== validation (from the status log)")
for line in open(f"{S}/status.log"):
    if " validate rc=" in line or "startup:" in line:
        print("  " + line.strip()[21:200])
