"""Compare the ladder campaign's arms. From the recipe root, on a copy of the arm directories:

  python3 -I runs/draft-ladder-20261119/compare.py <control arm> <arm> [<arm> ...] [--second-control <arm>]

Per solo prompt: whether the reply equals the control's, rounds, verified drafts, the fit's predicted decode time
(24.7 ms a round + 3.58 ms a verified draft, November 19) and the engine's decode rate. Then the sums over the
code-like solo prompts (every solo prompt but the prose one) and the bursts.
"""
import json
import sys

D = "runs/draft-ladder-20261119"
A, B = 0.0247, 0.00358
args = [a for a in sys.argv[1:] if not a.startswith("--")]
second = sys.argv[sys.argv.index("--second-control") + 1] if "--second-control" in sys.argv else None
if second in args:
    args.remove(second)
ctl, arms = args[0], args[1:]


def load(a):
    return json.load(open(f"{D}/{a}/probe.json"))


def solo(rows):
    return [r for r in rows if "width" not in r]


def nums(r):
    R = float(r["rounds"]); acc, ver = (float(x) for x in r["accepted"].split("/")); T = float(r["completion_tokens"])
    return dict(R=R, V=ver, T=T, pred=A * R + B * ver, meas=T / float(r["tok_per_s"]), rate=float(r["tok_per_s"]), tpr=T / R)


P = {a: load(a) for a in [ctl] + arms + ([second] if second else [])}
c = solo(P[ctl])
print(f"control {ctl}" + (f", second control {second}" if second else "") + f"; arms {arms}")
print("\n== solo prompts: tokens a round | verified drafts a round | predicted time vs control | engine rate vs control   (* = reply differs)")
tot = {a: dict(pred=0.0, meas=0.0, R=0.0, V=0.0, T=0.0) for a in P}
worst = {a: (0.0, "") for a in P}
for i, r0 in enumerate(c):
    n0 = nums(r0)
    code = r0["label"] != "prose greedy"
    line = f"{r0['label']:15s}{'s' if r0.get('sampled') else ' '} ctl {n0['tpr']:.2f} {n0['V'] / n0['R']:.2f} {n0['rate']:6.1f}"
    for a in P:
        r = solo(P[a])[i]
        assert r["label"] == r0["label"] and r.get("seed") == r0.get("seed"), (a, i, r["label"])
        n = nums(r)
        if code:
            for k in tot[a]:
                tot[a][k] += n[k]
        if a == ctl:
            continue
        d = 100 * (n["rate"] / n0["rate"] - 1)
        if code and d < worst[a][0]:
            worst[a] = (d, r0["label"])
        star = "" if r["sha"] == r0["sha"] else "*"
        line += f" | {a} {n['tpr']:.2f} {n['V'] / n['R']:.2f} pred {100 * (n['pred'] / n0['pred'] - 1):+.1f}% rate {d:+.1f}%{star}"
    print(line)
print("\n== code-like solo prompts, summed")
t0 = tot[ctl]
for a in P:
    t = tot[a]
    print(f"{a:12s} rounds {t['R']:.0f} verified drafts {t['V']:.0f} tokens {t['T']:.0f} | predicted {t['pred']:.2f} s ({100 * (t['pred'] / t0['pred'] - 1):+.2f}%)"
          f" | measured {t['meas']:.2f} s ({100 * (t['meas'] / t0['meas'] - 1):+.2f}%) = {t['T'] / t['meas']:.1f} tok/s | worst prompt rate {worst[a][0]:+.1f}% {worst[a][1]}")
print("\n== bursts: aggregate tok/s by wall | tokens per stream per round | verified drafts per stream per round   (* = a reply differs)")
for i, r0 in enumerate(P[ctl]):
    if "width" not in r0:
        continue
    line = f"{r0['label']:18s} ctl {float(r0['agg_tok_s_wall']):.1f} {float(r0['tokens_per_stream_round']):.2f} {float(r0['drafted_per_stream_round']):.2f}"
    for a in P:
        if a == ctl:
            continue
        r = P[a][i]
        star = "" if str(r["sha"]) == str(r0["sha"]) else "*"
        line += f" | {a} {float(r['agg_tok_s_wall']):.1f} ({100 * (float(r['agg_tok_s_wall']) / float(r0['agg_tok_s_wall']) - 1):+.1f}%) {float(r['tokens_per_stream_round']):.2f} {float(r['drafted_per_stream_round']):.2f}{star}"
    print(line)
same = {a: sum(str(r["sha"]) == str(r0["sha"]) for r, r0 in zip(P[a], P[ctl])) for a in P if a != ctl}
print("\nreplies equal to the control's:", {a: f"{n} of {len(P[ctl])}" for a, n in same.items()})
