"""The ladder campaign's final table. From the recipe root, on a copy of the arm directories:

  python3 -I runs/draft-ladder-20261119/final.py

Counts (rounds, verified drafts) are compared against m1-off, the new image with production's switches. Rates are
compared against production-fresh, the same probe on restored production in a process as young as the arms'.
Sums are over the 21 code-like solo prompts; each also has the per-prompt differences' mean and standard error.
"""
import json
import math
import os
import statistics as st

D = "runs/draft-ladder-20261119"
A, B = 0.0247, 0.00358
ARMS = [("t3-five", "five"), ("m2-shadow", "seven, 7th unused"), ("t2-seven", "seven"), ("t1-ladder", "ladder 0.43")]


def load(a):
    p = f"{D}/{a}/probe.json"
    return json.load(open(p)) if os.path.exists(p) else None


def solo(rows):
    return [r for r in rows if "width" not in r]


def n(r):
    R = float(r["rounds"]); acc, ver = (float(x) for x in r["accepted"].split("/")); T = float(r["completion_tokens"])
    return dict(R=R, V=ver, T=T, pred=A * R + B * ver, meas=T / float(r["tok_per_s"]))


def code(rows):
    return [r for r in solo(rows) if r["label"] != "prose greedy"]


ctl, fresh = load("m1-off"), load("production-fresh")
print("== proofs")
if fresh:
    same = sum((a["rounds"], a["accepted"], a["sha"]) == (b["rounds"], b["accepted"], b["sha"]) for a, b in zip(solo(ctl), solo(fresh)))
    print(f"production-fresh against m1-off: {same} of {len(solo(ctl))} solo rows equal in rounds, verified drafts and hash")
s04 = json.load(open("runs/suite-20261119/blocks/s04.probe.json"))
t2 = load("t2-seven")
if t2:
    ref = {(r["label"], r.get("seed")): r for r in s04 if "rounds" in r}
    hit = [(r["rounds"], r["accepted"], r["sha"]) == (ref[k]["rounds"], ref[k]["accepted"], ref[k]["sha"]) for r in solo(t2) if (k := (r["label"], r.get("seed"))) in ref]
    print(f"plain seven against the suite's block-7 arm: {sum(hit)} of {len(hit)} shared solo rows equal")
t3 = load("t3-five")
if t3:
    off = json.load(open("runs/draft-block-20261118/d0-off/probe.json"))
    ref = {(r["label"], r.get("seed")): r for r in off if "rounds" in r}
    hit = [(r["rounds"], r["accepted"], r["sha"]) == (ref[k]["rounds"], ref[k]["accepted"], ref[k]["sha"]) for r in solo(t3) if (k := (r["label"], r.get("seed"))) in ref]
    print(f"five against the draft-block campaign's switch-off arm: {sum(hit)} of {len(hit)} shared solo rows equal")
t1 = load("t1-ladder")
if t1 and t2:
    d = sum(a["accepted"].split("/")[1] != b["accepted"].split("/")[1] for a, b in zip(solo(t1), solo(t2)))
    print(f"ladder against plain seven: verified drafts differ on {d} of {len(solo(t1))} solo rows")
for a, _ in ARMS:
    rows = load(a)
    if rows:
        print(f"{a}: replies equal to the control's {sum(str(r['sha']) == str(r0['sha']) for r, r0 in zip(rows, ctl))} of {len(ctl)}")

print("\n== sums over the 21 code-like solo prompts")
c = code(ctl); f = code(fresh) if fresh else None
tc = {k: sum(n(r)[k] for r in c) for k in ("R", "V", "T", "pred", "meas")}
tf = {k: sum(n(r)[k] for r in f) for k in ("meas",)} if f else None
print(f"{'six (control counts)':22s} rounds {tc['R']:.0f}  verified drafts {tc['V']:.0f}  tokens/round {tc['T'] / tc['R']:.3f}  predicted {tc['pred']:.1f} s"
      + (f"  measured, fresh production {tf['meas']:.1f} s = {tc['T'] / tf['meas']:.1f} tok/s (the older m1-off process: {tc['meas']:.1f} s)" if tf else ""))
res = {}
for a, name in ARMS:
    rows = load(a)
    if not rows:
        continue
    x = code(rows)
    t = {k: sum(n(r)[k] for r in x) for k in ("R", "V", "T", "pred", "meas")}
    dp = [100 * (n(r)["pred"] / n(r0)["pred"] - 1) for r, r0 in zip(x, c)]
    line = (f"{name:22s} rounds {t['R']:.0f} ({100 * (t['R'] / tc['R'] - 1):+.1f}%)  verified {t['V']:.0f} ({100 * (t['V'] / tc['V'] - 1):+.1f}%)  tokens/round {t['T'] / t['R']:.3f}"
            f"  predicted {100 * (t['pred'] / tc['pred'] - 1):+.2f}% (per prompt {st.mean(dp):+.2f} ± {st.stdev(dp) / math.sqrt(len(dp)):.2f}, slower on {sum(v > 0 for v in dp)} of {len(dp)})")
    if f:
        dm = [100 * (n(r)["meas"] / n(r0)["meas"] - 1) for r, r0 in zip(x, f)]
        line += f"  measured {100 * (t['meas'] / tf['meas'] - 1):+.2f}% (per prompt {st.mean(dm):+.2f} ± {st.stdev(dm) / math.sqrt(len(dm)):.2f}) = {t['T'] / t['meas']:.1f} tok/s"
    res[a] = (x, t)
    print(line)
if "t1-ladder" in res and "t2-seven" in res:
    x1, x2 = res["t1-ladder"][0], res["t2-seven"][0]
    dp = [100 * (n(a)["pred"] / n(b)["pred"] - 1) for a, b in zip(x1, x2)]
    dm = [100 * (n(a)["meas"] / n(b)["meas"] - 1) for a, b in zip(x1, x2)]
    print(f"ladder against plain seven, paired by prompt: predicted {st.mean(dp):+.2f} ± {st.stdev(dp) / math.sqrt(len(dp)):.2f}%, measured {st.mean(dm):+.2f} ± {st.stdev(dm) / math.sqrt(len(dm)):.2f}%;"
          f" verified drafts {res['t1-ladder'][1]['V']:.0f} against {res['t2-seven'][1]['V']:.0f}")

print("\n== per prompt: measured decode time against fresh production (negative is faster)" if f else "\n== per prompt: predicted time against control")
base = f if f else c
key = "meas" if f else "pred"
for i, r0 in enumerate(base):
    line = f"{r0['label']:15s}{'s' if r0.get('sampled') else ' '} six {r0['tok_per_s']:>6s} tok/s"
    for a, name in ARMS:
        if a in res:
            line += f" | {name} {100 * (n(res[a][0][i])[key] / n(r0)[key] - 1):+5.1f}%"
    print(line)
print("\n== prose and bursts: rate against " + ("fresh production" if fresh else "control"))
b0 = fresh if fresh else ctl
for i, r0 in enumerate(b0):
    if "width" not in r0 and r0["label"] != "prose greedy":
        continue
    v0 = float(r0["agg_tok_s_wall"]) if "width" in r0 else float(r0["tok_per_s"])
    line = f"{r0['label']:18s} six {v0:6.1f}"
    for a, name in ARMS:
        rows = load(a)
        if rows:
            v = float(rows[i]["agg_tok_s_wall"]) if "width" in r0 else float(rows[i]["tok_per_s"])
            line += f" | {name} {v:6.1f} ({100 * (v / v0 - 1):+.1f}%)"
    print(line)
