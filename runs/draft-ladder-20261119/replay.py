"""Read a logged arm (TF_DS_DRAFT_LADDER_LOG=1) and replay gate rules on it. From the recipe root:

  python3 -I runs/draft-ladder-20261119/replay.py <arm>

Each [ladder] line is one pass scored by the round after it: the rows that round kept, the pass's own survival
through the drafts before the governed one, its own confidence for the last draft, and whether the governed draft
was scored and right. Lines are assigned to the probe's solo requests by the engine's done lines.
"""
import collections
import json
import math
import re
import sys

D = "runs/draft-ladder-20261119"
arm = sys.argv[1]
rows = [r for r in json.load(open(f"{D}/{arm}/probe.json")) if "width" not in r]
lines = open(f"{D}/{arm}/engine.log", errors="replace").read().splitlines()
kv = re.compile(r"(\w+)=(\S+)")
want = {(r["sha"], r["rounds"], str(r["completion_tokens"])): r for r in rows}
seg, cur = {}, []
for l in lines:
    if "[ladder]" in l:
        d = dict(kv.findall(l))
        own = [float(x) for x in d["own"].split(",")]
        bits = int(d["scored"], 16)
        cur.append(dict(kept=int(d["kept"]), ran=int(d["ran"]), offered=int(d["offered"]), surv=float(d["surv"]), own=own,
                        scored=bool(bits & (1 << 6)), right=bool(bits & (1 << 22))))
    elif " done " in l:
        d = dict(kv.findall(l))
        k = (d.get("sha"), d.get("rounds"), d.get("tokens"))
        if k in want and want[k]["label"] not in seg:
            seg[want[k]["label"] + ("/s" if want[k].get("sampled") else "")] = cur
        cur = []
print(f"{arm}: {sum(len(v) for v in seg.values())} scored passes over {len(seg)} of {len(rows)} solo requests")


def sig(x):
    return 1 / (1 + math.exp(-x))


print("\n== passes that ran seven drafts: share where six landed | seventh right once six landed | the pass's own figures")
allp = []
for name, ps in seg.items():
    w = [p for p in ps if p["ran"] >= 7]
    six = [p for p in w if p["kept"] >= 7]
    sc = [p for p in w if p["scored"]]
    allp += [(name, p) for p in w]
    if not w:
        continue
    print(f"{name:17s} wide passes {len(w):3d} of {len(ps):3d} | six landed {100 * len(six) / len(w):3.0f}% (own survival mean {sum(p['surv'] for p in w) / len(w):.2f})"
          f" | seventh right {100 * sum(p['right'] for p in sc) / max(len(sc), 1):3.0f}% of {len(sc):3d} (own confidence mean {sum(sig(p['own'][1]) for p in sc) / max(len(sc), 1):.2f})")
print("\n== calibration over every wide pass: by the pass's own survival through six, how often six landed")
for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .9), (.9, 1.01)):
    b = [p for _, p in allp if lo <= p["surv"] < hi]
    if b:
        print(f"  survival {lo:.1f}-{min(hi, 1):.1f}: {len(b):4d} passes, mean {sum(p['surv'] for p in b) / len(b):.2f}, six landed {100 * sum(p['kept'] >= 7 for p in b) / len(b):3.0f}%")
print("== ... and by the pass's own confidence for the seventh draft, how often it was right once six had landed")
for lo, hi in ((0, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)):
    b = [p for _, p in allp if p["scored"] and lo <= sig(p["own"][1]) < hi]
    if b:
        print(f"  confidence {lo:.2f}-{min(hi, 1):.2f}: {len(b):4d} scored, right {100 * sum(p['right'] for p in b) / len(b):3.0f}%")

print("\n== replay: offer the seventh when survival through six x the stream's running share >= bar (share from 0.7, weight 1/16)")
print("   per bar: offered passes, landed, and net decode time saved against never offering it (3.58 ms a verified draft, the prompt's own ms a token)")
tok_ms = {r["label"] + ("/s" if r.get("sampled") else ""): 1000 / float(r["tok_per_s"]) for r in rows}
for bar in (0.30, 0.35, 0.40, 0.43, 0.46, 0.50, 0.55, 0.60):
    tot = collections.Counter(); per = []
    for name, ps in seg.items():
        rate, off, land = 0.7, 0, 0
        for p in ps:
            if p["ran"] >= 7:
                if p["surv"] * min(max(rate, 0.02), 0.98) >= bar:
                    off += 1
                    land += p["kept"] >= 7 and p["right"]
                if p["scored"]:
                    rate += ((1.0 if p["right"] else 0.0) - rate) / 16
        net = land * tok_ms[name] - off * 3.58
        tot.update(off=off, land=land); tot["net"] += net
        per.append((name, off, land, net))
    line = " ".join(f"{n.split()[0]}{'/s' if n.endswith('/s') else ''} {o}/{l} {net:+.0f}" for n, o, l, net in per if o)
    print(f"  bar {bar:.2f}: offered {tot['off']:4d} landed {tot['land']:4d} ({100 * tot['land'] / max(tot['off'], 1):.0f}%) net {tot['net'] / 1000:+.2f} s | {line[:230]}")
