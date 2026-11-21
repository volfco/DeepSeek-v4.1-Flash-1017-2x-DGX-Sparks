"""Per-block table for suite-fused-next-20261121, with the previous suite's fused blocks (production's config today).
Run from the recipe root: python3 -I runs/suite-fused-next-20261121/table.py"""
import json, statistics as st, sys
sys.argv = ["x"]
S, P = "runs/suite-fused-next-20261121", "runs/suite-fused-20261121"
src = open(f"{S}/compare.py").read()
ns = {}
exec(src.split("B, F = arms")[0].replace('S = "runs/suite-fused-20261121"', f'S = "{S}"'), ns)  # helpers only
blocks = {}
for pre, d in ((S, "now"), (P, "prev")):
    for b in ("s01", "s02", "s03", "s04", "s05"):
        try: blocks[f"{d}:{b}"] = json.load(open(f"{pre}/blocks/{b}.json"))
        except FileNotFoundError: pass
pre, dec, rms = ns["prefill"], ns["decode"], ns["round_ms"]
med = lambda x: st.median(x) if x else float("nan")
def probe_solo(d):
    rows = d["cells"].get("probe", {}).get("rows", [])
    t = s = 0.0
    for r in rows:
        if "rounds" in r:
            t += float(r["completion_tokens"]); s += float(r["completion_tokens"]) / float(r["tok_per_s"])
    return t / s if s else float("nan")
def deep(d):
    return med(pre([{"cells": {"d": d["cells"].get("deep300", {})}, "done_lines": d.get("done_lines", {})}], "d", "300000"))
M = [("probe solo tok/s", probe_solo)]
for c in ("dec1", "dec4", "dec8"):
    M.append((f"{c} tok/s", lambda d, c=c: med(dec([d], c))))
for c in ("dec1", "dec4", "dec8"):
    M.append((f"{c} ms/round", lambda d, c=c: rms([d], c)[0]))
for k in ("t1", "t4"):
    M.append((f"think {k} tok/s", lambda d, k=k: med(dec([d], "think", k))))
for k in ("lc1", "lc4"):
    M.append((f"longctx {k} tok/s", lambda d, k=k: med(dec([d], "longctx", k))))
for sz in ("8000", "32000", "100000"):
    M.append((f"prefill {int(sz)//1000}K", lambda d, sz=sz: med(pre([d], "prefill_div", sz))))
M.append(("prefill 300K", deep))
pct = lambda a, b: f"{100*(a/b-1):+.2f}%"
cols = ["now:s01", "now:s02", "now:s04", "now:s03", "now:s05", "prev:s02", "prev:s03"]
print("| Metric | s01 base | s02 next | A | s04 base | s03 next | B | base-base | s05 next+PDL | s05 vs base mean | prev fused s02/s03 |")
print("| --- " + "| ---: " * 10 + "|")
for name, f in M:
    v = {c: f(blocks[c]) for c in cols if c in blocks}
    bm = (v["now:s01"] + v["now:s04"]) / 2
    print(f"| {name} | {v['now:s01']:.2f} | {v['now:s02']:.2f} | {pct(v['now:s02'], v['now:s01'])} | {v['now:s04']:.2f} | {v['now:s03']:.2f} | "
          f"{pct(v['now:s03'], v['now:s04'])} | {pct(v['now:s04'], v['now:s01'])} | {v['now:s05']:.2f} | {pct(v['now:s05'], bm)} | "
          f"{v['prev:s02']:.2f} / {v['prev:s03']:.2f} |")
print()
for c in cols[:5]:
    d = blocks[c]
    print(c, d["arm"], d.get("image"), d.get("status"), "arm_proof", d.get("arm_proof_problems"), "cells", len(d["cells"]), "/", len(d["cells_planned"]),
          "validate", d["cells"].get("validate", {}).get("rc"))
